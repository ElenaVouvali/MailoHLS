"""CPU integration tests with real PEFT, token deltas, and a tiny Llama."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from LLM_branch.common import device_adaptation as da


class DeviceAdaptationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        self.base = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        self.ids = torch.tensor([[1, 30, 2, 31]])

    def adapted(self):
        model = get_peft_model(copy.deepcopy(self.base), LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.1, target_modules=['q_proj', 'v_proj'],
            trainable_token_indices=[30, 31], task_type='CAUSAL_LM'), adapter_name='mailohls_base')
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'lora_B' in name or 'trainable_tokens_delta' in name:
                    p.add_(torch.randn_like(p) * 0.1)
        model.eval()
        before = model(self.ids).logits.detach()
        model.add_adapter('device_adapt', LoraConfig(r=2, lora_alpha=4,
                          target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        da.activate(model, trainable=True)
        model.eval()
        torch.testing.assert_close(before, model(self.ids).logits, rtol=0, atol=0)
        return model

    def contract(self, model):
        return {'stage': 'stage1', 'device_mode': 'device_adapt', da.KEY: da.trainables_contract(model),
                'device_adaptation': {'device': 'xcu50-fsvh2104-2-e',
                    'device_capacities': {'BRAM_18K': 2688, 'DSP': 5952, 'FF': 1743360, 'LUT': 871680},
                    'parent': {'adapter_sha256': 'test'}}}

    def test_gradient_freeze_save_reload_and_resume(self):
        model = self.adapted()
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model.train()
        model(self.ids, labels=self.ids).loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if da.residual_parameter(n)))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n in frozen))
        optimizer.step()
        for n, p in model.named_parameters():
            if n in frozen:
                torch.testing.assert_close(frozen[n], p, rtol=0, atol=0)
        model.eval()
        expected = model(self.ids).logits.detach()
        with tempfile.TemporaryDirectory() as tmp:
            contract = self.contract(model)
            model.save_pretrained(tmp, save_embedding_layers=False)
            (Path(tmp) / 'training_contract.json').write_text(json.dumps(contract))
            da.save_bundle_manifest(tmp, contract)
            loaded = da.load_bundle(copy.deepcopy(self.base), tmp)
            loaded.eval()
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            torch.testing.assert_close(expected, loaded(self.ids).logits, rtol=0, atol=0)
            resumed = da.load_bundle(copy.deepcopy(self.base), tmp, trainable=True)
            self.assertEqual(da.trainables_contract(resumed), contract[da.KEY])
            (Path(tmp) / 'device_adapt/adapter_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                da.load_bundle(copy.deepcopy(self.base), tmp)

    def test_reject_unfrozen_parent(self):
        model = self.adapted()
        next(p for n, p in model.named_parameters() if '.lora_A.mailohls_base.' in n).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'only'):
            da.trainables_contract(model)


if __name__ == '__main__':
    unittest.main()
