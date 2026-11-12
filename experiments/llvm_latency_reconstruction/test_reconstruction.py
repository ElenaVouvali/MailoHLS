import json
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LLM_branch/train"))
from prepare import materialize
from record_checkpoint import record
import torch
import train_SFT_xattn as original


class ReconstructionTests(unittest.TestCase):
    def test_measured_target_filter_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "memory").mkdir()
            (p / "memory/k.memory.pt").write_bytes(b"fixture-identity-only")
            source = "/*L1:*/ for(;;) {\n#pragma HLS pipeline II=auto{_PIPE_L1}\n}"
            (p / "sources.json").write_text(json.dumps({"templates": {"k": source}}))
            r = dict(device="xczu7ev-ffvc1156-2-e", clock_period=10, kernel_name="k",
                     source_key="k", latency=3, area=4, target="auto{_PIPE_L1} = 1")
            (p / "in.jsonl").write_text("\n".join(json.dumps(x) for x in
                [r, dict(r, device="xcu200-fsgd2104-2-e"), dict(r, latency=0)]) + "\n")
            report = materialize(p / "in.jsonl", p / "sources.json", p / "memory", p / "out.jsonl")
            self.assertEqual(report["examples"], 1)
            out = json.loads((p / "out.jsonl").read_text())
            self.assertEqual(out["input"], source)
            self.assertEqual((out["latency"], out["area"]), (3, 4))

    def test_absolute_target_routing_and_prompt_exclusion(self):
        ids = torch.tensor([[101, 7, 103, 8, 102]])
        slots = original.build_placeholder_slot_ids(ids, [101, 102, 103], torch.tensor([2]))
        self.assertEqual(slots.tolist(), [[0, 0, 3, 0, 2]])
        self.assertEqual(original.forward_fill_slot_ids(slots).tolist(), [[0, 0, 3, 3, 2]])

    def test_zero_gate_parity_and_invalid_memory(self):
        torch.manual_seed(123)
        layer = original.GatedCrossAttentionBlock(dim=8, dim_memory=64,
            heads=2, dim_head=4, enable_ff=False, attn_gate_init=0.0)
        x, memory = torch.randn(1, 3, 8), torch.randn(1, 64, 64)
        out = layer(x, memory, placeholder_slot_ids=torch.tensor([[1, 0, 0]]),
                    memory_mask=torch.ones(1, 64, dtype=torch.bool))
        torch.testing.assert_close(x, out, rtol=0, atol=0)
        layer.attn_gate.data.fill_(1)
        out = layer(x, memory, placeholder_slot_ids=torch.tensor([[1, 0, 0]]),
                    memory_mask=torch.zeros(1, 64, dtype=torch.bool))
        torch.testing.assert_close(x, out, rtol=0, atol=0)

    def test_complete_causal_rhs_includes_newline(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [{"1": 1, "2": 2, "\n": 0}[c] for c in text]}
        class Model:
            def __call__(self, input_ids, **kwargs):
                logits = torch.zeros(1, input_ids.shape[1], 6)
                for pos, target in [(1, 1), (2, 2), (3, 0)]:
                    logits[0, pos, target] = 3 + pos
                return SimpleNamespace(logits=logits)
        result = original.score_rhs_candidate_suffix(model=Model(), tok=Tokenizer(),
            base_input_ids=torch.tensor([[4, 5]]), base_attention_mask=torch.ones(1, 2),
            candidate_text="12\n")
        expected = torch.stack([torch.log_softmax(torch.tensor([v] + [0.] * 5), 0)[0]
                                for v in [4., 5., 6.]])
        self.assertAlmostEqual(result["mean_logprob"], expected.mean().item(), places=6)
        self.assertAlmostEqual(result["sum_logprob"], expected.sum().item(), places=6)

    def test_untrained_output_cannot_be_marked_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                record(Path(tmp))

    def test_missing_fusion_checkpoint_fails(self):
        with self.assertRaises(FileNotFoundError):
            original.load_partial_harp_xattn(None, "/unavailable/trained_fusion.pt", "TEST")

    def test_resumed_selection_preserves_previous_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            best = Path(tmp) / "best_custom_stage2"
            best.mkdir()
            (best / "best_selection_metrics.json").write_text(json.dumps(
                {"selection_score": .75, "step": 40}))
            callback = original.StageValSelectionCallback(None, [], {}, tmp,
                best_dir_name="best_custom_stage2")
            self.assertEqual((callback.best_score, callback.best_step), (.75, 40))

    def test_numpy_rng_checkpoint_restores_with_weights_only_loading(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        try:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            state = dict(python=random.getstate(), numpy=np.random.get_state(),
                         cpu=torch.get_rng_state())
            expected = (random.random(), np.random.rand(), torch.rand(3))
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(state, Path(tmp) / "rng_state.pth")
                trainer = object.__new__(original.LengthGroupedTrainer)
                trainer.args = SimpleNamespace(world_size=1, parallel_mode=None)
                with patch("torch.cuda.is_available", return_value=False):
                    trainer._load_rng_state(tmp)
                self.assertEqual(random.random(), expected[0])
                self.assertEqual(np.random.rand(), expected[1])
                torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        finally:
            random.setstate(before[0])
            np.random.set_state(before[1])
            torch.set_rng_state(before[2])

    def test_wrapped_peft_checkpoint_restores_after_parent_installation(self):
        from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
        config = LlamaConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        base = LlamaForCausalLM(config)
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        parent = original.get_peft_model(base, original.LoraConfig(r=2, lora_alpha=4,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        for name, value in parent.named_parameters():
            if "lora_" in name:
                value.data.uniform_(-.2, .2)

        def wrap(model):
            original.extend_instance(model, original.HARPLMMixin)
            model.set_decoder_layers_attr_name(original.infer_decoder_layers_attr_name(model))
            model.init_harp_flamingo(placeholder_token_ids=list(range(10, 74)),
                lang_hidden_size=16, mem_hidden_size=64, cross_attn_every_n_layers=1,
                gradient_checkpointing=False, xattn_heads=2, xattn_dim_head=4,
                xattn_ff_mult=1, only_attend_immediate_memory=True, mask_mode="segment")
            return model

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            parent.save_pretrained(folder / "parent")
            trained = wrap(parent)
            trained.save_pretrained(folder / "checkpoint")
            torch.save(original.get_harp_xattn_state_dict(trained), folder / "checkpoint/harp_xattn.pt")
            expected = {k: v.clone() for k, v in trained.state_dict().items()
                        if "lora_" in k or "gated_cross_attn_layer" in k}
            fresh_base = LlamaForCausalLM(config)
            fresh_base.load_state_dict(base_state, strict=True)
            resumed = wrap(original.PeftModel.from_pretrained(fresh_base, folder / "parent", is_trainable=True))
            original.load_partial_harp_xattn(resumed, str(folder / "checkpoint/harp_xattn.pt"), "TEST")
            trainer = original.Trainer(model=resumed, args=TrainingArguments(
                output_dir=str(folder / "trainer"), use_cpu=True, report_to="none"))
            trainer._load_from_checkpoint(str(folder / "checkpoint"))
            for key, value in expected.items():
                torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
