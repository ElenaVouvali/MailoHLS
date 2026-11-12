import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

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


if __name__ == "__main__":
    unittest.main()
