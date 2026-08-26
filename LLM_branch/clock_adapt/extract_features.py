"""Materialize tiny-selector features from structural memory packs.

The optional ``frozen_lm_scores`` field is produced by the expensive decoder
front-end; keeping it in this cache means selector training never updates or
re-runs the 6.7B model.
"""
import argparse, json
from pathlib import Path
import torch


def pooled_structural_memory(pack):
    kv = pack["node_embs"].float(); mask = pack["node_embs_mask"].bool()
    w = mask.to(kv.dtype).unsqueeze(-1)
    return (kv * w).sum(0) / w.sum().clamp_min(1.0)


def make_features(cases, memory_dir, default_scores=None):
    examples = []
    for case in cases:
        path = Path(memory_dir) / f"{case['kernel']}.memory.pt"
        if not path.exists(): continue
        vec = pooled_structural_memory(torch.load(path, map_location="cpu", weights_only=False))
        budget = case.get("resource_budget", {})
        vals = [float(budget.get(k, 0.0)) for k in ("lut", "ff", "dsp", "bram")]
        caps = [1.0, 1.0, 1.0, 1.0]
        features = torch.stack([torch.cat((vec, torch.tensor(
            vals + caps + [__import__('math').log2(float(clock) / 5.0)], dtype=torch.float32
        ))) for clock in case["available_clock_periods"]])
        scores = torch.tensor(case.get("frozen_lm_scores", default_scores or [0.0] * len(case["available_clock_periods"])), dtype=torch.float32)
        examples.append({"features": features, "scores": scores,
                         "label": case["available_clock_periods"].index(case["gold_clock_period"]),
                         "clocks": case["available_clock_periods"], "case": case})
    if not examples: raise ValueError("No cases matched memory packs")
    return examples


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases_jsonl", required=True); ap.add_argument("--memory_dir", required=True); ap.add_argument("--output", required=True)
    a=ap.parse_args(); cases=[json.loads(x) for x in open(a.cases_jsonl) if x.strip()]
    torch.save(make_features(cases, a.memory_dir), a.output); print(f"saved {a.output}")
if __name__ == "__main__": main()
