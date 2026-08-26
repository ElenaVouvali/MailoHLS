"""Materialize one-pass clock-selector features from structural memory packs."""
import argparse, json
from pathlib import Path
import torch
from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES


def pooled_structural_memory(pack):
    kv = pack["node_embs"].float(); mask = pack["node_embs_mask"].bool()
    w = mask.to(kv.dtype).unsqueeze(-1)
    return (kv * w).sum(0) / w.sum().clamp_min(1.0)


def make_features(cases, memory_dir):
    examples = []
    for case in cases:
        path = Path(memory_dir) / f"{case['kernel']}.memory.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing required structural memory: {path}")
        vec = pooled_structural_memory(torch.load(path, map_location="cpu", weights_only=False))
        budget = case.get("resource_budget", {})
        vals = [float(budget.get(k, 0.0)) for k in ("bram", "dsp", "ff", "lut")]
        capacities = DEVICE_RESOURCES.get(case['device'])
        if capacities is None: raise ValueError(f"Unknown device capacity: {case['device']}")
        caps = [__import__('math').log1p(float(capacities[k])) for k in ('BRAM_18K','DSP','FF','LUT')]
        features = torch.stack([torch.cat((vec, torch.tensor(
            vals + caps + [__import__('math').log2(float(clock) / 5.0)], dtype=torch.float32
        ))) for clock in case["available_clock_periods"]])
        examples.append({"features": features,
                         "label": case["available_clock_periods"].index(case["gold_clock_period"]),
                         "clocks": case["available_clock_periods"], "case": case})
    if not examples: raise ValueError("No cases matched memory packs")
    return examples


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases_jsonl", required=True); ap.add_argument("--memory_dir", required=True); ap.add_argument("--output", required=True)
    a=ap.parse_args(); cases=[json.loads(x) for x in open(a.cases_jsonl) if x.strip()]
    torch.save(make_features(cases, a.memory_dir), a.output); print(f"saved {a.output}")
if __name__ == "__main__": main()
