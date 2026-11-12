"""Reproduce the split, complete-context and frozen action-memory coverage audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LLM_branch/train"))
    import torch
    from transformers import AutoTokenizer
    import train_SFT_xattn as s
    from prepare import sha256

    rows = s.load_rows(str(a.runtime / "dataset.jsonl"))
    counts = Counter((r["kernel_name"], r["target"]) for r in rows)
    if max(counts.values()) != 1:
        raise ValueError("Duplicate assignments require grouping before a design split")
    parts = s.split_rows_random_design(rows, .1, .1, 123, True)
    split_path = a.runtime / "split.json"
    if split_path.exists():
        old = s.load_split_spec(str(split_path))
        actual = s.apply_split_spec(rows, old)
        if [[r["_jsonl_idx"] for r in x] for x in actual] != [[r["_jsonl_idx"] for r in x] for x in parts]:
            raise ValueError("Existing split does not match the specified protocol")
    else:
        s.save_split_spec(str(split_path), *parts)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.add_special_tokens({"additional_special_tokens":
        [g["token"] for g in s.GOALS.values()] + s.SOURCE_PLACEHOLDER_TOKENS + s.TARGET_PLACEHOLDER_TOKENS})
    report = dict(split_counts=list(map(len, parts)), selected_counts=[], lengths={},
                  memory_coverage={}, missing_slots={}, duplicate_assignments=0,
                  dataset_sha256=sha256(a.runtime / "dataset.jsonl"),
                  split_sha256=sha256(split_path), max_length=7168)
    for name, part in zip(["train", "validation", "test"], parts):
        selected, _ = s.select_goal_rows(part, "PARETO_LATENCY_EXTREME", 6, .25, .12, .6, 1.0)
        report["selected_counts"].append(len(selected))
        lengths = []
        for r in selected:
            prefix = tok(s.build_prompt(r["input"], r["obj_mode"]), add_special_tokens=False)["input_ids"]
            pack = s.build_deterministic_rhs_pack(r["input"], r["target"], tok)
            lengths.append((len(prefix) + len(pack.input_ids), r["kernel_name"]))
        if max(n for n, _ in lengths) > report["max_length"]:
            raise ValueError("Selected source/target would be truncated")
        report["lengths"][name] = sorted(lengths, reverse=True)[:10]
    for r in {x["kernel_name"]: x for x in rows}.values():
        pack = torch.load(a.runtime / "memory" / (r["kernel_name"] + ".memory.pt"),
                          map_location="cpu", weights_only=False)
        vectors, mask = pack["node_embs"], pack["node_embs_mask"]
        if tuple(vectors.shape) != (64, 64) or tuple(mask.shape) != (64,) or not torch.isfinite(vectors).all():
            raise ValueError("Invalid original LLVM memory")
        labels = {int(x) for x in re.findall(r"_L(\d+)", r["target"])}
        missing = sorted(x for x in labels if not mask[x-1])
        for i in mask.nonzero().flatten().tolist():
            if int(pack["labels"][i]) != i+1:
                raise ValueError("Original memory violates absolute-slot identity")
        report["memory_coverage"][r["kernel_name"]] = dict(required=len(labels), available=int(mask.sum()), missing=missing)
        if missing:
            report["missing_slots"][r["kernel_name"]] = missing
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["missing_slots"]:
        raise ValueError("Incomplete compiler-to-action bindings")
    print(json.dumps({k: report[k] for k in ("split_counts", "selected_counts", "missing_slots", "dataset_sha256", "split_sha256")}))


if __name__ == "__main__":
    main()
