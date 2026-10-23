#!/usr/bin/env bash
set -u
set -o pipefail

OUT=/home/ubuntu/runs/serrano_holdout/final_ablation_all_stages
python - "${OUT}" <<'PY'
import json, os, glob, sys

out_dir = sys.argv[1]
pred_dir = os.path.join(out_dir, "preds")
out_path = os.path.join(out_dir, "all_stages_all_objectives_9_for_hls.jsonl")

files = sorted(glob.glob(os.path.join(pred_dir, "*.jsonl")))

def stage_from_name(fn):
    b = os.path.basename(fn)
    if "_stage1_" in b:
        return "stage1"
    if "_stage2_" in b:
        return "stage2"
    if "_stage3_" in b:
        return "stage3"
    raise ValueError(f"Cannot infer stage from filename: {b}")

with open(out_path, "w", encoding="utf-8") as g:
    for fn in files:
        stage = stage_from_name(fn)
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cand = row["candidates"][0]
                out_row = {
                    "kernel_name": row["kernel_name"],
                    "obj_mode": row["obj_mode"],
                    "stage": stage,
                    "prediction": cand["canonical_prediction"],
                }
                g.write(json.dumps(out_row, ensure_ascii=False) + "\n")

print(f"Wrote synthesis-ready ablation file -> {out_path}")
PY

wc -l "${OUT}/cases/"*_one_per_kernel.jsonl
wc -l "${OUT}/all_stages_all_objectives_9_verbose.jsonl"
wc -l "${OUT}/all_stages_all_objectives_9_for_hls.jsonl"

