#!/usr/bin/env bash
set -euo pipefail
: "${U50_ADAPT_ROOT:?Set U50_ADAPT_ROOT to the adaptation root}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
cd "$repo_dir"

"$python_bin" -m Preprocessing.data_preprocess --mode llm \
  --input-dir "$U50_ADAPT_ROOT/raw_csv" \
  --output-dir "$U50_ADAPT_ROOT/preprocessed" --force

"$python_bin" -m Preprocessing.create_jsonl \
  --preprocessed-dir "$U50_ADAPT_ROOT/preprocessed" \
  --output "$U50_ADAPT_ROOT/artifacts/u50_10ns_sft.jsonl" --force

"$python_bin" -m Preprocessing.build_family_split \
  --dataset_jsonl "$U50_ADAPT_ROOT/artifacts/u50_10ns_sft.jsonl" \
  --output_json "$U50_ADAPT_ROOT/splits/u50_adaptation_split.json" \
  --seed 123 \
  --val_kernels 'machsuite-sort-radix,machsuite-viterbi,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0' \
  --test_kernels 'spcl_example_03'

echo "Prepared dataset and family split under $U50_ADAPT_ROOT"
