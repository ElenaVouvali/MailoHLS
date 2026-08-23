#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

seed="${MAILOHLS_SEED:-123}"
dataset="${MAILOHLS_SFT_DATASET:-$repo_root/artifacts/llm/mailohls_sft.jsonl}"
split="${MAILOHLS_FAMILY_SPLIT:-$repo_root/mailohls_runs/mailohls_final_family_split_s${seed}.json}"
validation="${MAILOHLS_VALIDATION_KERNELS:-machsuite-sort-radix,machsuite-viterbi,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0}"
test="${MAILOHLS_TEST_KERNELS:-serrano-kalman-filter}"
canonical_device="${MAILOHLS_GNN_TARGET_DEVICE:-xczu7ev-ffvc1156-2-e}"
canonical_clock="${MAILOHLS_GNN_CLOCK_NS:-10}"

python -u Preprocessing/data_preprocess.py --mode llm --force
python -u Preprocessing/create_jsonl.py --output "$dataset" --force
python -u -m Preprocessing.build_family_split \
  --dataset_jsonl "$dataset" \
  --output_json "$split" \
  --seed "$seed" \
  --val_kernels "$validation" \
  --test_kernels "$test"
python -u Preprocessing/data_preprocess.py \
  --mode gnn \
  --device "$canonical_device" \
  --clock-period-ns "$canonical_clock" \
  --exclude-kernels "$test" \
  --force

printf 'Shared final-experiment split: %s\n' "$split"
