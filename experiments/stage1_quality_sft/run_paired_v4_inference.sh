#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
OUT="${1:-$ROOT/artifacts/stage1_quality_sft/v4_paired_20261105}"
GPU_UUID="${GPU_UUID:-GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35}"
PYTHON="${STAGE1_PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
REQUESTS="$OUT/frozen_validation_requests.jsonl"
POLICY="$ROOT/experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy.json"
V4_ROOT="${V4_ROOT:-/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4}"
PROD="$ROOT/mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1"

if [[ ! -s "$REQUESTS" ]]; then
  echo "Missing authenticated frozen request file: $REQUESTS" >&2
  exit 2
fi

run_one() {
  local name="$1" adapter="$2" registry="$3" mode="$4"
  local args=(--stage stage1 --adapter_dir "$adapter"
    --directive_domain_registry_json "$registry"
    --application_dataset_dir "$ROOT/Data/ApplicationDataset"
    --input_jsonl "$REQUESTS" --output_jsonl "$OUT/predictions/${name}_${mode}.jsonl"
    --num_samples 1 --print_predictions)
  if [[ "$mode" == source_tool_v1 ]]; then
    args+=(--admissibility_policy source_tool_v1 --tool_policy_json "$POLICY")
  fi
  echo "[RUN] model=$name policy=$mode adapter=$adapter gpu=$GPU_UUID"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT" \
    "$PYTHON" -u -m LLM_branch.inference.eval_stage1_stage2_stage3 "${args[@]}" \
    >"$OUT/predictions/${name}_${mode}.log" 2>&1
  echo "[DONE] $OUT/predictions/${name}_${mode}.jsonl"
}

mkdir -p "$OUT/predictions"
for arm in single_reference multi_reference; do
  adapter="$V4_ROOT/$arm/run"
  run_one "$arm" "$adapter" "$adapter/directive_domain_registry.json" legacy
  run_one "$arm" "$adapter" "$adapter/directive_domain_registry.json" source_tool_v1
done
run_one production_latency "$PROD" "$ROOT/mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json" legacy
run_one production_latency "$PROD" "$ROOT/mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json" source_tool_v1
