#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
OUT="${1:-$ROOT/artifacts/stage1_quality_sft/v4_paired_20261105}"
GPU_UUID="${GPU_UUID:-GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb}"
PYTHON="${STAGE1_PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
REQUESTS="$OUT/frozen_validation_requests.jsonl"
POLICY="$ROOT/experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json"
V4_ROOT="${V4_ROOT:-/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4}"
PROD="$ROOT/mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1"

[[ -s "$REQUESTS" ]] || { echo "Missing frozen request file: $REQUESTS" >&2; exit 2; }
mkdir -p "$OUT/predictions"
run_one() {
  local name="$1" adapter="$2" registry="$3"
  local output="$OUT/predictions/${name}_source_tool_v2.jsonl"
  [[ ! -e "$output" && ! -e "$OUT/predictions/${name}_source_tool_v2.log" ]] || {
    echo "Refusing to overwrite prior v2 output: $output" >&2; exit 3;
  }
  CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT" \
    "$PYTHON" -u -m LLM_branch.inference.eval_stage1_stage2_stage3 \
    --stage stage1 --adapter_dir "$adapter" --directive_domain_registry_json "$registry" \
    --application_dataset_dir "$ROOT/Data/ApplicationDataset" --input_jsonl "$REQUESTS" \
    --output_jsonl "$output" --num_samples 1 --print_predictions \
    --admissibility_policy source_tool_v2 --tool_policy_json "$POLICY" \
    >"$OUT/predictions/${name}_source_tool_v2.log" 2>&1
}

run_one single_reference "$V4_ROOT/single_reference/run" "$V4_ROOT/single_reference/run/directive_domain_registry.json"
run_one multi_reference "$V4_ROOT/multi_reference/run" "$V4_ROOT/multi_reference/run/directive_domain_registry.json"
run_one production_latency "$PROD" "$ROOT/mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json"
