#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
OUT="${1:-$ROOT/artifacts/stage1_quality_sft/v4_paired_20261105/diagnostics}"
GPU_UUID="${GPU_UUID:-GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb}"
PYTHON="${STAGE1_PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
V4_ROOT="${V4_ROOT:-/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4}"
POLICY="$ROOT/experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy.json"
POLICY_V2="$ROOT/experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json"
GAN="$ROOT/artifacts/inference/quality_v4_gan_20261105_6cases"
KALMAN="$ROOT/artifacts/inference/quality_v4_kalman_20261104_6cases"

mkdir -p "$OUT"
run_pair() {
  local name="$1" requests="$2" app_root="$3"
  for arm in single_reference multi_reference; do
    local adapter="$V4_ROOT/$arm/run"
    for mode in legacy source_tool_v1 source_tool_v2; do
      local args=(--stage stage1 --adapter_dir "$adapter"
        --directive_domain_registry_json "$adapter/directive_domain_registry.json"
        --application_dataset_dir "$app_root" --input_jsonl "$requests"
        --output_jsonl "$OUT/${name}_${arm}_${mode}.jsonl" --num_samples 1 --print_predictions)
      if [[ "$mode" == source_tool_v1 ]]; then
        args+=(--admissibility_policy source_tool_v1 --tool_policy_json "$POLICY")
      elif [[ "$mode" == source_tool_v2 ]]; then
        args+=(--admissibility_policy source_tool_v2 --tool_policy_json "$POLICY_V2")
      fi
      CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT" \
        "$PYTHON" -u -m LLM_branch.inference.eval_stage1_stage2_stage3 "${args[@]}" \
        >"$OUT/${name}_${arm}_${mode}.log" 2>&1
    done
  done
}

run_pair gan "$GAN/requests.jsonl" "$GAN/metadata"
run_pair kalman "$KALMAN/requests.jsonl" "$ROOT/Data/ApplicationDataset"
