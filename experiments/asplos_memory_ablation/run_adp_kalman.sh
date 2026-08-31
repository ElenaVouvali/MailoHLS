#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-asplos_results/memory_ablation_adp_kalman_s123}"

SOURCE_CASES="mailohls_runs/stage1_pareto_adp_s123_inference_fix_full/selected_debug/test_selected_pareto_adp.jsonl"
CASES="${RUN_DIR}/cases/kalman_adp_18.jsonl"
CASE_MANIFEST="${RUN_DIR}/cases/manifest.json"

STAGE1="mailohls_runs/stage1_final_final_adp_s123/best_custom_stage1"
STAGE2="mailohls_runs/stage2_specified_adp_epoch9_s123_v3/best_custom_stage2"
DOMAINS="mailohls_runs/stage2_specified_adp_epoch9_s123_v3/directive_domain_registry.json"
ALIGNED="artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned"
ABLATION_ROOT="${RUN_DIR}/memory_banks"

mkdir -p "${RUN_DIR}/cases" "${RUN_DIR}/predictions" "${RUN_DIR}/logs"

"${PYTHON_BIN}" experiments/asplos_memory_ablation/prepare_cases.py \
  --input_jsonl "${SOURCE_CASES}" \
  --output_jsonl "${CASES}" \
  --manifest_json "${CASE_MANIFEST}" \
  --budgets_per_device_clock 3

if [[ ! -f "${ABLATION_ROOT}/zero/memory_manifest.json" ]]; then
  "${PYTHON_BIN}" GNN_branch/make_memory_ablation.py \
    --static_dir "${ALIGNED}" \
    --zero_out "${ABLATION_ROOT}/zero" \
    --shuffled_out "${ABLATION_ROOT}/shuffled" \
    --global_out "${ABLATION_ROOT}/global" \
    --local_out "${ABLATION_ROOT}/local" \
    --local_shuffled_out "${ABLATION_ROOT}/local_shuffled" \
    --seed 123
fi

run_stage1() {
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON_BIN}" -u \
    LLM_branch/inference/eval_stage1_stage2_stage3.py \
    --stage stage1 \
    --adapter_dir "${STAGE1}" \
    --directive_domain_registry_json "${DOMAINS}" \
    --input_jsonl "${CASES}" \
    --num_samples 1 \
    --output_jsonl "${RUN_DIR}/predictions/stage1.jsonl" \
    2>&1 | tee "${RUN_DIR}/logs/stage1.log"
}

run_stage2() {
  local name="$1"
  local memory_dir="$2"
  local allow_ablation="$3"
  local extra_args=()
  if [[ "${allow_ablation}" == "yes" ]]; then
    extra_args+=(--allow_memory_ablation)
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON_BIN}" -u \
    LLM_branch/inference/eval_stage1_stage2_stage3.py \
    --stage stage2 \
    --adapter_dir "${STAGE2}" \
    --directive_domain_registry_json "${DOMAINS}" \
    --memory_dir "${memory_dir}" \
    --input_jsonl "${CASES}" \
    --num_samples 1 \
    --output_jsonl "${RUN_DIR}/predictions/${name}.jsonl" \
    "${extra_args[@]}" \
    2>&1 | tee "${RUN_DIR}/logs/${name}.log"
}

run_stage1
run_stage2 zero "${ABLATION_ROOT}/zero" yes
run_stage2 shuffled "${ABLATION_ROOT}/shuffled" yes
run_stage2 aligned "${ALIGNED}" no

"${PYTHON_BIN}" experiments/asplos_memory_ablation/summarize_results.py \
  --result "stage1=${RUN_DIR}/predictions/stage1.jsonl" \
  --result "zero=${RUN_DIR}/predictions/zero.jsonl" \
  --result "shuffled=${RUN_DIR}/predictions/shuffled.jsonl" \
  --result "aligned=${RUN_DIR}/predictions/aligned.jsonl" \
  --summary_csv "${RUN_DIR}/summary.csv" \
  --synthesis_queue_jsonl "${RUN_DIR}/synthesis_queue.jsonl"

echo "[DONE] ${RUN_DIR}/summary.csv"
