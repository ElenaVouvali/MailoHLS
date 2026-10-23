#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/random_designsplit_3stage_3goals.sh            # run all stages: SFT + xattn + DPO
#   bash src/random_designsplit_3stage_3goals.sh sft        # only Stage 1 + Stage 2
#   bash src/random_designsplit_3stage_3goals.sh dpo        # only Stage 3 DPO
#
# Optional env:
#   GPU=0
#   RUN_ROOT=/home/ubuntu/runs/random_design_split_fixed
#   DPO_MODE=final   # final or smoke
#   SEED=123

MODE="${1:-all}"   # all | sft | dpo

if [[ "${MODE}" != "all" && "${MODE}" != "sft" && "${MODE}" != "dpo" ]]; then
  echo "[ERROR] MODE must be one of: all, sft, dpo"
  echo "Example: bash src/random_design_3stage_3goals.sh all"
  exit 1
fi

GPU="${GPU:-0}"

SFT_SCRIPT="src/train_SFT_xattn.py"
DPO_SCRIPT="src/train_DPO_harp_xattn.py"

DATASET="${DATASET:-/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl}"
MEMORY_DIR="${MEMORY_DIR:-/home/ubuntu/save/harp/memory_tokens/}"
MODEL="${MODEL:-deepseek-ai/deepseek-coder-7b-base}"

SEED="${SEED:-123}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/runs/random_design_split_16n_fixed}"
SPLIT_JSON="${RUN_ROOT}/splits/random_design_80_10_10.json"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/splits"

SUMMARY_LOG="${RUN_ROOT}/logs/_summary.log"
touch "${SUMMARY_LOG}"

# --------------------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------------------
echo "===================================================================================================="
echo "[PREFLIGHT]"
echo "[MODE]       ${MODE}"
echo "[GPU]        ${GPU}"
echo "[DATASET]    ${DATASET}"
echo "[MEMORY_DIR] ${MEMORY_DIR}"
echo "[MODEL]      ${MODEL}"
echo "[RUN_ROOT]   ${RUN_ROOT}"
echo "[SPLIT_JSON] ${SPLIT_JSON}"
echo "===================================================================================================="

if [[ ! -f "${DATASET}" ]]; then
  echo "[ERROR] Dataset not found: ${DATASET}"
  exit 1
fi

if [[ ! -d "${MEMORY_DIR}" ]]; then
  echo "[ERROR] Memory dir not found: ${MEMORY_DIR}"
  exit 1
fi

if ! compgen -G "${MEMORY_DIR}/*.memory.pt" > /dev/null; then
  echo "[ERROR] No .memory.pt files found in: ${MEMORY_DIR}"
  exit 1
fi

# --------------------------------------------------------------------------------------------------
# Common SFT + Stage2 args
# --------------------------------------------------------------------------------------------------
SFT_COMMON_ARGS=(
  --run_mode two_stage
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"

  --split_mode random_design
  --val_ratio 0.10
  --test_ratio 0.10
  --split_seed "${SEED}"
  --stratify_by_kernel

  --top_k 6
  --goal_domination_penalty 0.25
  --goal_max_dominated_gap 0.12
  --score_weight_min 0.6
  --score_weight_power 1.0

  --candidate_loss_weight 0.0
  --candidate_max_prefix_tokens 1536
  --candidate_keep_head_tokens 256

  --min_supervised_sites 2
  --min_site_coverage 0.85
  --selection_num_val_kernels 6

  --max_length 4096
  --epochs 4
  --batch_size 2
  --grad_accum 4
  --num_workers 4
  --group_by_length
  --gradient_checkpointing

  --lr_lora 5e-5
  --lr_embed 5e-5
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.05

  --mem_dim -1
  --require_pragma_free_memory
  --max_slots 64
  --every_n_layers 16
  --xattn_heads 4
  --xattn_dim_head 64
  --xattn_ff_mult 1

  --eval_steps 160
  --save_steps 160

  --stage2_epochs 4
  --stage2_eval_steps 160
  --stage2_save_steps 160
  --stage2_lr_xattn 1e-4
  --stage2_lr_gate 2e-4
  --stage2_lr_ff 0.0
  --stage2_lr_gate_ff 0.0

  --seed "${SEED}"
)

# --------------------------------------------------------------------------------------------------
# DPO config
# --------------------------------------------------------------------------------------------------

DPO_COMMON_ARGS=(
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"
  --sft_script "${SFT_SCRIPT}"

  --split_json "${SPLIT_JSON}"
  --split_mode random_design
  --split_seed "${SEED}"
  --stratify_by_kernel

  --top_k 6
  --goal_domination_penalty 0.25
  --goal_max_dominated_gap 0.12
  --score_weight_min 0.6
  --score_weight_power 1.0

  --dpo_chosen_top_k 3
  --dpo_hard_window 2
  --dpo_hard_negatives_per_chosen 2
  --dpo_medium_negatives_per_chosen 1
  --dpo_min_score_gap 0.03
  --dpo_hard_gap_max 0.15
  --dpo_medium_gap_max 0.35
  --dpo_min_primary_rel_gain 0.02
  --dpo_min_edit_distance 1
  --dpo_min_edit_frac 0.0
  --dpo_max_edit_frac 1.0

  --min_supervised_sites 2
  --min_site_coverage 0.85
  --selection_num_val_kernels 6
  --require_same_supervised_schema

  --beta 0.1
  --label_smoothing 0.0
  --sft_alpha 0.0

  --max_length 4096
  --batch_size 1
  --grad_accum 8
  --epochs 3
  --eval_steps 200
  --save_steps 200
  --logging_steps 10
  --num_workers 4
  --group_by_length
  --gradient_checkpointing

  --lr_lora 0.0
  --lr_xattn 1e-5
  --lr_gate 2e-5
  --lr_ff 0.0
  --lr_gate_ff 0.0
  --lr_embed 0.0

  --train_xattn_dpo
  --train_attn_gate_dpo

  --mem_dim -1
  --require_pragma_free_memory
  --max_slots 64
  --every_n_layers 16
  --xattn_heads 4
  --xattn_dim_head 64
  --xattn_ff_mult 1

  --save_total_limit 2
  --seed "${SEED}"
)

FAILURES=()

run_sft_goal() {
  local GOAL="$1"
  local TAG="$2"

  local LOG_FILE="${RUN_ROOT}/logs/${TAG}_stage1_stage2.log"
  local STAGE1_DIR="${RUN_ROOT}/${TAG}_stage1"
  local STAGE2_DIR="${RUN_ROOT}/${TAG}_stage2"

  local SPLIT_FLAG="--split_json"
  if [[ ! -f "${SPLIT_JSON}" ]]; then
    SPLIT_FLAG="--save_split_json"
  fi

  echo
  echo "===================================================================================================="
  echo "[START SFT+XATTN] ${GOAL}"
  echo "[TAG]    ${TAG}"
  echo "[SPLIT]  ${SPLIT_FLAG} ${SPLIT_JSON}"
  echo "[STAGE1] ${STAGE1_DIR}"
  echo "[STAGE2] ${STAGE2_DIR}"
  echo "[LOG]    ${LOG_FILE}"
  echo "===================================================================================================="

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${SFT_SCRIPT}" \
      "${SFT_COMMON_ARGS[@]}" \
      --objective "${GOAL}" \
      "${SPLIT_FLAG}" "${SPLIT_JSON}" \
      --stage1_output_dir "${STAGE1_DIR}" \
      --stage2_output_dir "${STAGE2_DIR}" \
      2>&1 | tee "${LOG_FILE}"; then
    echo "[FAIL] ${GOAL} SFT+XATTN" | tee -a "${SUMMARY_LOG}"
    FAILURES+=("${GOAL}:sft_xattn")
    return 1
  fi

  echo "[OK] ${GOAL} SFT+XATTN" | tee -a "${SUMMARY_LOG}"
  return 0
}

run_dpo_goal() {
  local GOAL="$1"
  local TAG="$2"

  local STAGE1_BEST_DIR="${RUN_ROOT}/${TAG}_stage1/best_custom_stage1"
  local STAGE2_CKPT="${RUN_ROOT}/${TAG}_stage2/best_custom_stage2/harp_xattn.pt"
  local STAGE3_DIR="${RUN_ROOT}/${TAG}_stage3"
  local STAGE3_LOG="${RUN_ROOT}/logs/${TAG}_stage3.log"

  echo
  echo "===================================================================================================="
  echo "[START DPO]  ${GOAL}"
  echo "[TAG]        ${TAG}"
  echo "[STAGE1]     ${STAGE1_BEST_DIR}"
  echo "[STAGE2]     ${STAGE2_CKPT}"
  echo "[STAGE3]     ${STAGE3_DIR}"
  echo "[LOG]        ${STAGE3_LOG}"
  echo "===================================================================================================="

  if [[ ! -d "${STAGE1_BEST_DIR}" ]]; then
    echo "[FAIL] Missing Stage1 best adapter: ${STAGE1_BEST_DIR}" | tee -a "${SUMMARY_LOG}"
    FAILURES+=("${GOAL}:missing_stage1_best")
    return 1
  fi

  if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "[FAIL] Missing Stage2 HARP checkpoint: ${STAGE2_CKPT}" | tee -a "${SUMMARY_LOG}"
    FAILURES+=("${GOAL}:missing_stage2_harp")
    return 1
  fi

  if [[ ! -f "${SPLIT_JSON}" ]]; then
    echo "[FAIL] Missing split json: ${SPLIT_JSON}" | tee -a "${SUMMARY_LOG}"
    FAILURES+=("${GOAL}:missing_split_json")
    return 1
  fi

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${DPO_SCRIPT}" \
      "${DPO_COMMON_ARGS[@]}" \
      --objective "${GOAL}" \
      --stage1_adapter_dir "${STAGE1_BEST_DIR}" \
      --stage2_harp_xattn_path "${STAGE2_CKPT}" \
      --output_dir "${STAGE3_DIR}" \
      2>&1 | tee "${STAGE3_LOG}"; then
    echo "[FAIL] ${GOAL} DPO" | tee -a "${SUMMARY_LOG}"
    FAILURES+=("${GOAL}:dpo")
    return 1
  fi

  echo "[OK] ${GOAL} DPO" | tee -a "${SUMMARY_LOG}"
  return 0
}

GOALS=(
  "PARETO_LATENCY_EXTREME:latency_extreme"
  "PARETO_KNEE:pareto_knee"
  "PARETO_AREA_EXTREME:area_extreme"
)

if [[ "${MODE}" == "all" || "${MODE}" == "sft" ]]; then
  for item in "${GOALS[@]}"; do
    GOAL="${item%%:*}"
    TAG="${item##*:}"
    run_sft_goal "${GOAL}" "${TAG}"
  done
fi

if [[ "${MODE}" == "all" || "${MODE}" == "dpo" ]]; then
  for item in "${GOALS[@]}"; do
    GOAL="${item%%:*}"
    TAG="${item##*:}"
    run_dpo_goal "${GOAL}" "${TAG}"
  done
fi

echo
echo "===================================================================================================="
echo "[DONE] Random design split 3-stage pipeline"
echo "[MODE] ${MODE}"
echo "[ROOT] ${RUN_ROOT}"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "[STATUS] All requested runs completed successfully"
else
  echo "[STATUS] Failures: ${FAILURES[*]}"
  exit 1
fi
echo "===================================================================================================="
