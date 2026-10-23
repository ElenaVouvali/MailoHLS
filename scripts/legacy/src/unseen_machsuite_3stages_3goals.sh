#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/unseen_machsuite_3stages_3goals.sh all    # Stage 1 + Stage 2 + Stage 3
#   bash src/unseen_machsuite_3stages_3goals.sh sft    # Stage 1 + Stage 2 only
#   bash src/unseen_machsuite_3stages_3goals.sh dpo    # Stage 3 only
#
# Optional env:
#   GPU=0
#   SEED=123
#   RUN_ROOT=/home/ubuntu/runs/unseen_machsuite_fixed
#
# Important:
#   This script leaves out the whole MachSuite suite from train/val.
#   Because train_SFT_xattn.py splits by family_id, the held-out test ids are:
#     machsuite_gemm
#     machsuite_md_knn
#     machsuite_sort_radix
#     machsuite_spmv_ellpack
#     machsuite_stencil2d
#     machsuite_stencil3d
#     machsuite_viterbi
#
# These cover:
#   machsuite-gemm-blocked
#   machsuite-gemm-ncubed
#   machsuite-md-knn
#   machsuite-sort-radix
#   machsuite-spmv-ellpack
#   machsuite-stencil2d
#   machsuite-stencil3d
#   machsuite-viterbi

MODE="${1:-all}"   # all | sft | dpo

if [[ "${MODE}" != "all" && "${MODE}" != "sft" && "${MODE}" != "dpo" ]]; then
  echo "[ERROR] MODE must be one of: all, sft, dpo"
  echo "Example: bash src/unseen_machsuite_3stages_3goals.sh all"
  exit 1
fi

GPU="${GPU:-0}"

SFT_SCRIPT="${SFT_SCRIPT:-src/train_SFT_xattn.py}"
DPO_SCRIPT="${DPO_SCRIPT:-src/train_DPO_harp_xattn.py}"

DATASET="${DATASET:-/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl}"
MEMORY_DIR="${MEMORY_DIR:-/home/ubuntu/save/harp/memory_tokens/}"
MODEL="${MODEL:-deepseek-ai/deepseek-coder-7b-base}"

SEED="${SEED:-123}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/runs/unseen_machsuite_fixed}"
SPLIT_JSON="${RUN_ROOT}/splits/family_leave_machsuite_out.json"

# Validation must NOT include any MachSuite family.
# Use non-MachSuite families for model selection.
VAL_FAMILIES="${VAL_FAMILIES:-rodinia_pathfinder;rodinia_kmeans}"

# Test families are family IDs, not raw kernel names.
TEST_FAMILIES="${TEST_FAMILIES:-machsuite_gemm;machsuite_md_knn;machsuite_sort_radix;machsuite_spmv_ellpack;machsuite_stencil2d;machsuite_stencil3d;machsuite_viterbi}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/splits"
SUMMARY_LOG="${RUN_ROOT}/logs/_summary.log"
touch "${SUMMARY_LOG}"

echo "===================================================================================================="
echo "[PREFLIGHT]"
echo "[MODE]          ${MODE}"
echo "[GPU]           ${GPU}"
echo "[DATASET]       ${DATASET}"
echo "[MEMORY_DIR]    ${MEMORY_DIR}"
echo "[MODEL]         ${MODEL}"
echo "[RUN_ROOT]      ${RUN_ROOT}"
echo "[SPLIT_JSON]    ${SPLIT_JSON}"
echo "[VAL_FAMILIES]  ${VAL_FAMILIES}"
echo "[TEST_FAMILIES] ${TEST_FAMILIES}"
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

if [[ ! -f "${SFT_SCRIPT}" ]]; then
  echo "[ERROR] Missing SFT script: ${SFT_SCRIPT}"
  exit 1
fi

if [[ ! -f "${DPO_SCRIPT}" ]]; then
  echo "[ERROR] Missing DPO script: ${DPO_SCRIPT}"
  exit 1
fi

# --------------------------------------------------------------------------------------------------
# Common Stage 1 + Stage 2 args
# --------------------------------------------------------------------------------------------------
SFT_COMMON_ARGS=(
  --run_mode two_stage
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"

  --split_mode family
  --val_families "${VAL_FAMILIES}"
  --test_families "${TEST_FAMILIES}"
  --split_seed "${SEED}"

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
  --every_n_layers 8
  --xattn_heads 4
  --xattn_dim_head 64
  --xattn_ff_mult 1

  --eval_steps 120
  --save_steps 120

  --stage2_epochs 4
  --stage2_eval_steps 120
  --stage2_save_steps 120
  --stage2_lr_xattn 1e-4
  --stage2_lr_gate 2e-4
  --stage2_lr_ff 0.0
  --stage2_lr_gate_ff 0.0

  --seed "${SEED}"
)

# --------------------------------------------------------------------------------------------------
# Common Stage 3 DPO args
# --------------------------------------------------------------------------------------------------
DPO_COMMON_ARGS=(
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"
  --sft_script "${SFT_SCRIPT}"

  --split_json "${SPLIT_JSON}"
  --split_mode family
  --val_families "${VAL_FAMILIES}"
  --test_families "${TEST_FAMILIES}"
  --split_seed "${SEED}"

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
  --every_n_layers 8
  --xattn_heads 4
  --xattn_dim_head 64
  --xattn_ff_mult 1

  --save_total_limit 2
  --seed "${SEED}"
)

FAILURES=()

print_split_summary() {
  if [[ ! -f "${SPLIT_JSON}" ]]; then
    echo "[WARN] Split JSON not found yet: ${SPLIT_JSON}"
    return 0
  fi

  python - "${DATASET}" "${SPLIT_JSON}" "${TEST_FAMILIES}" "${VAL_FAMILIES}" <<'PY'
import json, re, sys
from collections import Counter

dataset, split_json, test_families_arg, val_families_arg = sys.argv[1:5]

def normalize_name(s):
    return re.sub(r"[-\s]+", "_", s.strip().lower())

def family_id_from_kernel_name(name):
    s = normalize_name(name).replace("-", "_")
    if s.startswith("machsuite_gemm"):
        return "machsuite_gemm"
    if s.startswith("machsuite_"):
        parts = s.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return s
    if s.startswith("spcl_example"):
        return "spcl_example"
    if s.startswith("serrano_"):
        return "serrano_kalman_filter"
    if s.startswith("rodinia_"):
        rest = s[len("rodinia_"):]
        for special in ["cfd_flux", "cfd_step_factor", "lc_gicov", "lc_mgvf"]:
            if rest.startswith(special):
                return f"rodinia_{special}"
        algo = rest.split("_")[0]
        return f"rodinia_{algo}"
    return s

expected_test = {normalize_name(x) for x in test_families_arg.split(";") if x.strip()}
expected_val = {normalize_name(x) for x in val_families_arg.split(";") if x.strip()}

machsuite_fams = {
    "machsuite_gemm",
    "machsuite_md_knn",
    "machsuite_sort_radix",
    "machsuite_spmv_ellpack",
    "machsuite_stencil2d",
    "machsuite_stencil3d",
    "machsuite_viterbi",
}

if expected_val & machsuite_fams:
    raise SystemExit(f"[ERROR] VAL_FAMILIES contains MachSuite family ids: {sorted(expected_val & machsuite_fams)}")

rows = []
with open(dataset, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue
        r = json.loads(line)
        r["_jsonl_idx"] = idx
        r["_family"] = family_id_from_kernel_name(r["kernel_name"])
        rows.append(r)

idx_to_row = {r["_jsonl_idx"]: r for r in rows}

with open(split_json, "r", encoding="utf-8") as f:
    spec = json.load(f)

split_fams = {}
split_kernels = {}

for split_name, key in [
    ("train", "train_jsonl_idx"),
    ("val", "val_jsonl_idx"),
    ("test", "test_jsonl_idx"),
]:
    split_rows = [idx_to_row[i] for i in spec.get(key, []) if i in idx_to_row]
    fams = Counter(r["_family"] for r in split_rows)
    kernels = Counter(r["kernel_name"] for r in split_rows)
    split_fams[split_name] = set(fams.keys())
    split_kernels[split_name] = set(kernels.keys())

    print(f"[SPLIT] {split_name}: rows={len(split_rows)} families={len(fams)} kernels={len(kernels)}")
    print(f"[SPLIT] {split_name} top families: {fams.most_common(20)}")
    print(f"[SPLIT] {split_name} kernels: {sorted(kernels.keys())}")

train_fams = split_fams["train"]
val_fams = split_fams["val"]
test_fams = split_fams["test"]

leaked = (train_fams | val_fams) & machsuite_fams
if leaked:
    raise SystemExit(f"[ERROR] MachSuite leaked into train/val families: {sorted(leaked)}")

missing = expected_test - test_fams
if missing:
    raise SystemExit(f"[ERROR] Some expected MachSuite test families are missing from test split: {sorted(missing)}")

extra_machsuite_outside_test = machsuite_fams - test_fams
if extra_machsuite_outside_test:
    raise SystemExit(f"[ERROR] Some known MachSuite families are not in test split: {sorted(extra_machsuite_outside_test)}")

print("[OK] Split sanity passed: all MachSuite families are test-only.")
PY
}

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

  print_split_summary

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
    echo "[FAIL] Missing split JSON: ${SPLIT_JSON}" | tee -a "${SUMMARY_LOG}"
    FAILURES+=("${GOAL}:missing_split_json")
    return 1
  fi

  print_split_summary

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
  print_split_summary

  for item in "${GOALS[@]}"; do
    GOAL="${item%%:*}"
    TAG="${item##*:}"
    run_dpo_goal "${GOAL}" "${TAG}"
  done
fi

echo
echo "===================================================================================================="
echo "[DONE] Unseen MachSuite 3-stage pipeline"
echo "[MODE] ${MODE}"
echo "[ROOT] ${RUN_ROOT}"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "[STATUS] All requested runs completed successfully"
else
  echo "[STATUS] Failures: ${FAILURES[*]}"
  exit 1
fi
echo "===================================================================================================="
