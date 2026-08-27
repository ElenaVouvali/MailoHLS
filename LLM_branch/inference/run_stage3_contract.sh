#!/usr/bin/env bash
set -euo pipefail

# Contract-driven MailoHLS Stage 3.  The Stage-2 directory is the single
# source of truth for model/tokenizer/placement/routing/memory configuration.
: "${MAILOHLS_DATA:?set MAILOHLS_DATA}"
: "${MAILOHLS_SPLIT:?set MAILOHLS_SPLIT}"
: "${MAILOHLS_DOMAINS:?set MAILOHLS_DOMAINS}"
: "${MAILOHLS_MEMORY:?set MAILOHLS_MEMORY}"
: "${MAILOHLS_STAGE2:?set MAILOHLS_STAGE2 to the winning Stage-2 adapter}"
: "${MAILOHLS_STAGE3_OUT:?set MAILOHLS_STAGE3_OUT}"

# Stage-2 saves the shared directive registry beside checkpoint directories,
# while users commonly point MAILOHLS_STAGE2 at best_custom_stage2/. Resolve
# that layout automatically, without bypassing the hash validation in DPO.
if [[ ! -f "${MAILOHLS_DOMAINS}" ]]; then
  PARENT_DOMAINS="$(dirname "${MAILOHLS_STAGE2}")/directive_domain_registry.json"
  if [[ -f "${PARENT_DOMAINS}" ]]; then
    export MAILOHLS_DOMAINS="${PARENT_DOMAINS}"
  else
    echo "Missing directive domain registry: ${MAILOHLS_DOMAINS}" >&2
    echo "Also checked: ${PARENT_DOMAINS}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
MODE="${STAGE3_MODE:-preflight}"
GPU="${STAGE3_GPU:-0}"
MAX_STEPS="${STAGE3_MAX_STEPS:-60}"

case "${MODE}" in
  preflight|train) ;;
  *) echo "STAGE3_MODE must be preflight or train" >&2; exit 2 ;;
esac

if [[ "${MODE}" == "train" ]] && [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing publishable Stage-3 training from a dirty tracked tree." >&2
  exit 2
fi

test -f "${MAILOHLS_STAGE2}/training_contract.json"
test -f "${MAILOHLS_STAGE2}/structural_xattn.pt"
test -f "${MAILOHLS_MEMORY}/memory_manifest.json"
mkdir -p "${MAILOHLS_STAGE3_OUT}"

ARGS=(
  --dataset "${MAILOHLS_DATA}"
  --split_json "${MAILOHLS_SPLIT}"
  --directive_domain_registry_json "${MAILOHLS_DOMAINS}"
  --memory_dir "${MAILOHLS_MEMORY}"
  --stage2_adapter_dir "${MAILOHLS_STAGE2}"
  --sft_script "${REPO_ROOT}/LLM_branch/train/train_SFT_xattn_new.py"
  --output_dir "${MAILOHLS_STAGE3_OUT}"
  --top_k 6
  --dpo_chosen_top_k "${STAGE3_CHOSEN_TOP_K:-1}"
  --dpo_hard_window 8
  --dpo_hard_negatives_per_chosen 2
  --dpo_medium_negatives_per_chosen 1
  --dpo_min_score_gap 0.02
  --dpo_hard_gap_max 0.15
  --dpo_medium_gap_max 0.35
  --dpo_min_primary_rel_gain 0.02
  --dpo_max_edit_distance "${STAGE3_MAX_EDIT_DISTANCE:-8}"
  --require_same_supervised_schema
  --beta "${STAGE3_BETA:-0.5}"
  --label_smoothing 0
  --sft_alpha "${STAGE3_SFT_ALPHA:-0.02}"
  --dpo_logp_reduction mean
  --train_xattn_dpo
  --train_attn_gate_dpo
  --lr_xattn "${STAGE3_LR_XATTN:-5e-5}"
  --lr_gate "${STAGE3_LR_GATE:-2e-5}"
  --lr_lora 0
  --lr_embed 0
  --lr_ff 0
  --lr_gate_ff 0
  --batch_size "${STAGE3_BATCH_SIZE:-1}"
  --grad_accum "${STAGE3_GRAD_ACCUM:-8}"
  --max_steps "${MAX_STEPS}"
  --eval_steps "${STAGE3_EVAL_STEPS:-20}"
  --selection_eval_steps "${STAGE3_SELECTION_EVAL_STEPS:-${STAGE3_EVAL_STEPS:-20}}"
  --selection_early_stopping_patience "${STAGE3_SELECTION_EARLY_STOPPING_PATIENCE:-1}"
  --selection_early_stopping_min_step "${STAGE3_SELECTION_EARLY_STOPPING_MIN_STEP:-20}"
  --save_steps "${STAGE3_SAVE_STEPS:-20}"
  --logging_steps "${STAGE3_LOGGING_STEPS:-5}"
  --num_workers "${STAGE3_NUM_WORKERS:-0}"
  --gradient_checkpointing
  --save_selection_debug
  --seed 123
)

if [[ "${STAGE3_TRAIN_LORA:-0}" == "1" ]]; then
  echo "LoRA is locked frozen for the final focused-DPO ablation." >&2
  exit 2
fi
if [[ "${STAGE3_REQUIRE_CHOSEN_RANK0:-1}" == "1" ]]; then
  ARGS+=(--dpo_require_chosen_rank0)
fi
if [[ "${STAGE3_REUSE_PAIR_CACHE:-0}" == "1" ]]; then
  ARGS+=(--reuse_pair_cache)
fi
if [[ "${STAGE3_REUSE_SELECTION_CACHE:-0}" == "1" ]]; then
  ARGS+=(--reuse_selection_cache)
fi

if [[ "${MODE}" == "preflight" ]]; then
  python -u -m LLM_branch.train.train_DPO_harp_xattn \
    "${ARGS[@]}" --pair_build_only \
    2>&1 | tee "${MAILOHLS_STAGE3_OUT}/preflight.log"
else
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u -m LLM_branch.train.train_DPO_harp_xattn \
    "${ARGS[@]}" \
    2>&1 | tee "${MAILOHLS_STAGE3_OUT}/train.log"
fi
