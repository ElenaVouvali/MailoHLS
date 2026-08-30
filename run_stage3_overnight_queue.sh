#!/usr/bin/env bash
set -uo pipefail

# Sequential, unattended Stage-3 replication queue.  Each run gets a fresh
# output directory so stale selection checkpoints cannot contaminate it.
export MAILOHLS_DATA="${MAILOHLS_DATA:-artifacts/llm/mailohls_sft.jsonl}"
export MAILOHLS_SPLIT="${MAILOHLS_SPLIT:-mailohls_runs/mailohls_final_family_split_s123.json}"
export MAILOHLS_DOMAINS="${MAILOHLS_DOMAINS:-mailohls_runs/stage2_specified_adp_epoch9_s123_v3/directive_domain_registry.json}"
export MAILOHLS_MEMORY="${MAILOHLS_MEMORY:-artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned}"
export MAILOHLS_STAGE2="${MAILOHLS_STAGE2:-mailohls_runs/stage2_specified_adp_epoch9_s123_v3/best_custom_stage2}"
GPU="${STAGE3_GPU:-0}"

run_stage3() {
  local seed="$1" out="$2"
  mkdir -p "$out"
  if [[ -f "$out/training_contract.json" && -f "$out/train.log" ]]; then
    echo "[QUEUE] Refusing existing output: $out" >&2
    return 2
  fi
  MAILOHLS_STAGE3_OUT="$out" \
  STAGE3_MODE=train STAGE3_GPU="$GPU" STAGE3_SEED="$seed" STAGE2_SEED=123 \
  STAGE3_MAX_STEPS=40 STAGE3_EVAL_STEPS=20 STAGE3_SELECTION_EVAL_STEPS=20 \
  STAGE3_SELECTION_EARLY_STOPPING_PATIENCE=1 STAGE3_SELECTION_EARLY_STOPPING_MIN_STEP=20 \
  STAGE3_PAIR_UNIT=semantic_action STAGE3_SEMANTIC_ACTION_ONLY=1 \
  STAGE3_MIN_ACTION_DISTANCE=1 STAGE3_MAX_ACTION_DISTANCE=1 \
  STAGE3_MIN_EDIT_DISTANCE=1 STAGE3_MAX_EDIT_DISTANCE=0 \
  STAGE3_MAX_REFERENCE_MARGIN=0.05 STAGE3_REQUIRE_CHOSEN_RANK0=1 \
  STAGE3_REUSE_PAIR_CACHE=0 STAGE3_REUSE_SELECTION_CACHE=0 \
  bash LLM_branch/inference/run_stage3_contract.sh \
  2>&1 | tee "$out/launcher.log"
  return "${PIPESTATUS[0]}"
}

run_stage3 789 "mailohls_runs/stage3_adp_semantic_action_s789_rep2" || exit $?
run_stage3 123 "mailohls_runs/stage3_adp_semantic_action_s123_rep_final" || exit $?

echo "[QUEUE] Stage-3 replication queue completed"
