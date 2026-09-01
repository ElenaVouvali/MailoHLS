#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after applying stage3_latency_required_fix.patch.
export MAILOHLS_DATA="${MAILOHLS_DATA:-artifacts/llm/mailohls_sft.jsonl}"
export MAILOHLS_SPLIT="${MAILOHLS_SPLIT:-mailohls_runs/mailohls_final_family_split_s123.json}"
export MAILOHLS_STAGE2="${MAILOHLS_STAGE2:-mailohls_runs/stage2_latency_ce_rank010_s123_v2/best_custom_stage2}"
export MAILOHLS_DOMAINS="${MAILOHLS_DOMAINS:-mailohls_runs/stage2_latency_ce_rank010_s123_v2/directive_domain_registry.json}"
export MAILOHLS_MEMORY="${MAILOHLS_MEMORY:-artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned}"
export MAILOHLS_STAGE3_OUT="${MAILOHLS_STAGE3_OUT:-mailohls_runs/stage3_latency_semantic_action_s123_final}"
export STAGE3_GPU="${STAGE3_GPU:-0}"

common=(
  STAGE3_SEED=123 STAGE2_SEED=123
  STAGE3_MAX_STEPS=40 STAGE3_EVAL_STEPS=20
  STAGE3_SELECTION_EVAL_STEPS=20
  STAGE3_SELECTION_EARLY_STOPPING_PATIENCE=1
  STAGE3_SELECTION_EARLY_STOPPING_MIN_STEP=20
  STAGE3_PAIR_UNIT=semantic_action STAGE3_SEMANTIC_ACTION_ONLY=1
  STAGE3_MIN_ACTION_DISTANCE=1 STAGE3_MAX_ACTION_DISTANCE=1
  STAGE3_MIN_EDIT_DISTANCE=1 STAGE3_MAX_EDIT_DISTANCE=0
  STAGE3_MAX_REFERENCE_MARGIN=0.05 STAGE3_REQUIRE_CHOSEN_RANK0=1
)

case "${1:-preflight}" in
  preflight)
    env "${common[@]}" STAGE3_MODE=preflight \
      STAGE3_REUSE_PAIR_CACHE=0 STAGE3_REUSE_SELECTION_CACHE=0 \
      bash LLM_branch/inference/run_stage3_contract.sh
    ;;
  train)
    test -s "${MAILOHLS_STAGE3_OUT}/pair_debug/train_pairs.jsonl"
    env "${common[@]}" STAGE3_MODE=train \
      STAGE3_REUSE_PAIR_CACHE=1 STAGE3_REUSE_SELECTION_CACHE=1 \
      bash LLM_branch/inference/run_stage3_contract.sh
    ;;
  *)
    echo "usage: $0 [preflight|train]" >&2
    exit 2
    ;;
esac
