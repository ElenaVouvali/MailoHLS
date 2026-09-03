#!/usr/bin/env bash
set -euo pipefail

export MAILOHLS_DATA="${MAILOHLS_DATA:-artifacts/llm/mailohls_sft.jsonl}"
export MAILOHLS_SPLIT="${MAILOHLS_SPLIT:-mailohls_runs/mailohls_final_family_split_s123.json}"

# Final AREA Stage-2 parent.
export MAILOHLS_STAGE2="${MAILOHLS_STAGE2:-mailohls_runs/stage2_area_gate2e4_s123_final_rescue_v4/best_custom_stage2}"

# Registry belonging to exactly that Stage-2 lineage.
export MAILOHLS_DOMAINS="${MAILOHLS_DOMAINS:-mailohls_runs/stage2_area_gate2e4_s123_final_rescue_v4/directive_domain_registry.json}"

# Final frozen GNN structural memory.
export MAILOHLS_MEMORY="${MAILOHLS_MEMORY:-artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned}"

# New clean AREA Stage-3 output.
export MAILOHLS_STAGE3_OUT="${MAILOHLS_STAGE3_OUT:-mailohls_runs/stage3_area_semantic_action_s789_final}"

export STAGE3_GPU="${STAGE3_GPU:-0}"

common=(
  # Keep target/budget construction exactly tied to Stage-2 seed 123.
  # Only the Stage-3 optimization seed changes.
  STAGE3_SEED=789
  STAGE2_SEED=123

  # Same focused run length that worked for final ADP.
  STAGE3_MAX_STEPS=40
  STAGE3_EVAL_STEPS=20
  STAGE3_SELECTION_EVAL_STEPS=20
  STAGE3_SAVE_STEPS=20

  # Stop quickly if preference alignment does not improve the parent.
  STAGE3_SELECTION_EARLY_STOPPING_PATIENCE=1
  STAGE3_SELECTION_EARLY_STOPPING_MIN_STEP=20

  # Whole semantic action, not unrelated RHS tokens.
  STAGE3_PAIR_UNIT=semantic_action
  STAGE3_SEMANTIC_ACTION_ONLY=1
  STAGE3_MIN_ACTION_DISTANCE=1
  STAGE3_MAX_ACTION_DISTANCE=1

  # Permit any number of member-field changes inside that single semantic action.
  STAGE3_MIN_EDIT_DISTANCE=1
  STAGE3_MAX_EDIT_DISTANCE=0

  # Focus DPO on pairs not already confidently solved by frozen Stage-2.
  STAGE3_MAX_REFERENCE_MARGIN=0.05
  STAGE3_REQUIRE_CHOSEN_RANK0=1
)

case "${1:-preflight}" in
  preflight)
    env "${common[@]}" \
      STAGE3_MODE=preflight \
      STAGE3_REUSE_PAIR_CACHE=0 \
      STAGE3_REUSE_SELECTION_CACHE=0 \
      bash LLM_branch/inference/run_stage3_contract.sh
    ;;

  train)
    test -s "${MAILOHLS_STAGE3_OUT}/pair_debug/train_pairs.jsonl"

    env "${common[@]}" \
      STAGE3_MODE=train \
      STAGE3_REUSE_PAIR_CACHE=1 \
      STAGE3_REUSE_SELECTION_CACHE=1 \
      bash LLM_branch/inference/run_stage3_contract.sh
    ;;

  *)
    echo "usage: $0 [preflight|train]" >&2
    exit 2
    ;;
esac
