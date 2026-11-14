#!/usr/bin/env bash
# No automatic fallback or automatic training: choose the named cohort explicitly.
set -euo pipefail
MODE="${MODE:-audit}"
[[ "$MODE" =~ ^(audit|smoke|train|resume)$ ]] || { echo "Invalid MODE $MODE" >&2; exit 2; }
PYTHON="${PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
ROOT="${ROOT:-/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final}"
PARENT="${PARENT:-/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run}"
FILM_CP="${FILM_CP:-/home/elvouvali/MailoHLS/worktrees/gnosis-stage1-film-ab/artifacts/gnosis_stage1_ab/stage1_ab_pilot_20261112_retry2/B/checkpoint-0600}"
GNOSIS_BANK="${GNOSIS_BANK:-$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_bank_original_anchors}"
GMT_MEMORY="${GMT_MEMORY:-$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/memory_B}"
GPU_UUID="${GPU_UUID:-GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb}"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/campaign}"
ENTRY="${ENTRY:-$ROOT/experiments/film_gmt_stage2_final/train.py}"
COHORT="${COHORT:?Set COHORT=original-film-31+2 to select the qualified fallback; expanded-recovery remains blocked}"
[[ "${ALLOW_EXPLICIT_SUBSET:-0}" != 1 ]] || { echo 'Subset bypass is forbidden' >&2; exit 2; }
for variable in FILM_CP GNOSIS_BANK GMT_MEMORY; do
  [[ -n "${!variable}" ]] || { echo "Set $variable to a verified absolute path" >&2; exit 2; }
done
case "$MODE" in
 audit) OUT="$RUN_ROOT/audit" ;;
 smoke) OUT="$RUN_ROOT/smoke" ;;
 train) OUT="$RUN_ROOT/final" ;;
 resume) OUT="$RUN_ROOT/final" ; [[ -n "${RESUME_FROM:-}" ]] || { echo "Set RESUME_FROM" >&2; exit 2; } ;;
esac
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
ARGS=(
  --mode "$MODE" --cohort "$COHORT" --repo "$ROOT" --parent "$PARENT"
  --film-checkpoint "$FILM_CP" --gnosis-bank "$GNOSIS_BANK"
  --gmt-memory "$GMT_MEMORY" --gpu-uuid "$GPU_UUID"
  --output "$OUT" --seed 123 --steps 450
  --contexts 0 --grad-accum 4 --max-length 7168
  --lr-xattn 1e-4 --lr-gate 2e-4
  --gate-open-steps 60 --gate-open-lr 8e-4 --probe-sft 3
  --rank-budget .20 --rank-clock .05
  --eval-interval 150 --val-sft 24 --val-pair-groups 8
  --min-train-families 10 --min-val-families 2
)
if [[ "$MODE" == resume ]]; then ARGS+=(--resume-from "$RESUME_FROM"); fi
"$PYTHON" -u "$ENTRY" "${ARGS[@]}"
