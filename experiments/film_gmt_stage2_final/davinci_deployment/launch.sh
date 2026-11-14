#!/usr/bin/env bash
set -euo pipefail
DEPLOY=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
export ROOT="$DEPLOY/code"
export PYTHON="$DEPLOY/venv/bin/python"
export PARENT=/home/evouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run
export FILM_CP="$DEPLOY/assets/film/B/checkpoint-0600"
export GNOSIS_BANK="$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_bank_original_anchors"
export GMT_MEMORY="$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/memory_B"
export GPU_UUID=GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f
export ENTRY="$DEPLOY/adapter/entry.py"
export COHORT=original-film-31+2
export RUN_ROOT="${RUN_ROOT:-$DEPLOY/runs/campaign_s123_v1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"
exec bash "$ROOT/experiments/film_gmt_stage2_final/commands.sh"
