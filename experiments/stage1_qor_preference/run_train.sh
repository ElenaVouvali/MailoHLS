#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT"
cd "$ROOT"
/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python -u -m experiments.stage1_qor_preference.train \
  2>&1 | tee -a "$ROOT/artifacts/stage1_qor_preference/v1_20261106/training.log"
