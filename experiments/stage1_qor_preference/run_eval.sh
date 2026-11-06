#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT" MPLCONFIGDIR=/tmp/stage1-qor-matplotlib
cd "$ROOT"
PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
"$PYTHON" -u -m experiments.stage1_qor_preference.evaluate infer54 2>&1 | tee -a artifacts/stage1_qor_preference/v1_20261106/inference54.log
"$PYTHON" -m experiments.stage1_qor_preference.evaluate summarize54
"$PYTHON" -u -m experiments.stage1_qor_preference.evaluate gemv 2>&1 | tee -a artifacts/stage1_qor_preference/v1_20261106/gemv.log
