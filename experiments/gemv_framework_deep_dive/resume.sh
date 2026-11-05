#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/artifacts/gemv_framework_deep_dive/v4_20261106}"
PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
export CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/gemv-matplotlib
cd "$ROOT"
"$PYTHON" -u -m experiments.gemv_framework_deep_dive.run --out "$OUT" 2>&1 | tee "$OUT/resume.log"
"$PYTHON" -m experiments.gemv_framework_deep_dive.analyze --out "$OUT"
"$PYTHON" -m experiments.gemv_framework_deep_dive.audit --out "$OUT"
