#!/usr/bin/env bash
set -euo pipefail
cd /home/elvouvali/MailoHLS/worktrees/historical-mlir-gmt-gemv
export PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
export CPU_PYTHON=/home/elvouvali/.hls-llm/bin/python
export PARENT=/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run
export HISTORICAL_ASSETS_ROOT=/home/elvouvali/MailoHLS
export GEXF_DIR="$PWD/GNN_branch/MLIR_graphs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LIONO_GPU_UUID=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35
export MAX_LENGTH=7168
PKG=/home/elvouvali/MailoHLS/experiment_packages/historical_gmt_ab_liono
RUN_ID=historical_gmt_ab_liono_a2_smoke_retry2_20261112 STEPS=2 GMT_STEPS=40 SEEDS=123 STAGE2_EVAL=1 bash "$PKG/run_liono.sh"
echo 'SMOKE_SUCCEEDED_STARTING_FULL'
RUN_ID=historical_gmt_ab_liono_a2_full_20261112 STEPS=300 GMT_STEPS=600 SEEDS=123,456 STAGE2_EVAL=50 bash "$PKG/run_liono.sh"
