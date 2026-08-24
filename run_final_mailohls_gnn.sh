#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EXP="${EXP:-gnn_final_reference_delta_hardware_regression_s123}"
CUDA_VISIBLE_DEVICES="$GPU" \
python -u GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --target perf \
  --target_mode reference_delta \
  --baseline_manifest GNN_branch/baselines/neutral_vitis_2021_1.csv \
  --development_exclude_kernels rodinia_lud_1_tiling_0 \
  --target_device xczu7ev-ffvc1156-2-e \
  --clock_period_ns 10.0 \
  --vitis_hls_version 2021.1 \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --standardize_targets \
  --kernel_balanced_loss \
  --rank_aux_weight 0 \
  --resource_aux_weight 1.0 \
  --checkpoint_objective hardware_regression \
  --gnn_type transformer \
  --D 64 \
  --num_layers 4 \
  --graph_attention_heads 1 \
  --dropout 0.2 \
  --jkn_mode max \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --lr 3e-5 \
  --weight_decay 1e-4 \
  --scheduler plateau \
  --warmup_epochs 3 \
  --plateau_patience 4 \
  --plateau_factor 0.5 \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 1e-4 \
  --epoch_num 100 \
  --num_workers 2 \
  --eval_num_workers 0 \
  --split_json mailohls_runs/mailohls_final_family_split_s123.json \
  --experiment_name "$EXP" \
  --random_seed 123
