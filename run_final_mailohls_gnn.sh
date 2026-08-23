#!/usr/bin/env bash
set -euo pipefail

# Run from the MailoHLS repository root after applying the GNN patch and
# passing: python -m pytest -q GNN_branch/tests
#
# Deliberately NO --force_regen: the patch changes training/selection logic,
# not the MLIR graph or tensor feature schema.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python GNN_branch/main_GNN.py \
  --dataset mlir \
  --split_json mailohls_runs/mailohls_final_family_split_s123.json \
  --target perf area \
  --target_mode kernel_center \
  --center_aux_weight 0.25 \
  --response_aux_weight 1.0 \
  --standardize_targets \
  --kernel_balanced_loss \
  --kernel_grouped_sampling \
  --kernels_per_batch 4 \
  --points_per_kernel 4 \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --samples_per_kernel_per_epoch 128 \
  --rank_aux_weight 0.10 \
  --rank_temperature 1.0 \
  --rank_tie_relative 0.05 \
  --resource_aux_weight 0.10 \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --checkpoint_objective embedding_rank \
  --min_rank_tau 0.20 \
  --gnn_type transformer \
  --D 64 \
  --num_layers 4 \
  --graph_attention_heads 1 \
  --dropout 0.2 \
  --jkn_mode max \
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
  --target_device xczu7ev-ffvc1156-2-e \
  --clock_period_ns 10.0 \
  --vitis_hls_version 2021.1 \
  --experiment_name gnn_final_mailohls_rank_s123 \
  --random_seed 123
