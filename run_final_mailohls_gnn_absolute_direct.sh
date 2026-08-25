#!/usr/bin/env bash
set -euo pipefail

# Final structural-GNN attempt using the empirically stronger absolute-QoR,
# direct-head formulation. This deliberately preserves the current dataset,
# feature schema, resource-budget validation, checkpoint contracts, and
# provenance machinery. Reference-delta anchoring is not used in absolute mode.

SPLIT="mailohls_runs/mailohls_final_family_split_s123.json"
EXPERIMENT="${EXPERIMENT:-gnn_final_absolute_direct_rank_s123}"
STAGE1_OUT="${STAGE1_OUT:-mailohls_runs/stage1_final_final_adp_s123}"
BUDGET_BANK="${RESOURCE_BUDGET_BANK:-${STAGE1_OUT}/validation_resource_budget_bank.json}"

[[ -f "${SPLIT}" ]] || { echo "Missing ${SPLIT}" >&2; exit 2; }
[[ -f "${BUDGET_BANK}" ]] || {
  echo "Missing exact Stage-1 validation budget bank: ${BUDGET_BANK}" >&2
  echo "Start the final Stage-1 run first, or set RESOURCE_BUDGET_BANK." >&2
  exit 2
}

if [[ -d "Checkpoints/${EXPERIMENT}" ]]; then
  echo "Refusing to reuse existing experiment directory: Checkpoints/${EXPERIMENT}" >&2
  echo "Choose a fresh name with EXPERIMENT=<new-name>." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python -u GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --target perf area \
  --target_mode absolute \
  --target_device xczu7ev-ffvc1156-2-e \
  --clock_period_ns 10.0 \
  --vitis_hls_version 2021.1 \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --standardize_targets \
  --qor_output_init_scale 1.0 \
  --kernel_balanced_loss \
  --kernel_grouped_sampling \
  --kernels_per_batch 16 \
  --development_exclude_kernels rodinia_lud_1_tiling_0 \
  --points_per_kernel 4 \
  --samples_per_kernel_per_epoch 128 \
  --batch_size 64 \
  --grad_accum_steps 1 \
  --rank_aux_weight 0.05 \
  --pairwise_delta_weight 0 \
  --rank_tie_relative 0.05 \
  --resource_aux_weight 0.10 \
  --resource_budget_bank "${BUDGET_BANK}" \
  --resource_budget_count 16 \
  --resource_budget_min_fraction 0.05 \
  --resource_boundary_tolerance 0.02 \
  --checkpoint_objective qualified_lexicographic \
  --min_rank_tau 0.20 \
  --max_kernel_zero_baseline_ratio 1.25 \
  --kernel_zero_baseline_additive_tolerance 0.001 \
  --epoch_num 40 \
  --lr 3e-5 \
  --scheduler plateau \
  --warmup_epochs 0 \
  --plateau_patience 4 \
  --plateau_factor 0.5 \
  --early_stopping_patience 8 \
  --early_stopping_min_delta 1e-4 \
  --split_json "${SPLIT}" \
  --num_features 403 \
  --edge_dim 82 \
  --random_seed 123 \
  --num_workers 2 \
  --eval_num_workers 0 \
  --experiment_name "${EXPERIMENT}"
