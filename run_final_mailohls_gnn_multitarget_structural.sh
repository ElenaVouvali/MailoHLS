#!/usr/bin/env bash
set -euo pipefail

# Final recovery GNN challenger: absolute QoR across every measured
# device/clock target, selected solely for held-out structural ranking.
# All generated data lives in isolated versioned directories so the existing
# single-target epoch-9 checkpoint remains reproducible and exportable.

SPLIT="mailohls_runs/mailohls_final_family_split_s123.json"
EXPERIMENT="${EXPERIMENT:-gnn_final_multitarget_structural_s123}"
STAGE1_OUT="${STAGE1_OUT:-mailohls_runs/stage1_final_final_adp_s123}"
BUDGET_BANK="${RESOURCE_BUDGET_BANK:-${STAGE1_OUT}/validation_resource_budget_bank.json}"
PREPROCESSED_DIR="${MULTITARGET_PREPROCESSED_DIR:-artifacts/gnn_datasets/v13_all_targets_s123/preprocessed_CSVS}"
CACHE_DIR="${MULTITARGET_MLIR_CACHE_DIR:-artifacts/gnn_datasets/v13_all_targets_s123/MLIR_dataset/all_kernels}"

[[ -f "${SPLIT}" ]] || { echo "Missing ${SPLIT}" >&2; exit 2; }
[[ -f "${BUDGET_BANK}" ]] || {
  echo "Missing exact Stage-1 validation budget bank: ${BUDGET_BANK}" >&2
  exit 2
}
if [[ -d "Checkpoints/${EXPERIMENT}" ]]; then
  echo "Refusing to reuse existing experiment directory: Checkpoints/${EXPERIMENT}" >&2
  echo "Choose a fresh name with EXPERIMENT=<new-name>." >&2
  exit 2
fi

if [[ ! -f "${PREPROCESSED_DIR}/preprocessing_manifest.json" ]]; then
  python -u Preprocessing/data_preprocess.py \
    --mode gnn \
    --all-targets \
    --exclude-kernels serrano-kalman-filter \
    --output-dir "${PREPROCESSED_DIR}" \
    --force
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python -u GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --target perf area \
  --target_mode absolute \
  --multi_target_qor \
  --preprocessed_csv_dir "${PREPROCESSED_DIR}" \
  --mlir_dataset_cache_dir "${CACHE_DIR}" \
  --force_regen \
  --target_device xczu7ev-ffvc1156-2-e \
  --clock_period_ns 10.0 \
  --vitis_hls_version 2021.1 \
  --gnn_type transformer \
  --graph_attention_heads 4 \
  --graph_residual_beta \
  --graph_layer_norm \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --standardize_targets \
  --dropout 0.10 \
  --qor_output_init_scale 0.10 \
  --kernel_balanced_loss \
  --kernel_grouped_sampling \
  --kernels_per_batch 4 \
  --points_per_kernel 2 \
  --samples_per_kernel_per_epoch 64 \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --rank_aux_weight 0.05 \
  --rank_tie_relative 0.05 \
  --pairwise_delta_weight 0 \
  --resource_aux_weight 0.10 \
  --resource_budget_bank "${BUDGET_BANK}" \
  --resource_budget_count 16 \
  --resource_budget_min_fraction 0.05 \
  --resource_boundary_tolerance 0.02 \
  --checkpoint_objective structural_rank \
  --min_rank_tau 0.20 \
  --epoch_num 45 \
  --lr 5e-4 \
  --scheduler plateau \
  --warmup_epochs 5 \
  --plateau_patience 5 \
  --plateau_factor 0.5 \
  --early_stopping_patience 20 \
  --early_stopping_min_delta 1e-4 \
  --split_json "${SPLIT}" \
  --num_features 403 \
  --edge_dim 82 \
  --random_seed 123 \
  --num_workers 2 \
  --eval_num_workers 0 \
  --experiment_name "${EXPERIMENT}"
