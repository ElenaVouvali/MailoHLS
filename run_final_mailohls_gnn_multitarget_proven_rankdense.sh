#!/usr/bin/env bash
set -euo pipefail

# Controlled multi-target follow-up to absolute rank-dense.  The only
# scientific change versus the rank-dense contract is --multi_target_qor and
# its isolated all-target v13 data; sampler geometry and optimization remain
# identical.
SPLIT="mailohls_runs/mailohls_final_family_split_s123.json"
EXPERIMENT="${EXPERIMENT:-gnn_final_multitarget_proven_rankdense_s123}"
STAGE1_OUT="${STAGE1_OUT:-mailohls_runs/stage1_final_final_adp_s123}"
BUDGET_BANK="${RESOURCE_BUDGET_BANK:-${STAGE1_OUT}/validation_resource_budget_bank.json}"
PREPROCESSED_DIR="${MULTITARGET_PREPROCESSED_DIR:-artifacts/gnn_datasets/v13_all_targets_s123/preprocessed_CSVS}"
CACHE_DIR="${MULTITARGET_MLIR_CACHE_DIR:-artifacts/gnn_datasets/v13_all_targets_s123/MLIR_dataset/all_kernels}"

[[ -f "${SPLIT}" ]] || { echo "Missing ${SPLIT}" >&2; exit 2; }
[[ -f "${BUDGET_BANK}" ]] || { echo "Missing exact Stage-1 validation budget bank: ${BUDGET_BANK}" >&2; exit 2; }
if [[ -d "Checkpoints/${EXPERIMENT}" ]]; then
  echo "Refusing to reuse existing experiment directory: Checkpoints/${EXPERIMENT}" >&2
  exit 2
fi
if [[ ! -f "${PREPROCESSED_DIR}/preprocessing_manifest.json" ]]; then
  python -u Preprocessing/data_preprocess.py --mode gnn --all-targets \
    --exclude-kernels serrano-kalman-filter --output-dir "${PREPROCESSED_DIR}" --force
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python -u GNN_branch/main_GNN.py \
  --dataset mlir --subtask train --target perf area --target_mode absolute \
  --multi_target_qor --preprocessed_csv_dir "${PREPROCESSED_DIR}" \
  --mlir_dataset_cache_dir "${CACHE_DIR}" --force_regen \
  --target_device xczu7ev-ffvc1156-2-e --clock_period_ns 10.0 \
  --vitis_hls_version 2021.1 --gnn_type transformer --loss smooth_l1 \
  --graph_attention_heads 1 \
  --smooth_l1_beta 0.5 --standardize_targets --dropout 0.2 \
  --qor_output_init_scale 1.0 --kernel_balanced_loss --kernel_grouped_sampling \
  --kernels_per_batch 8 --points_per_kernel 8 \
  --samples_per_kernel_per_epoch 128 --batch_size 64 --grad_accum_steps 1 \
  --rank_aux_weight 0.05 --rank_tie_relative 0.05 \
  --pairwise_delta_weight 0 --resource_aux_weight 0.10 \
  --resource_budget_bank "${BUDGET_BANK}" --resource_budget_count 16 \
  --resource_budget_min_fraction 0.05 --resource_boundary_tolerance 0.02 \
  --checkpoint_objective structural_rank --min_rank_tau 0.20 \
  --epoch_num 40 --lr 3e-5 --scheduler plateau --warmup_epochs 0 \
  --plateau_patience 4 --plateau_factor 0.5 --early_stopping_patience 8 \
  --early_stopping_min_delta 1e-4 --split_json "${SPLIT}" \
  --development_exclude_kernels rodinia_lud_1_tiling_0 \
  --num_features 403 --edge_dim 82 --random_seed 123 \
  --num_workers 2 --eval_num_workers 0 --experiment_name "${EXPERIMENT}"
