#!/usr/bin/env bash
set -euo pipefail

# Final MailoHLS structural-GNN training run.
# Run from the MailoHLS repository root after applying and testing
# the final Stage-1/GNN patch.
#
# The neutral references supervise reference-delta QoR during GNN training.
# They are NOT consumed when static_pre_npt structural memory is later exported.

MANIFEST="GNN_branch/baselines/neutral_vitis_2021_1.csv"
SPLIT="mailohls_runs/mailohls_final_family_split_s123.json"
EXPERIMENT="${EXPERIMENT:-gnn_final_anchored_pairwise_s123}"
STAGE1_OUT="${STAGE1_OUT:-mailohls_runs/stage1_final_final_adp_s123}"
BUDGET_BANK="${RESOURCE_BUDGET_BANK:-${STAGE1_OUT}/validation_resource_budget_bank.json}"

[[ -f "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 2; }
[[ -f "${SPLIT}" ]] || { echo "Missing ${SPLIT}" >&2; exit 2; }
[[ -f "${BUDGET_BANK}" ]] || {
  echo "Missing exact Stage-1 validation budget bank: ${BUDGET_BANK}" >&2
  echo "Start the final Stage-1 run first, or set RESOURCE_BUDGET_BANK." >&2
  exit 2
}

# The checked-in development manifest currently lacked these two development
# kernels. generate_neutral_baselines.py appends requested kernels without
# deleting already authenticated successful rows.
missing=()
for kernel in spcl_example_01; do
  if ! grep -q "^${kernel},success," "${MANIFEST}"; then
    missing+=("${kernel}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  IFS=,
  echo "Neutral reference(s) missing for: ${missing[*]}" >&2
  echo "Generate them with Vitis HLS 2021.1 before training:" >&2
  echo "  python GNN_branch/generate_neutral_baselines.py \\" >&2
  echo "    --kernels ${missing[*]} \\" >&2
  echo "    --output ${MANIFEST} \\" >&2
  echo "    --work-dir neutral_baseline_build_missing" >&2
  exit 2
fi

if [[ -d "Checkpoints/${EXPERIMENT}" ]]; then
  echo "Refusing to reuse existing experiment directory: Checkpoints/${EXPERIMENT}" >&2
  echo "Rename/remove it or choose a new --experiment_name for a scratch run." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python -u GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --target perf area \
  --target_mode reference_delta \
  --baseline_manifest "${MANIFEST}" \
  --target_device xczu7ev-ffvc1156-2-e \
  --clock_period_ns 10.0 \
  --vitis_hls_version 2021.1 \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --standardize_targets \
  --qor_output_init_scale 0.1 \
  --kernel_balanced_loss \
  --kernel_grouped_sampling \
  --kernels_per_batch 4 \
  --development_exclude_kernels rodinia_lud_1_tiling_0 \
  --points_per_kernel 4 \
  --samples_per_kernel_per_epoch 128 \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --rank_aux_weight 0 \
  --pairwise_delta_weight 0.05 \
  --pairwise_delta_start_epoch 3 \
  --pairwise_delta_ramp_epochs 3 \
  --pairwise_calibration_stable_epochs 1 \
  --pairwise_calibration_tolerance 0.20 \
  --rank_tie_relative 0.05 \
  --resource_aux_weight 0.10 \
  --resource_budget_bank "${BUDGET_BANK}" \
  --resource_budget_count 16 \
  --resource_budget_min_fraction 0.05 \
  --resource_boundary_tolerance 0.02 \
  --checkpoint_objective qualified_lexicographic \
  --min_rank_tau 0.20 \
  --max_kernel_zero_baseline_ratio 1.10 \
  --kernel_zero_baseline_additive_tolerance 0.001 \
  --epoch_num 15 \
  --lr 3e-5 \
  --scheduler plateau \
  --warmup_epochs 3 \
  --plateau_patience 4 \
  --plateau_factor 0.5 \
  --early_stopping_patience 4 \
  --early_stopping_min_delta 1e-4 \
  --split_json "${SPLIT}" \
  --num_features 403 \
  --edge_dim 82 \
  --random_seed 123 \
  --num_workers 2 \
  --eval_num_workers 0 \
  --experiment_name "${EXPERIMENT}"
