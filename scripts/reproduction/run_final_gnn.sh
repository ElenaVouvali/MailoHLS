#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

seed="${MAILOHLS_SEED:-123}"
split="${MAILOHLS_FAMILY_SPLIT:-$repo_root/mailohls_runs/mailohls_final_family_split_s${seed}.json}"
batch_size="${MAILOHLS_GNN_BATCH_SIZE:-8}"
kernels_per_batch="${MAILOHLS_GNN_KERNELS_PER_BATCH:-4}"
points_per_kernel="${MAILOHLS_GNN_POINTS_PER_KERNEL:-2}"
grad_accum_steps="${MAILOHLS_GNN_GRAD_ACCUM_STEPS:-8}"
target_device="${MAILOHLS_GNN_TARGET_DEVICE:-xczu7ev-ffvc1156-2-e}"
target_clock="${MAILOHLS_GNN_CLOCK_NS:-10}"
experiment="${MAILOHLS_GNN_EXPERIMENT:-gnn_final_canonical_s${seed}}"

if [[ ! -f "$split" ]]; then
  printf 'Required shared experiment split does not exist: %s\n' "$split" >&2
  exit 1
fi

python -u GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --device cuda:0 \
  --split_json "$split" \
  --force_regen \
  --target_device "$target_device" \
  --clock_period_ns "$target_clock" \
  --target perf area \
  --target_mode absolute \
  --checkpoint_objective absolute \
  --D 64 \
  --num_layers 4 \
  --graph_attention_heads 1 \
  --jkn_mode max \
  --dropout 0.20 \
  --standardize_targets \
  --kernel_balanced_loss \
  --kernel_grouped_sampling \
  --batch_size "$batch_size" \
  --grad_accum_steps "$grad_accum_steps" \
  --kernels_per_batch "$kernels_per_batch" \
  --points_per_kernel "$points_per_kernel" \
  --samples_per_kernel_per_epoch 128 \
  --rank_aux_weight 0 \
  --resource_aux_weight 0.10 \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --lr 0.00003 \
  --weight_decay 0.0001 \
  --scheduler plateau \
  --warmup_epochs 3 \
  --plateau_patience 4 \
  --early_stopping_patience 15 \
  --epoch_num 100 \
  --num_workers 2 \
  --eval_num_workers 0 \
  --random_seed "$seed" \
  --experiment_name "$experiment" \
  "$@"
