#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

seed="${MAILOHLS_SEED:-123}"
split="${MAILOHLS_FAMILY_SPLIT:-$repo_root/mailohls_runs/mailohls_final_family_split_s${seed}.json}"
batch_size="${MAILOHLS_GNN_BATCH_SIZE:-8}"
kernels_per_batch="${MAILOHLS_GNN_KERNELS_PER_BATCH:-4}"
experiment="${MAILOHLS_GNN_EXPERIMENT:-gnn_final_multitarget_s${seed}}"

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
  --multi_target_qor \
  --target perf area \
  --target_mode absolute \
  --checkpoint_objective absolute \
  --D 64 \
  --num_layers 4 \
  --graph_attention_heads 4 \
  --graph_residual_beta \
  --graph_layer_norm \
  --dropout 0.10 \
  --standardize_targets \
  --kernel_balanced_loss \
  --kernel_grouped_sampling \
  --batch_size "$batch_size" \
  --kernels_per_batch "$kernels_per_batch" \
  --points_per_kernel 2 \
  --samples_per_kernel_per_epoch 64 \
  --rank_aux_weight 0.05 \
  --rank_temperature 0.5 \
  --rank_tie_epsilon 0.02 \
  --resource_aux_weight 0.10 \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --scheduler plateau \
  --warmup_epochs 5 \
  --plateau_patience 5 \
  --early_stopping_patience 20 \
  --epoch_num 150 \
  --num_workers 2 \
  --eval_num_workers 0 \
  --random_seed "$seed" \
  --experiment_name "$experiment" \
  "$@"
