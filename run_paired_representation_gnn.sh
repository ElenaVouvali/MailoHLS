#!/usr/bin/env bash
# Matched HARP-Rep vs Structured-MLIR surrogate experiment for ASPLOS.
# Only REPRESENTATION and its graph tensor cache are allowed to change.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

REPRESENTATION="${REPRESENTATION:-harp}"
case "$REPRESENTATION" in harp|mlir) ;; *) echo "REPRESENTATION must be harp or mlir" >&2; exit 2;; esac
SEED="${SEED:-123}"
EPOCHS="${EPOCHS:-45}"
NUM_LAYERS="${NUM_LAYERS:-4}"
EXPERIMENT="${EXPERIMENT:-paired_${REPRESENTATION}_multitarget_structural_s${SEED}_development_locked}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
MULTITARGET_PREPROCESSED_DIR="${MULTITARGET_PREPROCESSED_DIR:-artifacts/gnn_datasets/v13_all_targets_s123/preprocessed_CSVS}"
CACHE_ROOT="${PAIRED_CACHE_ROOT:-artifacts/gnn_datasets/paired_representation_s123/development_locked}"
MLIR_CACHE="${MLIR_CACHE:-${CACHE_ROOT}/mlir_dataset/all_kernels}"
HARP_CACHE="${HARP_CACHE:-${CACHE_ROOT}/harp_dataset/all_kernels}"
SPLIT_JSON="${SPLIT_JSON:-mailohls_runs/mailohls_final_family_split_s123.json}"
RESOURCE_BUDGET_BANK="${RESOURCE_BUDGET_BANK:-mailohls_runs/stage1_final_final_adp_s123/validation_resource_budget_bank.json}"
HARP_GRAPH_DIR="${HARP_GRAPH_DIR:-GNN_branch/HARP_graphs}"
MLIR_GRAPH_DIR="${MLIR_GRAPH_DIR:-GNN_branch/MLIR_graphs}"

for f in "$SPLIT_JSON" "$RESOURCE_BUDGET_BANK"; do
  [[ -f "$f" ]] || { echo "Missing required paired-control input: $f" >&2; exit 2; }
done
[[ -d "$MULTITARGET_PREPROCESSED_DIR" ]] || { echo "Missing preprocessed QoR dir: $MULTITARGET_PREPROCESSED_DIR" >&2; exit 2; }
[[ -d "$HARP_GRAPH_DIR" ]] || { echo "Missing HARP graph dir: $HARP_GRAPH_DIR" >&2; exit 2; }
[[ -d "$MLIR_GRAPH_DIR" ]] || { echo "Missing MLIR graph dir: $MLIR_GRAPH_DIR" >&2; exit 2; }

REGEN=()
if [[ "$REBUILD_DATASET" == "1" ]]; then REGEN+=(--force_regen); fi

# Architecture and learning objective are intentionally common across both arms.
COMMON=(
  --subtask train
  --dataset "$REPRESENTATION"
  --target perf area
  --target_mode absolute
  --multi_target_qor
  --preprocessed_csv_dir "$MULTITARGET_PREPROCESSED_DIR"
  --split_json "$SPLIT_JSON"
  --test_kernels serrano-kalman-filter
  --val_kernels machsuite-sort-radix,machsuite-viterbi,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0
  --target_device xczu7ev-ffvc1156-2-e
  --clock_period_ns 10.0
  --mlir_graph_dir "$MLIR_GRAPH_DIR"
  --mlir_dataset_cache_dir "$MLIR_CACHE"
  --harp_graph_dir "$HARP_GRAPH_DIR"
  --harp_dataset_cache_dir "$HARP_CACHE"
  --D 64
  --num_layers "$NUM_LAYERS"
  --dropout 0.2
  --gnn_type transformer
  --graph_attention_heads 1
  --jkn_mode max
  --standardize_targets
  --kernel_balanced_loss
  --kernel_grouped_sampling
  --kernels_per_batch 16
  --points_per_kernel 4
  --samples_per_kernel_per_epoch 128
  --batch_size 64
  --grad_accum_steps 1
  --rank_aux_weight 0.05
  --rank_temperature 1.0
  --rank_tie_relative 0.05
  --resource_aux_weight 0.1
  --resource_budget_bank "$RESOURCE_BUDGET_BANK"
  --resource_budget_count 16
  --resource_budget_min_fraction 0.05
  --resource_boundary_tolerance 0.02
  --checkpoint_objective qualified_lexicographic
  --min_rank_tau 0.2
  --max_kernel_zero_baseline_ratio 1.25
  --loss smooth_l1
  --smooth_l1_beta 0.5
  --lr 3e-5
  --weight_decay 1e-4
  --scheduler plateau
  --plateau_patience 4
  --plateau_factor 0.5
  --early_stopping_patience 8
  --early_stopping_min_delta 1e-4
  --epoch_num "$EPOCHS"
  --random_seed "$SEED"
  --experiment_name "$EXPERIMENT"
  --num_workers 2
  --eval_num_workers 0
)

printf '[PAIRED] representation=%s seed=%s epochs=%s experiment=%s\n' "$REPRESENTATION" "$SEED" "$EPOCHS" "$EXPERIMENT"
printf '[PAIRED] MLIR cache=%s\n' "$MLIR_CACHE"
printf '[PAIRED] HARP cache=%s\n' "$HARP_CACHE"

PYTHONHASHSEED="$SEED" \
python -u GNN_branch/main_GNN.py "${COMMON[@]}" "${REGEN[@]}"
