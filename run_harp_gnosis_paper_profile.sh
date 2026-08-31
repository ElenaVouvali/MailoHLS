#!/usr/bin/env bash
# HARP-GNΩSIS (our port), distinct from HARP-Rep.
# This maps the released HARP training profile (6 GNN layers, dropout 0.1,
# lr=1e-3, cosine+linear warmup, weight_decay=1e-4, batch=64, MSE) onto the
# current MailoHLS/HARP graph backend and the frozen GNOSIS family split.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SEED="${SEED:-123}"
EPOCHS="${EPOCHS:-1500}"
EXPERIMENT="${EXPERIMENT:-harp_gnosis_paper_profile_s${SEED}}"
CANONICAL_PREPROCESSED_DIR="${HARP_CANONICAL_PREPROCESSED_DIR:-GNN_branch/Data/preprocessed_CSVS}"
SPLIT_JSON="${SPLIT_JSON:-mailohls_runs/mailohls_final_family_split_s123.json}"
CACHE_ROOT="${HARP_PAPER_CACHE_ROOT:-artifacts/gnn_datasets/harp_gnosis_paper_profile_s${SEED}}"
MLIR_CACHE="${MLIR_CACHE:-${CACHE_ROOT}/canonical_point_source}"
HARP_CACHE="${HARP_CACHE:-${CACHE_ROOT}/harp_dataset}"
HARP_GRAPH_DIR="${HARP_GRAPH_DIR:-GNN_branch/HARP_graphs}"
MLIR_GRAPH_DIR="${MLIR_GRAPH_DIR:-GNN_branch/MLIR_graphs}"

[[ -d "$CANONICAL_PREPROCESSED_DIR" ]] || { echo "Missing canonical 10-ns GNOSIS CSV dir: $CANONICAL_PREPROCESSED_DIR" >&2; exit 2; }
[[ -f "$SPLIT_JSON" ]] || { echo "Missing split: $SPLIT_JSON" >&2; exit 2; }

REGEN=()
if [[ "${REBUILD_DATASET:-1}" == "1" ]]; then REGEN+=(--force_regen); fi

PYTHONHASHSEED="$SEED" \
python -u GNN_branch/main_GNN.py \
  --subtask train \
  --dataset harp \
  --target perf \
  --target_mode absolute \
  --preprocessed_csv_dir "$CANONICAL_PREPROCESSED_DIR" \
  --split_json "$SPLIT_JSON" \
  --test_kernels serrano-kalman-filter \
  --val_kernels machsuite-sort-radix,machsuite-viterbi,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0 \
  --target_device xczu7ev-ffvc1156-2-e \
  --clock_period_ns 10.0 \
  --mlir_graph_dir "$MLIR_GRAPH_DIR" \
  --mlir_dataset_cache_dir "$MLIR_CACHE" \
  --harp_graph_dir "$HARP_GRAPH_DIR" \
  --harp_dataset_cache_dir "$HARP_CACHE" \
  --D 64 \
  --num_layers 6 \
  --dropout 0.1 \
  --gnn_type transformer \
  --jkn_mode max \
  --batch_size 64 \
  --rank_aux_weight 0 \
  --pairwise_delta_weight 0 \
  --resource_aux_weight 1.0 \
  --checkpoint_objective hardware_regression \
  --loss mse \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --warmup linear \
  --warmup_epochs 3 \
  --epoch_num "$EPOCHS" \
  --random_seed "$SEED" \
  --experiment_name "$EXPERIMENT" \
  --num_workers 2 \
  --eval_num_workers 0 \
  "${REGEN[@]}"
