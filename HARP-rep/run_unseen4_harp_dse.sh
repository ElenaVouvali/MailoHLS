#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/elvouvali/HARP-paper-rep-unseen4
SRC="$ROOT/src"

REG="$ROOT/final_unseen4_checkpoints/harp_regression_unseen4_s123_best_val.pth"
CLS="$ROOT/final_unseen4_checkpoints/harp_classifier_unseen4_s123_best_val.pth"

OUTROOT="$ROOT/paper_rep_logs/dse_unseen4"

V100_UUID=$(
  nvidia-smi --query-gpu=uuid,name --format=csv,noheader |
  awk -F', ' '$2 ~ /V100/ {print $1; exit}'
)

if [ -z "$V100_UUID" ]; then
    echo "FATAL: could not locate Tesla V100"
    exit 70
fi

VISIBLE_GPU=$(
  CUDA_VISIBLE_DEVICES="$V100_UUID"   python -c 'import torch; print(torch.cuda.get_device_name(0))'
)

echo "Selected GPU UUID=$V100_UUID"
echo "CUDA cuda:0=$VISIBLE_GPU"

if [[ "$VISIBLE_GPU" != *V100* ]]; then
    echo "FATAL: selected device is not a V100"
    exit 71
fi

mkdir -p "$OUTROOT"

test -f "$REG"
test -f "$CLS"

cd "$SRC"

for K in covariance jacobi-2d syr2k trmm-opt
do
    STAMP=$(date +%Y%m%d_%H%M%S)
    OUT="$OUTROOT/${K}_${STAMP}.out"

    # Write this BEFORE launching so the active run is always discoverable.
    echo "$OUT" > "$OUTROOT/${K}.latest_out"

    echo "============================================================"
    echo "[$(date)] START HARP-DSE: $K"
    echo "REG=$REG"
    echo "CLS=$CLS"
    echo "OUT=$OUT"
    echo "GPU=$V100_UUID ($VISIBLE_GPU)"
    echo "============================================================"

    START=$(date +%s)

    CUDA_VISIBLE_DEVICES="$V100_UUID" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=123 \
    python -u -W ignore main.py \
      --task regression \
      --subtask dse \
      --target_kernel "$K" \
      --all_kernels "" \
      --model_path "$REG" \
      --class_model_path "$CLS" \
      --device cuda:0 \
      > "$OUT" 2>&1

    CODE=$?
    END=$(date +%s)

    echo "$CODE" > "$OUTROOT/${K}.exitcode"
    echo "$((END - START))" > "$OUTROOT/${K}.wall_seconds"

    if [ "$CODE" -ne 0 ]; then
        echo "FATAL: $K exited with code $CODE"
        exit "$CODE"
    fi

    RUN_DIR=$(
        grep 'Logging to ' "$OUT" |
        tail -1 |
        sed 's/.*Logging to //'
    )

    echo "$RUN_DIR" > "$OUTROOT/${K}.run_dir"

    if [ ! -f "$RUN_DIR/$K.pickle" ]; then
        echo "FATAL: expected $RUN_DIR/$K.pickle was not produced"
        exit 20
    fi

    echo "[$(date)] FINISHED HARP-DSE: $K"
    echo "RUN_DIR=$RUN_DIR"

    grep -E \
      'Explored [0-9]+ points|Best Results Found:' \
      "$RUN_DIR/log.txt" | tail -10

done

echo "============================================================"
echo "[$(date)] ALL FOUR HARP DSE RUNS COMPLETE"
echo "============================================================"
