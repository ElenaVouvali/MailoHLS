#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/elvouvali/HARP-paper-rep-unseen4
SRC="$ROOT/src"

REG="$ROOT/final_unseen4_checkpoints/harp_regression_unseen4_s123_best_val.pth"
CLS="$ROOT/final_unseen4_checkpoints/harp_classifier_unseen4_s123_best_val.pth"

OUT="$ROOT/external_harp_dse_results"
mkdir -p "$OUT"

# Previously verified V100 on liono.
V100_UUID="GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb"

export CUDA_VISIBLE_DEVICES="$V100_UUID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=123
export DGLBACKEND=pytorch

# HARP_CUDA_LIBRARY_PATH_FIX
HARP_ENV=/home/elvouvali/miniforge3/envs/harp-rep
TORCH_LIB="$HARP_ENV/lib/python3.7/site-packages/torch/lib"
export LD_LIBRARY_PATH="$HARP_ENV/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python - <<'PYCUDA'
import torch
import torch_sparse
import torch_scatter
import torch_geometric

print("[CUDA PREFLIGHT] torch =", torch.__version__)
print("[CUDA PREFLIGHT] torch CUDA =", torch.version.cuda)
print("[CUDA PREFLIGHT] torch_sparse =", torch_sparse.__version__)
print("[CUDA PREFLIGHT] torch_scatter =", torch_scatter.__version__)
print("[CUDA PREFLIGHT] torch_geometric =", torch_geometric.__version__)
print("[CUDA PREFLIGHT] CUDA =", torch.cuda.is_available())

assert torch.cuda.is_available()
print("[CUDA PREFLIGHT] GPU =", torch.cuda.get_device_name(0))
print("[CUDA PREFLIGHT] PASS")
PYCUDA

if [ $? -ne 0 ]; then
    echo "FATAL: PyG/CUDA preflight failed"
    exit 90
fi


cd "$SRC" || exit 1

echo "============================================================"
echo "External HARP zero-shot DSE"
echo "Started: $(date -Is)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Regression: $REG"
echo "Classifier: $CLS"
echo "============================================================"

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total \
  --format=csv || true

run_kernel () {
    K="$1"
    STDOUT="$OUT/${K}.stdout.log"
    EXITFILE="$OUT/${K}.exitcode"
    LOGDIRFILE="$OUT/${K}.logdir"

    echo
    echo "################################################################"
    echo "START $K : $(date -Is)"
    echo "################################################################"

    python -u -W ignore main.py \
      --task regression \
      --subtask dse \
      --target_kernel "$K" \
      --all_kernels "" \
      --model_path "$REG" \
      --class_model_path "$CLS" \
      --graph_type extended-pseudo-block-connected-hierarchy \
      --dataset harp \
      --v_db v21 \
      --random_seed 123 \
      --device cuda:0 \
      2>&1 | tee "$STDOUT"

    CODE=${PIPESTATUS[0]}
    echo "$CODE" > "$EXITFILE"

    LOGDIR=$(
      grep -m1 '^Logging to ' "$STDOUT" \
      | sed 's/^Logging to //'
    )

    if [ -n "$LOGDIR" ]; then
        echo "$LOGDIR" > "$LOGDIRFILE"

        echo
        echo "HARP log directory:"
        echo "$LOGDIR"

        if [ -f "$LOGDIR/${K}.pickle" ]; then
            cp \
              "$LOGDIR/${K}.pickle" \
              "$OUT/${K}_top10.pickle"

            echo "COPIED TOP10: $OUT/${K}_top10.pickle"
        else
            echo "WARNING: $LOGDIR/${K}.pickle not found"
        fi
    else
        echo "WARNING: could not identify HARP log directory"
    fi

    echo
    echo "FINISH $K : $(date -Is)"
    echo "EXIT $K = $CODE"
    echo

    return "$CODE"
}

# Higher-priority external application first.
run_kernel gan
GAN_CODE=$?

# Continue to Kalman even if GAN fails.
run_kernel kalman
KALMAN_CODE=$?

{
    echo "gan=$GAN_CODE"
    echo "kalman=$KALMAN_CODE"
    echo "finished=$(date -Is)"
} > "$OUT/queue_summary.txt"

echo
echo "============================================================"
echo "QUEUE COMPLETE"
cat "$OUT/queue_summary.txt"
echo "============================================================"
