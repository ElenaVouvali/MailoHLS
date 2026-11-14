#!/usr/bin/env bash
# Build all nine compiler MLIR graphs using the exact historical GNN training graph generator.
# Must run on a machine with cgeist and native MLIR analysis Python bindings.
set -euo pipefail
ROOT=${1:?Usage: bash prepare_stage2_eval_graphs.sh MAILOHLS_WORKTREE STAGED_SOURCES OUTPUT_GRAPH_DIR}
SOURCES=${2:?Usage: bash prepare_stage2_eval_graphs.sh MAILOHLS_WORKTREE STAGED_SOURCES OUTPUT_GRAPH_DIR}
OUT=${3:?Usage: bash prepare_stage2_eval_graphs.sh MAILOHLS_WORKTREE STAGED_SOURCES OUTPUT_GRAPH_DIR}
ROOT=$(realpath "$ROOT"); SOURCES=$(realpath "$SOURCES")
GENERATOR="$ROOT/Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py"
[[ -s "$GENERATOR" ]] || { echo "Historical generator missing: $GENERATOR" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite graph output: $OUT" >&2; exit 2; }
: "${CGEIST:=cgeist}"
command -v "$CGEIST" >/dev/null 2>&1 || { echo "cgeist missing; set CGEIST=/absolute/path/to/cgeist" >&2; exit 2; }

# Prefer an explicitly selected Python environment; otherwise try known installations.
if [[ -n "${GRAPH_PY:-}" ]]; then
  PY="$GRAPH_PY"
else
  PY=""
  for candidate in "$(command -v python)" /home/elvouvali/miniforge3/envs/*/bin/python; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c 'from mlir import ir; from mlir._mlir_libs import _mailohls_analysis; import networkx' >/dev/null 2>&1; then
      PY="$candidate"
      break
    fi
  done
fi
[[ -n "$PY" && -x "$PY" ]] || { echo "No Python with MLIR Python bindings + _mailohls_analysis; set GRAPH_PY to the previously used MLIR environment" >&2; exit 2; }
"$PY" -c 'from mlir import ir; from mlir._mlir_libs import _mailohls_analysis; import networkx' || exit 2

# GNN checkpoint contract pins the generator bytes: do not substitute the newer one.
"$PY" - "$ROOT" "$GENERATOR" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,generator=map(Path,sys.argv[1:])
contract=json.loads((root/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/gnn_checkpoint_contract.json').read_text())
expected=contract['training_artifacts']['files']['mlir_graph_gen.py']['sha256']
actual=hashlib.sha256(generator.read_bytes()).hexdigest()
if actual!=expected: raise RuntimeError(f'Historical graph generator SHA changed: {actual} != {expected}')
print('[GENERATOR] authenticated historical MLIR generator SHA:',actual)
PY

EXTRA_CFLAGS=()
if [[ -n "${HLS_INCLUDE_DIR:-}" ]]; then
  [[ -d "$HLS_INCLUDE_DIR" ]] || { echo "Invalid HLS_INCLUDE_DIR=$HLS_INCLUDE_DIR" >&2; exit 2; }
  EXTRA_CFLAGS+=("--cflag=-I$HLS_INCLUDE_DIR")
fi

mkdir -p "$OUT"
for K in GAN Kalman chstone-aes chstone-jpeg covariance jacobi-2d rosetta-3d-rendering syr2k trmm-opt; do
  folder="$SOURCES/$K"
  SRC=""
  [[ -f "$folder/source.cpp" ]] && SRC="$folder/source.cpp"
  [[ -f "$folder/source.c" ]] && SRC="$folder/source.c"
  [[ -n "$SRC" && -s "$folder/kernel_info.txt" ]] || { echo "Incomplete input for $K" >&2; exit 2; }
  echo "[GRAPH] $K -> $OUT/$K.gexf"
  # The historical generator rejects non-exact action mapping and missing native analysis.
  # --cflag supplies local headers. Site IDs are NOT substituted/renumbered.
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$GENERATOR" "$SRC" \
    --cgeist "$CGEIST" \
    --cflag="-I$folder" \
    "${EXTRA_CFLAGS[@]}" \
    --output "$OUT/$K.gexf" \
    --analysis-output "$OUT/$K.analysis.json" \
    --mlir-output "$OUT/$K.mlir"
done
printf '[GRAPHS COMPLETE] %s\n' "$OUT"
