#!/usr/bin/env bash
set -euo pipefail
ROOT="${MAILOHLS_ROOT:-$(pwd)}"
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CGEIST:=cgeist}"
export PYTHONHASHSEED=0
for K in chstone-aes chstone-jpeg rosetta-3d-rendering; do
  SRC=""
  [[ -f "${PKG}/${K}/kernel.c" ]] && SRC="${PKG}/${K}/kernel.c"
  [[ -f "${PKG}/${K}/kernel.cpp" ]] && SRC="${PKG}/${K}/kernel.cpp"
  if [[ -z "${SRC}" ]]; then
    echo "[ERROR] ${K} is not materialized. Run: python ${PKG}/materialize_upstream.py" >&2
    exit 2
  fi
  OUTDIR="${PKG}/${K}/code_to_memory_outputs/gexf"
  mkdir -p "${OUTDIR}"
  echo "[MLIR] ${K}"
  python "${ROOT}/GNN_branch/mlir_graph_gen.py" \
    "${SRC}" \
    --cgeist "${CGEIST}" \
    --output "${OUTDIR}/${K}_mlir.gexf"
done
echo "[DONE] Validated MLIR graphs generated."
echo "[NOTE] Rosetta uses ap_int.h; make the Vitis HLS include directory visible to your normal cgeist/Polygeist environment if it is not already on the include path."
echo "Next: export .pt + structural memory using the exact GNN checkpoint/normalization used by the final Stage-2 checkpoint."
