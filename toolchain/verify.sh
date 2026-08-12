#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
. "$SCRIPT_DIR/lock.env"

usage() { echo "usage: $0 <patched-source-dir> <build-dir> [smoke-output-dir]" >&2; exit 2; }
[[ $# -ge 2 && $# -le 3 ]] || usage
SOURCE_DIR=$(realpath -- "$1")
BUILD_DIR=$(realpath -- "$2")
SMOKE_DIR=$(realpath -m -- "${3:-$BUILD_DIR/mailohls-smoke}")
PYTHON=${PYTHON:-python3.11}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
PYTHON_ROOT="$BUILD_DIR/tools/mlir/python_packages/mlir_core"

(cd "$SCRIPT_DIR" && sha256sum --check SHA256SUMS)
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$POLYGEIST_COMMIT"
test "$(git -C "$SOURCE_DIR/llvm-project" rev-parse HEAD)" = "$LLVM_PROJECT_COMMIT"

probe=$(mktemp -d)
trap 'rm -rf -- "$probe"' EXIT
git clone --no-checkout "$SOURCE_DIR" "$probe/polygeist" >/dev/null
git -C "$probe/polygeist" checkout --detach "$POLYGEIST_COMMIT" >/dev/null
for patch in "$SCRIPT_DIR"/patches/*.patch; do
  git -C "$probe/polygeist" apply --check "$patch"
  git -C "$probe/polygeist" apply "$patch"
done

PYTHONPATH="$PYTHON_ROOT" "$PYTHON" -S -c \
  'from mlir import ir; from mlir._mlir_libs import _mailohls_analysis; print(_mailohls_analysis.__file__)'
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" --target \
  check-mailohls-analysis check-polygeist-python check-cgeist check-polygeist-opt

mkdir -p "$SMOKE_DIR"
source_file="$REPO_ROOT/Data/ApplicationDataset/machsuite-gemm-blocked/gemm.c"
PYTHONHASHSEED=0 PYTHONPATH="$PYTHON_ROOT" "$PYTHON" \
  "$REPO_ROOT/GNN_branch/mlir_graph_gen.py" "$source_file" \
  --cgeist "$BUILD_DIR/bin/cgeist" \
  --output "$SMOKE_DIR/bbgemm.gexf" \
  --mlir-output "$SMOKE_DIR/bbgemm.mlir" \
  --analysis-output "$SMOKE_DIR/bbgemm.analysis.json"
test -s "$SMOKE_DIR/bbgemm.gexf"
echo "MailoHLS verification complete: $SMOKE_DIR/bbgemm.gexf"
