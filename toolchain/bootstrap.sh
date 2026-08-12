#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/lock.env"

usage() {
  echo "usage: $0 <source-dir> <build-dir>" >&2
  echo "Environment: PYTHON, CC, CXX and BUILD_JOBS may select installed tools." >&2
  exit 2
}
[[ $# -eq 2 ]] || usage
SOURCE_DIR=$(realpath -m -- "$1")
BUILD_DIR=$(realpath -m -- "$2")
PYTHON=${PYTHON:-python3.11}
CC=${CC:-cc}
CXX=${CXX:-c++}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

for tool in git cmake ninja sha256sum "$PYTHON" "$CC" "$CXX"; do
  command -v "$tool" >/dev/null || { echo "error: required tool not found: $tool" >&2; exit 1; }
done
[[ "$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "$PYTHON_VERSION" ]] || {
  echo "error: Python $PYTHON_VERSION is required; set PYTHON to that interpreter" >&2; exit 1;
}
$PYTHON -c "import pybind11; assert pybind11.__version__ == '$PYBIND11_VERSION', pybind11.__version__" || {
  echo "error: $PYTHON needs pybind11==$PYBIND11_VERSION" >&2; exit 1;
}

[[ ! -e "$SOURCE_DIR" ]] || { echo "error: source destination already exists: $SOURCE_DIR" >&2; exit 1; }
[[ ! -e "$BUILD_DIR" ]] || { echo "error: build destination already exists: $BUILD_DIR" >&2; exit 1; }

(cd "$SCRIPT_DIR" && sha256sum --check SHA256SUMS)
git clone --recursive "$POLYGEIST_URL" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --detach "$POLYGEIST_COMMIT"
git -C "$SOURCE_DIR" submodule update --init --recursive
actual_llvm=$(git -C "$SOURCE_DIR/llvm-project" rev-parse HEAD)
[[ "$actual_llvm" == "$LLVM_PROJECT_COMMIT" ]] || {
  echo "error: llvm-project is $actual_llvm, expected $LLVM_PROJECT_COMMIT" >&2; exit 1;
}

for patch in "$SCRIPT_DIR"/patches/*.patch; do
  git -C "$SOURCE_DIR" apply --check "$patch"
  git -C "$SOURCE_DIR" apply "$patch"
done

cmake -S "$SOURCE_DIR/llvm-project/llvm" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="$LLVM_ENABLE_PROJECTS" \
  -DLLVM_EXTERNAL_PROJECTS=polygeist \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR="$SOURCE_DIR" \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" \
  '-DLLVM_LIT_ARGS=-sv;--filter-out=CUDA/' \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v "$PYTHON")" \
  -DPOLYGEIST_ENABLE_CUDA=OFF -DPOLYGEIST_ENABLE_ROCM=OFF \
  -DPOLYGEIST_ENABLE_POLYMER=OFF
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" --target \
  cgeist polygeist-opt MLIRPythonModules PolygeistPythonModules

echo "MailoHLS toolchain built successfully"
echo "export CGEIST=$BUILD_DIR/bin/cgeist"
echo "export PYTHONPATH=$BUILD_DIR/tools/mlir/python_packages/mlir_core"
