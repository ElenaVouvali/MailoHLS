#!/usr/bin/env bash
set -euo pipefail

NAME="$1"
DIR="$2"
MODE="$3"

cd "$DIR"

if [ "$NAME" = "gan" ] || [ "$NAME" = "kalman" ]; then

    ROOT="/home/elvouvali/HARP-paper-rep-unseen4"

    LLVM10="/home/elvouvali/miniforge3/envs/harp-llvm10/bin"
    CXX="$LLVM10/clang++"

    APINT_INC="$ROOT/external_harp_ready/toolchain_headers/apint_include"

    test -x "$CXX" || {
        echo "ERROR: LLVM10 clang++ missing: $CXX"
        exit 2
    }

    test -f "$APINT_INC/ap_int.h" || {
        echo "ERROR: ap_int.h missing"
        exit 3
    }

    test -f "$APINT_INC/ap_fixed.h" || {
        echo "ERROR: ap_fixed.h missing"
        exit 4
    }

    DEBUG=()
    if [ "$MODE" = "multi_modality" ]; then
        DEBUG=(-g)
    fi

    echo "[HARP-EXTERNAL]"
    echo "  kernel   = $NAME"
    echo "  compiler = $CXX"
    echo "  LLVM     = 10"
    echo "  ap_inc   = $APINT_INC"

    "$CXX" \
      -x c++ \
      -std=c++14 \
      -Wno-unknown-pragmas \
      -emit-llvm \
      -fno-discard-value-names \
      "${DEBUG[@]}" \
      -I. \
      -I"$APINT_INC" \
      -S -c \
      "$NAME.c" \
      -o "$NAME.ll"

else

    # Preserve the released HARP behavior for existing benchmark kernels.
    if command -v clang-13 >/dev/null 2>&1; then
        CC=clang-13
    elif [ -x "${CONDA_PREFIX:-}/bin/clang-13" ]; then
        CC="$CONDA_PREFIX/bin/clang-13"
    elif [ -x "${CONDA_PREFIX:-}/bin/clang" ]; then
        CC="$CONDA_PREFIX/bin/clang"
    else
        echo "ERROR: clang-13 not found"
        exit 5
    fi

    if [ "$MODE" = "multi_modality" ]; then
        "$CC" \
          -emit-llvm \
          -fno-discard-value-names \
          -g \
          -S -c \
          "$NAME.c" \
          -o "$NAME.ll"
    else
        "$CC" \
          -emit-llvm \
          -fno-discard-value-names \
          -S -c \
          "$NAME.c" \
          -o "$NAME.ll"
    fi
fi
