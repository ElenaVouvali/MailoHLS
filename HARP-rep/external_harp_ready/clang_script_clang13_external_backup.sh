#!/usr/bin/env bash
set -euo pipefail

NAME="$1"
DIR="$2"
MODE="$3"

cd "$DIR"

if [ "$NAME" = "gan" ] || [ "$NAME" = "kalman" ]; then

    CONDA_BIN="${CONDA_PREFIX:?CONDA_PREFIX is not set}/bin"

    if [ -x "$CONDA_BIN/clang++" ]; then
        CXX="$CONDA_BIN/clang++"
    elif [ -x "$CONDA_BIN/clang++-13" ]; then
        CXX="$CONDA_BIN/clang++-13"
    else
        echo "ERROR: Clang++ 13 not found in $CONDA_BIN"
        exit 2
    fi

    ROOT="/home/elvouvali/HARP-paper-rep-unseen4"
    APINT_INC="$ROOT/external_harp_ready/toolchain_headers/apint_include"

    test -f "$APINT_INC/ap_int.h" || {
        echo "ERROR: $APINT_INC/ap_int.h missing"
        exit 3
    }

    test -f "$APINT_INC/ap_fixed.h" || {
        echo "ERROR: $APINT_INC/ap_fixed.h missing"
        exit 4
    }

    DEBUG=()
    if [ "$MODE" = "multi_modality" ]; then
        DEBUG=(-g)
    fi

    echo "[HARP-EXTERNAL]"
    echo "  kernel   = $NAME"
    echo "  compiler = $CXX"
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

    if command -v clang-13 >/dev/null 2>&1; then
        CC=clang-13
    elif [ -x "${CONDA_PREFIX:-}/bin/clang" ]; then
        CC="$CONDA_PREFIX/bin/clang"
    else
        echo "ERROR: clang not found"
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
