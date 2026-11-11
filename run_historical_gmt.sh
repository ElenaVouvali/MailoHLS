#!/usr/bin/env bash
set -euo pipefail

# Run INSIDE the isolated historical research worktree on liono.
ROOT="${MAILOHLS_HISTORICAL_ROOT:-/home/elvouvali/MailoHLS/worktrees/historical-mlir-gmt-gemv}"
cd "$ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# The base Miniforge Python 3.13 has no research dependencies. Reuse the
# existing Liono environment rather than installing packages into base.
PYTHON="${PYTHON:-/home/elvouvali/.hls-llm/bin/python}"
MODULE=experiments.historical_mlir_gmt_gemv.historical_gmt_slots
OUT=artifacts/historical_gmt_slots_v1

printf 'Python interpreter: %s\n' "$PYTHON"
"$PYTHON" -c 'import torch, torch_geometric, networkx, numpy; print("Torch",torch.__version__,"CUDA",torch.cuda.is_available())'
"$PYTHON" -m unittest discover -s experiments/historical_mlir_gmt_gemv -p 'test_historical_gmt_slots.py' -v
"$PYTHON" -m "$MODULE" selftest
if [[ "${1:-}" == "--test-only" ]]; then
  exit 0
fi
# Materialize the authenticated historical source snapshot before locating its GEXF corpus.
"$PYTHON" -c 'from experiments.historical_mlir_gmt_gemv.historical import prepare; prepare()'

# Search ONLY known locations for the authentic historical MLIR compiler graphs.
# Don't quietly train on a random unrelated graph collection.
GRAPH_DIR="${HISTORICAL_TRAIN_GRAPH_DIR:-}"
if [[ -z "$GRAPH_DIR" ]]; then
  for candidate in \
     "$ROOT/artifacts/historical_mlir_gmt_gemv/historical/GNN_branch/MLIR_graphs" \
     "/home/elvouvali/MailoHLS/GNN_branch/MLIR_graphs" \
     "$ROOT/GNN_branch/MLIR_graphs"; do
    if [[ -d "$candidate" ]] && compgen -G "$candidate/*.gexf" >/dev/null; then
      GRAPH_DIR="$candidate"; break
    fi
  done
fi

if [[ -z "$GRAPH_DIR" ]] || [[ ! -d "$GRAPH_DIR" ]]; then
  echo 'ERROR: historical training-kernel .gexf corpus not found.' >&2
  echo 'Locate it with: find /home/elvouvali/MailoHLS -type f -name "*_mlir.gexf" | head' >&2
  echo 'Then run: HISTORICAL_TRAIN_GRAPH_DIR=/absolute/path/to/MLIR_graphs bash run_historical_gmt.sh' >&2
  echo 'For a clearly labeled GEMV-only technical pilot, see README; do not treat it as transferable.' >&2
  exit 2
fi

printf 'Historical training graph directory: %s\n' "$GRAPH_DIR"
"$PYTHON" -m "$MODULE" prepare --out "$OUT" --training-dir "$GRAPH_DIR"
"$PYTHON" -m "$MODULE" train --out "$OUT" --steps 600 --seed 123 --device auto
"$PYTHON" -m "$MODULE" evaluate --out "$OUT"

printf '\nCOMPLETE: outputs in %s\n' "$ROOT/$OUT"
printf 'Inspect training_receipt.json, evaluation.json, memory_manifest.json, historical_role_cosines.png\n'
