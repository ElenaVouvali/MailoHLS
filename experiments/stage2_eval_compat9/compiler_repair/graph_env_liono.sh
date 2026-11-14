# Source this file in the shell that will run prepare_stage2_eval_graphs.sh.
# Existing Liono toolchain; no packages installed and no binaries rebuilt.
export POLYGEIST_BUILD=/home/elvouvali/tools/Polygeist/build-mailohls-assertions
export GRAPH_PY=/home/elvouvali/.mlir-python311/bin/python
export MLIR_PYTHON="$GRAPH_PY"
export MLIR_PYTHON_ROOT="$POLYGEIST_BUILD/tools/mlir/python_packages/mlir_core"
export CGEIST="$POLYGEIST_BUILD/bin/cgeist"
export PYTHONPATH="$MLIR_PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$POLYGEIST_BUILD/bin:$PATH"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
