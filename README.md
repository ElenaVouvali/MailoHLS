# MailoHLS

MailoHLS learns from HLS programs, pragma configurations, and synthesis QoR.
The current MLIR path converts each C/C++ kernel into one deterministic semantic
graph, reuses that graph across the kernel's design points, and trains a GNN to
encode program structure and pragma effects.  A later LLM stage will combine the
GNN representation with device- and frequency-aware optimization objectives.

## Pipeline status

| Stage | Purpose | Status |
|---|---|---|
| MLIR graph generation | Represent control, SSA data flow, loops, calls, memory, dependences, and HLS actions | Implemented |
| QoR preprocessing | Build an unambiguous reference-target supervision table | Implemented |
| MLIR tensor dataset | Convert GEXF graphs and design points to compact PyG tensors | Implemented |
| GNN training | Learn structural and pragma-aware kernel representations | Experimental |
| LLM integration | Add target-aware reasoning and optimization | Next stage |

## Repository layout

```text
Data/
â”œâ”€â”€ ApplicationDataset/          # kernel source, headers, kernel_info.txt
â”œâ”€â”€ ApplicationAPLMapping/       # CSV directive column -> action mapping
â”œâ”€â”€ ApplicationInformation.csv   # batch-generation metadata
â”œâ”€â”€ CSVS/                        # raw multi-target synthesis measurements
â””â”€â”€ preprocessed_CSVS/           # GNN reference-target measurements

Preprocessing/
â””â”€â”€ data_preprocess.py

GNN_branch/
â”œâ”€â”€ mlir_graph_gen.py             # one C/C++ kernel -> one semantic GEXF
â”œâ”€â”€ generate_mlir_dataset.py      # validated 55-kernel graph driver
â”œâ”€â”€ mlir_data.py                  # GEXF + QoR -> compact PyG dataset
â”œâ”€â”€ main_GNN.py                   # training/inference entry point
â”œâ”€â”€ train_GNN.py
â”œâ”€â”€ model.py
â””â”€â”€ MLIR_graphs/                  # generated graphs and manifest
```

## 1. MLIR graph generation

### Toolchain

Use `cgeist`, MLIR Python bindings, and the MailoHLS compiled analysis
extension from the same Polygeist build:

```bash
export POLYGEIST_BUILD="$HOME/tools/Polygeist/build-mailohls-assertions"
export CGEIST="$POLYGEIST_BUILD/bin/cgeist"
export MLIR_PYTHON="$HOME/.mlir-python311/bin/python"
export MLIR_PYTHON_ROOT="$POLYGEIST_BUILD/tools/mlir/python_packages/mlir_core"
export PYTHONHASHSEED=0
export PYTHONPATH="$MLIR_PYTHON_ROOT"
```

Sanity check:

```bash
"$CGEIST" --help | grep mailohls-action-manifest

"$MLIR_PYTHON" - <<'PY'
from mlir import ir
from mlir._mlir_libs import _mailohls_analysis
print("MLIR bindings: OK")
print("MailoHLS analysis:", _mailohls_analysis.__file__)
PY
```

### Inputs

Each application directory contains its source and a `kernel_info.txt` file.
The first non-empty line names the top-level function.  Remaining lines declare
loop or array actions, for example:

```text
workload
L1,loop,100
L2,array,buffer,1,1024
```

The corresponding source contains exact labels such as `L1: for (...)` and
`L2: float buffer[1024]`.  Generation fails when an action cannot be mapped
exactly once; it does not guess from loop order or nearby source locations.

### Generate graphs

One kernel:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
  --force --keep-mlir --only rodinia-knn-1-tiling
```

All configured kernels:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
  --force --keep-mlir
```

The batch is valid only when `GNN_branch/MLIR_graphs/generation_manifest.csv`
reports success for every configured application.  The driver also checks graph
schema, compiler and binding hashes, source/action mappings, module verification,
and deterministic output metadata.

### Graph semantics

Each GEXF is a directed multigraph.  Parallel edges are intentional because two
nodes may have several simultaneous relations.

| Graph element | Information represented |
|---|---|
| Operation/value nodes | MLIR operations, SSA definitions, block arguments, constants, and types |
| Structural edges | Regions, blocks, control order, loop nesting, and calls |
| SSA edges | Operand/result def-use and loop-carried values |
| Memory edges | Allocations, views, exact aliases, uncertain alias pairs, effects, and accesses |
| Dependence edges | Proven affine RAW/WAR/WAW relations or explicitly marked conservative uncertainty |
| Action nodes | PIPELINE, UNROLL, and ARRAY_PARTITION scopes grounded in labeled source and MLIR |
| Features | Loop/trip-count data, access/effect facts, dependence certainty/distance, and provenance |

## 2. QoR preprocessing for the GNN

The raw CSVs intentionally retain all devices and clock periods for the future
target-aware LLM stage.  The GNN supervision view uses one reference target:

- device: `xczu7ev-ffvc1156-2-e`
- clock period: `10.0 ns` (100 MHz)

Repeated measurements of the same effective pragma configuration are aggregated
with the median before Pareto weights are computed.

```bash
python Preprocessing/data_preprocess.py
```

This updates `Data/preprocessed_CSVS/`.  Do not replace `Data/CSVS/`; it remains
the multi-target source of truth.

## 3. Build tensors and train the GNN

For a kernel-disjoint experiment, keep Kalman for final testing and use the
same validation families as the LLM pipeline:

```bash
PYTHONHASHSEED=0 python GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --force_regen \
  --random_seed 123 \
  --val_kernels machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0 \
  --test_kernels serrano-kalman-filter
```

`--force_regen` validates the graph manifest, fits categorical encoders only on
training kernels, and rebuilds `GNN_branch/MLIR_dataset/all_kernels/`.  Training
uses validation loss for checkpoint selection and evaluates the held-out test
kernel once after training.

## Reproducibility contract

- Keep `PYTHONHASHSEED=0` and `--random_seed` fixed.
- Keep graph generation and the compiled analysis extension pinned together.
- Regenerate tensors after changing graphs, preprocessing, feature schemas, or
  the reference QoR target.
- Never use the Kalman test result to select hyperparameters or checkpoints.
- Record the Git commits, tool hashes, manifest, split lists, and random seed for
  every reported experiment.

## Next stages

The next work will document trained-checkpoint evaluation, extraction of
pragma-disabled structural GNN embeddings, and target-aware LLM integration.
