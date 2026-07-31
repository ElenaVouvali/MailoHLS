# Representative MailoHLS MLIR graphs

These three graphs demonstrate complementary parts of the MLIR representation.
They are more informative than selecting three kernels that differ only in size.

| Graph | Structure represented | Nodes | Edges | Native dependence result |
|---|---|---:|---:|---|
| [Rodinia KNN tiling](rodinia-knn-1-tiling.gexf) | Tiled helper-call pipeline, local scratchpad arrays, nested affine loops, and a loop-carried distance reduction | 200 | 768 | 4 access pairs proven independent; 11 conservative dependence edges |
| [MachSuite Viterbi](machsuite-viterbi.gexf) | Dynamic-programming table with nested affine loops and strong cross-iteration memory structure | 251 | 1,059 | 14 proven dependence edges, 26 pairs proven independent, 64 conservative edges |
| [Serrano Kalman](serrano-kalman-filter.gexf) | Call-rich C++ kernel with record-backed storage, derived MemRef views, alias classes, and mixed affine/non-affine behavior | 490 | 1,955 | 11 proven dependence edges, 15 pairs proven independent, 46 conservative edges |

## What a graph contains

The files are directed NetworkX `MultiDiGraph` GEXF graphs. Parallel edges are
intentional: the same two nodes can have more than one program relationship.

Node kinds:

- `type=0`: MLIR operation (`affine.for`, `affine.load`, `func.call`, etc.);
- `type=1`: SSA operation result or block argument;
- `type=2`: scalar, type, shape, and affine-access feature;
- `type=4`: one MailoHLS scope node per real MLIR block;
- `type=100`: pragma placeholder attached to an exact `Lk` action;
- `type=104`: source array-action scope.

Important edge flows:

- `0`: control order and explicit block successors;
- `1`: SSA producer/consumer relations;
- `2`: call-to-callee and actual-to-formal relations;
- `6`: loop nesting;
- `7`: source array scope and its accesses;
- `8`: operation/region entry and exit;
- `9`: MemRef view, exact MustAlias, and pairwise MayAlias relations;
- `10`: read/write operation to affected memory value;
- `11`: loop initialization, backedge, and loop-result flow;
- `12`: RAW/WAR/WAW memory dependence with proof or fallback metadata;
- `200`: MailoHLS pragma-to-scope attachment.

## How to read the certainty metadata

`certainty="proven"` means native MLIR affine analysis established the
dependence. `certainty="may"` means the graph intentionally retains a possible
dependence because aliasing, SCF control, or a non-affine access prevented a
proof. A proven `no_dependence` query emits no edge and is recorded in graph
metadata. Unsupported analysis is never treated as independence.

## Reproducibility note

The committed snapshots record the exact generator and tool hashes in the GEXF
graph metadata. Any edit to `mlir_graph_gen.py`, including this documentation
cleanup, changes the generator hash. Regenerate these examples (and eventually
the complete dataset) before presenting them as outputs of the cleaned script.

From the repository root, with the audited Polygeist/MLIR environment exported:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$MLIR_PYTHON_ROOT" "$MLIR_PYTHON" \
  GNN_branch/generate_mlir_dataset.py \
  --force --keep-mlir --continue-on-error \
  --only rodinia-knn-1-tiling machsuite-viterbi serrano-kalman-filter
```

The run is ready to share only when all three manifest rows are `generated`,
their `schema_version` matches the scripts, and their embedded
`generator_sha256` equals the SHA-256 of the checked-in generator.
