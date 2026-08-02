# MailoHLS MLIR Graph Generation

This document describes the production graph-generation stage of the MailoHLS MLIR branch. It is intentionally limited to the transformation from a labeled HLS C/C++ application to a validated deterministic GEXF graph. Dataset tensorization and GNN training are separate stages.

## Status and pinned implementation

This README was prepared against the repository state inspected on **2026-08-01**:

- Repository: `ElenaVouvali/MailoHLS`
- Commit: `b366d5fc7c34e416a832894b13ddd8c213c52ac4`
- Graph generator: `GNN_branch/mlir_graph_gen.py`
- Dataset driver: `GNN_branch/generate_mlir_dataset.py`
- Current graph schema: `mailohls-mlir-graph-v6-root-uncertainty`
- Required native-analysis schema: `mailohls-native-analysis-v3`

The committed `generation_manifest.csv` reports **55 generated graphs**. A generated manifest row is not, by itself, sufficient for training: each graph must also have matching APL metadata and usable QoR design points during `mlir_data.py` processing.

---

## 1. Purpose

The graph generator preserves the MLIR-level information needed for HLS QoR prediction while aligning every tunable MailoHLS action with a precise semantic scope. The production representation includes:

- MLIR operations and one node per SSA value;
- typed SSA definition-use relations;
- structured regions, block adjacency, and direct loop hierarchy;
- loop bounds, steps, trip counts, and loop-carried values;
- calls, actual-to-formal argument links, and returned values;
- physical memory roots, views, alias facts, effects, and accesses;
- compiler-proven affine memory dependences;
- explicit root-level uncertainty when a dependence cannot be proven;
- PIPELINE, UNROLL, and ARRAY_PARTITION action nodes;
- deterministic IDs, edge ordering, and provenance hashes.

The generator deliberately fails rather than guessing an action mapping.

---

## 2. Repository layout

Expected locations:

```text
MailoHLS/
├── GNN_branch/
│   ├── mlir_graph_gen.py
│   ├── generate_mlir_dataset.py
│   ├── mlir_data.py
│   ├── config.py
│   ├── MLIR_graphs/
│   │   ├── <application>.gexf
│   │   └── generation_manifest.csv
│   └── Data/
│       ├── ApplicationDataset/
│       │   └── <application>/
│       │       ├── <source>.c|cpp
│       │       ├── headers...
│       │       └── kernel_info.txt
│       ├── ApplicationInformation.csv
│       ├── ApplicationAPLMapping/
│       └── preprocessed_CSVS/
└── ...
```

`ApplicationInformation.csv` provides the application name, source filename, extension, and top-level function used by the 55-kernel batch driver.

---

## 3. Required toolchain

The production path requires a mutually compatible set of:

1. Polygeist/cgeist built with the MailoHLS action-manifest support.
2. MLIR Python bindings from the same LLVM/MLIR build.
3. The native `_mailohls_analysis` Python extension.
4. Python packages used by the scripts, including NetworkX.
5. `PYTHONHASHSEED` set before Python starts.

Recommended environment variables:

```bash
export CGEIST=/absolute/path/to/cgeist
export MLIR_PYTHON=/absolute/path/to/python
export MLIR_PYTHON_ROOT=/absolute/path/to/mlir/python_packages
export PYTHONHASHSEED=0
```

Sanity checks:

```bash
"$CGEIST" --version

PYTHONPATH="$MLIR_PYTHON_ROOT" "$MLIR_PYTHON" - <<'PY'
from mlir import ir
from mlir._mlir_libs import _mailohls_analysis
print("MLIR Python bindings: OK")
print("MailoHLS native binding:", _mailohls_analysis.__file__)
PY
```

Do not mix Python bindings, cgeist, and `_mailohls_analysis` from different LLVM/MLIR builds.

---

## 4. Input contract

Each application directory must contain:

- exactly the source file named by `ApplicationInformation.csv`;
- all required local headers;
- `kernel_info.txt`;
- labeled action points in the C/C++ source.

### 4.1 `kernel_info.txt`

The first non-empty line is the top-level function. Subsequent lines define actions.

Loop action:

```text
L1,loop,BOUND_OR_EXTENT
```

Array action:

```text
L2,array,local_array,1,1024
```

For a multidimensional array:

```text
L3,array,tile,1,32,2,32
```

The historical loop value is retained as audit metadata. The actual trip count is derived from MLIR/native analysis when provable.

### 4.2 Source labels

Supported loop forms include:

```cpp
L1: for (...) {
```

```cpp
/*L1:*/ for (...) {
```

```cpp
L1: LOOP_NAME: for (...) {
```

Supported local-array form:

```cpp
L2: float local_array[1024];
```

The labels define action identity and source position. Program semantics still come from verified MLIR, not from the lightweight source parser.

### 4.3 Action consistency requirements

Generation fails when:

- a `kernel_info.txt` action is absent from the source;
- the source action kind disagrees with `kernel_info.txt`;
- an action ID appears more than once;
- a loop action cannot be matched to exactly one MLIR loop;
- an array action cannot be grounded in one physical allocation/root;
- any expected action is unmatched or duplicated.

---

## 5. Frontend policy

The generator invokes cgeist with a fixed HLS-oriented policy:

```text
-O0
-scal-rep=0
-print-debug-info
-memref-fullrank
-raise-scf-to-affine
-function=<top-level-function>
-mailohls-action-manifest=<temporary-manifest>
```

Functions that own labeled actions are additionally marked `noinline`. This prevents inlining from duplicating loop actions or deleting/merging local array buffers that must remain identifiable.

The intended MLIR level is:

```text
Affine + SCF + MemRef + Arith + Func
```

Eligible static loops/accesses may be represented as Affine. Unsupported or dynamic structured control remains SCF. The flow does not lower the design to LLVM/CF before graph extraction.

---

## 6. Generate one graph

The batch driver is preferred because it applies the repository metadata and strict validation consistently:

```bash
cd ~/MailoHLS

PYTHONHASHSEED=0 \
PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
  --force \
  --keep-mlir \
  --only rodinia-knn-1-tiling
```

A direct generator invocation is useful for debugging one application:

```bash
cd ~/MailoHLS

PYTHONHASHSEED=0 \
PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/mlir_graph_gen.py \
  GNN_branch/Data/ApplicationDataset/<application>/<source>.cpp \
  --kernel <top-level-function> \
  --output GNN_branch/MLIR_graphs/<application>.gexf \
  --cgeist "$CGEIST"
```

The exact source path may differ if the repository is using another supported `ApplicationDataset` root.

Debug-only options such as action fallbacks or conservative-only analysis must not be used for production training graphs.

---

## 7. Generate the complete 55-graph set

```bash
cd ~/MailoHLS

PYTHONHASHSEED=0 \
PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
  --force \
  --keep-mlir \
  --continue-on-error
```

The batch driver:

1. reads the authoritative kernel set from `GNN_branch/config.py::ALL_KERNEL`;
2. resolves each source and top-level function through `ApplicationInformation.csv`;
3. computes source, action, cgeist, generator, and binding hashes;
4. writes a candidate graph to a temporary path;
5. reopens and strictly validates the candidate;
6. atomically replaces the old graph only when validation succeeds;
7. writes `GNN_branch/MLIR_graphs/generation_manifest.csv`.

A failed forced regeneration does not destroy a previously valid graph.

---

## 8. Graph representation

The output is a directed NetworkX multigraph. Parallel edges are intentional because a node pair can simultaneously participate in different relations.

### 8.1 Node types

| Numeric type | Meaning |
|---:|---|
| `0` | MLIR operation |
| `1` | SSA value |
| `2` | immediate or bounded semantic feature |
| `4` | MLIR block/pseudo scope |
| `100` | MailoHLS pragma placeholder |
| `104` | array-action scope |

### 8.2 Edge flows

| Flow | Meaning |
|---:|---|
| `0` | operation order / control |
| `1` | SSA data relation or semantic feature input |
| `2` | call, actual-to-formal, or return-to-call |
| `4` | block-scope membership |
| `5` | real MLIR block adjacency |
| `6` | direct loop parent-child hierarchy |
| `7` | array scope to physical root/access |
| `8` | region entry/exit |
| `9` | memory view or alias relation |
| `10` | read/write access to a memory root |
| `11` | loop-carried value relation |
| `12` | compiler-proven RAW/WAR/WAW dependence |
| `13` | root-mediated unresolved dependence feature |
| `200` | MailoHLS pragma/action relation |

Each edge also carries stable position/role metadata. Dependence edges carry proof status and, when available, dependence depth, distance, and loop-carried information.

---

## 9. SSA and structural modeling

The generator indexes the whole module before adding semantic edges:

1. Discover all function symbols.
2. Assign deterministic function, block, operation, and SSA identities.
3. Create one node per MLIR `OpResult` or `BlockArgument`.
4. Add operation-result and value-operand edges with exact positions.
5. Check graph use counts against MLIR’s SSA use information.
6. Add operation order, explicit CFG successors, and region entry/exit.
7. Add loop iter-argument, yield, result, and backedge relations.
8. Add direct calls, actual-to-formal links, and return-to-call links.

The graph validator rejects duplicate SSA identities, missing definitions, and use-count mismatches.

---

## 10. Loop action mapping

A loop action is resolved primarily from its exact source file and line. Column distance selects the closest loop operation on that line. The source loop ordinal is used only as a tie-breaker after exact file/line matching.

Production graphs must use exact source-location resolutions. Generic function/ordinal fallback exists only for debugging and is rejected by the batch validator.

PIPELINE and UNROLL placeholders are attached bidirectionally to the exact loop operation. The loop operation stores the action ID, action source location, and resolution provenance.

---

## 11. Array action mapping

Array mapping follows the current strict policy:

1. Search globally for an allocation at the exact source declaration location.
2. When the declaration location is unavailable, resolve the owning MLIR function using an exact source-function relation.
3. Restrict candidates to allocations with the exact structured MemRef shape.
4. Compare each candidate root’s MLIR access source lines with the source uses of the named array.
5. Accept only one candidate with positive, unique source-use overlap.
6. Otherwise fail and print candidate allocations, shapes, locations, and access lines.

A successful ARRAY_PARTITION action creates:

- one pragma node;
- one `array_scope` node;
- a bidirectional pragma-scope relation;
- a relation to the physical memory root;
- relations to every resolved read/write access rooted at that allocation.

There is no first-N textual-use truncation.

---

## 12. Memory and dependence semantics

The native binding supplies:

- operation memory effects;
- derived-view relations;
- pairwise alias classifications;
- affine dependence queries;
- affine loop trip counts;
- coverage and verification metadata.

The Python graph builder checks that native operation identities exactly match its MLIR operation index before using these results.

### Proven dependence

A proven affine RAW, WAR, or WAW relation becomes a direct flow-12 edge. The edge is marked `certainty=proven` and contains available distance/depth/loop-carried metadata.

### Proven independence

No dependence edge is added. The count is stored in graph provenance.

### Unknown or unsupported dependence

The generator does **not** add a speculative operation-to-operation dependence. It creates a bounded uncertainty feature and connects it to the relevant physical root through flow 13. The edge stores the reason and possible loop-carried status.

This preserves uncertainty without claiming unsupported causality.

---

## 13. Block scopes and loop hierarchy

After semantic graph construction:

- one pseudo-scope node is created per real MLIR block;
- every member is connected bidirectionally to its block scope;
- scope-to-scope edges follow real region/CFG block adjacency;
- direct parent-child loop hierarchy edges are added;
- no all-pairs block clique or redundant transitive loop-ancestor clique is created.

These structural scopes support the existing MailoHLS masks and message-passing interface without changing the GNN architecture.

---

## 14. Validation

Before a candidate is accepted, the generator validates:

- contiguous node IDs;
- required node and edge attributes;
- absence of isolated nodes;
- one node per structural SSA identity;
- exact SSA definition and use-count invariants;
- valid node types and edge flow types;
- placeholder rather than concrete pragma values;
- one loop scope per loop action;
- one physical root per array action;
- complete expected action set;
- exact action-resolution policy;
- native-analysis schema and coverage;
- proven-dependence and uncertainty cardinalities;
- source, action, cgeist, generator, binding, and MLIR hashes.

The batch driver reopens the serialized GEXF and repeats the reusable-graph checks before atomic replacement.

---

## 15. Determinism and provenance

Determinism is enforced through:

- `PYTHONHASHSEED`;
- deterministic traversal and stable function-local identities;
- structural node signatures and iterative relabeling;
- canonical node and parallel-edge sorting;
- fixed frontend policy;
- content hashes embedded in the GEXF metadata envelope.

The GEXF graph name contains a deterministic JSON metadata envelope beginning with:

```text
mailohls-meta-v1:
```

Important provenance fields include:

- graph schema;
- top-level kernel;
- source and `kernel_info.txt` hashes;
- cgeist, generator, MLIR, and native-binding hashes;
- action resolutions;
- native coverage;
- dependence, independence, and uncertainty counts;
- alias classifications and fallback reasons.

---

## 16. Inspect the manifest

```bash
column -s, -t < GNN_branch/MLIR_graphs/generation_manifest.csv | less -S
```

Count accepted rows:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path("GNN_branch/MLIR_graphs/generation_manifest.csv")
rows = list(csv.DictReader(p.open()))
ok = [r for r in rows if r["status"] in {"generated", "skipped"}]
print(f"accepted={len(ok)} total={len(rows)}")
for r in rows:
    if r["status"] not in {"generated", "skipped"}:
        print(r["app_name"], r["status"], r.get("detail", ""))
PY
```

The expected production result is 55 accepted applications and no failed or stale row.

---

## 17. Inspect one GEXF

```bash
python - <<'PY'
import json
import networkx as nx
from pathlib import Path

path = Path("GNN_branch/MLIR_graphs/rodinia-knn-1-tiling.gexf")
g = nx.read_gexf(path)
print("graph:", path.name)
print("nodes:", g.number_of_nodes())
print("edges:", g.number_of_edges())
print("multigraph:", g.is_multigraph())

prefix = "mailohls-meta-v1:"
name = str(g.graph.get("name", ""))
assert name.startswith(prefix)
metadata = json.loads(name[len(prefix):])
for key in (
    "schema_version",
    "kernel",
    "action_resolutions",
    "native_analysis_coverage",
    "proven_dependence_edge_count",
    "proven_independent_count",
    "unresolved_dependence_query_count",
    "root_uncertainty_feature_count",
):
    print(f"{key}: {metadata.get(key)}")
PY
```

---

## 18. Handoff to MLIR dataset creation

After all 55 GEXFs pass generation checks, create the tensor dataset with `mlir_data.py` through the repository’s normal `main_GNN.py` path.

Before the first complete build:

```bash
cd ~/MailoHLS
rm -rf GNN_branch/MLIR_dataset/all_kernels_tmp
```

Then run with a fresh feature schema and encoders:

```bash
PYTHONHASHSEED=0 \
python GNN_branch/main_GNN.py \
  --dataset mlir \
  --force_regen
```

Exact invocation details depend on the active environment and other `config.py` flags.

The dataset build must report:

- 55 encoded static graph packs;
- no missing APL mapping;
- no missing or ambiguous preprocessed CSV;
- no graph skipped because it has zero valid design points;
- finite node, edge, pragma, and target tensors;
- every nonzero directive reaching at least one action scope.

Do not start a long training run until the dataset preflight and a forward/backward smoke test have succeeded.

---

## 19. Recommended pre-training gates

### Gate A — Graph coverage

- 55 manifest rows accepted.
- 55 readable graph files.
- schema and native-analysis version consistent across all files.
- every action resolution exact.

### Gate B — Dataset coverage

- 55 graph `.pt` packs.
- 55 point `.pt` packs.
- at least one usable point per graph.
- no APL/CSV mismatch.

### Gate C — Dynamic directive injection

For representative loop and array actions:

- zero design point gives a zero per-node vector;
- PIPELINE/UNROLL values reach only the intended loop scope;
- ARRAY_PARTITION type/factor/dimension reach only the intended array scope;
- no nonzero point produces an all-zero `X_pragma_per_node` tensor.

### Gate D — Model smoke test

Run one forward and backward pass on:

- a compact graph, such as `machsuite-gemm-blocked`;
- a medium graph, such as `rodinia-knn-1-tiling`;
- the largest graph, `rodinia_lud_1_tiling_0`.

Record peak GPU memory and batch time before choosing the production batch size.

### Gate E — Evaluation split

Use kernel- or family-disjoint splits for generalization claims. A random design-point split over a dataset that reuses one static graph per kernel is an interpolation baseline and can leak kernel/graph identity between training and test samples.

---

## 20. Known limitations and follow-up work

1. **Uncertainty-heavy kernels.** Some complex kernels produce many unresolved native dependence queries. These are represented safely, but the root-level summary compresses query multiplicity and endpoint context.
2. **Array provenance naming.** The current strict source-use-overlap path records the accepted resolution as `source_location`. A future schema could distinguish `source_declaration_location` from `source_access_overlap` for clearer auditing.
3. **Flat pragma ordering.** `mlir_data.py` currently sorts pragma dictionary keys lexicographically. This is deterministic, but `L10` sorts before `L2`; use an action-aware natural order before relying on the flat pragma vector as semantic input.
4. **Large-graph batching.** LUD and Streamcluster require explicit memory profiling.
5. **Legacy flags.** Avoid old CLI boolean definitions that use `type=bool`; use explicit `store_true/store_false` flags in future cleanup.
6. **GAE compatibility.** Keep `gae_T=False` unless the MLIR `.pt` dataset path is explicitly adapted to the old `.klepto` workflow.
7. **Aggregate audit.** Add a machine-readable report aggregating node/edge counts, action coverage, dependence coverage, and uncertainty ratios over all 55 graphs.

These limitations do not invalidate the current graph representation, but items 3, 4, and the evaluation-split policy should be resolved before interpreting final GNN results.

---

## 21. Reproducibility checklist

Record the following with every complete graph generation:

- Git commit SHA;
- LLVM/MLIR/Polygeist commit SHA;
- cgeist SHA-256;
- `_mailohls_analysis` SHA-256;
- Python and NetworkX versions;
- `PYTHONHASHSEED`;
- generation command;
- `generation_manifest.csv`;
- all 55 GEXFs;
- preserved MLIR files when `--keep-mlir` is used;
- stdout/stderr logs;
- aggregate graph audit report.

---

## 22. Flowchart

The current production flow is provided in four forms:

```text
mlir_graph_generation_flowchart.mmd   Mermaid source
mlir_graph_generation_flowchart.dot   Graphviz source
mlir_graph_generation_flowchart.svg   scalable rendered diagram
mlir_graph_generation_flowchart.png   raster preview
```

The SVG is recommended for documentation because it remains readable when zoomed.