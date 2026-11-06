# GEMV compiler-evidence coverage

The counts below come from `artifacts/rc3_fusion_foundation/v1_20261107/gemv_rc3_original.pt`, produced by the frozen V4b encoder from the authenticated RC3 graph. A is the current exact-slot baseline; B changes only which action vectors the decision can address; C adds selected node and edge evidence. These are memory-access counts, not evidence of information quality.

## Per-action coverage

| Decision | Authenticated compiler evidence | A accessible keys | B accessible keys | C evidence keys | Direct evidence / remaining limit |
|---|---|---:|---:|---:|---|
| L1 — A partition | Proven memory root `memref<32x32xf32>`; affine load map `(d0,d1)->(d0,d1)` with induction loops L4/L5; relation to loop hierarchy | 1 (L1) | 2 (L1,L4) | 10 | Typed access witness plus action/root/loop values. The complete access map and operand positions remain in provenance metadata; a 128D vector does not guarantee lossless recovery. |
| L2 — x partition | Proven root `memref<32xf32>`; affine load `x[j]`, with L5 induction index | 1 (L2) | 2 (L2,L4) | 10 | Access role/map and loop context are explicit in edge payload. Does not establish a required number of hardware ports. |
| L3 — y partition | Source action is `y[32]`; compiler action target resolves to `memref<f32>` (scalarized storage); stores/reads and return/alias evidence are retained with status | 1 (L3) | 2 (L3,L4) | 14 | C retains one proven dependence witness in the graph scope. Source extent and compiler scalar storage are both recorded; the memory encoder is not a correctness or observability analysis. |
| L4 — outer loop | Static trip count 32, depth 0, child L5; nested body operations and accesses | 1 (L4) | 5 (L1–L5) | 36 | C includes loop/control nodes, accesses, arithmetic, and carried-state witnesses in scope. L4 sees its nested computation, but recurrence belongs to L5's loop-carried proof. |
| L5 — inner reduction | Static trip count 32; `iter_args`, `arith.mulf`, `arith.addf`, `affine.yield`; proven loop-carried SSA witness | 1 (L5) | 4 (L1,L2,L4,L5) | 30 | C provides typed SSA/edge evidence. This is a scalar recurrence witness, not a memory-dependence claim. |

The original graph has 47 globally selected evidence records; diagnostic graph counts are 82 for `reduce32`, 67 for `independent32`, and 82 for `reduce32_rename`. The schema capacity is 128; larger graphs fail with an overflow error and require an explicit capacity/schema update. No key is silently truncated.

## What each interface makes visible

- **A — exact action memory:** one 128D frozen action vector at each absolute matched Lk slot. It is compressed compiler context, with no typed compiler witness supplied at attention time.
- **B — relational action memory:** the identical action vectors as A; the boolean relation mask limits a decision to the RC3-proven action neighbors. This provides relational access to peer action vectors but not a textual/compiler-fact key.
- **C — action plus typed evidence:** identical action vectors plus a variable-length list of selected compiler node vectors and deterministic edge-witness vectors. The visible evidence types and known-field masks identify evidence classes. Exact facts and provenance remain in sidecar metadata, not as learned source names, line numbers, or graph IDs.

All routes retain the 64-slot absolute Lk coordinate system. Only the first five slots are matched for this graph. `possible`, `not_established`, and missing evidence stay explicit; no missing edge is interpreted as proof of independence. The selected V4b GNN was trained with ADP-oriented physical-QoR supervision and is frozen for interface diagnostics only, not asserted as a latency-optimal encoder.

## Diagnostic inputs

The initialized `reduce32` is a scalar reduction with initialized inputs and observable reduction outputs. `independent32` has different iteration/output semantics and is not presented as an equivalent rewrite. `reduce32_rename` is the semantics-preserving variable rename control; source names and node IDs change while action IDs remain L1–L5. Their graph/source/metadata/MLIR hashes and per-memory hashes are in `manifests/provenance.json` and the ignored artifact `manifest.json`.
