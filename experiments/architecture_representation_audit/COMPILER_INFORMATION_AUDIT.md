# Compiler-information audit

## Status and provenance

**Completed static schema/cohort audit and stratified archive-to-MLIR identity checks; full compiler rerun is still pending.** For all 55 exact-source measured graphs, the archived `raw_mlir/<graph-stem>.mlir` bytes match the graph's recorded `mlir_sha256` (55/55, no missing or mismatching MLIR member). A sampled gemm-ncubed graph with proven scalar `loop_carried` witnesses contains the corresponding MLIR loop `iter_args` and affine loop/load/store constructs; sampled gemm-blocked proven and possible dependence records are tied to compiler operation ordinals and distinguish a proven affine WAW relation from possible alias-mediated WAR evidence. This is a provenance/structure cross-check, not an independent rerun of the pinned compiler analyses. The authenticated population is the RC3 v2.2 archive (`salt-hls-graphs-v22-20261027-rc3.tar.gz`, SHA-256 `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`). The row-level graph/source/analysis provenance is in [`results/cohort_manifest.json`](results/cohort_manifest.json). RC3 provenance distinguishes compiler analysis from source metadata; `possible`, `proven`, `not_established`, and missing evidence are not interchangeable.

## Information path and coverage

| Fact | Compiler/graph representation | Tensor path | Confidence / limitation |
|---|---|---|---|
| Source action → compiler target | Action coverage record plus proven `targets` relation | Absolute-slot action point and `action_node_index` | Matched and unresolved masks retained; mapping failure is not repaired by slot compaction |
| Loop hierarchy | `loop_child`, `owns_region`, `contains_block`; loop context/depth | Typed graph edges plus node `loop_depth` value/known pair; action semantic summary may include bounded loop facts | Local relations are explicit. Full hierarchy can exceed the two-layer receptive field; direct action descriptors provide a separate path |
| Static trip count | Compiler-proven action coverage / loop attributes (`trip_count_static`, `trip_count`) | Node numerical `trip_count` value/known pair; action-demand/semantic summaries also carry selected loop facts | In `structural_memory_v1.py`, `known['trip_count'] = attrs.get('trip_count_static') == 1` means **a statically known trip-count flag**, not that the trip count equals one. The numeric value is carried separately. This is semantically consistent if `trip_count_static` is the documented binary flag. |
| Scalar loop-carried SSA | Explicit `loop_carried` edge, carried SSA nodes, carrying-loop identity and status | Relation embedding plus edge attributes; selected recurrence nodes/edges in Memory C when proven | Proven positive labels exist; no edge cannot be used as a negative label |
| Memory dependence / alias | `depends_on` with evidence status, dependence kind, source analysis; possible relations separately represented | Relation type plus edge categorical/status features and semantic edge features | A possible dependence stays possible. No explicit `not_established` edge does not establish independence |
| Memory object / dimensions | `memory_object` nodes with shape, root-resolution status; source action binding may contain dimensions | Node volume/rank and source-volume value/known pairs; categorical object/root types; action semantic summary | Shape source must be retained in provenance; symbolic/unknown dimensions are not fabricated as numeric dimensions |
| Affine access and read/write role | `memory_access` relation with access indices/status/affine map and access kind | Edge base features and semantic features; action-level semantic summary | Full affine expression is not necessarily a scalar node feature. Edge payload/metadata contain more detail than learned vector input |
| Arithmetic and array-access association | Operation/SSA operand/definition edges and affine `memory_access` witnesses | Typed nodes/relations, plus action semantic summary when locally queried | No general guarantee that action vector retains the full operand/access subgraph |

The tensorizer uses 18 numerical node properties, each as a value/known pair (`NODE_DIM=36`), categorical node embeddings (five 8D embeddings), 4 evidence-status one-hots plus 20 base edge numerical features, and semantic edge features. V4b adds 40D direct action-demand and the versioned semantic-action feature vector at action nodes before the first message-passing layer. The latter is a direct bypass around graph distance, but remains a fixed summary, not a lossless graph encoding.

Unknown numeric fields are encoded as zero plus a separate known bit. This avoids treating missing as zero in principle; it does not guarantee the learned model uses the known bit correctly. The graph JSON and sidecar provenance retain more detail than the tensor consumed by the GNN.

## RC3 source evidence and limitations

Coverage on the measured 55: 331/339 loop actions have proven static trip counts; 61 have explicit proven scalar loop-carried SSA evidence; 1,447 proven and 8,131 possible dependence edges; 2,205 memory-access edges; 1,949 operations with an affine map. The absent proven recurrence/dependence edge is unknown unless a separate compiler analysis explicitly establishes independence. The corpus-level audit script preserves evidence status rather than synthesizing negatives.

The paired `trip_count_static` flag and `trip_count` value are different schema fields. Audit consumers should test the flag for knownness and report the associated value, not compare that value to `1` as a knownness test. This distinction is checked in the existing C memory implementation and tensorizer; no production or source schema changes were made.

## Raw evidence revalidation still required

The archive contains provenance hashes for source, metadata, and MLIR, and the action evidence is internally authenticated. The marker inspection above did not rerun the entire pinned compiler pipeline or prove every witness direction. Before using recurrence/dependence probes as ground truth beyond explicit graph witnesses, independently reparse representative MLIR and verify source-analysis origin and relation orientation. Until then, explicit evidence is a valid compiler-recorded label, while missing evidence remains unknown.
