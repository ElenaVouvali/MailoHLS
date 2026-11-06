# Structural-memory routing comparison

## Provenance correction

The earlier table in this report was not an observed `action_relation_mask`; it applied a historical routing policy to presumed action relationships. The implementation in `GNN_branch/build_structural_memory.py` is explicitly tied to the prior numeric-node/GEXF graph contract (`flow` IDs, pragma and pseudo-block nodes, `array_scope` anchors). It must not be called the RC3 mask. A direct attempt to run it on the repository's historical GEMV paper-subgraph GEXF fails: `L4: expected exactly one loop pseudo scope, found 0`. Therefore no historical GEMV eligible-key matrix is authenticated from that artifact. Exact-slot routing is representation-independent; it is simply the one key with the same absolute action ID.

The RC3-derived table below is independently reconstructed from `paper_figures/gemv_rc3_20261105/gemv.canonical.json` (SHA-256 `7d295d...cb43a6`, graph identity `42fe84...abae21`), preserving absolute IDs L1–L5. Evidence used: five proven `targets` bindings; L4 `loop_child` L5; proven `memory_access` edges from op6→A(L1), op7→x(L2), op12→y(L3), and op15→y(L3); operation `loop_context` values place A/x reads inside L4 and L5, the y store in L4, and return read after the loops. The inter-operation dependence op15 depends_on op12 is proven, but does not create an inter-action loop-to-loop dependence: it is outside/inside L4 and both accesses concern L3. The L5 recurrence is proven by loop_carried edges carrying on op5, but is intra-action. Neither is projected to a cross-action dependence relation. Alias evidence is not promoted to an action relation here.

For an RC3 routing candidate, array/loop access is symmetrized as an action eligibility relation when a compiler-proven access to an action's memory root occurs in that loop region; hierarchy retains direction. This explicit projection is a proposed RC3 policy derived from RC3 facts, not an already deployed schema. Self keys are always available.

| Query action | Exact-slot keys | RC3-derived eligible keys | RC3 relation evidence | Count (exact / RC3-derived) |
|---|---|---|---|---:|
| L1 (A) | L1 | L1, L4 | self; L4 region has proven A read (op6) | 1 / 2 |
| L2 (x) | L2 | L2, L4 | self; L4 region has proven x read (op7) | 1 / 2 |
| L3 (y) | L3 | L3, L4 | self; L4 contains the proven y store (op12); return read op15 is outside L4 | 1 / 2 |
| L4 (outer) | L4 | L1, L2, L3, L4, L5 | self; accesses L1/L2/L3; child L5 | 1 / 5 |
| L5 (inner) | L5 | L1, L2, L4, L5 | self; accesses L1/L2; parent L4 | 1 / 4 |

RC3-derived total eligible keys across five query rows: 15. Counts are active slots only. The graph also has action-local compiler evidence that this key mask does not express: A/x affine accesses, L5 SSA recurrence and arithmetic/dataflow, and L3's dependence witness. Provenance remains attached to compiler edges/nodes, not implied by eligibility.

## Historical policy comparison

| Property | Historical builder (`build_structural_memory.py`) | RC3-derived candidate above |
|---|---|---|
| Graph contract | Numeric GEXF node IDs; `flow` encodings; pragma/array_scope/pseudo scopes | RC3 canonical IDs, typed `relation_type`, `evidence_status`, op `loop_context`, action target bindings |
| Self / hierarchy | self, nearest labelled loop parent/child | self, explicit L4 parent / L5 child from proven `loop_child` |
| Array relation | loop region traversal over historical flow edges, then array-scope memory access | RC3 loop-context / region evidence matched to proven action-root `memory_access` |
| Dependence projection | only proven inter-action loop-owner dependences | no cross-action dependence for this graph; recurrence stays on L5, L3 RAW stays compiler-local |
| GEMV mask status | not authenticated; historical subgraph fails required pseudo-scope check | reconstructed from RC3 artifact; schema/policy proposal, not implemented in production |

The old bit values (self=1, parent=2, child=4, array=8, dependence forward=16, reverse=32) are not automatically portable to an RC3 mask. Define and version an RC3 relation schema before serializing these relations. Exact-slot routing has one key per query (5 total across queries); relational routing has the counts above, but relation eligibility is not attention selection or evidence use.

## Decision relevance and limits

Current five action vectors can encode compiler evidence after direct action feature injection and graph propagation. The CPU perturbation audit in `CURRENT_GNN_INFORMATION_AUDIT.md` shows action-embedding sensitivity to several compiler feature groups, but not improved decisions. RC3-derived routing can make loop/array action relationships addressable; it still cannot route to the actual load, multiply, add, loop-carried SSA witness, or dependence witness. For L1 partitioning and L5 pipeline/unroll choices, evidence memory is the only compared option here that makes those records individually addressable.

The existing structural cross-attention implementation may distribute attention over multiple eligible keys per head; it is not necessarily a hard one-key selector. Its output is still a weighted context aggregation, so distinct evidence can be mixed. Key-count/type coverage and per-evidence attention diagnostics should be reported. Eligibility does not prove use.
