# RC3 graph reachability audit

## Message-passing semantics

The current `SmallGraphEncoder` uses 66-dimensional numeric node features (18 numeric quantities, each value plus known mask), five 8-wide categorical embeddings (40 total), and 2 typed `TransformerConv` layers at hidden width 64, with 16-wide typed/evidence-bearing edge features. Its tensor graph adds reverse message edges, except `no_alias` negative evidence is held separately. Action-demand and semantic feature projections are added directly to action-node hidden states before message passing. Output node width is 128. Thus one layer can aggregate one-hop neighbor states; two layers can aggregate paths of at most two edges. Distances below are shortest paths in the canonical evidence graph treating directed relations as traversable for audit purposes; actual message direction is encoded by typed forward/reverse relations and can constrain what reaches an anchor in each layer.

## Distances and reachable evidence

The following shortest paths use the canonical graph as an undirected audit distance. Separately, the tensorizer computes action semantic summaries before message passing by selecting relevant compiler edges/operations for the action. Those summaries are direct inputs; they bypass path length but average supported fields and do not preserve each witness separately.

| Anchor | Target | Nearest operation | Nearest SSA value | Nearest control/block node | Notes |
|---|---:|---:|---:|---:|---|
| L1 | 1 | 2 (`op6`, A load) | 2 (A root) | 3 | Arithmetic path continues through load results and operands; inner loop/recurrence are farther. |
| L2 | 1 | 2 (`op7`, x load) | 2 (x root) | 3 | Arithmetic path continues through load result and operands. |
| L3 | 1 | 2 (`op12`, nearby output/init op) | 2 (y root) | 3 | Init/output access operations are reachable from the root; the exact source/compiler rank discrepancy is direct metadata. |
| L4 | 1 (loop op4) | 1 (anchor itself) | 2 (nearest scalar SSA) | 2 | Its region and contained body provide additional evidence; y access/arithmetic are a few hops away. |
| L5 | 1 (loop op5) | 1 (anchor itself) | 2 (nearest scalar SSA) | 2 | Loop-carried value is represented by an edge incident on the loop and is close; A/x access operations are within a few hops. |

Distances use undirected shortest paths over all declared canonical edges, to quantify graph proximity only. They are not the directional receptive field of the actual typed message-passing graph. For example, the array access operations are two hops from the array action anchor (target-to-memory, then memory-access), while an associated arithmetic operation requires additional operand/result paths. The operation-level dependence and loop-carried edges are separately typed.

Distance is not proof of information retention: evidence can be attenuated, overwritten, collapsed by aggregation, or inaccessible due to edge orientation/typed transformations. For L1–L3, action kind, source/compiler match, source/compiler shape features and action-specific demand are direct. Supported affine/access fields can also enter a direct mean-pooled semantic summary from associated access-operation edges. L4/L5 receive trip and loop-demand values directly; supported recurrence/dependence semantic fields can enter their direct summaries. The two GNN layers add one-/two-hop typed neighborhoods from action node states. Individual evidence nodes at distance 1–2 can be reached nominally within these layers, subject to edge direction and typed transformations; longer paths require more layers. Arithmetic-to-access wiring and exact SSA/dependence witnesses can be indirectly mixed or lost in pooled action summaries and message aggregation. Four layers increase nominal reach from two to four hops only; they do not guarantee retention or usefulness. Four layers enlarge nominal reach to 4 edges; they do not guarantee the model preserves or composes those facts.

The shortest paths were computed over the 66-node canonical graph, using declared graph edges, with source and endpoint named by semantic role above. The full canonical artifact remains authoritative for parallel edges and exact edge witnesses. This table is a structural-distance calculation, not a claim about learned effective receptive fields.

## Evidence directness and missingness

- Direct at action: action kind, source/compiler match status, compiler dimensions/rank where represented, proven trip count and validity, action-loop hierarchy metadata if included in demand descriptors.
- Direct but aggregated: supported action-edge semantic summary over compiler evidence selected for its target; this includes supported access/affine and recurrence/dependence fields.
- Requiring propagation for individual structure: operation identities and pairwise incidence; which product/access operands feed which arithmetic operation; ordered SSA producer-to-yield chain; individual dependence witness connections; loop body structure.
- Not represented as a stable action-aligned feature: arbitrary full source/MLIR witness text, complete affine expressions/iteration domains, provenance chains, and a guaranteed explicit “unresolved dependence” token for every missing relation. Canonical JSON has more detail than the fixed tensor contract.

The tensorizer’s validity masks distinguish many unknown numerical values from numeric zero. Its semantic edge summary is mean-pooled into action nodes before message passing for supported edges incident to an array root or loop-context operations, including supported affine/access and loop-carried/dependence features. Exact witness identity and full witness structure remain graph-level. Do not equate absent edges with no dependence; use explicit evidence status and completeness indicators only where compiler analysis supplies them.
