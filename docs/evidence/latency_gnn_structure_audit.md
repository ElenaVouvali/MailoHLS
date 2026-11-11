# Structural information audit

The RC3 release archive and its canonical members are authenticated by their original SHA256 values. The model still has four edge-aware TransformerConv stages of width 64, JK-max, one candidate-conditioned post-directive stage, and a `[first-layer | JK]` 128-D memory. `GNN_branch/model.py` is unchanged from `20028e7`.

| Evidence | Authenticated availability | Numerical/categorical encoding | Accessible prediction/export path |
|---|---|---|---|
| Loop hierarchy | RC3 `contains_*`, `owns_region`, `loop_child` relations | Typed forward/reverse edges; nesting depth with known indicators | Four static edge-aware layers, JK, candidate-conditioned post stage and graph readout |
| Bounds/stride/trip count | `loop_lower`, `loop_upper`, `loop_step`, `trip_count`, proven and TCAD counts | Wrong historical adapter aliases repaired; finite numerical values and known indicators | Static node features; operation/action targets route evidence to action slots |
| Operations/SSA | Operation text, SSA categories, value types and use counts | Training-only categorical vocabulary, node categories; exact `ssa_use_count` field repaired | Static encoder features and message passing; unknown held-out tokens remain unknown |
| Memory objects/shapes/accesses | Memory nodes, rank, shape, types, access/resolution relations | Rank, static volume and known indicators; type categories and typed access edges | Static encoder and candidate-conditioned propagation. Individual shape/stride/offset facts are not all independent numerical channels; raw metadata is retained |
| Dependence/evidence | Dependence, alias, access and evidence-status relations | Relation and evidence channels with reverse-edge identities | Edge-aware attention/messages. Detailed relation records also accompany exports; not every dependence-distance scalar is a direct tensor channel |
| Exact registered source actions | `kernel_info.txt`, labeled source and APL mapping; bound compiler action nodes where available | Absolute Lk slots; source action kinds/metadata; complete candidate vector | Candidate injection and complete-assignment side channel, including unresolved bindings |
| Pipeline/unroll/partition scopes | Registered loop/array action kinds and actual compiler `targets` edges | Action kinds, typed target relations, candidate features after static encoding | One post-directive stage plus full candidate summary; absent bindings are never guessed |
| Readout | Existing `PaperRC3Net` mean readout and historical modules | Same width/layers/heads | Candidate-conditioned QoR/resource heads; targets enter heads only |

A real compatibility check found that the original RC3 archive has no action node for L1, L2, L3, L4 or L6 in each of `rodinia-hotspot-2-pipeline` and `rodinia-hotspot-3-unroll`. These ten registered array actions still have authentic source/APL measurements. Masking them out of structural memory caused the maintained compiler-relation router to reject real action queries.

Tensor schema v3 therefore adds **isolated source-only action nodes** in an adapter copy. Their array metadata comes from authenticated `kernel_info.txt`. The original compiler graph, archive hashes, compiler bindings and dependence relations are unchanged. No target edge or dependence proof is invented. These slots use the same four-layer static encoder and have identity-only memory visibility. The source-action, compiler-origin, available-memory, compiler-bound and unbound masks remain explicit. This is a shared input/compatibility repair for both final arms, not a new GNN architecture.

The preserved v2 A attempt is trained but unqualified for freeze. Both final arms start fresh under v3 after the four-kernel validation amendment was frozen. The repaired bank passed real maintained XAttn forward/backward for every registered source action, zero-gate identity, finite/nonzero gradients, static candidate/target/QoR/budget invariance and complete unbound-action candidate handling.

Memory RMS statistics use only training kernels, with equal kernel weighting. Exports contain frozen weights' static `[conv1 | JK]` vectors and authenticated relation/action identities; no candidate values, target contexts, budgets or measured labels enter memory construction. A compatible standalone bank does not establish an improvement in downstream LLM decisions.
