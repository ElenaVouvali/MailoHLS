# Current V4b GNN information audit

## Checkpoint and inference evidence

The authenticated files are available locally at the paths recorded by the existing manifest: `/home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37/compiler_selected.pt` and `/tmp/gemv_rc3_embedding_input_20261105.pt`. SHA-256 values rechecked: checkpoint `ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162` (compiler arm, selected step 1500); cache `85fa99822dc82375b77142546ef67076705a4069387bff47f9d5bf82d1ea5fe7`. Cache provenance graph identity is `42fe84cb8b8c2f5665dc016fcfb6a969f93b848a3407c7343100b6b4bcabae21`, canonical artifact hash `7d295d791f8438eeda62a6b230ab80c8d3a051ecda43b822c7a3827ec0cb43a6`, and tensorizer hash matches V4b manifest `06cd45a1c3269497d2a0a2674d69c4021b1aa1e674591896d98fb1a5acbe7c80`. The input has 71 nodes, 282 edges, five active absolute action slots, `x=[71,36]`, `action_demand=[1,64,82]`, `action_semantic=[1,64,108]`, and `edge_attr=[282,132]`. Model config verified from `models.py`/checkpoint: compiler-only V4bGraphModel, hidden 64, two typed TransformerConv layers, output 128, no hardware messages; runtime used CPU. Training receipt records source commit `f248497d4d8b51313fa9683aa64e0b701c6c2f39`, seed 123, 1500 updates, and protocol fingerprint `09b0f68a3bd80e4bc777daf3fb0f0c99f6b6b9b9112039eb7f852e588fda1ba5`. GEMV remains out-of-cohort; it adds zero measured examples.

| Arm / profile | L1 norm | L2 | L3 | L4 | L5 | Same-checkpoint target delta Frobenius |
|---|---:|---:|---:|---:|---:|---:|
| compiler-only, U200 | 10.2443 | 10.1915 | 10.0451 | 9.9478 | 10.4591 | 0.000000 |
| hardware-message, U200 | 11.1992 | 11.2130 | 8.9434 | 7.6324 | 7.7784 | 0.051337 |
| target-FiLM, U200 | 6.7702 | 6.7818 | 6.8072 | 6.7486 | 6.9407 | 0.000272 |

Compiler-only output is exactly profile invariant in the exports; that is expected for a compiler-only path. Hardware-message and FiLM profile deltas are small here and do not establish target sensitivity quality. The exported action vectors are 128-dimensional and preserve five absolute action slots. Exact vectors are in the per-arm `*_active.csv`; norms summarize scale only.

## Controlled in-memory masking results

For U200 at 10 ns, the compiler-only encoder was run on CPU in inference mode. Each group was masked on a clone of the prepared graph by zeroing its selected value and validity fields in node numeric pairs, action-demand pairs, action semantic summaries and relevant edge semantic features; arithmetic/SSA also neutralized op-type categories. Graph topology and unrelated features were retained. These are representation ablations, not compiler claims or physically realizable source transformations. Groups overlap where the schema contains evidence in more than one place.

| Masked authenticated feature group | L1 Δ norm (% baseline) | L2 | L3 | L4 | L5 | Five-action Δ Frobenius |
|---|---:|---:|---:|---:|---:|---:|
| Array shape/rank/volume | 1.406 (13.7%) | 1.187 (11.6%) | 1.020 (10.2%) | 0.848 (8.5%) | 0.799 (7.6%) | 2.405 |
| Access/affine/root | 4.387 (42.8%) | 4.423 (43.4%) | 4.024 (40.1%) | 2.965 (29.8%) | 2.820 (27.0%) | 8.470 |
| Loop/trip | 0.046 (0.4%) | 0.044 (0.4%) | 0.265 (2.6%) | 2.265 (22.8%) | 2.591 (24.8%) | 3.452 |
| Arithmetic/SSA | 0.141 (1.4%) | 0.139 (1.4%) | 0.403 (4.0%) | 9.323 (93.7%) | 6.296 (60.2%) | 11.258 |
| Dependence | 1.267 (12.4%) | 1.220 (12.0%) | 1.185 (11.8%) | 3.654 (36.7%) | 3.681 (35.2%) | 5.604 |

Mask definition: shape zeroed node numeric pairs for `memory_volume`, `memory_rank`, `source_volume`, action-demand `access_dimensions`, and semantic `index_result_rank` / `affine_result_count` fields. Access/affine/root masked action-demand read/write counts, witness/root/access/affine-index count fields, and action/edge semantic `operand_position_*`, `index_*`, and `affine_*` fields. Loop/trip masked node `loop_depth`/`trip_count`, action-demand `trip_count`/`loop_depth`, and carrying-loop/status/forwarding/state semantic fields. Arithmetic/SSA masked arithmetic demand counters, loop-carried/SSA-state demand, carrying/forwarding/state semantic fields, and operation-family categories (`op_type` set to neutral `none`). Dependence masked proven/possible RAW/WAR/WAW demand fields, unresolved/incomplete root evidence, dependence/loop-carried semantic fields, and dependence/loop-carried base edge attributes. In every group, numeric values and their paired known bits were zeroed together. Typed graph topology and relation IDs were retained, so these are feature ablations, not edge-deletion tests. Group overlap is intentional and means deltas are not additive.

Deltas establish sensitivity of this checkpoint’s action embeddings to these input ablations. They do not establish semantic decoding, causal importance under valid transformations, or decision quality. The small L1/L2 arithmetic/SSA delta is especially not evidence that arithmetic is irrelevant: those actions receive direct access/semantic summaries, and their output can be insensitive after aggregation. Weights are fixed; no retraining occurred.

Downstream candidate-score comparison was unavailable: the prepared GEMV cache contains graph tensors and profile features only; V4b provenance explicitly reports zero measured GEMV examples, and the compiler checkpoint requires candidate/demand tensors for `from_memory`. No authenticated GEMV candidate/QoR batch exists in the inspected manifest/run. Candidate scores are therefore omitted rather than fabricated.

## Evidence status

- **Implemented and verified:** existing V4b checkpoint and prepared GEMV cache load on CPU with strict model state matching; base action embeddings reproduce the committed compiler-only export norms.
- **Numerical experiment:** controlled in-memory feature-group masks and per-action embedding deltas in the table above.
- **Not available:** authenticated GEMV candidate tensors / measured QoR batch for downstream candidate-score evaluation.
- **Theoretical observation:** direct summaries and two-layer message passing can both contribute to the observed changes; these group masks do not isolate those mechanisms.
- **Proposed:** future action-evidence memory and matched routing/encoder comparisons described in the other reports.

## What is known vs inferred

Direct input: common categorical/numeric node features; per-action demand features and an action semantic vector added directly at action nodes; typed graph edges and edge evidence; compiler graph only in the compiler-only arm. `semantic_tensor_features_v22.action_edge_summary` pools supported semantic edge rows for relevant array-access operations or loop-context operations, including supported affine/access and loop-carried/dependence indicators, before message passing. Thus some of the group sensitivity is direct compressed evidence, not necessarily evidence acquired only by propagation. Learned information: the two-layer encoder aggregates neighboring node states over at most two edges and projects each node to 128 dimensions. V4b training uses candidate-independent physical memory and pooled program state; candidate/demand features enter later local and QoR/preference heads. The checkpoint contract and provenance code bind tensor schema, source/cache identities, and checkpoint hashes; dimension matching alone is insufficient. The V4b objective is learned from measured QoR/pair-preference supervision within its authenticated cohort, not from an explicit per-feature reconstruction target.

The stored embeddings demonstrate the model emits distinct action vectors. They do not identify whether a dimension represents a trip count, recurrence, affine access, or dependence. The controlled masks above show the learned embedding changes, but embedding distance cannot establish better directive choices, lower latency, or robust semantics. No downstream candidate scores are authenticated for this standalone GEMV graph.

## Historical model compatibility

`GNN_branch/model.py` is the historical four-layer GNN with max Jumping Knowledge over layer outputs. It is V1-schema/preprocessing and uses a different feature contract and historical target/supervision. `mlir_graph_v2.py` explicitly marks V2 tensors incompatible with V1 checkpoints. Reusable ideas: multi-layer readout/JK, typed-edge propagation, and ablation methodology. Historical weights and feature projections cannot be applied to RC3. A four-layer RC3 model must be freshly trained on identical RC3 tensors and labels/cohorts to the two-layer comparator.
