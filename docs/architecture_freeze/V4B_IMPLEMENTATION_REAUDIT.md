# Corrected V4b implementation re-audit

Verdict: no incremental hardware-message or FiLM advantage demonstrated under the historical ADP protocol. This does not reject hardware-aware memory for latency Stage2.

All saved source hashes in the corrected full37 protocol match this implementation. All 3,664 cache measurements were reauthenticated against source/table/graph/split identities. The cache SHA is `4a74291644d1cd4ccbd523c3156b75d2cf6211007810c4ff478bc81ed4eaf356`. See [raw re-audit](../../experiments/final_architecture_20261109/results/v4b_reaudit.json).

`V4bGraphModel.encode` removes hardware edges only for compiler memory; `from_memory` concatenates `graph.hardware_context` into both heads for both arms. Budget is separate and absent from physical predictions. `SmallGraphEncoder` executes exactly two residual typed TransformerConv layers with reverse derived edges. On machsuite-gemm-blocked, all five action nodes are one message hop from hardware nodes. Removing hardware edges leaves 256 edges versus 322. Hardware-feature gradient norm is 17.560793 with messages and exactly zero without them. This verifies transmission, not useful optimization.

The fast `(kernel,config_hash)` cache equals fresh `(kernel,device,clock,config_hash)` tensors exactly for 96 candidate contexts across both devices and three clocks. Candidate demand remains source/candidate-derived. No cache fix was warranted. Compiler memory is invariant; hardware and FiLM memory vary. The direct numeric head ablation changes predictions with memory fixed. H0/H1/H2/H3 therefore have distinct information paths; H2/H3 here are inference diagnostics, not separately trained arms.

FiLM modulates matched action vectors; pooled program vectors remain unchanged. Its selected raw gate is 0.00445602, effective gate 0.00445599; modulation/base norm ratio 0.00240950 on the first audited context. In two disposable CPU diagnostic updates, gate gradient at zero is -0.000657884; MLP gradient norm moves from zero to 6.50771062e-06. These two updates establish gradient mechanics only. They are excluded from GPU/full-training spend and no model is saved or promoted.

Historical 1,500-update arms share exact sampled identities and compatible initial states. FiLM selects step250; a plateau was not established. The capacity head uses191,604 parameters versus FiLM191,667 and selects step500. Full model compiler/hardware each127,538. Recovered historical weights and receipts are hashed, with source commits in the JSON. No gate-gradient trajectory was logged historically; do not reconstruct one.

| Predictor | Feasible-selection failure (family macro) | Conditional ADP regret | Combined |
|---|---:|---:|---:|
| compiler | 0.210734 | 0.526370 | 0.737105 |
| hardware | 0.168362 | 0.623194 | 0.791556 |
| no_graph | 0.109040 | 0.849426 | 0.958466 |
| target_film | 0.125989 | 0.560131 | 0.686120 |
| wider compiler head | 0.109040 | 0.523430 | 0.632469 |

Each result has three validation kernels, two families,18 physical contexts,90 correlated budget views. Lower combined score means failure+conditional regret; it is not unconditional latency. No graph improves both coverage and regret over every control. Zero/shuffled/wrong-device memory controls were replayed; wrong-device memories retain the true numeric head target. These models evaluate supplied designs and do not generate directives.

305 training and5 validation selected observations have all-zero resource percentages; the corrected ADP ordering excludes them globally while physical regression retains proxy zero labels. Supplement reports normalization sensitivity after excluding them, without retraining. Exact integer counts and timing compliance remain unavailable here.

| Concern | Verdict | Limitation |
|---|---|---|
| Target path and hardware gradients | VERIFIED | One real kernel, six physical contexts |
| Shared initialization and schedule | VERIFIED | One historical seed |
| Real candidate cache equivalence | VERIFIED | 96 comparisons; source version pinned |
| Global zero-ADP correction | VERIFIED | Regression still includes ambiguous proxy labels |
| FiLM gate mechanism | VERIFIED | Historical convergence INCONCLUSIVE |
| Latency-oriented matched training | NOT_TESTED | Requires informative development cohort |
| Latency Stage2 advantage | NOT_TESTED | Standalone head is a different task |
