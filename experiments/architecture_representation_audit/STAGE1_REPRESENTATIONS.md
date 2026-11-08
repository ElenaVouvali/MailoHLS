# Stage-1 representation analysis

## Completed evidence and limits

The available 32-layer study is the existing GEMV full-depth inference-only analysis in [`experiments/gemv_stage1_full_depth/STAGE1_32_LAYER_ANALYSIS.md`](../gemv_stage1_full_depth/STAGE1_32_LAYER_ANALYSIS.md) on the authenticated reference branch. It captures the post-self-attention residual at exact directive-prefix decision positions, uses the production constrained candidate scorer, and reports 189 controlled activation-patching records. The checkpoint/tokenizer/adapter identity is recorded there. Raw activation chunks are not available in this audit worktree and were not copied or regenerated.

This is useful decision-boundary evidence, not the requested cross-kernel 32-layer property-decoding study. No comparable exact action-aligned, compiler-labeled hidden-state matrix exists across the 55 RC3 cohort members; the held-out family split is only meaningful for GNN, not for Stage-1 source prompts. The V100S is reserved by the independent Stage-2 run, and this investigation has not loaded a model or used a GPU.

## Existing GEMV observations (not generalization claims)

| Evidence | Result | What it supports |
|---|---:|---|
| Layer width/depth | 4,096 / 32 | Correct model layer contract |
| Hardware-context comparisons | 18 requests; 2 devices × 3 clocks × 3 resource fractions plus a historical condition | Prompt context changes states and candidate preferences in this GEMV setup |
| Candidate margin comparisons | Active candidate set matched 117/117 field contrasts; winner changes in two L5 budget cases | Hardware prompt can affect actual constrained scores/selections |
| Activation patches | 189 records; placebo exactly zero; 4 preference flips | Decision states can causally affect matched candidate-score ordering in sampled cases |
| Structural source contrasts | Reduction, independent-iteration program, and rename control | Source changes affect representations/preferences; rename sensitivity cautions against invariance claims |

The referenced report found late-layer device/budget state response, clock response concentrated around layers 9–15 plus a late contribution, and small patch-induced ranking changes. Hidden-state movement does not establish that a compiler property is linearly accessible or that cross-attention at that layer is useful. The source-contrast pair changes algorithm/output behavior; the rename control itself shifts a decision. Neither is a robust multi-family semantic probe.

## Required cross-kernel analysis remains open

No 32-layer compiler-property probe, family-bootstrap result, lexical baseline, random-label control, capacity-matched GNN comparison, semantic invariance estimate, or cross-kernel causal intervention was run. Consequently, there is no empirically justified layer-by-property heatmap for this cohort and no selected fusion layer. The historical GEMV shortlist is input to a later experiment protocol only, not a layer choice.

When GPU capacity is available, capture only exact directive-decision positions and the post-self-attention residual boundary, stream compact activations, and verify hook-on/off scorer equivalence. Join with RC3 labels only for source-identity and action correspondence that can be proven. Fit probes by family-disjoint splits, with the frozen test family untouched and family-level uncertainty. Use the original GEMV findings as a prior case study rather than as evidence for a multi-kernel claim.
