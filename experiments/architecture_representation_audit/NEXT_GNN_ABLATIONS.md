# Next GNN ablations (proposal; none trained here)

## Prerequisite

First fix and regression-test RC3 action reconciliation and evidence overflow on the exact 55-source cohort. Keep this work in a separate reviewed implementation change. No evidence may be silently truncated; preserve source graph identity, absolute action IDs, status/known masks, and deterministic provenance. Rebuild and authenticate all C vectors/masks before any architecture comparison.

## Matched encoder comparison

| Arm | Encoder | Question |
|---|---|---|
| E0 | Existing two-layer V4b, hidden 64, 128D output | Reference behavior |
| E1 | Historical four-layer + Jumping Knowledge, capacity-matched to the same 128D action output | Does larger receptive field improve property recovery when parameter count and supervision are controlled? |
| E2 | Two-layer V4b with direct action summaries only, no graph messages | How much recoverability comes from action-demand/semantic bypass? |
| E3 | Parameter-matched MLP/control without typed graph propagation | Does graph structure help beyond feature capacity? |

Use identical authenticated RC3 tensors, family split, supervision examples, label masks, optimizer, update budget, early-stop rule, and latency-focused measured objective/resource constraints. The existing V4b checkpoint can be evaluated frozen as a reference but is ADP-oriented, so it is not a latency-trained comparator. Fresh arms should share the same initialized projection/head contract where dimensions allow, and report parameter counts and compute cost.

## Diagnostics required per arm

1. Property recovery at R0 raw fields/local context, projected input, each message layer, final nodes, action vectors, and C evidence vectors.
2. Family-disjoint validation; report the held-out test family once. Use family bootstrap only where enough independent families exist; otherwise mark uncertainty as underpowered.
3. Unknown-aware classification for proven recurrence positives vs explicit established negatives only; if negatives are absent, do not fit a binary task. Trip count/depth use numeric targets and known masks. Affine role and access classification keep unknown/possible separate.
4. Direct-feature, constant, random-label, and capacity-matched controls; tune regularization only on validation families.
5. Witness-specific distance interventions on graph copies, compared with matched irrelevant-node interventions. These remain diagnostic unless the source is recompiled and compiler facts reauthenticated.
6. Memory-C coverage and per-action witness coverage as mandatory outcomes, not post-hoc exclusions.
7. Latency-focused held-out preference/feasibility evaluation under resource constraints. Keep QoR/ADP results separate; do not infer latency utility from embedding probes.

## Interpretation rule

Only test E1 after E0 diagnostics show a specific property is present in raw/tensor features but poorly recovered at R2/R3/action output, and action-evidence distances support a propagation hypothesis. If action summaries explain the available recovery, a deeper encoder is not justified. If recovery exists at action nodes but not in C, prioritize evidence selection/vector/routing rather than encoder depth. A future four-layer + JK result is an alternative hypothesis, not a default architecture.
