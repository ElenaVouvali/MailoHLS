# Next latency-aligned GNN and Stage-2 ablations

No model training is performed by this package. The current frozen V4b encoder emits bit-identical compiler memory for tested U200/10 ns and ZCU104/3.33 ns contexts because hardware edges are excluded. The following specification keeps that static compiler-memory interface frozen and separates information recovery, physical latency prediction, feasibility, and generated directives.

## GNN comparison to run later

Use the authenticated RC3 graph set, fixed family-disjoint train/validation/test splits, absolute action slots, and identical hardware-context inputs for all arms:

| Arm | Encoder | Purpose |
|---|---|---|
| G0 | Existing two-layer V4b RC3 GNN | Frozen diagnostic baseline; ADP-oriented, not latency-qualified |
| G1 | Capacity-matched two-layer RC3 encoder | Tests whether current representation capacity is useful under latency supervision |
| G2 | Four-layer RC3 encoder | Depth comparison with parameter/compute reporting |
| G3 | Four-layer RC3 plus Jumping Knowledge | Tests whether retaining multiple propagation depths helps |
| G4 (optional) | Direct action/compiler feature head without message passing | Measures what structured compiler inputs provide before GNN propagation |

Primary target: predict latency and rank complete configurations **subject to measured resource feasibility**. Retain physical-QoR prediction only as optional auxiliary supervision. Do not choose the encoder from the old ADP-only ranking. Report compiler-fact recovery separately from latency regret, feasibility classification, and constrained directive generation. Match family splits, candidate pools, optimizer budget, and selection rules across arms. Include a source/contract and graph hash manifest, and select on validation families only.

## Future Stage-2 fusion experiment

Freeze the authenticated latency V4 Stage-1 checkpoint; initialize every arm from identical Stage-1 weights/tokenizer and use fresh cross-attention weights. Historical V1 ADP cross-attention is not compatible with the V4b/RC3 memory contract. Keep training requests, objective, examples, update count, seeds, and family splits fixed.

| Arm | Memory / control | Question |
|---|---|---|
| S0 | No-memory Stage-1 | Baseline |
| S1 | Exact action memory A | Does action-aligned frozen compiler memory help? |
| S2 | RC3 relational action memory B | Does compiler-derived routing help independently of vector content? |
| S3 | Typed action/evidence memory C | Do explicit compiler witnesses help beyond action vectors? |
| S4 | Zero-valued memory with identical masks | Is any gain caused by added parameters or route topology? |
| S5 | Shuffled/mismatched graph memory | Does alignment to the correct program matter? |
| S6 | Capacity-matched non-informative memory | Does a same-size learned context explain the effect? |

Compare the historical four-layer schedule `[8,16,24,32]` with four-layer alternatives sampled from the full-depth supported shortlist `[12,16,24,28,32]`; hold the number of attention blocks and parameter count fixed. A one-layer diagnostic can be added as a pilot, but does not replace matched four-layer arms. The present CPU interface tests do not establish a best layer.

First fusion boundary should be the existing `post_self_attention_residual`: (u=h+SA(Norm(h))), then fuse memory before the layer's post-attention normalization/MLP. Preserve the GNN output contract and the same memory tensor across attention placement arms. Initial experiment should use static compiler memory; do not add target-conditioned gating or hardware-conditioned GNN memory until matched static-memory gains are established. After that, test target-conditioned gate and memory conditioning separately.

## Machine-readable protocol

`manifests/latency_evaluation_protocol.json` records frozen arms, primary objective, splits, metrics, and controls. It is a protocol template, not a training receipt or a claim that the cohort already has all required labels.
