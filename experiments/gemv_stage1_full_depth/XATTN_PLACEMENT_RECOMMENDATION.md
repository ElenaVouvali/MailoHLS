# Structural cross-attention placement and interface recommendation

## Evidence-ranked layer shortlist

The measured post-self-attention residual is exactly the insertion boundary supported by `StructuralPostSelfAttentionResidual`: a pre-hook on `post_attention_layernorm` observes `u`, cross-attention produces `v=u+r`, and an MLP hook adds the same residual so the native outer residual yields `v + MLP(Norm(v))`. The post-decoder output and final normalized LM-head vector were captured separately and are not substituted for this boundary.

**Primary Stage-2 layer tests:** 12, 16, 24, 28, and 32 post-self-attention residual; retain layer 8 as a historical comparator. Layer 12/16 sample clock-sensitive middle depth, 24 has patch-induced flips on near-tied budget preferences, and 28/32 carry the largest late target movements; 32 showed the largest median sampled patch-margin effect. The activation patch only sampled `[1,4,8,12,16,20,24,28,32]`, so evidence for intervening layers is descriptive only. This is a shortlist for training ablation, not an optimal layer claim.

Across the matched R1-donor patches, the same-state placebo had zero score effect. Donor replacement changed the margin and flipped only four of 189 sampled pairs, mainly where the unpatched margin was near zero. Layer 8’s median change is near the smallest sampled; this is a reason to include it as a control, not to reject it. Additional matched trained-fusion ablations are needed.

## Memory and routing contract

Read-only V4b RC3 tensorization supplies 36 numerical node features, five categorical embeddings, typed-edge attributes, action-demand `[B,64,82]`, action-semantic `[B,64,108]`, and registered/matched/unresolved/source-only absolute-slot masks. Its encoder outputs action memory `[B,64,128]` and pooled program memory `[B,128]`; the saved GEMV arrays confirm `(64,128)`, `(128,)`, masks `(64,)`, and active slots L1–L5. `Lk` is absolute slot `k−1`; do not compact the five actions. These are V4b research encodings, not production V1-compatible learned weights.

The production Stage-2 interface consumes `[B,S,D]` memory and a `[B,S]` mask; optional compiler-relational routing supplies an `[B,S,S]` action relation mask. Exact-slot mode gives one query exactly one key. With one key, softmax attention is identically 1, so the operation behaves as a query-independent transformed action vector followed by a learned scalar gate. Relational routing can expose several verified action memories, allowing query-conditioned attention weights. Whether those additional keys help L1/L5 must be measured against exact-slot, not assumed.

The available production ADP contract is a different objective/checkpoint: `PARETO_ADP`, 128D/64 slots, `post_self_attention_residual`, compiler-relational routing, layers [8,16,24,32], and parent Stage-1 adapter SHA beginning `222f3c24`; it is not a compatible latency-V4 Stage-2 model. Reinitialize fresh attention/gates for a V4 latency study; do not reuse that checkpoint’s attention weights as if trained for V4. V4b `[B,64,128]` dimensional agreement does not establish compatible graph schema, memory manifest or embedding semantics with production V1.

## Recommended ablation order

1. **Static compiler-only memory first.** Hold the LLM target prompt fixed and compare frozen V4/no memory against authentic RC3 compiler action memory. This isolates structural information; no target-conditioned GNN values are required to explain the observed prompt sensitivity.
2. **Information controls.** Add zero-information, within-program compatible-action shuffle, and a capacity-matched content-free projection. Include loop-depth/trip-only memory against full recurrence/access semantics. Preserve absolute slots and explicit matched masks.
3. **Routing.** Compare exact-slot (one memory key) with V4b compiler-relational masks (multiple compiler-linked action keys), same encoder/output width and trainable parameter count. Report keys-per-query and attention entropy.
4. **Gate/context.** Start with the existing scalar per-layer gate. Only after an authentic-memory effect is established, test a context-conditioned gate with parameter-matched control; the target information already enters the LLM, so target-conditioned gating is a hypothesis, not a current requirement.
5. **Hardware-conditioned GNN memory last.** The encoder can include a target overlay, but Stage-1 already responds to device/clock/budget. Add target to memory only after static compiler-memory gains survive shuffled controls and the target/structure interaction is replicated.

Use frozen V4 and identical Stage-1 inputs, constrained decoding, and candidate policy. Do not use this descriptive GEMV source as a QoR benchmark. The meaningful outcomes are held-out directive ranking, legality, and eventually successful measured synthesis under a verified physical context.
