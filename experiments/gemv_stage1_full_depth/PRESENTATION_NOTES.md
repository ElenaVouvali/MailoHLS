# Presentation notes

## 1. The task

**Use:** Figure 8 and Directive Grid.

The same five actions must satisfy a source, objective, device, clock and requested-resource context. The run covered 18 target rows plus a separate historical 20% case. Predictions are source/tool-legal, but no physical synthesis was performed.

**Next question:** Which target changes actually alter the selected directive?

## 2. Hardware enters twice

**Use:** Figure 6 and HARDWARE_ENCODING_AUDIT.md.

Device, clock and objective use learned special-token rows; resource counts/percentages use ordinary tokenizer pieces. The saved rows moved relative to exact resize-time initialization, while numbers are not dedicated tokens.

**Next question:** Does this prompt information reach the directive states?

## 3. 32-layer target path

**Use:** Figure 1 and Figure 4.

For every request we recorded decision states at post-attention, post-block and final normalized boundaries. Target changes move states across layers; natural and controlled-prefix paths separate direct prompt effects from autoregressive carryover.

**Next question:** At which points do state changes alter candidate preferences?

## 4. Directives vs margins

**Use:** Figure 3 and DIRECTIVE_GRID.md.

Device and clock one-factor comparisons leave winners unchanged. XCU200 at 50% changes L5 PIPE and UNROLL; many other contexts change candidate margins without changing the chosen RHS.

**Next question:** Are these preference shifts mediated by particular layers?

## 5. L5 structural control

**Use:** Figure 2 and Figure 5.

RC3 verifies a carried scalar for reduction and no explicit scalar loop-carried result for independent products. Under a common prefix the unroll ranking differs, while a semantics-preserving rename also changes states and PIPE preference.

**Next question:** Can compiler memory supply invariant recurrence/access summaries?

## 6. Layer causal test

**Use:** Figure 9 and layer_evidence.csv.

At nine spread-out layers, R1 decision activations were patched into matched target requests. Same-state placebo was exactly null; four of 189 pair preferences flipped, with small starting margins.

**Next question:** Which insertion layers survive a trained fusion ablation?

## 7. Structural interface

**Use:** XATTN_PLACEMENT_RECOMMENDATION.md.

V4b memory preserves absolute L1–L64 slots and provides 128D action vectors. Exact-slot attention has one key and cannot select among memories; compiler-relational routing can expose verified neighbors.

**Next question:** Does authentic compiler-only memory beat shuffled/capacity controls?

## 8. Next Stage-2 test

**Use:** Figure 8 and Stage-2 recommendation.

Start with static compiler-only memory at layers 12,16,24,28,32 and layer8 as historical comparator. Compare exact-slot vs relational routing, no-memory and shuffled controls; keep hardware-conditioned memory separate.

**Next question:** Does compiler structure improve legal held-out choices and then measured QoR?
