# Stage-1 representation behavior across all 32 layers

## Study and authentication

This is an inference-only investigation of V4 multi-reference latency Stage-1. The adapter SHA-256 is `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`; the saved training contract pins DeepSeek-Coder-6.7B-Base revision `ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912`. Inference ran on Tesla V100S UUID `GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb` using the existing NF4 double-quantized loader with FP16 compute. The output preserved the checkpoint, tokenizer, special-token mapping, directive registry, prompt schema, and production decoder unchanged.

The frozen matrix contains 18 requests over two devices × three clock periods × three uniform resource fractions, plus a separately labelled historical 20% request and a 24-row matched source-structure subset. Every official result has 13 fields. All rows passed source/tool V2 validation. No GNN or cross-attention was loaded, and no synthesis/QoR was measured.

## What is captured

The DeepSeek-Coder backbone has 32 layers of width 4096 (verified by the loaded model and captured tensor shapes). At each exact directive-prefix position, the tracer saved: (1) the pre-hook input to the native `post_attention_layernorm`, which is the post-self-attention residual `u`; (2) the decoder block output residual; and (3) the final RMS-normalized vector entering the LM head. The source marker positions are auxiliary and occur before their source declarations, so they are not treated as complete action summaries. The directive states occur after the source has been read and immediately before candidate scoring. The candidate score still comes from the real serialized candidate continuation plus newline; a prefix vector alone is not treated as a full candidate likelihood.

Per request the raw chunk has shape `[1690,4096]` float32: all 13 natural decision sites, five source markers, eight controlled-prefix decision sites, two residual boundaries at every layer, and the normalized LM-head input. The 43 request chunks and exact token/index metadata are under `artifacts/gemv_stage1_full_depth/v1_20261106/hidden_states/`. Hook-on/off full decode equivalence passed on the R1 reference, and every traced decision-site scorer call matched the corresponding unhooked score within the experiment tolerance.

## Hardware sensitivity and directive preferences

A nonzero change in a hidden state is observed for all tested target contrasts. The table below gives median post-self-attention relative L2 movement across directive sites and layers; cosine remains high, which does not mean that the target is ignored. Natural and controlled prefixes are equal for device and clock contrasts because upstream directives stayed equal. Budget contrasts show additional natural-prefix movement at later fields when preceding choices diverge.

| Contrast family | Natural median Δh | Controlled median Δh | Natural median cosine | Interpretation |
|---|---:|---:|---:|---|
| device | 0.0602 | 0.0612 | 0.99807 | representation responds; score relevance is assessed separately below |
| clock | 0.0287 | 0.0282 | 0.99940 | representation responds; score relevance is assessed separately below |
| budget | 0.0295 | 0.0279 | 0.99640 | representation responds; score relevance is assessed separately below |

Controlled-prefix focus: L1 type/factor/dimension, L2 factor, L4 pipeline/unroll, and L5 pipeline/unroll. The complete natural-prefix grid covers all 13 fields. In the one-factor official comparisons, device and clock leave all 13 selected RHS values unchanged; a 10 ns budget change alters two L5 choices only at XCU200/50% (PIPE_L5 `0→1`, UNROLL_L5 `2→0`). Other 10 ns budget comparisons are invariant. The full factorial grid is not monotonic: ZCU104/3.33 ns/25% changes L2 factor, L3 partition tuple, and L5 unroll relative to R1, but this combines clock and budget effects and cannot assign the cause to either one. See [DIRECTIVE_GRID.md](DIRECTIVE_GRID.md) for all 18 configurations.

All target comparisons used the same active candidate set: 117/117 field contrasts. The production top-two mean-logprob margin changed in 13 device, 52 clock, and 52 budget field comparisons. Top-two ranking order changed in 1, 2, and 5 fields, respectively; selected RHS changed in 0, 0, and 2 fields. Median absolute margin shifts were device 0.0302, clock 0.0238, budget 0.0317. These are ranking-score changes, not calibrated probabilities or QoR predictions.

## Layer-resolved findings

Full 32-layer tables are in `layer_pairwise_metrics.csv`; Figure 1 shows natural hardware contrasts by layer/action and Figure 4 separates natural from controlled prefixes. Device and budget effects grow toward late layers (layers 28–32); clock effects are strongest around layers 9–15 with a second late contribution. Across controlled focus states, median movement is about 0.0612 for device and 0.0282 for clock. These aggregates do not select a fusion layer by cosine distance.

A targeted read-only patch experiment replaces one receiver request’s post-self-attention residual at the directive token with the aligned controlled R1 vector, then invokes the unchanged production candidate scorer. Candidate domains and the prior prefix remain matched; the same-condition activation is the placebo. Across 189 layer/action/target patch records, placebo margin change was exactly zero. Donor patching changed candidate margins by a median absolute 0.0078 log-probability/token and flipped 4/189 sampled candidate-pair preferences (3 L2 factor flips and 1 L5 pipeline flip), all on the budget contrast. The flips occur at small receiver margins; they establish that these layer activations can affect the scorer, not that the resulting directive is physically better. See `activation_patching.json` and Figure 9.

## Structural source contrast and action-level interpretation

The original authenticated C source is unchanged. The extra diagnostic sources are newly compiled initialized programs: `reduce32` retains a scalar accumulation; `independent32` emits a per-iteration product and zero final `y` (a different algorithm/output); `reduce32_rename` is a variable-renaming control. All preserve five source-action registrations, compile with the pinned RC3 pipeline, and pass their own deterministic numeric oracle. The independent program is not claimed equivalent to reduction.

RC3 MLIR for `reduce32` has an inner `affine.for` with `iter_args`, `arith.addf`, and `affine.yield`; the independent version has no inner loop-carried scalar result. This is a verified scalar SSA distinction, not a proof about memory independence. The rename control retains the same recurrence and numerical behavior. Source and graph hashes are listed in `source_variant_manifest.json`.

Under a common R1 directive prefix, the L5 unroll argmax is 4 for the reduction and 2 for independent iterations; the L5 pipeline argmax remains 1 but its margin changes. The rename control makes PIPE_L5 narrowly prefer 0 and keeps unroll 4. In natural predictions, all three choose PIPE_L5=1/UNROLL_L5=0. This is source-contrast sensitivity and conditional candidate preference, but the recurrence change also changes computation, output behavior, source text, and tokenization. The rename contrast is a warning that lexical variation itself can move states and decisions. At R1, L1 partition dimension ranks 1 over 2 for reduce32 and independent32, while the rename control selects dimension 2. Therefore the observation does not establish access-direction reasoning; it demonstrates that this decision is not invariant to a meaning-preserving lexical change.

No cross-kernel probe was trained: a single GEMV source and its small variant set cannot support generalization claims. Trip count, loop depth, affine access role, recurrence, and array shape are available as explicit compiler facts in RC3, but decodability as a semantic variable is not established by these within-program contrasts. A hidden vector can respond to a source edit without representing the intended property in a reusable way.

## What is established and what remains open

**Established:** target prompts alter the LLM’s contextual states at all decision sites; target changes alter some actual production score margins; a small set of budget-sensitive winners changes; R1 activation patches can change matched directive scores and occasionally their ranking; compiler-verified source changes and a syntax control move L5 states/preferences.

**Not established:** target-dependent outputs improve physical QoR; any directive is optimal or feasible; the LLM has a robust, syntax-invariant internal representation of recurrence or array access; a particular layer is optimal for cross-attention; static GNN memory improves Stage-1 decisions. The original GEMV’s uninitialized inputs and partial output observability further limit physical interpretation.

## Short layer shortlist for Stage-2 experiments

Based on controlled target movement, the targeted score patch, and source sensitivity—not activation distance alone—test post-self-attention residual insertion at **layers 12, 16, 24, 28, and 32**, with layer 8 as an explicit historical low/contrast point. The uniform patch sweep sampled layers 1,4,8,12,16,20,24,28,32; layer 12 and 16 capture the clock-sensitive middle, layer 24 includes the sampled budget-induced preference reversals, and layers 28/32 capture late device/budget response. Layer 32 had the largest median patch-margin shift; this is not proof of optimality. The historical set [8,16,24,32] remains a baseline, not a conclusion.

A minimal future sequence is: first freeze V4 and compare no memory against static compiler-only action memory; then compare exact-slot and compiler-relational routing with identical memory/parameter budgets; include zero-information and within-kernel action-shuffled memories. Keep target inputs only in the LLM initially. Test target-conditioned gate and hardware-conditioned GNN memory as separate later ablations. These distinguish complementary compiler facts from merely additional parameters or target-conditioned memory.
