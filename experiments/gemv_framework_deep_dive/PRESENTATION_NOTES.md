# GEMV V4 Stage-1: seven-slide narrative

All figures below contain actual completed-run evidence or an explicitly labeled
architecture/source diagram. Stems have PNG and SVG files under
`artifacts/gemv_framework_deep_dive/v4_20261106/figures/`.

## Slide 1 — Five source actions form one coupled decision problem

**Takeaway:** Complete legal directives and demonstrated hardware quality are
different outcomes.

**Figure:** `fig1_actions`.

Three arrays contribute nine partition fields and two loops contribute four
PIPE/UNROLL fields. The inner loop has a floating-point accumulator recurrence;
uninitialized local inputs and only y[0] being observable make this a representation
study rather than a numerical GEMV benchmark.

**Next question:** How can target sensitivity be tested while keeping the source fixed?

## Slide 2 — Three contrasts share one reference

**Takeaway:** Four matched requests separate budget, clock and device contrasts.

**Figure:** `fig2_requests`.

R2 reduces the ZCU104 budget to integer-rounded 20%; R3 changes its specified
clock to 3.33 ns; R4 changes the FPGA and corresponding full capacities. All
requests use one authenticated V4 adapter, identical source/metadata, PARETO_LATENCY
and the original legacy greedy candidate decoder.

**Next question:** Which fields actually change under these constraints?

## Slide 3 — Budget changes five fields; clock and device keep the argmax

**Takeaway:** The observed adaptation is specific to the requested contrast.

**Figure:** `fig3_directives` (13-field matrix); report section 4 for the five-action table.

R2 changes A's dimension, x/y factors and inner PIPE/UNROLL. R3/R4 retain every
R1 directive, and R1/R4 reproduce the historical V4 outputs exactly; all four
configurations pass source and V2 tool-policy validation, with no synthesis claim.

**Next question:** Does an unchanged directive imply unchanged internal states?

## Slide 4 — The internal target response exists beyond final choices

**Takeaway:** Fixed argmax decisions coexist with changed representations and rankings.

**Figures:** `fig8_final_decision_states` for the general final-state view;
`fig4_similarity_L1`, `fig4_similarity_L4`, `fig4_similarity_L5` for layer details.

The final pre-normalization A-factor residual moves by 9.60% for the device
contrast, despite retaining factor 2. Clock/device each change three full candidate
rankings without changing the winning values. These final residuals precede the
model's final RMS norm; they are distinct from the exact normalized pre-logit states.

**Next question:** How much of the budget response comes from previous generated values?

## Slide 5 — A frozen prefix changes the interpretation of L5

**Takeaway:** Sequential output context is a major confound in natural hidden-state comparisons.

**Figure:** `fig5_movement`, with the natural/controlled L5 panels in
`fig4_similarity_L5`.

At layer24/post-layer, the budget contrast moves L5 UNROLL by 0.750193 naturally
(cosine 0.708378), versus 0.049462 with earlier fields fixed to R1 (cosine
0.998788). Only one of five official budget-related argmax flips remains under
the frozen prefix; this control does not identify which prior field caused each change.

**Next question:** Do the corresponding candidate scores explain that distinction?

## Slide 6 — The same candidate domain supports different preferences

**Takeaway:** Candidate scores reveal both a direct target response and strong prefix dependence.

**Figure:** `fig6_candidates`, preferably crop the A-dimension and L5-unroll columns.

A dimension changes from 1 to 2 even with controlled earlier assignments, with
R2 margin 0.022713. Natural R2's L5 UNROLL is 0, but controlled R2 prefers 2 over
4 by 0.216918; all candidate sets are unchanged, and controlled L5 PIPE's lead
for 0 is only 0.000515. These are token scores, not hardware-quality predictions.

**Next question:** What explicit compiler information could be tested at these sites?

## Slide 7 — Define a structural intervention from actual compiler evidence

**Takeaway:** Compiler memory should be audited as additional evidence, not presumed to repair a missing concept.

**Figure:** `fig7_architecture`, with report section 7's compiler comparison table.

The authenticated 66-node/106-edge graph represents A's affine accesses, loop
nesting/trip counts, and three proven scalar-SSA forwarding edges for the inner
accumulator. No GNN fusion was used. Before attributing a deficit to structural
memory, the most informative next experiment swaps earlier changed prefix fields
individually to identify which feedback drives the observed L5 response.

**Next question:** Which prior directive changes cause the L5 preference shift,
and which verified action relations would a subsequent fusion test need to address?
