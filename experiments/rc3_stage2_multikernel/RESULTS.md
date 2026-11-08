# Results status

## Data and memory preparation

The authenticated V4 data/RC3 graph audit selected one source kernel from each
of 17 training families and retained all three validation kernels from the two
existing held-out families. The resulting pilot has 1,164 training and 54
validation rows. The compiler-only V4b GNN was frozen and used to build Memory C
for all 20 selected kernels. Memory packs contain 42–877 evidence records
(maximum combined key count 941); action slots remain absolute L1–L64. Every
pack is source/graph/checkpoint authenticated and static across hardware
requests.

See [COVERAGE.md](COVERAGE.md) for the scope and limits of the measured-derived
labels. In particular, achieved HLS clock estimates and exact resource counts
are unavailable in the curated SFT rows, so no timing-qualified or physical-QoR
claim is made here.

## Five-step layer-32 smoke

The production trainer loaded the authenticated Stage-1 adapter and fresh
layer-32 `post_self_attention_residual` module, verified zero-gate start, and
completed five optimizer updates on the V100S. A variable-length Memory C
batching error (`S=119` versus `S=573`) was fixed by right-padding each batch
to its longest pack and false-masking padded keys and relation rows/columns.
Focused CPU tests pass (7 passed, 3 skipped).

At the sharing preflight, the target V100S had 14,487 MiB free, with a Ray worker
using 18,004 MiB; the other Ray worker was on the A2. Explicit Ray-sharing mode
allowed the run without modifying or stopping either process. The Ray worker
released the V100S during model loading, so the five optimizer updates ran
after that worker had exited. One transient `nvidia-smi` driver-communication
failure occurred during final validation; telemetry recovered and the trainer
finished normally without system changes.

| Smoke metric | Step 0 | Step 5 |
|---|---:|---:|
| Held-out Viterbi decision-site accuracy (101 sites) | 0.52451 | 0.52451 |
| Mean candidate margin (96 non-forced sites) | -0.022090 | -0.022200 |
| Validation loss | 0.706614 | 0.706600 |
| Exact full-design accuracy in selected subset | 0.0 | 0.0 |

These near-identical validation numbers show no measurable gain in five
updates. The gate moved from 0 to -0.00029197; its initial gradient norm was
8.47e-4. Between step 0 and step 5, the Q/KV/output projection weight L2
changes were 0.02246, 0.01115, and 0.02481, respectively, and attention norm
weight/bias L2 changes were 0.000312/0.000467. The frozen Stage-1 adapter hash
remained exactly `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`.
The saved step-5 checkpoint includes optimizer, scheduler, scaler, RNG, and
structural weights. Step-0 and step-5 val-selection files are both retained.

The small smoke-only selection subset was one held-out kernel (Viterbi) across
its device/clock groups. It does not constitute multi-family generalization or
physical-QoR evidence.

## Four-layer pilot

The bounded pilot completed 20 optimizer updates using fresh gated
`post_self_attention_residual` modules at layers `[8,16,24,32]`. The Stage-1
LLM, its LoRA and special-token embeddings, and the V4b GNN remained frozen.
The run took 1,688.1 seconds. Training loss was 0.374891. Evaluation loss on
the recorded validation batches changed from 0.706614 (step 0) to 0.706459
(step 10) and 0.706453 (step 20).

The family-held-out checkpoint selection comparison used two kernels from
Viterbi and Pathfinder across the configured device/clock cases. At step 0 and
step 20, mean decision-site accuracy was 0.293505, key decision accuracy was
0.264154, exact-design accuracy was 0, and schema/source semantic compliance
was 1.0. The mean candidate margin changed slightly from -0.485576 to
-0.486175. Thus the small loss decrease did not improve the declared
validation-selection metrics. The training run demonstrates a runnable
multi-kernel Stage-2 path, not useful generalization or improved QoR.
`best_custom_stage1` points to the zero-gate step-0 model under the saved
selection contract; the step-20 checkpoint remains available for diagnostics.

The four-request constrained-decoder comparison used the authenticated
Viterbi RC3 Memory C pack and identical source/action metadata for ZCU104/10 ns,
ZCU104/3.33 ns, XCU200/10 ns, and XCU200/3.33 ns. Stage-1, zero-gate C, and
step-20 C each produced complete 17-field assignments for all four requests.
All 12 assignments passed the repository's source-derived TCAD tuple
validator. Zero-gate and step-20 predictions matched the Stage-1 assignment
for all four requests; Stage-1 changed two of its four assignments across the
target contexts. This is a small held-out diagnostic, not a QoR test.

| Action | ZCU104 / 10 ns | ZCU104 / 3.33 ns | XCU200 / 10 ns | XCU200 / 3.33 ns |
|---|---|---|---|---|
| L1 | complete, factor 0, dim 2 | same | same | same |
| L2 | pipeline 0, unroll 2 | same | same | same |
| L3 | pipeline 1, unroll 0 | same | same | same |
| L4 | pipeline 1, unroll 0 | same | same | same |
| L5 | pipeline 0, unroll 2 | same | same | same |
| L6 | pipeline 1, unroll 0 | same | pipeline 0, unroll 4 | same as ZCU104 / 10 ns |
| L7 | pipeline 1, unroll 0 | same | same | same |
| L8 | pipeline 0, unroll 4 | same | pipeline 1, unroll 0 | same as ZCU104 / 10 ns |

This confirms target-conditioned variation in the frozen Stage-1's R3
assignment, while Stage-2 step 20 preserves those same winners. These
predictions are not labeled optimal; no matching physical receipts are present
for this validation source/request set.

The first zero-gate production run exposed a residual-dtype mismatch: its
cross-attention block cast the native residual stream before adding the
structural result. Independent Stage-1 reloads themselves were bitwise stable
at the saved field-score level, so that mismatch was not ordinary reload
noise. The residual path now casts only the attention branch and adds its
gated update in the original stream dtype. After this correction, the actual
production decoder matched Stage-1 exactly over all four requests: all
directive assignments, all 68 selected field scores, and all four sequence
sum log-probabilities were identical (maximum absolute difference 0.0). A
focused mixed-dtype exact-identity test was added. The original pre-fix output
files are retained as historical diagnostic artifacts; use the
`*_residual_fix_predictions.jsonl` outputs for current comparisons.
For the corrected step-20 run, all four complete assignments still matched
Stage-1. Across 68 decisions the mean absolute selected-field score change was
0.000827 (maximum 0.006966); the mean absolute change in winner-minus-runner-up
margin was 0.002426 (maximum 0.013573). This is measurable score movement
without a selected-directive change. Current inference emits aggregate
attention diagnostics, not provenance-resolved per-key weights; this pilot
therefore does not establish which individual evidence records caused the
score movement.

The complete assignments are in the ignored artifacts
`artifacts/rc3_stage2_multikernel/v1/eval/stage1_predictions.jsonl`,
`stage2_zero_gate_predictions.jsonl`, and `stage2_step20_predictions.jsonl`.
The frozen input and hashes are in `viterbi_four_targets.jsonl` and
`manifest.json`. This worktree has no matching prepared `reduce32` RC3 memory
pack or measured multi-kernel synthesis receipts, so it makes no reduce32
structure/QoR claim. Configurations without exact synthesis receipts have
unknown physical QoR.

## Resume

The completed smoke is at
`/home/elvouvali/mailohls_runs/rc3_stage2_multikernel/v2/smoke_layer32`.
The completed four-layer pilot is at
`/home/elvouvali/mailohls_runs/rc3_stage2_multikernel/v3/pilot_layers8_16_24_32`.
If it must be resumed after interruption, use the trainer's standard resume
checkpoint mechanism with `checkpoint-10` or `checkpoint-20` and the exact
saved training contract. For a fresh run after the designated V100S is idle:

```bash
/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python -m experiments.rc3_stage2_multikernel.train --stage pilot --run-dir /home/elvouvali/mailohls_runs/rc3_stage2_multikernel/fresh --execute
```

The wrapper validates the GPU UUID/process list. Add `--allow-gpu-sharing` only
when sharing with Ray workers is intended; it permits only Ray workers on the
assigned V100S and requires at least 12 GiB free. A V100S Ray worker was
present at preflight and was not interrupted. It exited during model loading,
so optimizer updates occurred after it released the GPU. A separate four-
request Stage-1 inference completed while the Ray worker was using the V100S,
showing that inference sharing was feasible in that observed memory state.
Sharing can still cause OOM/contention if Ray expands its allocation. The pilot
uses two held-out validation kernels and does not establish improved hardware
quality without exact synthesis receipts.
