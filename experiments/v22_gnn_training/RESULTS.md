# V4 experimental results

## Bounded matched pilot — 2026-11-01

A small family-disjoint probe completed on the NVIDIA A2. This is the first
completed V4 training result in this branch; it supersedes the earlier
“training not run” status below but does not replace the historical V3 results.

- Source commit: `815c8b3` (`research/gnn-refinement`).
- V4 tensor schema: `salt-hls-target-tensors-v4-semantic-wide-affine-experimental`.
- Frozen graph archive SHA256:
  `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
- Dataset SHA256: `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`.
- Seed-123 family split SHA256:
  `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`.
- Protocol SHA256:
  `a8e03450dd265432d1932ee8f0ea5e904545dea52128aa6d94f4101e1e212e90`.
- Cohort manifest SHA256:
  `80ec5ef03d3585d53cb1da520e50dae1706290a4c1c182ce452dd26945271822`.
- GPU: A2 UUID `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`.
- Both arms completed 20 optimizer updates. GPU phase: 3.53 seconds; no error or OOM. Peak CUDA allocation was not instrumented.
- Initial model-state hashes matched exactly; per-update sampled identities matched exactly.
- Each arm has a fixed-step probe checkpoint, not a production or validation-selected model.

### Frozen small cohort

| Partition | Kernels | Families | Device contexts | Candidates |
|---|---|---|---:|---:|
| Train | GEMM-NCubed, SpMV-ELLpack, Stencil2D, MD-KNN, Rodinia Dilate | 5 training families | 10 (ZCU104/U200 at 10 ns) | 80 |
| Validation | MachSuite Viterbi, Rodinia PathFinder baseline | `machsuite_viterbi`, `rodinia_pathfinder` | 4 (ZCU104/U200 at 10 ns) | 32 |

Each context has eight candidates. Training candidates were selected by a
deterministic greedy Hamming-distance procedure over complete directive
assignments with hash tie-breaks. Validation candidates were the first eight
configuration hashes; validation QoR did not affect cohort selection. The two
validation kernels are family-disjoint from training and from each other.
Viterbi and PathFinder measured outcomes vary within these selected pools. At
50% capacity, PathFinder/ZCU104 has only five of eight candidates measured
feasible; therefore feasibility is observable in only a small part of this
validation cohort. The full 104,147-record authenticated dataset was not used
for fitting, and no test rows were included.

The pilot intentionally used only the 10 ns clock: it compared the two devices
while keeping frequency fixed and the cohort small. It therefore did not test
clock-conditioned behavior at 3.33 or 5 ns. Those contexts exist in the
authenticated dataset, but shared complete-assignment coverage for the selected
validation kernels has not yet been checked. See
[`NEXT_STEPS_BRIEF.md`](NEXT_STEPS_BRIEF.md) for a bounded follow-up plan.

The initial candidate cohort used Sort-Radix but the preflight correctly
rejected it: its source registration includes L6 but the authenticated action
mapping has no measured column for L6. Its failed attempt and audit receipt
remain outside the final cohort; it was replaced with fully mapped Viterbi
from a different validation family. No label was fabricated.

### One-time feature preparation

The final cache at `/tmp/v22-gnn-v4-fast-probe-s123-r4/` was prepared from
RC3 and the authenticated seed-123 tables. Times were measured by the probe:

| Stage | Seconds |
|---|---:|
| Authenticated data and graph loading | 150.82 |
| Graph tensorization (28 target/budget graph tensors) | 9.82 |
| Candidate feature preparation (112 records) | 38.78 |
| Cache serialization | 0.26 |
| Total preparation including equality checks and setup | 248.84 |

All 28 graph tensors, 112 complete candidate vectors and 112 candidate-demand
descriptors passed exact tensor equality against the existing V4 APIs. Graph
and candidate tensors are reused during both arms and validation. The old
thousands-candidate repeated validation path is not used. The machine-readable
selection and identity are in
[`results/fast_probe_s123_cohort_manifest.json`](results/fast_probe_s123_cohort_manifest.json).

### Matched validation results

All values below use the same 32 measured configurations (four kernel/device
contexts), with QoR normalization fitted only on the 80 training candidates.
Latency ranking statistics are defined in all four contexts. Feasibility
counts aggregate the four contexts at both 100% and 50% budgets (64
candidate-budget outcomes). The top-k statistics cover eight
context-budget groups; each source pool contains eight candidates, so top-10
is trivially the entire measured pool and is not an informative result.

| Metric | Numerical | Hardware messages | Hardware − numerical |
|---|---:|---:|---:|
| Latency MAPE (macro) | 0.8123 | 0.8156 | +0.0034 |
| Latency MAE, ms (macro) | 19.590 | 19.618 | +0.029 |
| Resource MAE / capacity (macro) | 0.04893 | 0.04897 | +0.00004 |
| Latency Spearman (macro, 4/4 defined) | 0.1171 | 0.1026 | −0.0145 |
| Latency Kendall tau (macro) | 0.1809 | 0.1815 | +0.0007 |
| Top-1 feasible measured-pool regret | 0.3683 | 5.6788 | +5.3105 |
| Top-3 feasible measured-pool regret | 0.0000 | 0.3264 | +0.3264 |
| Feasibility accuracy (both budgets) | 0.9531 | 0.9531 | 0.0000 |
| False-feasible candidate-budget outcomes | 3 / 64 | 3 / 64 | 0 |
| No predicted-feasible contexts | 0 / 8 | 0 / 8 | 0 |
| Latency top-1 false-feasible selections | 0 / 8 | 0 / 8 | 0 |

Latency regret is relative to the best measured feasible candidate in the
selected eight-candidate pool, not the full GNΩSIS candidate pool or an
unmeasured global optimum. The absolute errors are high. Resource MAE by
physical unit `[BRAM18K, DSP, FF, LUT]` was approximately `[16.545, 5.802,
22051, 57562]` for numerical and `[16.545, 5.865, 22057, 57613]` for hardware.
Normalized resource errors were about `[2.281, 0.210, 2.720, 14.362]` and
`[2.281, 0.213, 2.721, 14.373]` percentage points, respectively. These small
validation counts make macro estimates highly sensitive to one context.

Per-context latency values (full-capacity primary scenario):

| Kernel / device | Numerical: MAPE / Spearman / regret | Hardware: MAPE / Spearman / regret |
|---|---:|---:|
| Viterbi / U200 | 0.6805 / −0.286 / 1.438 | 0.6860 / 0.262 / 0.019 |
| Viterbi / ZCU104 | 0.7815 / 0.548 / 0.000 | 0.7852 / −0.238 / 22.661 |
| PathFinder / U200 | 0.8878 / 0.503 / 0.000 | 0.8901 / 0.503 / 0.000 |
| PathFinder / ZCU104 | 0.8994 / 0.299 / 0.071 | 0.9013 / 0.419 / 0.071 |

The hardware arm helps substantially in one Viterbi/U200 context and harms
Viterbi/ZCU104; the two PathFinder top-1 regrets are unchanged. This is not a
consistent advantage. Both arms select feasible top-1 designs in all eight
context-budget groups, but the validation cohort has few infeasible candidates
and feasibility accuracy is not a strong discriminator here.

### Hardware-message ablation

With the trained hardware model held fixed, disabling hardware message edges
changed action embeddings in every validation context (relative active-action
L2 shift about 0.52; pooled-program L2 shifts 0.076–0.153). Hardware relation
rows received finite nonzero gradient norms (`6.61e-5` to `2.34e-4`); the
numerical arm's hardware-relation gradient norm was zero. Mean absolute
prediction changes `[latency ms, BRAM, DSP, FF, LUT]` were approximately
`[0.0153, 0, 0.0071, 2.002, 170.49]`.

The ablation changed latency Spearman from −0.0351 with messages disabled to
0.1026 enabled, but changed the selected latency candidate in **0 of 4**
full-capacity validation contexts. Its latency MAPE shifted only from 0.8175
to 0.8156 and measured-pool latency regret remained 5.6877. This shows that
the pathway is active and can alter ranking scores; it does not demonstrate
improved candidate selection. The ablation is not a separately trained
numerical model.

### Training trajectory and interpretation

The numerical arm's first/last update losses were 0.3625/0.5307 (minimum
0.2736); the hardware arm's were 0.3631/0.5342 (minimum 0.2716). Losses did
not decrease monotonically over these 20 randomly sampled updates. The run is
too short to claim convergence or robust learning. It does establish that both
matched arms can execute end-to-end, produce finite updates, validate on a
small measured pool, and expose a working hardware-message pathway.

**Scientific conclusion:** hardware messages changed representations and
some rank scores, but this pilot does not show consistent improvement in QoR,
feasible selection, or measured-pool regret over numerical target conditioning.
The unusually large hardware-arm Viterbi/ZCU104 regret dominates its average.
Because this is one seed, only 20 updates, two validation families and four
device/kernel contexts, it also cannot establish that hardware messages are
ineffective in a properly trained model.

## Earlier V4 smoke and preserved evidence

The prior V100S smoke remains at
`/home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-v100s-smoke-s123-r3/`. It made
five numerical-arm updates then failed at first validation because
`Counter` was not imported. No hardware arm or checkpoint selection completed.
The missing import is fixed in the probe source. The earlier full validation
attempt was not resumed and no old run files were overwritten.

The first bounded cohort attempt found that Sort-Radix L6 has no authenticated
measurement column; the second attempt corrected family-label resolution;
the third corrected the canonical resource-order reference. All audit files
were preserved in their separate `/tmp/v22-gnn-v4-fast-probe-s123*` output
directories. The final r4 run completed successfully.

## Test record

In the isolated Python 3.10.20 / PyTorch 2.6.0+cu124 / PyG 2.6.1 environment:

```text
8 fast-probe tests: passed (the added output-protection test ran separately)
48 combined fast-probe, training-contract, semantic-tensor and action-demand tests: passed before that final test was added
py_compile for fast_probe.py and train.py: passed
Git diff whitespace check: passed
```

The earlier historical V4 tests and their skips are described in
[`AUDIT.md`](AUDIT.md); they are not counted as freshly rerun here. No full
63-graph regeneration, test-split evaluation, test-label access, HLS synthesis,
Stage-1/Stage-2 training, or production checkpoint modification occurred.

## Decision

`NO_USEFUL_HELD_OUT_LEARNING_OBSERVED`.

The bounded pilot is technically successful and the hardware path is
computationally active, but the loss did not trend downward and the validation
errors and measured-pool regrets remain poor. The experiment therefore does
not show useful held-out QoR learning or a reliable hardware-message advantage.
It also does not establish that longer, properly selected training cannot
learn. The next useful step, if pursued, is a predeclared modest multi-seed
comparison with more validation families and a candidate set that provides
stronger feasibility variation—not another large all-candidate validation
loop. No such run is launched by this report.

## Historical V4 preflight (superseded current status)

Before the bounded pilot, the 8,000-update protocol had not run. An earlier
V100S smoke and an A2 availability check were blocked; those records remain
historical evidence only and are not the current experiment status. The
historical full-data audit identity and exclusions remain unchanged.

## Fixed-context learnability and six-context target probe — 2026-11-01

This follow-up tests candidate-conditioned learnability, target response, and
the incremental value of hardware-resource messages. It does not modify RC3,
the V4 tensor schema, production checkpoints, or Stage-2.

### Fixed-context learnability

The context is `machsuite-stencil2d / zcu104 / 10 ns` in the training split.
Twenty-four distinct measured assignments were chosen by deterministic
complete-assignment Hamming diversity, without QoR-based selection. One
hardware-enabled `CandidateQoRModel` was trained with regression-only
SmoothL1, AdamW `3e-4`, weight decay `1e-4`, seed 123, and no ranking loss.
This is an in-sample memorization diagnostic.

| Step | Standardized train SmoothL1 | Latency MAE (ms) | Latency MAPE | Train Spearman | Predicted-best regret |
|---:|---:|---:|---:|---:|---:|
| Constant mean baseline | 0.3363 | 0.4023 | — | — | — |
| 1 | 0.3293 | 0.3804 | — | 0.388 | 1.05% |
| 100 | 0.3199 | 0.3761 | — | 0.397 | 1.05% |
| 200 | 0.1925 | 0.2679 | 0.8013 | 0.510 | 1.05% |
| 1,000 | 0.0940 | 0.2479 | 0.6848 | 0.701 | 1.05% |

The 1,000-update run started fresh from the same seed; its first 200 updates
match the separate 200-update trajectory at recorded checkpoints. Standardized
loss fell 72% and latency MAE fell 38% relative to the constant baseline. The
predicted best was 1.05% slower than the pool optimum, not the exact winner.
The model therefore shows candidate-dependent memorization on this set, not
held-out learning. The best-candidate identity did not change from step 1,
which qualifies the top-1 result. GPU time: 3.57 s (200 updates) and 15.25 s
(1,000); peak CUDA allocation: 23,386,112 bytes. Weights remain in
`/tmp/v22-v4-fixed-learnability-s123-1000-20261101/`.

### Authenticated coverage and frozen cohort

The RC3 archive (`feb2c2d4…a1b73`), dataset (`afdf4af8…d5ab820`) and seed-123
family split (`d0886fa8…ec360c`) match the previous audit. Only the five
training kernels and two validation kernels entered this coverage analysis;
Kalman/test measurements were excluded. Every training kernel has at least 28
exact assignments shared across all six target contexts. PathFinder has 14
six-way assignments; Viterbi has none, so its pool is selected independently
by hash for each context and paired results use exact overlaps only.

| Partition | Contexts | Configurations | Selection |
|---|---:|---:|---|
| Train: 5 kernels/families | 30 | 480 | 16 exact assignments/kernel shared across six targets; Hamming-diverse, QoR-blind |
| Validation: Viterbi, PathFinder | 12 | 180 | 14 PathFinder assignments shared six ways; 16 Viterbi hashes/context; QoR-blind |

Exact overlaps, context counts, QoR variation, feasibility and source/graph
identities are in [`results/target_coverage_audit_s123.json`](results/target_coverage_audit_s123.json).
The exact selection identities are in
[`results/target_condition_probe_s123_manifest.json`](results/target_condition_probe_s123_manifest.json).

### Three-arm held-out probe

Each arm ran 100 matched updates, eight candidates/update, AdamW `3e-4`, weight
decay `1e-4`, clipping at 1, and the existing `SmoothL1 + 0.1 * ranking_loss`.
Numerical and hardware GNNs share the same 116,081 parameters, initial state
SHA256 `f4f5f98a…e2f89956`, and sample trace. The candidate-only MLP uses
candidate, analytical demand and numerical target inputs but no graph; its
338,053 parameters make it a simple control, not a parameter-matched ablation.
All checkpoints are fixed at update 100; validation did not select a model.

| Validation metric, macro over 12 contexts | Candidate + target | Numerical GNN | Hardware GNN |
|---|---:|---:|---:|
| Latency MAPE | 0.6888 | 0.6126 | 0.6066 |
| Latency MAE (ms) | 6.4723 | 6.5538 | 6.3968 |
| Resource MAE / capacity `[BRAM,DSP,FF,LUT]` | `[.02395,.00071,.04305,.21402]` | `[.02395,.00104,.04314,.21046]` | `[.02395,.00104,.04314,.21095]` |
| Latency Spearman / Kendall | .2360 / .2231 | .0791 / .0494 | .1473 / .1098 |
| Latency top-1 / top-3 measured-pool regret | 7.3810 / .2755 | .2865 / .0605 | 2.3451 / .0029 |
| Area / ADP Spearman | .2831 / −.2787 | .3308 / .3879 | .3571 / .3275 |
| Area top-1 regret (11/12 defined) | 4.1838 | 1.7682 | 1.5688 |
| ADP top-1 regret (11/12 defined) | 7.6971 | .6811 | 11.6841 |
| 100% budget accuracy / always-feasible baseline | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| 50% budget accuracy / always-feasible baseline | .816 / .816 | .816 / .816 | .816 / .816 |
| 50% false-feasible / false-infeasible candidates | 31 / 0 | 31 / 0 | 31 / 0 |
| 50% contexts selecting actually feasible design | 9/12 | 10/12 | 9/12 |

Regret is measured-pool regret, not regret against every HLS design. The
Viterbi/U200/5 ns measured area and ADP optima are exactly zero, so relative
regret is undefined there. The corrected summary sets that context to null and
uses 11/12 contexts; the raw report is retained separately and its original
epsilon-denominator values must not be averaged. All arms match an
always-feasible classifier at 50%, so feasibility accuracy is not learned
evidence here.

### Target contrasts, budget semantics and message ablation

Exact paired validation comparisons contain 91 assignment-pair observations
across 10 nonempty clock groups and 45 across 5 device groups. They overlap and
come from only two kernels. Full-budget macro sign accuracy over comparison
groups is 0.621/0.867/0.467 for clock changes (candidate-only/numerical/hardware)
and 0/0/0 for device changes. Clock paired-latency-delta MAE remains about
6.9 ms. Viterbi overlaps are sparse; PathFinder supplies most pairs. These are
descriptive, not independent statistical trials.

For the same candidate/device/clock, changing only the requested budget
changed predicted physical latency by 0.0124 ms (candidate-only), 0.1105 ms
(numerical), and 0.1417 ms (hardware) on average. Measured physical QoR labels
are held constant across budget requests. This is a model-conditioning defect
for the desired physical-QoR semantics. A future versioned experiment should
predict QoR from program/candidate/device/clock and derive feasibility against
the requested budget; no graph or V4 schema was changed here.

Hardware relation rows received finite mean gradient norm `1.04e-4`; the
numerical arm's corresponding relation-row gradient was zero. Disabling
hardware edges changed action memories (mean relative L2 shift 0.509) and the
selected latency candidate in 10/12 contexts. Macro latency Spearman changed
from −0.220 without the edges to 0.147 with them, while top-1 latency regret
worsened from 0.697 to 2.345. The path is active but its selection effect is
mixed, not a reliable benefit. This is an inference-time ablation, not a
separately trained numerical model.

The run used Python 3.10 / PyTorch 2.6.0+cu124 / PyG 2.6.1 and A2 UUID
`GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`. Data loading took 94.96 s,
feature preparation 136.59 s and all GPU training/evaluation 7.56 s; peak CUDA
memory was not instrumented. Weights and caches remain outside Git in
`/tmp/v22-v4-target-conditioned-probe-s123-20261101-r2/`. The tracked corrected
summary is [`results/target_condition_probe_s123_summary.json`](results/target_condition_probe_s123_summary.json);
raw results are in [`results/target_condition_probe_s123_raw.json`](results/target_condition_probe_s123_raw.json).

A later corrected-report rerun was refused by the safety guard because the A2
had become occupied by `VLLM::EngineCore`; no process was signalled. The
completed earlier checkpoints were preserved. The zero-optimum correction is
unit-tested and applied in the tracked summary; it does not alter model
weights. This remains one seed, 100 updates, two validation families and a
small pool. It cannot establish broad generalization, synthesis QoR, or
Stage-2 benefit.

**Conclusion:** the existing model can partially memorize one fixed context.
The 100-update cross-family models remain weak; target-conditioned device
response is not reliable; hardware messages offer no consistent selection
advantage. V4 embeddings are not ready for Stage-2. The next bounded experiment
is 300–600 updates on this same frozen three-arm cohort when the A2 is free,
with no simultaneous change to graph features, loss or selection. If
budget-induced QoR drift persists, test a separately versioned
physical-QoR/feasibility separation.

## Corrected 300-update continuation and budget separation (2026-11-01)

This section supersedes the earlier target-difference **sign** summary only;
it does not alter historical raw receipts or weights. Target differences are
now considered meaningful only when
`abs(ΔL) > max(1e-5 ms, 0.01 * max(abs(L1),abs(L2)))`. The `1e-5 ms` absolute
tolerance is one 10 ns cycle expressed in milliseconds. Near-zero pairs remain
in paired-delta MAE/RMSE but are excluded from sign accuracy. Exact assignment
hashes are required. Pair observations overlap across device/clock groups and
are descriptive, not independent samples.

The Viterbi/U200/5 ns zero area/ADP optimum comes from nine source CSV rows
whose four utilization fields are present and literally `0`. Preprocessing
preserves those values; percentage-to-capacity conversion consequently yields
zero resource counts, area proxy and ADP. The CSV contains no sub-percent raw
counts or precision metadata, so exact zero cannot be distinguished from
rounded/truncated sub-percent use. No conversion defect is evidenced. Labels
remain unchanged; relative regret at zero optimum is undefined, with absolute
measured-pool regret reported separately. See
[`results/zero_area_measurement_audit.json`](results/zero_area_measurement_audit.json).

### Matched validation: historical 100 updates vs new 300 updates

The three arms used the frozen 480-row train / 180-row validation cohort, seed
123, same candidate sampling trace, 300 updates, and the existing optimizer
and `SmoothL1 + 0.1 ranking` loss. Numerical and hardware GNNs shared initial
parameter SHA256 `f4f5f98a…e2f89956`. The new run was freshly initialized, not
resumed from step 100. All 180 validation rows are from Viterbi and PathFinder;
the Kalman test split was not loaded. Table values are macro averages over 12
program/device/clock contexts. Relative area/ADP regrets omit the one
zero-optimum context.

| Arm/checkpoint | Latency MAE ms | MAPE | Latency Spearman | Top-1 regret | Top-3 regret | Area Spearman | ADP Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate + target, 100 | 6.4723 | 0.6888 | 0.2360 | 7.3810 | 0.2755 | 0.2831 | −0.2787 |
| Numerical GNN, 100 | 6.5538 | 0.6126 | 0.0791 | 0.2865 | 0.0605 | 0.3308 | 0.3879 |
| Hardware GNN, 100 | 6.3968 | 0.6066 | 0.1473 | 2.3451 | 0.0029 | 0.3571 | 0.3275 |
| Candidate + target, 300 | 17.9620 | 6.9100 | 0.0996 | 5.5892 | 0.2028 | −0.1157 | −0.1616 |
| Numerical GNN, 300 | 5.9332 | 1.0676 | 0.3131 | 2.7901 | 0.3575 | 0.5101 | 0.2171 |
| Hardware GNN, 300 | 5.8389 | 1.0013 | 0.3916 | 2.3451 | 0.1775 | 0.5372 | 0.2880 |

More updates improved mean latency Spearman for both GNNs, but did not
consistently improve all metrics. Hardware top-1 regret remained unchanged;
its MAPE rose. Predicted latency ranges remain compressed against measured
ranges, especially on Viterbi. Candidate-only's 300-update result was worse
than its 100-update result and is reported rather than omitted. Fixed
Stencil2D training diagnostics show the regression SmoothL1 at step 300 was
0.423 (numerical) and 0.445 (hardware), versus 0.387 for both near
initialization; training does not show a clear fixed-context fit improvement.
The 300-update training/evaluation took 20.66 GPU seconds; peak allocation was
121.6 MB. Data load was 94.28 CPU seconds and feature preparation 144.47 CPU
seconds.

### Corrected target contrasts at full budget

There are 91 exact clock-pair observations across 10 overlapping groups and
two kernels. All 91 exceed the meaningful-difference threshold. Device
comparisons have 45 observations, but only 3 are meaningful (all Viterbi); 42
PathFinder device pairs are negligible. Per-kernel macro results:

| Arm, 300 updates | Clock Δ MAE ms | Clock sign accuracy | Device Δ MAE ms | Meaningful device pairs | Device sign accuracy |
|---|---:|---:|---:|---:|---:|
| Candidate + target | 7.795 | 0.042 | 8.865 | 3/45 | 0.000 |
| Numerical GNN | 7.292 | 0.958 | 8.313 | 3/45 | 0.000 |
| Hardware GNN | 7.400 | 0.042 | 8.319 | 3/45 | 0.000 |

The numerical GNN often predicts the clock-change direction correctly, but
its delta magnitude error remains large. Hardware GNN's clock sign result is
poor at 300 updates. There is insufficient meaningful device-pair coverage to
claim device conditioning. Corrected historical 100-update clock sign
accuracy is 0.083 / 0.917 / 0.042 (candidate/numerical/hardware); device sign
accuracy is zero for the three meaningful pairs in all arms. These replace
the old uncorrected sign rates.

### Feasibility and physical-QoR/budget separation

At 50% capacity, 149/180 rows are feasible and 31 are infeasible. Every arm
predicts all rows feasible: accuracy 0.8278, exactly the always-feasible
baseline, with 31 false-feasible and zero false-infeasible. At exploratory
10%, 83/180 are feasible; all arms still predict every row feasible (accuracy
0.4611). Feasibility is not learned.

For an identical program, candidate, device and clock, changing only the
requested budget changes original-V4 mean latency predictions by 0.4121 ms
(numerical) and 0.6291 ms (hardware) for 100% vs 50%, and 0.7577/1.0978 ms for
100% vs 10%. The measurements themselves do not change with this external
feasibility request. This confirms budget/QoR conflation.

A separately versioned 300-update ablation feeds full-capacity graph context
to the physical-QoR model and uses budgets only for feasibility/ranking. The
resulting budget-only latency drift is below `5e-7 ms`, numerical roundoff
from batched calls. This enforces the physical invariant but does not reliably
improve selection:

| Formulation | Arm | 100%-50% latency drift | Latency MAE ms | Spearman | Top-1 regret | 50% accuracy |
|---|---|---:|---:|---:|---:|---:|
| Original V4 | Numerical | 0.4121 | 5.9332 | 0.3131 | 2.7901 | 0.8278 |
| Original V4 | Hardware | 0.6291 | 5.8389 | 0.3916 | 2.3451 | 0.8278 |
| Budget-independent | Numerical | <0.0000005 | 6.0124 | 0.2860 | 0.4514 | 0.8278 |
| Budget-independent | Hardware | <0.0000005 | 5.9024 | 0.2426 | 6.4573 | 0.8278 |

The hardware-edge inference ablation for original V4 changes mean latency
Spearman from 0.3916 (enabled) to 0.3486 (disabled), and top-1 regret from
2.3451 to 2.7901. In the separated model, disabling edges changes Spearman
from 0.2426 to 0.2568 and top-1 regret from 6.4573 to 2.6183. The relation
path is active (mean relation-gradient norm `2.57e-4`, max `1.32e-3`) but not
consistently useful. The ablation is inference-only, not a separately trained
numerical model.

The completed original-V4 attempt-03 source commit is `dd31514`; its results,
manifest and checkpoints are stored outside Git at
`/home/elvouvali/mailohls-gnn-runs/v4-corrected-300-s123-20261101/attempt-03`.
Results SHA256:
`fec061f0dde2a72db6e99ca8d2eded3a3c7be1e908e268763c2ee57f6229107e`; manifest
SHA256: `d52e8fd7f161e42205ce0c66d78c44f3e40dbbf8b3b6d6c75cffb5680c13a225`.
The budget-independent attempt-04 is in the adjacent
`attempt-04-budget-independent` directory; result SHA256
`338b978fca62482c951be145e1811dd7c829c1b014eff350dce97b8e7c936c90`, manifest
SHA256 `b1dda912241fc10e5db2b2f449bc13d053fec7d2fbe23cf9e69d974649dd3209`.
Both used the A2 UUID `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`; each CPU
preparation was under four minutes and GPU work under 21 seconds. No test data
were used.

A conditional 600-update launch was initially refused before feature
preparation or GPU training because the research worktree had pending
documentation edits; the clean-source guard behaved as intended. The attempt
directory `/home/elvouvali/mailohls-gnn-runs/v4-corrected-300-s123-20261101/attempt-05-600-updates`
contains its data-audit receipt and must be preserved. It has zero optimizer
updates and is not a training result. A fresh retry is only valid after the
documentation is committed, in a new output directory, with a new recorded
source commit.

Attempt-02 trained and saved weights but failed before producing an
authenticated results report. Its model files differ from attempt-03 because
CUDA execution was not bitwise reproducible; only attempt-03 is treated as the
complete original-V4 run. No metrics are inferred from attempt-02.

**Conclusion:** there is partial candidate-ranking signal, and hardware
messages sometimes affect predictions, but reliable QoR calibration,
feasibility and device conditioning are not demonstrated. The budget split
fixes an input-contract inconsistency but does not prove better learning. V4
remains unsuitable for Stage-2 integration. The next single experiment should
be a clean-source 600-update convergence check, justified by falling aggregate
training loss and improved 300-step latency ranking, with the original cohort,
seed, loss, optimizer and schedule unchanged. The first launch was blocked
before training by the clean-source guard. Do not add graph depth or relations.
Paired target-difference supervision is deferred until this convergence check
and fixed-context fit are understood.
