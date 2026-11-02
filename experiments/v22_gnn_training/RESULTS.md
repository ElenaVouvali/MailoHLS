# V4 experimental results

**Latest result:** the 2026-11-01 three-arm directive-preference probe completed
with 300 updates per arm on the A2. Hardware messages did not improve feasible
ADP selection over compiler-only memory; FiLM's small preference-accuracy
gain came with worse feasible selection/regret. All arms missed the 21
meaningful validation budget winner changes, while validation had no meaningful
device/clock winner changes. Classification:
`TARGET_PREFERENCE_COVERAGE_INSUFFICIENT`. The latest full report is the final
section below. Historical V4 QoR experiments are preserved unchanged.

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

### Fresh 600-update convergence check

After committing the documentation so the clean-source guard could pass, a
fresh 600-update run was executed from pinned initialization (not resumed),
same seed/cohort/optimizer/loss, on the same A2. Data load took 94.04 s,
feature preparation 143.55 s, and all three arms' GPU training/evaluation
44.17 s. Maximum allocated GPU memory was 121.6 MB. It used source commit
`2e2617f00f53a5df3b2757b0477c0c46ad70e2ea`, seed 123, 480 training and 180
validation measurements, and no test data. Result SHA256
`0605618f86a542f86e0d1f6fd176841e3e7cc7c271d951ffabeccf68b0fa0bf8`, manifest
SHA256 `16148104959661fde4f1bf2835bfb2fab87df513ef31b61cf97cbd1afda5d89b`.
The sampling trace SHA256 is `58a91c753d683912083bcb852208d36a18011f57f48ff6baa11c80c8175736d2`;
initial graph-arm state SHA256 is the same `f4f5f98a…e2f89956` used by the
matched numerical/hardware arms.

| Arm | Updates | Train loss first → last | Fixed-context latency MAE, step 0 → 600 | Fixed-context rank, step 0 → 600 | Validation latency MAE ms | MAPE | Spearman | Top-1 regret | Top-3 regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate + target | 600 | 0.359 → 0.262 | 1.299 → 0.167 | .693 → .839 | 13.367 | 4.710 | .189 | 7.372 | .035 |
| Numerical GNN | 600 | 0.353 → 0.296 | 1.146 → 0.344 | .360 → .722 | 5.931 | .497 | .161 | 2.649 | .0069 |
| Hardware GNN | 600 | 0.354 → 0.298 | 1.160 → 0.341 | .445 → .513 | 6.067 | .639 | .243 | 2.618 | .0039 |

The fixed-context results show that both graph models can fit this selected
training context by step 600, unlike the mixed 300-step intermediate. This is
training-set learnability only. Held-out results did not improve uniformly:
hardware latency Spearman fell from .392 at 300 to .243 at 600, while top-1
regret changed from 2.345 to 2.618; numerical Spearman fell from .313 to .161,
while MAE stayed about 5.93 ms. Candidate-only substantially deteriorated.
The extra updates therefore do not establish better generalization.

At full budget, 600-step clock sign accuracy / paired-delta MAE is .958 / 7.32
ms (numerical) and .595 / 7.37 ms (hardware). Device sign accuracy is 1.0 / 0.0
respectively, but these scores use only three meaningful Viterbi pairs; 42
PathFinder pairs remain negligible. Large delta errors persist. At 50% budget
all models still have TP=149, FP=31, FN=TN=0 (accuracy .8278, same as the
always-feasible baseline). Physical latency drift from changing only budget
remains .184 ms numerical and .300 ms hardware, so the original formulation
still violates the physical-QoR/budget invariant.

The 600 run satisfies the fixed-context learning gate but not the held-out
reliability gate. It does not justify new graph semantics or deeper message
passing. Paired target-difference supervision is now a reasonable *next
single-variable experiment* for the clock axis: the authenticated training
cohort has 16 exact assignment identities per each of five kernels across six
device/clock contexts, so potential supervision is 480 clock pairs and 240
device pairs before filtering. However, the validation device axis still has
only three meaningful pairs, so only clock differences can be evaluated with
any descriptive coverage; device-generalization claims must remain open.
Predeclare pair aggregation/normalization and coefficient before training.

## Directive-preference structural-memory probe (2026-11-01)

This is the latest experiment. It asks whether candidate-independent
compiler/hardware memory improves selection of feasible measured directive
assignments, rather than whether a GNN can regress absolute QoR. It preserves
the V4 RC3 graph/tensor contracts and all earlier result sections.

### Authenticated cohort and preference construction

| Property | Training | Validation |
|---|---:|---:|
| Kernels / families | 14 / 12 | 3 / 2 |
| Authenticated measurement rows | 1,344 | 288 |
| Distinct complete assignment hashes in partition | 520 | 94 |
| Preference contexts (program × target × budget × objective) | 1,260 | 270 |
| Non-tie preference pairs | 124,399 | 24,061 |
| Feasible-over-infeasible pairs | 18,378 | 4,881 |
| Feasible objective pairs | 106,021 | 19,180 |
| Near ties excluded | 17,000 | 4,232 |
| Both-infeasible pairs excluded | 9,801 | 4,107 |

Each measured row comes from the authenticated seed-123 split and a complete
source-legal assignment. Selected contexts cover ZCU104/U200 at 3.33/5/10 ns,
with 16 measured assignments where available. The frozen roster uses graph,
action and target-support coverage, not validation QoR. Validation kernels are
MachSuite Viterbi and two Rodinia PathFinder implementations: two families,
not three. The budget levels (100/75/50/25/10%) create correlated views of the
same physical measurements, not additional syntheses. Kalman/test data were
not loaded.

Within one fixed program/target/budget/objective pool, a feasible design is
preferred over an infeasible one. Pairs where both designs are infeasible are
omitted. Among feasible designs, an objective gap greater than 5% defines a
strict pair; near ties are omitted. ADP is primary; the existing area and
latency objectives are supplementary. The known Viterbi/U200/5 ns zero-area
guard remained active; no selected pair triggered its ambiguous-zero
exclusion, and source measurements were unchanged.

### Target-dependent preference coverage

Exact assignment intersections show which axes have meaningful preference
changes in validation. A meaningful change requires a different best measured
configuration with a gap greater than 5%.

| Axis | Meaningful winner changes | Meaningful invariants | Winner changes that are near ties | Unsupported empty pools |
|---|---:|---:|---:|---:|
| Device | 0 | 12 | 42 | 6 |
| Clock | 0 | 55 | 8 | 3 |
| Budget | 21 | 173 | 51 | 24 |
| Objective | 31 | 18 | 158 | 6 |

There is no validation evidence for device- or clock-driven winner changes.
Budget and objective comparisons do have meaningful reversals and can test
whether the decision head responds correctly. These comparisons overlap in
assignment pools and are not independent samples.

### Matched training protocol

| Arm | Memory path | Trainable parameters |
|---|---|---:|
| Compiler-only | V4 compiler messages; hardware-node edges removed | 107,565 |
| Target-FiLM | Compiler memory plus zero-gated authenticated profile/clock modulation | 171,694 |
| Hardware messages | Existing hardware/resource nodes and typed edges | 107,565 |

All arms used seed 123, the same training groups, candidate pools, target,
budget and objective contexts, AdamW (`lr=3e-4`, `weight_decay=1e-4`), and 300
updates. Pair classes were balanced in minibatches. Shared parameter
initialization hashes and sampled-pair trace hashes matched across arms.
Checkpoint selection used validation-only ADP preference accuracy,
macro-averaged across the two validation families; all arms selected step 100.
No parameter-count-matched FiLM control was run.

The training-only fixed-context diagnostic used 16 GEMM-NCubed/U200/10 ns
assignments and 115 strict pairs. In 100 updates, pair loss changed from
0.69303 to 0.68623 while accuracy fell from 0.6957 to 0.6522. It does not
establish fixed-context preference learning. Full-training sampled losses
declined below initial values, but validation selection scores were highest at
step 100 and lower at steps 200/300. The selected checkpoint was step 100 for
all arms, so additional updates did not improve the predeclared validation
criterion.

### Held-out preference and measured-pool results

All aggregate metrics use the same 270 validation decision contexts. Pair
accuracy is macro-averaged within family then across the two families. Regret
is relative to the best measured feasible assignment in the selected pool,
not a global optimum. Top-1 regret excludes infeasible selections and contexts
with no eligible measured feasible candidate; those outcomes are separate.

| Primary ADP metric | Compiler-only | Target-FiLM | Hardware messages |
|---|---:|---:|---:|
| ADP pair accuracy, family macro | 0.7222 | 0.7246 | 0.7126 |
| ADP feasible top-1 success | 85 / 88 (96.59%) | 83 / 88 (94.32%) | 85 / 88 (96.59%) |
| ADP false-feasible selections | 3 / 88 | 5 / 88 | 3 / 88 |
| ADP empty feasible pools | 2 / 90 | 2 / 90 | 2 / 90 |
| ADP top-1 measured-pool relative regret | 0.3412 | 0.5319 | 0.4439 |
| ADP top-3 measured-pool relative regret | 0.1764 | 0.2045 | 0.1874 |

All-objective pair accuracy was 0.6714 compiler-only, 0.6824 FiLM and 0.6663
hardware messages. Across all objectives, selected actual feasibility was
255/264, 249/264 and 253/264. Six of 270 contexts had no measured feasible
candidate; the remaining 9/15/11 infeasible selections in supported contexts
are false-feasible decisions.

| Validation family | Model | ADP pair accuracy | Feasible top-1 | Top-1 regret | Top-3 regret |
|---|---|---:|---:|---:|---:|
| MachSuite Viterbi | Compiler-only | 0.6338 | 27 / 28 | 0.0659 | 0.0236 |
| MachSuite Viterbi | Target-FiLM | 0.6419 | 27 / 28 | 0.0659 | 0.0047 |
| MachSuite Viterbi | Hardware messages | 0.6081 | 27 / 28 | 0.0659 | 0.0236 |
| Rodinia PathFinder | Compiler-only | 0.8106 | 58 / 60 | 0.4694 | 0.2451 |
| Rodinia PathFinder | Target-FiLM | 0.8074 | 56 / 60 | 0.7566 | 0.2944 |
| Rodinia PathFinder | Hardware messages | 0.8170 | 58 / 60 | 0.6199 | 0.2623 |

Hardware messages improve PathFinder ADP pair accuracy by about 0.0065 over
compiler-only, but worsen its top-1 regret by about 0.1505. They reduce
Viterbi pair accuracy by about 0.0257. This is not a consistent selection
improvement. FiLM's ADP pair-accuracy gain over compiler-only is 0.0024, with
worse feasible top-1 and regret.

### Target-conditioned preference response

Among the 21 meaningful budget-driven winner changes, all arms correctly
identified **zero**. Compiler-only remained invariant-but-wrong in 20 and
changed to the wrong winner in one; FiLM did so in 15 and 6; hardware messages
in 19 and 2. On meaningful budget invariants, the correct complete winner was
selected in 52/173 contexts by compiler-only, 52/173 by FiLM and 22/173 by
hardware messages. For objectives, all arms missed the 31 meaningful winner
changes and correctly selected only 2/18 meaningful invariants.

Validation contained no meaningful device or clock winner changes, so these
axes cannot test learned changes. Of 12 meaningful device invariants, each arm
selected the complete winner in 3. For clock, compiler-only and FiLM got 16/55
invariants right and hardware messages got 4/55. Near-tie changes were not
counted as correct target response.

### Memory and hardware-path ablations

The same selected weights were evaluated with correct, zeroed, shuffled,
active-slot-global-mean and target-mismatched memory. Entries below are
primary ADP pair accuracy / top-1 regret.

| Model | Correct | Zero | Shuffled | Global mean | Target-mismatched |
|---|---:|---:|---:|---:|---:|
| Compiler-only | 0.7222 / 0.3412 | 0.6939 / 0.3512 | 0.7254 / 0.3412 | 0.7230 / 0.3412 | 0.7222 / 0.3412 |
| Target-FiLM | 0.7246 / 0.5319 | 0.6745 / 0.5481 | 0.7327 / 0.5077 | 0.7225 / 0.4145 | 0.7246 / 0.5319 |
| Hardware messages | 0.7126 / 0.4439 | 0.6777 / 0.5562 | 0.7132 / 0.4439 | 0.7103 / 0.4439 | 0.7126 / 0.4439 |

Zeroing memory changed top-1 selections in 31/270 compiler-only, 116/270
FiLM and 107/270 hardware-message contexts, showing sensitivity to nonzero
memory. Shuffling slots changed only 5, 8 and 5 top-1 choices respectively;
its pair accuracy was slightly higher than correct memory in each arm. The
measured pool does not show a robust selection benefit from correct
per-action alignment. FiLM target-mismatching changed no top-1 or complete
candidate order; mean score change was `8.5e-8`. Hardware target-mismatching
changed complete order in 13/270 contexts but no top-1, and changed one top-3
membership; mean score change was `1.19e-4`. This is weak target-memory
sensitivity, not useful target conditioning. The FiLM gate remained near zero
(`0.0011` at its selected step 100); hardware relation embeddings had finite
nonzero sampled gradients (`1.09e-4` at step 1, `2.43e-4` at step 300).
Neither observation proves decision utility.

### Runtime, provenance and artifacts

The completed run used source commit
`fdd66e63db1ec40cb6ed11e23590ee93541c07b2`, A2 UUID
`GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`, Python 3.10.20, PyTorch
2.6.0+cu124 and PyG 2.6.1. CPU authentication/cohort/feature preparation
took 906.91 s; candidate-demand construction accounted for 672.13 s,
graph tensorization 75.17 s, graph enrichment 40.63 s, equality checks 84.68 s
and cache serialization 2.29 s. GPU training, validation and memory ablations
took 73.50 s; peak CUDA allocation was 242,267,136 bytes (231.1 MiB). Each
full 270-context validation took about 1.4 s because features were reused.

| Artifact | Identity |
|---|---|
| Dataset SHA256 | `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820` |
| Seed-123 split SHA256 | `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c` |
| RC3 graph release SHA256 | `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73` |
| Cohort manifest identity | `2ac3074d770c98169a52d0d27935d1e5eec67fff4db57b0ff010111d8f5b297d` |
| Protocol SHA256 | `93cba7f8a7a1057c7b149776d7f5fb25de396a7535f643ceb7d8d2b29c8940e5` |
| Compact results SHA256 | `2b559fc95d420e57ae389874322e471fb8e70edf297a4f567e70c1d7393497ba` |
| Raw results SHA256 (outside Git) | `11a9755dd9263d0dcdf7a6b5ed32886b253e5718f47c489276ed9b00ab52fb6a` |

Selected checkpoints are all from step 100:

| Checkpoint | SHA256 | Durable path |
|---|---|---|
| `compiler_selected.pt` | `4078026800d035f7f5b16d15cfbaee91e9717b60bc76fa7b2ee3ca7170b098f3` | `/home/elvouvali/mailohls-gnn-runs/directive-preference-v1-20261101-fdd66e6-attempt01/compiler_selected.pt` |
| `target_film_selected.pt` | `f29ad1ea30907d265f97ef72b1833683816d424b811aaac6645e35ed82f5b0aa` | `/home/elvouvali/mailohls-gnn-runs/directive-preference-v1-20261101-fdd66e6-attempt01/target_film_selected.pt` |
| `hardware_messages_selected.pt` | `c67f66e1ac7fa192ac81d8e68f6a1b001b7c5f3580c0d2862bccd3678cf12173` | `/home/elvouvali/mailohls-gnn-runs/directive-preference-v1-20261101-fdd66e6-attempt01/hardware_messages_selected.pt` |

The compact report is
[`results/directive_preference_v1/compact_results.json`](results/directive_preference_v1/compact_results.json).
The full raw report, cache and model weights remain outside Git. The exact run
exited successfully (`0`). Earlier attempts remain preserved: one reached the
diagnostic then failed Tensor-to-JSON serialization; one exposed an unset
`-inf` progress score after one compiler-arm update; a launch-only attempt was
refused because its PID file made the output directory nonempty. None
overwrote the completed experiment.

### Conclusion and next experiment

**Classification: `TARGET_PREFERENCE_COVERAGE_INSUFFICIENT`.** The held-out
data contain 21 meaningful budget reversals, which all arms missed, but no
meaningful device or clock winner reversals. Target-aware memory variants did
not outperform compiler-only memory in feasible ADP selection; hardware
messages had worse top-1 regret, while FiLM's small pair-accuracy gain came
with worse feasible selection. Validation has only two families and one seed.
The scorer shows some pairwise signal, but the fixed-context diagnostic did
not improve accuracy and selection still incurs substantial regret.

No larger FPGA graph, deeper GNN, Stage-2 integration or refinement scorer is
justified. The next single useful study is an authenticated, family-disjoint
validation expansion, selected without QoR, that contains meaningful matched
device/clock preference changes. Keep the encoder frozen for that coverage
audit; do not train another variant until the evaluation can measure the
claimed target effect. A head-only candidate/target control is also needed
before claiming compiler structure adds value. V4 is not ready for Stage-2
cross-attention integration.

## V4b full-cohort action-aligned preference study (2026-11-02)

### Status

**Classification: `EXPERIMENT_INCOMPLETE_RESOURCE_LIMIT`.** The authenticated
full-cohort eligibility, exact target-pair, budget-winner, candidate-pool and
selected-pool preference-pair audits completed on CPU. The sandbox cannot
access CUDA devices directly, but direct host inspection found the V100S idle
and the A2 occupied by a separate Ray worker. A V100S one-update smoke was
started; it spent 31:25 elapsed (34:17 CPU) repeatedly preparing candidate
features and had not reached GPU execution. The identified V4b process (PID
2250277) received SIGINT at the 30-minute preparation cap and exited cleanly.
No checkpoint or optimizer update was produced. The previous V4 numbers above
remain historical and are not V4b results.

### Eligibility and source-action exclusions

| Property | Training | Validation | Test reservation |
|---|---:|---:|---:|
| Original split kernels | 50 | 4 | 1 |
| Strict complete-assignment eligible kernels | 37 | 3 | 0 |
| Eligible families | 17 | 2 | 0 |
| Eligible measured rows | 88,466 | 8,976 | Not loaded for V4b |

The full authenticated inventory remains 55 kernels and 150,026 aggregated
design points. Fourteen of the 54 non-Kalman kernels are excluded because at
least one source-registered action has no measured APL/CSV directive field.
No existing alternate authenticated mapping or proof of a fixed applied
directive value was found. Those states remain unknown; no disabled defaults
were inserted. The excluded IDs are recorded individually in the eligibility
receipt (including Sort-Radix L6). The eligible row total is 97,442.

### Full exact-assignment target and budget coverage

| Comparison | Paired kernel/target contexts | Exact assignment-pair count* | Meaningful ADP winner reversals at full budget | Meaningful reversals at any budget | Validation reversals |
|---|---:|---:|---:|---:|---:|
| ZCU104 ↔ U200, same clock | 104/120 | 3,605 | 2 | 6 contexts across 3 families | 0 |
| Clock pairs, same device | 209/240 | 9,733 | 0 | 1 context in 1 family | 0 |

*Assignment-pair counts overlap across device/clock comparisons and are not
independent observations. Paired contexts span 19 families overall; the
validation partition spans two families. All 5% winner-margin classifications
and 1%/5%/10% physical-delta sensitivity counts are in the machine-readable
receipt.

Across the five requested resource budgets there were 98 raw winner changes
in 2,400 budget-pair rows, but only eight rows met the 5% meaningful-margin
rule. Those eight are repeated budget comparisons from two unique
kernel/device/clock contexts in two families. The exact audit therefore shows
that the available data contain some target-dependent preference changes, but
they are sparse and clustered. The held-out validation families contain no
meaningful device, clock, or budget winner reversal in the full exact-pool
audit. A learned target-adaptive preference claim cannot be evaluated
reliably on this split.

The primary 16-candidate manifest was selected without looking at QoR: support
across target contexts first, followed by deterministic complete-directive
Hamming diversity and hash tie-breaks. It covers 240 physical contexts (222
training and 18 validation), selecting 3,376 training and 288 validation
candidate observations. Pools range from 7 to 16 candidates; no configurations
were fabricated. The full exact-pool coverage audit is separate from these
selected training/evaluation pools.

The selected-pool ADP pair audit creates pairs only within one program/device/
clock/budget context. Feasible candidates outrank infeasible candidates;
both-infeasible pairs and objective near-ties are omitted. The ambiguous
Viterbi/U200/5 ns all-zero resource rows are excluded from ADP comparisons
where they could determine the ordering, without modifying source measurements.
Physical QoR is reused unchanged for all budget views.

### V4b model controls and execution status

The isolated implementation contains a budget-invariant physical-QoR head
with feasibility calculated afterward, a candidate/target control with no
learned structural-memory encoder, compiler-only and hardware-message V4
encoders, zero-gated target-FiLM modulation of compiler memory, and a
compiler-only action-difference comparator. The candidate/target control
shares the same authenticated analytical candidate-demand descriptors and
action masks; “no graph” means no learned graph-memory encoder, not that the
analytical demand descriptors are graph-free. The pair head is antisymmetric
and returns zero for identical candidates. The shared hybrid objective is
physical-QoR SmoothL1 plus a predeclared 0.1 pairwise logistic loss. Parameter
counts are 126,930 (candidate/target control), 127,538 (compiler), 191,667
(target-FiLM), 127,538 (hardware), and 152,319 (compiler plus difference
comparator). FiLM and difference arms have extra capacity; no parameter-
matched controls for those added modules were run.

The first smoke exposed repeated authenticated overlay construction in the
existing `PreparedData.candidate_inputs()` call: one overlay was rebuilt for
each candidate. V4b first cached one overlay per kernel, but a second attempt
still spent 30:11 elapsed (33:20 CPU) in candidate preparation before GPU
work. Both attempts were gracefully interrupted at the preflight cap and
created no checkpoints. Inspection then found that candidate-demand analysis
also reran program validation, static-action extraction, relation queries and
loop-semantics queries for each assignment. The current V4b cache reuses one
overlay per kernel, analysis results per authenticated graph identity, and
candidate vectors/demands per `(kernel, complete-assignment hash)` across
physical targets. These are memoized outputs from the existing APIs; no
features or labels changed. The tensor API's candidate-demand vector contains
program/directive features; target metadata is discarded. One authentic
assignment matched exactly across U200/10 ns and U200/3.33 ns, and two
additional candidate vectors/demands matched exactly between cached and
uncached calls. In a one-kernel CPU microbenchmark, eight demand vectors took
0.520 s uncached and 0.071 s with static analysis/query caching (7.32x); this
small-kernel timing is not a full-cohort estimate. The optimized full-cohort
preparation has not yet been timed.

These are implementation/test findings, not selection results. Neither
compiler graph memory nor hardware messages have yet been trained on the full
cohort under V4b. The A2 availability gate stopped before training, so there
are no checkpoints, training curves, validation preference accuracy, feasible
selection rates, measured-pool regrets, or memory-alignment ablations to
report. The next gate is to expose an independently available A2 and run the
frozen five-arm command in `v4b/README.md`; do not use the reserved V100S.

| V4b arm | Held-out ADP pair accuracy | Feasible top-1 success | ADP top-1 regret | Feasible top-3 success/regret | Status |
|---|---:|---:|---:|---:|---|
| Candidate/target control | Not run | Not run | Not run | Not run | No checkpoint |
| Compiler V4 | Not run | Not run | Not run | Not run | No checkpoint |
| Target-FiLM V4 | Not run | Not run | Not run | Not run | No checkpoint |
| Hardware-message V4 | Not run | Not run | Not run | Not run | No checkpoint |
| Compiler action-difference | Not run | Not run | Not run | Not run | No checkpoint |

The full CPU audit completed in 78.35 seconds in the pinned environment; that
is data authentication/coverage time. Two selected-pool preparation attempts
exceeded 30 minutes before reaching GPU work. The current graph-analysis cache
is a subsequent code fix and must be timed in a fresh smoke before any
optimizer run. GPU training time, model validation time, and peak GPU
allocation remain unmeasured.

### Reproducibility receipts

The CPU audit receipts are tracked under
[`v4b/receipts/`](v4b/receipts/): eligibility, the 16-candidate cohort
manifest, the selected-pool preference-pair audit, and exact target/budget
coverage. They authenticate dataset SHA256
`afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`, split
SHA256 `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`,
and RC3 graph archive SHA256
`feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`. The
eligibility receipt identifies source commit `729c10ba06b9e7d2110b3c593376a877e53bb2ad`
and hashes the audit, preference builder, model implementation, and runner.
Receipt-file SHA256 values are recorded in `AUDIT.md`. No HLS
synthesis, Kalman/test-label evaluation, Stage-2/3 training, or production
artifact change occurred.
