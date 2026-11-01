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
