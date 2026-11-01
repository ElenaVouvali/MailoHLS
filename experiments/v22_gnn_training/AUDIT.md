# V4 implementation and data audit

## Scope and provenance

This audit distinguishes the completed bounded V4 GPU probe from historical V3
receipts and from the abandoned full-run smoke. The current branch is
`research/gnn-refinement`, with probe code at commit `815c8b3`. The V4 result
summary and fixed cohort manifest are checked in alongside this audit; model
weights and feature caches remain outside Git.

The frozen graph is MLIR schema `salt-hls-mlir-structural-v2.2`, release RC3.
Its archive SHA256 was independently checked as
`feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`. The
fresh extracted read-only release is `/tmp/salt-v22-training-release`. The
CPU audit output is `/tmp/v22-gnn-v4-verification-20261101/data-audit.json`;
the audit stdout receipt is `audit.stdout.json` in the same directory.

## Fresh V4 data audit

Command, from the repository worktree:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.audit \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-gnn-v4-verification-20261101/data-audit.json
```

Result: `pass_with_explicit_exclusions`.

| Check | V4 run result |
|---|---:|
| Frozen graph snapshots / GNΩSIS kernels | 63 / 55 |
| Reconstructed distinct GNΩSIS rows | 150,026 |
| Source-domain-validated assignments | 149,227 |
| Eligible measured records | 104,147 |
| Split eligible records | 88,466 train / 8,976 validation / 6,705 test |
| Source/domain exclusions | 45,080 missing APL measurement fields + 799 invalid full tuples |
| Source-measured unresolved-action rows retained with masks | 8,258 |
| External/unapproved target rows admitted | 0 |

Identity hashes reproduced:

| Identity | SHA256 |
|---|---|
| Selected dataset | `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820` |
| Selected manifest | `7a9540e3412621849db2e83d8c0c4a4ba0ac820e87685e29f9071f0981ecc01b` |
| Preprocessing manifest | `b9c44ab46ff850cc8ebd5b7f471a680ca0c3ed8796e9fdd9a84281bb2a54a5b7` |
| Seed-123 family split | `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c` |
| RC3 graph release manifest | `1c75bf22bd1bc6263e68118a8438ccf3bbf1eb8e7d675149f010d6f2a0a1a6e3` |

The clean worktree initially lacked ignored preprocessed CSVs. The audit was
re-run only after restoring tables whose SHA256 matched the committed manifest;
that first environmental failure is not counted as a data-contract pass or a
semantic test failure. No source, measurements, split or graph was changed.

## Matched-arm and leakage contract

V4 schema: `salt-hls-target-tensors-v4-semantic-wide-affine-experimental`.
Frozen graph schema: `salt-hls-mlir-structural-v2.2`. The V4 change widens
affine-map semantic support; the frozen RC3 graph and feature widths remain
unchanged.

The numerical arm retains hardware nodes for stable node indices but removes
all directed message edges with a hardware endpoint. `SmallGraphEncoder`
encodes nodes, but `memories()` pools only compiler-node categories. Therefore
disconnected hardware node values cannot enter the numerical arm's pooled
program representation or action embeddings through message passing. Both
arms still receive the same explicit numerical target context in the
candidate head. Hardware nodes can affect action embeddings in the hardware
arm only through typed forward/reverse hardware relations. Candidate directive
values and candidate-demand descriptors enter only the candidate-conditioned
head; measured QoR, winners and split membership are not static graph
features. Normalization is fitted on training rows.

A synthetic CPU gradient-path regression confirms that the typed hardware
relation embeddings receive finite nonzero gradients when those messages are
present, and exactly zero gradients for those relation rows after the
numerical-arm edge filter. This verifies the software path only; it is not a
V4 GPU update or a learning result.

New audit hooks record initial state hashes and per-update context,
configuration hashes, objective and budget fraction. At run time the hardware
arm refuses to proceed if initialization or sampled identities differ from
the numerical arm. Hashing the initial state establishes initialized
parameter equality; it does not guarantee bitwise-equal CUDA training updates.
GPU nondeterminism must be reported if present.

## Current CPU tests

Interpreter: Python 3.10.20, PyTorch 2.6.0+cu124, PyG 2.6.1, SciPy 1.15.3.
This isolated environment does not expose CUDA to the sandbox. In this task,
`test_v22_gnn_training_contracts.py` plus `test_v22_gnn_resume.py` completed
with 25 passed; `test_target_graph_v22.py` completed with 58 passed; and
`test_mlir_graph_v2.py`, `test_structural_audit.py`, and
`test_directive_domains.py` completed with 86 passed and 15 skipped. The
semantic-feature and action-demand suites completed as part of an earlier
focused run (35 passed across all four files); the two semantic-feature and
action-demand modules were unchanged afterward. Updated training-contract and
resume coverage was rerun in the 25-test command. The 15 skips are due to unavailable
compiled R1.2 fixtures, external GAN/graph archives, a historical V2
comparison, and optional saved local research artifacts; they are not counted
as successful semantic validation. The A2 profile, V4 GPU gradient preflight
and graceful GPU interruption/resume test have not run in this task. The
bounded probe did run actual GPU updates and recorded finite hardware-relation
gradients; it did not substitute for those separate preflight tests.

## Historical GPU availability evidence (pre-pilot)

The fresh host-level inventory at 2026-11-01 09:05 UTC showed:

| GPU | UUID | State |
|---|---|---|
| A2 | `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35` | occupied by `./cudaa`, PID 1181701, using 1,220 MiB; 13,650 MiB free but utilization 100% |
| V100S | `GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb` | occupied by the independent Stage-1 Python process PID 1128864, using 13,402 MiB; utilization 100% |

The safe V4 smoke launcher was retried with no shared-GPU override. It refused
to start because the A2 had active compute process PID 1181701. It exited
before creating the requested run directory; no trainer, checkpoint or
training state was created. Neither GPU process was signaled or modified.
Although free memory exceeded the launcher threshold, active compute and
100% utilization independently make the A2 unavailable under this protocol.

## Historical V3 receipts (not V4 validation)

The previous V3 audit reported 115 relevant tests passing and an A2 profile
with 656,113,664 peak allocated CUDA bytes on AES. It also reported finite
candidate, demand and hardware-relation gradients in real-candidate updates.
Those receipts are retained in Git history and are not evidence that V4 passed
GPU preflight or training. V4 requires fresh GPU smoke and interruption/resume
validation.

## Completed bounded V4 probe

The current authenticated run and exact output paths are:

- Cohort/cache: `/tmp/v22-gnn-v4-fast-probe-s123-r4/`
- Training receipts/checkpoints: `/tmp/v22-gnn-v4-fast-probe-s123-r4/training/`
- Tracked summary: `results/fast_probe_s123_summary.json`
- Tracked selected-context manifest: `results/fast_probe_s123_cohort_manifest.json`
- Source commit: `815c8b3`; protocol hash: `a8e03450dd265432d1932ee8f0ea5e904545dea52128aa6d94f4101e1e212e90`

The final cohort contains 80 training and 32 validation candidates in 10 train
and 4 validation kernel/device contexts. Training families are GEMM, MD-KNN,
SpMV-ELLpack, Stencil2D and Rodinia Dilate. Validation families are MachSuite
Viterbi and Rodinia PathFinder. Both devices are represented at 10 ns. The test
split is not loaded into the selected feature cache or run inputs. Validation
candidates are chosen by configuration-hash order only; their measured QoR is
used only for evaluation.

Preparation took 248.84 seconds total, including 150.82 seconds for the
existing full authentication/data reconstruction, 9.82 seconds for 28 graph
tensors and 38.78 seconds for 112 candidate feature preparations. Once
prepared, training and validation reuse the exact tensors. Exact checks passed
for 28 graph tensors, 112 complete assignment vectors and 112 candidate-demand
descriptors. The cache binds RC3, dataset/split hashes, action/source mappings,
V4 tensor and demand schemas, and implementation source hashes.

The A2 was freshly identified and idle immediately before launch; the V100S
was left untouched. No Stage-1 trainer was found by process-name check. The
probe completed two matched arms, 20 updates each, in 3.53 seconds total GPU
phase time. Initial model hashes and sample traces matched. The protocol and
checkpoint SHA256 values are in the tracked JSON summary. Peak CUDA allocation
was not recorded; no hardware-resource measurements were generated.

The old V100S smoke is preserved at
`/home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-v100s-smoke-s123-r3/`: five
numerical updates completed before evaluator validation failed with
`NameError: Counter is not defined`. The missing import is in the current
source. It was not resumed or overwritten. A separate initial cohort attempt
failed because Sort-Radix L6 has no source-authenticated measurement column;
subsequent cohort uses fully mapped Viterbi. Those failed preparation receipts
are retained in their separate r1/r2/r3 directories.

Focused verification in the pinned isolated Python 3.10.20 environment:
`48 passed` across the fast-probe, V4 training-contract, semantic-tensor and
action-demand suites; `py_compile` and `git diff --check` passed. Previously
completed broad suites are historical and not recounted as fresh.

The hardware path has finite gradients and visibly changes action embeddings.
However, primary matched-model ranking, QoR and top-k measured-pool results do
not show consistent selection benefit; one validation context has very large
latency regret. The one-seed, 20-update pilot is not evidence of broad
generalization, training convergence, synthesis QoR, or Stage-2 improvement.

## Learnability and target-conditioning follow-up (2026-11-01)

The branch was fetched and verified at `c8f823f63c557e60bcf0057d7bfcd1363d77a1ae`
in the clean `/tmp/MailoHLS-gnn-refinement` worktree. The original checkout,
Stage-1 and refinement worktrees were not changed. The active implementation
uses RC3 archive SHA256
`feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`, dataset
SHA256 `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`,
and split SHA256 `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`.
The new coverage report authenticates exact graph/source/metadata identities
for five train and two validation kernels and records an empty test-kernel
list. Kalman/test measurements were not selected or used for fitting or
evaluation.

### Target coverage and leakage boundaries

The new audit uses exact complete-assignment hashes to establish shared target
designs. Every training kernel has at least 16 assignments shared across
ZCU104/U200 at 3.33/5/10 ns. PathFinder has 14 assignments shared across all
six contexts; Viterbi has none, so its candidates are hash-selected by context
and paired calculations use exact overlaps only. The target coverage JSON
reports per-context candidate counts, overlap, measured latency/resource
variation, feasibility at 100/50/10%, action registration/match counts, and
source/graph identities. Physical measurement identity excludes request
budget; the evaluator changes feasibility limits while retaining the same
physical QoR labels.

The frozen cohort contains 480 training records (30 kernel/device/clock
contexts) and 180 validation records (12 contexts). Training assignments are
selected from the six-way intersection by directive Hamming diversity, with
no QoR label used. Validation selection is hash-only and likewise QoR-blind.
QoR normalization is fitted on selected training records. Static graph memory
does not receive candidate directives, candidate demand, QoR or split
membership. Candidate and analytical demand tensors enter only the QoR
prediction pathway. Test rows do not enter the selected records or feature
tensors.

### Fixed-context learnability

The reproduction runs use new output directories:

```bash
CUDA_VISIBLE_DEVICES=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 PYTHONPATH=. \
  /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.learnability_probe \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-v4-fixed-learnability-repro-200 \
  --kernel machsuite-stencil2d --profile zcu104 --clock-ns 10 \
  --candidates 24 --updates 200 --gpu-name A2 \
  --gpu-uuid GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 \
  --preparation-limit-seconds 600 --gpu-limit-seconds 1200

CUDA_VISIBLE_DEVICES=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 PYTHONPATH=. \
  /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.learnability_probe \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-v4-fixed-learnability-repro-1000 \
  --kernel machsuite-stencil2d --profile zcu104 --clock-ns 10 \
  --candidates 24 --updates 1000 --gpu-name A2 \
  --gpu-uuid GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 \
  --preparation-limit-seconds 600 --gpu-limit-seconds 1200
```

The tracked manifests/results record exact selected assignment hashes, data
identities, hashes for the V4 graph/feature implementation used by the run,
baselines, checkpoint identities and update history. The learnability helper's
own source hash was added to the script after these runs, so it is not claimed
as an authenticated field in those historical receipts. Weights remain in
`/tmp`. The 1,000-update run starts fresh with the same seed; its first 200
updates match the separate 200-update trajectory. It is training-set
memorization only.

### Matched three-arm experiment

The completed run used the A2 UUID
`GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`; the V100S was not used. Authenticated
data loading took 94.96 s, feature preparation 136.59 s, and all three
100-update arms plus validation 7.56 s total GPU time. Peak CUDA memory was not
instrumented. Numerical and hardware GNNs share a 116,081-parameter initial
state and exact sample trace. The candidate+target MLP has 338,053 parameters,
so it is a simple non-graph reference rather than a matched-capacity ablation.
The 180 validation rows come only from Viterbi and PathFinder; Kalman was not
used.

Reproduction command (use a new output directory and verify A2 availability
immediately before launching):

```bash
CUDA_VISIBLE_DEVICES=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 PYTHONPATH=. \
  /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.target_condition_probe \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-v4-target-conditioned-probe-reproduction \
  --gpu-name A2 --gpu-uuid GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 \
  --updates 100 --candidates-per-update 8 \
  --preparation-limit-seconds 600 --gpu-limit-seconds 1200
```

The completed raw result is preserved in `/tmp/v22-v4-target-conditioned-probe-s123-20261101-r2/`.
Its SHA256, exact cohort and corrected, denominator-aware summary are tracked
in `results/`. The raw result's near-zero area/ADP relative regrets in
Viterbi/U200/5 ns must not be averaged: that measured optimum is exactly zero.
The corrected summary marks those regrets undefined and reports 11/12 defined
contexts. A focused test covers this rule.

The first three-arm attempt trained and checkpointed all arms but failed while
serializing a tuple-keyed intermediate prediction map; it is preserved in its
original `/tmp` directory. The successful `-r2` attempt fixed the report
boundary. A later corrected-report rerun was refused by the GPU guard because
the A2 was occupied by `VLLM::EngineCore` (10,528 MiB, observed PID 1655798).
No process was signalled, and the previous completed checkpoints/results were
preserved.

### Verification and limits

The latest focused run of the new tests plus V4 training-contract tests passed
23 tests. Semantic-tensor/action-demand tests passed separately (12 tests); a
combined run before the near-zero helper change passed 34. Python compilation
passed for the new probe modules. After the final changes, the complete relevant
focused suite passed **120 tests** and `git diff --check` passed. No graph,
target overlay, V4 feature schema, production GNN, Stage-2,
measurements or training split changed. This one-seed, 100-update, two-family
validation does not establish broad transfer or Stage-2 benefit.

## Follow-up audit: corrected 300-step runs (2026-11-01)

The metric fix uses exact candidate-assignment identity and paired source
latencies. Meaningful latency change is
`abs(ΔL) > max(1e-5 ms, .01 * max(abs(L1),abs(L2)))`; `1e-5 ms` is one 10 ns
cycle. The corrected complete-pool report contains 91 clock-pair observations
across 10 overlapping groups and 45 device-pair observations across 5 groups.
These correspond to only two validation kernels. Clock: 91 meaningful, 0
negligible. Device: 3 meaningful, 42 negligible. The three meaningful device
pairs are from Viterbi; PathFinder's measured changes are below threshold.
Overlapping assignments are not independent observations. The previous
historical sign summary was computed with a defective threshold and must not
be cited; corrected historical 100-step evaluations are included in the
attempt-03 `results.json`.

Zero-area provenance audit: 9 Viterbi/U200/5 ns source CSV rows have all four
utilization columns present and literally equal to zero. The preprocessed
table preserves these values and the established percentage/capacity
conversion yields the zero area proxy and ADP. Raw CSV provides no count-level
resource precision, so rounding versus true zero is unresolved. No conversion
bug was found and source labels were not changed. Audit receipt:
`results/zero_area_measurement_audit.json` (raw CSV SHA256
`2e074830a6ad6597eab08d88264d5a8cbf51c47dbebe9369578a22c307cf507d`,
preprocessed CSV SHA256
`8fa65d6659b5d42fdf0826b2af38e26c1bef56ae5da835de6cbabedd2d6be4de`).

### New 300-update original V4 run

- Branch source: commit `dd31514eaeef933c12032c8cd93ca8ae6d339fa2`.
- Dataset SHA256: `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`.
- Split SHA256: `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`.
- RC3 archive SHA256: `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
- Cohort: same seed-123 480 training / 180 validation rows, five train and two validation families; test/Kalman excluded.
- Manifest SHA256: `d52e8fd7f161e42205ce0c66d78c44f3e40dbbf8b3b6d6c75cffb5680c13a225`.
- Result SHA256: `fec061f0dde2a72db6e99ca8d2eded3a3c7be1e908e268763c2ee57f6229107e`.
- Output: `/home/elvouvali/mailohls-gnn-runs/v4-corrected-300-s123-20261101/attempt-03`.
- A2 UUID `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`; 300 updates/arm, 20.66 s GPU total, 121.6 MB maximum allocation; 94.28 s dataset loading and 144.47 s features.
- Numerical/hardware initial parameter hash and sampled training trace matched. CUDA output checkpoints were not bitwise identical across retry attempts; attempt-02 has no completed report and contributes no results.

The new run still predicts every validation candidate feasible at 50% and
10%, matching always-feasible accuracy (149/180 and 83/180 respectively).
Budget-only physical latency drift at 100%-to-50% was 0.412 ms numerical and
0.629 ms hardware. Hardware edge inference ablation shifts latency Spearman
0.392 to 0.349 and top-1 regret 2.345 to 2.790, a modest effect on these
contexts, not a stable benefit.

### Separately versioned physical-QoR/budget ablation

The next run used schema
`salt-hls-v4-physical-qor-budget-independent-v1`, same candidate cohort,
sample trace and matched initial graph-model hash. It used a separate output
directory and fresh initialization. Budget is excluded from the physical
prediction graph path; resource feasibility still uses requested budgets.
Result SHA256 `338b978fca62482c951be145e1811dd7c829c1b014eff350dce97b8e7c936c90`,
manifest SHA256 `b1dda912241fc10e5db2b2f449bc13d053fec7d2fbe23cf9e69d974649dd3209`,
output `/home/elvouvali/mailohls-gnn-runs/v4-corrected-300-s123-20261101/attempt-04-budget-independent`.
Feature prep was 142.71 s and GPU work 18.70 s. Budget-only latency drift was
at most `4.61e-7 ms`, attributable to batch floating-point variation. This
correctness property is achieved, but held-out hardware selection worsened
(latency Spearman 0.243, top-1 regret 6.457); no predictive improvement is
claimed.

Both runs kept the graph archive, source/action registry, measured labels,
family split, model architecture and production artifacts unchanged. No
Stage-2, test, synthesis or full-corpus regeneration was performed.

Focused verification after the metric fix passed **49 tests** across
`test_v22_target_contrasts.py`, `test_v22_gnn_training_contracts.py`,
`test_v22_gnn_resume.py`, `test_target_graph_semantic_tensor_v22.py`, and
`test_action_demand_tensor_v22.py` using the pinned Python 3.10 environment.
Python compilation and `git diff --check` also passed. A 600-step launch was
blocked by the code's clean-source guard because documentation edits were
uncommitted; it made no optimizer updates. This is preserved as a failed
preflight receipt, not an experiment. The retry completed in a fresh output
after commit `2e2617f00f53a5df3b2757b0477c0c46ad70e2ea`.

## 600-update run receipt

Output: `/home/elvouvali/mailohls-gnn-runs/v4-corrected-300-s123-20261101/attempt-06-600-updates`.
Manifest SHA256 `16148104959661fde4f1bf2835bfb2fab87df513ef31b61cf97cbd1afda5d89b`,
results SHA256 `0605618f86a542f86e0d1f6fd176841e3e7cc7c271d951ffabeccf68b0fa0bf8`.
Data-load/feature times were 94.04/143.55 s; GPU train/evaluate was 44.17 s;
peak allocation 121.6 MB. It started fresh, not from optimizer state. Initial
parameter hash, seed and cohort match the 300-update study.

At step 600, fixed-context latency MAE was 0.344 ms for both graph arms (from
1.146 numerical / 1.160 hardware at initialization). Held-out hardware latency
Spearman/top-1 regret were 0.243/2.618, versus 0.392/2.345 at step 300;
numerical values were 0.161/2.649 versus 0.313/2.790. All arms still had 31
false-feasible outputs at 50%. Clock sign accuracy was 0.958 numerical and
0.595 hardware, but paired-delta MAE remained about 7.3 ms. Device sign used
only three meaningful observations and is not a general result. These
receipts support fixed-context learnability but not reliable held-out QoR.
