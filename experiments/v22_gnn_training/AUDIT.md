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
