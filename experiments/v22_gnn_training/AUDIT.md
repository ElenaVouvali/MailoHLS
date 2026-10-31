# V4 implementation and data audit

## Scope and provenance

This audit distinguishes the V4 CPU checks completed in this worktree from
historical V3 GPU receipts. No V4 GPU run has been attempted. Current branch at
start of this work: `research/gnn-refinement`, commit
`958c99b9fab2ee29b2b61f863b5654c5fc1fe409`. V4 source changes made during this
task are described in the current Git diff and are not covered by earlier V3
receipts.

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
and true GPU interruption/resume test have not run in this task.

## Current GPU availability evidence

The most recent host-level GPU inventory available to this task showed:

| GPU | UUID | State |
|---|---|---|
| A2 | `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35` | occupied by unrelated `ray::RayWorkerWrapper.__ray_call__`, PID 1110073, using about 9,552 MiB; utilization 98% |
| V100S | `GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb` | occupied by Stage-1 Python process PID 1100139, using about 6,972 MiB; utilization 100% |

No process was signaled or modified. The sandbox's direct `nvidia-smi` call
cannot access the host driver; the table is inherited from the just-prior
authorized host-level inventory and is not a fresh sandbox observation. The
training launcher was not invoked. Host RAM and disk were adequate at that
inventory time, but shared CPU/I/O availability does not make the occupied A2
safe to use.

## Historical V3 receipts (not V4 validation)

The previous V3 audit reported 115 relevant tests passing and an A2 profile
with 656,113,664 peak allocated CUDA bytes on AES. It also reported finite
candidate, demand and hardware-relation gradients in real-candidate updates.
Those receipts are retained in Git history and are not evidence that V4 passed
GPU preflight or training. V4 requires fresh GPU smoke and interruption/resume
validation.
