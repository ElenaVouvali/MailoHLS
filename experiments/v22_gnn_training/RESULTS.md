# V4 experimental results

## Current result: experiment incomplete

The V4 matched training experiment has **not** run. There are no V4 optimizer
updates, validation-selected checkpoints, test predictions or measured-pool
QoR metrics. Therefore no claim can be made that hardware-resource messages
improve QoR prediction, ranking, feasibility or design selection.

The fresh CPU audit reproduced the frozen graph, dataset and split identities
and passed with the declared exclusions. Completed CPU test commands in the
isolated Python 3.10.20 / PyTorch 2.6.0 / PyG 2.6.1 environment:

| Command group | Result |
|---|---:|
| `test_v22_gnn_training_contracts.py` + `test_v22_gnn_resume.py` | 25 passed |
| `test_target_graph_v22.py` | 58 passed |
| `test_mlir_graph_v2.py` + `test_structural_audit.py` + `test_directive_domains.py` | 86 passed, 15 skipped |
| Semantic tensor + action-demand tests | included in a completed 35-test focused run across four files; these two modules were unchanged afterward |

The three completed command groups above total 169 passes and 15 skips; the
semantic tensor/action-demand focused run separately reported 35 passes; its
updated training-contract/resume files were rerun in the 25-test command.
`py_compile` and `git diff --check`
also passed after the changes. CUDA-dependent V4 smoke/profile tests were not
run. A broader combined invocation was manually stopped after partial progress
and is not counted as a test result.

## GPU blocker observed

The latest available host-level inventory showed the designated NVIDIA A2
(`GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`) occupied by an unrelated
`ray::RayWorkerWrapper.__ray_call__` process, PID 1110073, at 98% utilization
and about 9,552 MiB allocated. The reserved V100S
(`GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb`) was occupied by the independent
Stage-1 Python process, PID 1100139, at 100% utilization and about 6,972 MiB.
No process was touched. The V4 launcher and profiler were not run because the
A2 availability gate failed. Direct sandbox `nvidia-smi` access is blocked;
these figures are from the just-prior authorized host inventory rather than a
new direct sandbox query.

The GPU interruption/resume smoke and both-arm short smoke therefore remain
outstanding. Full 8,000-update-per-arm training was not launched. There is no
V4 checkpoint to select or resume.

## What was validated on CPU

The authenticated audit report is at
`/tmp/v22-gnn-v4-verification-20261101/data-audit.json`. It reproduced:

| Item | Result |
|---|---:|
| RC3 graphs / GNΩSIS kernels | 63 / 55 |
| Distinct reconstructed rows | 150,026 |
| Source-domain-valid assignments | 149,227 |
| Eligible measurements | 104,147 |
| Train / validation / test | 88,466 / 8,976 / 6,705 |
| Excluded missing APL measurements / illegal complete tuples | 45,080 / 799 |
| External device/clock rows included | 0 |

The evaluator now retains per-context and aggregate feasibility outcomes,
false-feasible selections, no-prediction/no-reference cases, top-1/3/5/10
measured-pool shortlist success and conditional regret, explicit undefined
Spearman/Kendall reasons, and full-vs-half-budget prediction shifts. The
predeclared validation selection score is unchanged; these extensions are
reporting-only. Synthetic unit cases exercise failure denominators and do not
represent trained-model results.

## Historical evidence kept separate

The former V3 A2 memory profile and measured-candidate gradient receipts remain
valid only for their recorded V3 source/schema. A previous V100S attempt was
incomplete. Neither yields V4 training or a paired V4 comparison. No test
labels, test metrics, HLS synthesis results or model weights were generated in
this V4 attempt.

## Scientific decision

`CONTROLLED_EXPERIMENT_INCOMPLETE`.

The primary research question remains unanswered because GPU preflight and
matched training did not run. The sole next action is to repeat the A2
availability check and, once that designated GPU is idle, complete the V4
paired smoke, graceful interruption/resume test, and then the frozen matched
training protocol in [`RUN_PROTOCOL.md`](RUN_PROTOCOL.md). The V100S is not a
fallback. No architectural change is justified from the available evidence.
