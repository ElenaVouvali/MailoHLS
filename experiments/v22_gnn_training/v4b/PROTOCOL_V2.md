# V4b evaluation/resume protocol v2 — 2026-11-02

This protocol repairs the existing unexecuted V4b runner. Original V4 source,
checkpoints and experiment definitions remain unchanged. Earlier V4b audit
receipts and incomplete smoke attempts remain historical. V2 fingerprints
runtime sources independently of the frozen v1 candidate manifest.

The primary comparison is frozen before any new validation predictions:
37 training kernels / 17 families, three validation kernels / two families,
seed 123, 16 QoR-blind candidates per physical context (7–16 available),
ZCU104/U200 × 3.33/5/10 ns, budgets 1/.75/.5/.25/.1. No Kalman labels are
parsed. Missing actions remain unknown. No optional recovered cohort is admitted.

All five arms use independently optimized fresh models: no learned graph
memory, compiler, target-FiLM, hardware-message, compiler action-difference.
Compiler/hardware/difference encoders share initialization, never optimizer
state. The no-graph arm has separate parameters and initialization. All arms
receive the same analytical candidate descriptors; this isolates learned graph
memory, not all compiler information. Encoder geometry remains the RC3/V4
2-layer/64-wide/2-head/128-output/64-slot contract.

Training is bounded to 1,500 updates per arm, two physical observations and one
preference pair per update. One uniformly sampled training family, then one
nonempty context/budget, then one eligible pair is drawn using a shared seed-123
trace. Large families therefore do not get proportionally more updates.
AdamW lr 3e-4, weight decay 1e-4, gradient clip 1.0. Loss is training-only
normalized log1p physical SmoothL1 + **0.1** mean logistic pair loss. Feasible
beats infeasible; both-infeasible and <=5% ADP gaps are excluded. The coefficient
and architecture are not tuned on validation. Validation is every 250 updates.
Checkpoint selection minimizes family-macro selection failure plus conditional
feasible ADP top-1 regret; missing feasible regret contributes 1. Pair accuracy
breaks equal scores, then earliest step wins. Conditional regrets must always
be read alongside selection failure and feasible counts.

Pointwise physical predictions have no budget input. Feasibility uses predicted
physical counts versus explicit fraction × capacity. A separate budget-aware
preference pathway ranks predicted-feasible candidates. The difference arm
compares candidate-conditioned action states, uses symmetric attention inputs
and a bias-free signed projection. Both pair antisymmetry and identical-pair
zero are enforced. Pool ranking uses all-pairs net wins, O(N² × 64); ties use
canonical hashes. Cycles are possible. Standard pointwise scores induce a
transitive special case of the same aggregation rule. No objective token
enters structural memory.

Evaluate every validation context even if it has no meaningful pair. Report
per-context, per-family and family-macro results, physical errors, exact matched
physical-target delta errors, candidate feasibility confusion, top-1/top-3
measured-pool regret, and meaningful budget winner changes. Resource precision
is TP/(TP+FP), recall TP/(TP+FN), false-feasible FP/(FP+TN), false-infeasible
FN/(FN+TP). Undefined denominators are null. All-zero Viterbi/U200/5 ns resource
rows are omitted from ADP ordering, retaining their original physical labels.
The oracle diagnostic changes only the feasibility filter, never model inputs.
Budgets are correlated views, not independent replications. No significance
claim follows from two held-out families or one seed.

Postselection diagnostics use zero, fixed-seed shuffled, global-mean, and
opposite-device memory (same program/clock). Zero also removes global pooled
memory; slot shuffle/mean retain pooled state. The first hash-ordered measured
training pair differing at one matched action is selected without QoR and
perturbed by +0.1 per memory coordinate at that slot and one unrelated slot.
Report score changes; attention alone is not causal evidence.

The default is the A2 UUID `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35` was initially authorized. The user subsequently explicitly authorized the
V100S instead (`ok do it in V100 instead then`). Its required UUID is
`GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb`; host inspection must confirm it is
idle. This explicit override applies only to this V4b execution. Check host GPU processes, free memory,
disk and UUID immediately before CUDA. No sharing or fallback is permitted.
CPU authentication, tests and a fixed-context sanity check precede training.
A one-update smoke in a fresh output must complete before the 1,500-update run.
Limit preparation attempts to 30 minutes and the initial training attempt to
60 minutes wall time; interruption must save optimizer/RNG/sample position.
These are resource limits, not convergence criteria.

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.v4b.train_probe \
  --release-dir /tmp/salt-v22-training-release \
  --authenticated-audit experiments/v22_gnn_training/data-audit.json \
  --manifest experiments/v22_gnn_training/v4b/receipts/preference_cohort_manifest.json \
  --output /durable/new/output --prepare-only --prepared-cache /tmp/v4b-prepared-v2.pt
```

CPU cache publication records source, manifest and environment hashes. Treat
pickle caches as trusted local artifacts only. For CUDA omit `--prepare-only`,
set `CUDA_VISIBLE_DEVICES` and `SALT_HLS_GPU_UUID` to the explicitly selected GPU UUID, pass matching `--gpu-name V100S
--gpu-uuid GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb` for the authorized override, and use
`--updates 1 --validate-every 1` for the smoke, and a separate output for
`--updates 1500 --validate-every 250`. Resume the same invocation with `--resume`;
protocol changes are rejected. Atomic training states are saved every 25 updates
and at validation/safe interruption. Completed checkpoints are never overwritten
by a new experiment. Large caches, weights and GPU logs stay outside Git.

CompareXplore references inspected directly on 2026-11-02:
[src/model.py](https://github.com/NVlabs/CompareXplore/blob/main/src/model.py),
[src/node_att_diff.py](https://github.com/NVlabs/CompareXplore/blob/main/src/node_att_diff.py),
[src/pairwise.py](https://github.com/NVlabs/CompareXplore/blob/main/src/pairwise.py).
NodeAttDiff attends to candidate embeddings concatenated with their signed
difference and aggregates per graph. Its attention is not inherently swap
symmetric. The model combines pointwise regression and comparison losses;
pairwise.py forms within-kernel pairs with configurable pragma differences.
V4b uses these concepts only; no upstream implementation is copied.
Fetched SHA256 values, respectively:
`f1b2c96787bef206737da7454a504a8ea93b4cd1e8c734ed77cc44dad7057480`,
`2594f4fb781b2582152b8ea34c10965d60fb9bd84c52dd94753b27bc65cb7b33`,
`2090424c55134e157fe364bb2ad2aa5ae08c2443e7c435362ad2a2bc8a6ab55d`.
