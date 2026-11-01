# Next-step brief: continue the bounded V4 learnability study

## Current evidence (2026-11-01)

The existing V4 candidate-conditioned GNN can partially memorize a fixed
24-design Stencil2D/ZCU104/10 ns training set. At 1,000 regression-only updates,
standardized train SmoothL1 fell from the constant baseline 0.3363 to 0.0940;
latency MAE fell from 0.4023 ms to 0.2479 ms; train Spearman reached 0.701.
Its predicted best was 1.05% slower than the measured best and never selected
the exact winner. This demonstrates limited in-sample candidate learnability,
not generalization.

A matched 100-update experiment trained candidate+target, numerical GNN and
hardware-message GNN arms on 480 assignments across five training families
and six target contexts. It evaluated 180 measured configurations across
Viterbi and PathFinder, both held-out families, without using Kalman/test data.
The hardware model slightly improved latency MAPE and latency top-3 regret
over the numerical model, but had much worse top-1 latency regret. Feasibility
at 50% capacity exactly matched the always-feasible baseline. Device-change
latency signs were wrong in the paired validation groups. The hardware edge
path changes embeddings and outputs, but its ablation is mixed and does not
show a consistent selection benefit.

The probe also found budget-induced changes to predicted physical QoR for the
same candidate/device/clock: mean latency shifts were 0.0124 ms for the
candidate-only model, 0.1105 ms for the numerical GNN, and 0.1417 ms for the
hardware GNN. Measured physical QoR is independent of a later request budget.
If this persists, test a separately versioned physical-QoR/feasibility split;
do not alter the current V4 schema retroactively.

The A2 was later observed occupied by `VLLM::EngineCore` (PID 1655798 at the
last check). Do not signal or share it. The V100S is reserved for Stage-1.
Check both GPUs immediately before any new run.

## Single next experiment

When the A2 is actually idle, continue with **300–600 updates per arm** on the
same frozen three-arm cohort and same seed 123. This tests whether the poor
family-held-out results were mainly caused by only 100 updates. Reuse the
authenticated graph release, cohort manifest, feature logic, candidate pools,
optimizer, loss, and validation metrics. Do not tune the cohort or model on
validation QoR. Preserve the completed 100-step run and all checkpoints.

Use a fresh output directory and the existing V4 environment. First check
`nvidia-smi`, disk/RAM, and process identity. Never use `--allow-shared-gpu` or
fall back to the V100S. If the A2 is occupied, stop at CPU-only checks.

Example command (after confirming availability; `--output` must be new):

```bash
CUDA_VISIBLE_DEVICES=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 PYTHONPATH=. \
  /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.target_condition_probe \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-v4-target-conditioned-probe-s123-300step \
  --gpu-name A2 --gpu-uuid GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 \
  --updates 300 --candidates-per-update 8 \
  --preparation-limit-seconds 600 --gpu-limit-seconds 1200
```

This current entry point trains from the pinned seed at the requested update
count rather than resuming the 100-step optimizer state. It is a deterministic
longer-budget comparison, not an optimizer-state resume. Keep the 100-step
results as the baseline and compare each arm on identical validation contexts.
Do not run a second seed unless this continuation shows clear learning and
resource limits permit it.

If longer training improves rankings but hardware messages remain inconsistent,
do not add graph layers. If the physical QoR budget drift remains, design one
separately versioned physical-QoR/feasibility ablation after the continuation.
Stage-2 integration remains deferred: V4 candidate-conditioned QoR prediction
does not establish that its candidate-independent action memory improves LLM
directive generation.

## Reproducibility artifacts

- Target coverage: [`results/target_coverage_audit_s123.json`](results/target_coverage_audit_s123.json)
- Exact target cohort: [`results/target_condition_probe_s123_manifest.json`](results/target_condition_probe_s123_manifest.json)
- Raw and corrected results: [`results/target_condition_probe_s123_raw.json`](results/target_condition_probe_s123_raw.json), [`results/target_condition_probe_s123_summary.json`](results/target_condition_probe_s123_summary.json)
- Fixed-context learnability: [`results/fixed_context_learnability_s123_results.json`](results/fixed_context_learnability_s123_results.json)
- Full methods and interpretation: [`README.md`](README.md), [`RESULTS.md`](RESULTS.md), [`AUDIT.md`](AUDIT.md)

Production remains on the V1 GNN. Do not export V4 checkpoints or action
memories into Stage-2.

## Superseding status after corrected 300-step study

The recommendation above has been executed and superseded. Original V4 was
trained for 300 matched updates/arm in
`/home/elvouvali/mailohls-gnn-runs/v4-corrected-300-s123-20261101/attempt-03`;
a separate physical-QoR/budget-independent 300-step ablation is in
`attempt-04-budget-independent`. Both used the A2, same frozen seed-123 cohort,
and less than 21 seconds of GPU work. See `RESULTS.md` for run and checkpoint
hashes.

The key corrected facts are: (1) the numerical GNN correctly predicts the
direction of clock changes in 95.8% of meaningful paired observations, but
paired-delta MAE is 7.29 ms; (2) only 3/45 device contrasts are meaningful, so
device dependence cannot be assessed; (3) all models predict every 50% and
10% candidate feasible, equal to the always-feasible baseline; (4) the
hardware arm's 300-step ranking improves over its 100-step ranking, but
top-1 regret stays 2.345 and the result is not consistent across all metrics;
(5) separating feasibility budgets from physical prediction enforces
budget-invariant QoR (drift below `5e-7 ms`) but does not improve hardware-arm
selection.

The 600-update run was completed in a clean-source fresh output (not resumed);
the earlier dirty-tree attempt-05 was preserved and made zero updates. At 600,
fixed-context latency MAE improved from about 1.15 to 0.34 ms, confirming
candidate learnability. Held-out metrics were mixed or worse than at 300, and
feasibility remained the always-feasible baseline. No graph or encoder changes
are justified. The next single experiment should be a paired target-difference
loss ablation on the same training cohort, with one control arm and pair
selection/normalization/coefficient predeclared. Training coverage contains
480 potential exact assignment clock pairs and 240 device pairs before
filtering, but held-out device coverage has only three meaningful pairs.
Evaluate clock-pair response descriptively and do not make a device-transfer
claim. Keep Stage-2 integration deferred.
