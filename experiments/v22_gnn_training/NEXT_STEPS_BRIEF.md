# Next-step brief: target-aware directive-preference evidence

## Current decision after the preference probe (2026-11-01)

The three-arm, 300-update directive-preference run is complete and documented
in [`RESULTS.md`](RESULTS.md). On 270 shared validation contexts, hardware
messages did not beat compiler-only memory on ADP pair accuracy or measured
top-1/top-3 regret. FiLM's small ADP pair-accuracy gain came with fewer
feasible top-1 selections and higher regret. All three arms missed all 21
meaningful validation budget-driven winner changes. The validation data contain
no meaningful device- or clock-driven winner changes, so those axes cannot
support a response claim. Two validation families and one seed are insufficient
for a general conclusion.

### Next single step

Perform a label-independent coverage audit for a family-disjoint validation
cohort that has exact matched candidate assignments and meaningful measured
preference changes across device and clock. Keep the encoder frozen during
this audit. Do not launch another training variant until the validation set
contains evidence that can test the target-conditioning claim.

Before claiming that compiler structure itself contributes, add a small
candidate-plus-numerical-target scorer control with the same decision head and
protocol. The current zero-memory ablation is only an inference-time
distribution-shift diagnostic, not an independently trained no-graph baseline.
Do not build a richer FPGA graph, add GNN layers, integrate into Stage 2, or
modify the refinement controller on this evidence.

The selected V4 memory remains candidate-independent `[64,128]` action slots,
but it is a new feature space and cannot be substituted into V1-trained
cross-attention weights. A future Stage-2 test would retrain/adapt cross-
attention and compare prompt-only, V1 memory, V4 compiler memory and V4 target
memory on identical directive cases. A future refinement policy may use the
validated memory only to prioritize legal compiler-verified proposals; Vitis
measurements must still determine acceptance. Neither integration is part of
this result.

The current classification is **`TARGET_PREFERENCE_COVERAGE_INSUFFICIENT`**.
This reflects absent meaningful held-out device/clock reversals; it does not
hide that the available budget reversals were all missed. The result supports
no present hardware-message selection benefit.

## Historical status before the directive-preference experiment

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

## V4b full-cohort action-preference status (2026-11-02)

The 2026-11-01 V4 analysis above remains historical. The separate V4b CPU
eligibility and target-preference audit is complete; its manifests and
machine-readable receipts are under `v4b/receipts/`. It confirms 37 usable
training kernels and three validation kernels (19 families total, 17 train / 2
validation), with 14 non-Kalman kernels excluded for unknown registered-action
values. Exact measured pools contain some target-dependent preferences but
very sparse meaningful winner changes: two full-budget device contexts and no
full-budget clock contexts; the held-out families have no meaningful device,
clock or budget winner reversals.

V4b implements physical-QoR/budget separation, complete-assignment ADP pair
construction, candidate/target and compiler controls, target-FiLM modulation,
hardware messages, and an antisymmetric action-difference head. A direct host
query found the V100S idle (A2 was occupied). Two one-update smokes were
started there, but each exceeded 30 minutes in CPU preparation before GPU
work and was gracefully interrupted by verified PID. V4b now caches one
overlay per kernel, static program/action analysis per graph identity, and
candidate vectors/demands per complete assignment across targets. CPU
regressions and an authenticated cross-clock example check equality against
the existing path. The newest cache has not yet been timed, and no V4b
checkpoint or selection result exists.
The current conservative classification is
`EXPERIMENT_INCOMPLETE_RESOURCE_LIMIT`.

Next: rerun the one-update smoke on an explicitly selected, independently idle
GPU using the cached feature path, then launch the frozen five-arm schedule
only if preparation and validation complete within the resource budget. Do not
inspect Kalman, expand the candidate pool based on validation labels, or
integrate memory into Stage-2. Because
validation has no meaningful target preference reversal, an additional
family-disjoint target-sensitive validation cohort may be required before
claiming that target modulation learns correct device/clock changes.
