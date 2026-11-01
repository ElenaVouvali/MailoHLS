# Next-step brief: clock-conditioned V4 GNN study

This note is a handoff for the next SALT-HLS GNN research session. It records
the completed small pilot, its deliberate limits, and a bounded next experiment
that directly tests device and clock conditioning. It is not authorization to
run without first checking the current branch, worktrees, GPU processes, source
hashes, and data identities.

## Copy-ready task for the next ChatGPT session

> Work on `research/gnn-refinement` in a dedicated clean worktree. Read this
> brief and `experiments/v22_gnn_training/{README.md,RESULTS.md,AUDIT.md}` before
> changing anything. Do not touch `main`, the Stage-1 branch, the refinement
> branch, production V1 GNN weights, Stage-2, frozen RC3 graphs, historical
> measurements, or the completed fast-probe outputs. Do not run synthesis.
>
> The completed V4 pilot compared the existing numerical-conditioned and
> hardware-message models for 20 updates each. It used five training kernels,
> two held-out kernels, two devices, **only the 10 ns clock**, and 8 measured
> candidates per kernel/device context. Hardware messages affected action
> embeddings and some ranking scores, but did not consistently improve
> measured-pool selection. The models had high validation errors and the pilot
> did not establish useful held-out learning. The pilot did not test whether
> the model responds correctly to changing clock period.
>
> Prepare a small, family-disjoint follow-up using the same authenticated GNΩSIS
> dataset, seed-123 family split, RC3 graph release and V4 tensorizer. Include
> ZCU104 and U200 at each authenticated clock of 3.33, 5 and 10 ns. Start from
> the five existing training kernels and two validation kernels unless their
> authenticated coverage fails: train on GEMM-NCubed, SpMV-ELLpack, Stencil2D,
> MD-KNN and Rodinia Dilate; validate on MachSuite Viterbi and Rodinia PathFinder
> baseline. This gives 30 training kernel/device/clock groups and 12 validation
> groups before candidate selection. Keep the held-out test split unused.
>
> For each validation kernel, first determine whether complete directive
> assignments are measured in all six device/clock contexts. If so, choose a
> small deterministic hash-ordered intersection of shared assignments, without
> looking at their QoR labels during selection. This permits paired comparisons
> of the same assignment across clock and device. If the intersection is too
> small, report that and do not claim a paired target-response result. For
> training contexts, use a fixed, deterministic assignment-diversity procedure
> and only training labels for any feasibility balancing. Keep measured QoR
> outcomes attached only as prediction labels. Preserve the source-aware
> directive-domain validation and unresolved-action masks.
>
> Reuse `fast_probe.py`, `CandidateQoRModel`, the V4 tensorizer, ranking loss,
> and one-time feature cache. Do not create a second trainer or rebuild the
> graph architecture. Keep numerical and hardware arms identical in source,
> candidates, numerical device/clock/budget context, initialization, sample
> sequence, optimizer, update count, and evaluation. The only model input
> difference is hardware-resource message edges. Freeze a new manifest before
> training. Bound CPU preparation at 10 minutes and total GPU work at 20
> minutes; stop if either limit is reached.
>
> A reasonable diagnostic is 100 updates per arm for two predeclared paired
> seeds (123 and 456), if the first preflight fits the limits. Use the same
> candidate sets per seed and arm. Do not tune on test data or select a favorable
> seed. Report each seed separately and paired aggregates. If 100 updates do
> not show useful learning, do not add loss balancing or architectural changes
> in the same experiment.
>
> Report latency/resource errors, per-context ranking, measured-pool top-1 and
> top-3 regret, and feasibility at 100% and 50% budgets, with explicit empty or
> undefined cases. For target conditioning, report same-assignment QoR and
> feasibility predictions across 3.33/5/10 ns and across devices; distinguish
> genuine measured changes from predicted changes. For budget changes, physical
> QoR for a fixed candidate/device/clock should remain the same target label;
> budget changes only feasibility. Include a hardware-edge ablation after
> training, but do not treat it as a substitute for the separately trained
> numerical arm. State that measured-pool ranking is not synthesis or global
> optimization.
>
> Do not claim clock generalization beyond represented clocks, unseen-device
> transfer, Stage-2 benefit, or production readiness. Push only validated
> research code and documentation to `research/gnn-refinement`; leave model
> weights and large caches outside Git.

## Why the first pilot used only 10 ns

The request was to replace an impractical thousands-candidate validation loop
with a genuinely inexpensive experiment. The pilot therefore held clock fixed
at 10 ns while varying the two devices. It had 5 training kernels × 2 devices
and 2 validation kernels × 2 devices: 10 training contexts, four validation
contexts, 80 training candidates and 32 validation candidates. Both arms
completed in 3.53 seconds on the A2 after a 248.84-second one-time preparation.

This choice bounded the pilot and let it test the hardware-message pathway
without simultaneously expanding the target-context grid. It also means the
pilot cannot answer whether the learned representations or predictions respond
appropriately to 3.33 ns versus 5 ns versus 10 ns. It tested device differences
at 10 ns only. The next experiment should explicitly include all six
device/clock combinations and paired measured designs where the data permits.

## Current evidence to carry forward

- Research branch commit after the pilot: `c39f600448f9d869b26590190f0ef6af9a32ff70`.
- Graph archive SHA256: `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
- Dataset SHA256: `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`.
- Family split SHA256: `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`.
- Pilot protocol SHA256: `a8e03450dd265432d1932ee8f0ea5e904545dea52128aa6d94f4101e1e212e90`.
- Full numeric results and limitations: [`RESULTS.md`](RESULTS.md).
- Implementation and data audit: [`AUDIT.md`](AUDIT.md).
- Fixed pilot cohort and summary: `results/fast_probe_s123_cohort_manifest.json` and `results/fast_probe_s123_summary.json`.

The 3.33 ns and 5 ns contexts exist in the authenticated dataset according to
the full data audit, but their complete shared-candidate coverage for this
selected cohort has not yet been checked. Candidate intersections and QoR
variation must be checked before freezing a follow-up manifest; do not assume
the requested paired design pool exists.
