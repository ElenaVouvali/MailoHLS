# V4b v2 execution and scientific status

The full-cohort data audit is reproduced. The revised runner fixes unexecuted
V4b evaluation, device-transfer and resumption defects. GPU evaluation is
pending; no new selection advantage is established.

## Authentication and coverage

Local and remote `research/gnn-refinement` HEAD initially matched
`aa8e4fa7c8a15c315149586dc2abf9de8e580d88`. The supplied workspace was detached
at `9a4533a` with substantial unrelated deletions; work used the existing branch
worktree `/tmp/MailoHLS-gnn-refinement`. Five existing dirty report/receipt edits
were preserved and excluded from this change's commit.

`receipts_v2/authentication.json` reconstructs all 54 non-Kalman kernels and
143,321 rows, checking source/APL/CSV/graph hashes. The original 55-kernel,
150,026-row inventory is reconciled using the historical authenticated test
count, without parsing Kalman labels. Strict usable data remain 37 training
kernels and three validation kernels: 88,466 + 8,976 = 97,442 rows, 17 + 2
families. Including the reserved historical test count gives 104,147 strict
records; none of those test observations enters V4b.

Fourteen non-test kernels have missing measured fields for registered actions.
The authentication receipt includes exact action IDs, source lines and APL
text per exclusion. Source placeholders do not prove an applied directive
state. No alternate mapping is registered in the authenticated selected
contract. Recovery requires authenticated configuration traces or separately
versioned new measurements. No B-class fixed-state recovery is admitted.

The recomputed QoR-blind pools exactly match the frozen manifest: 240 physical
contexts and 3,664 selected observations. Full exact intersections reproduce
the original coverage audit: 104/120 supported device pairs (3,605 matching
assignments) and 209/240 clock pairs (9,733 matching assignments). At 5%, two
full-budget device contexts and zero clock contexts have meaningful winner
changes. Across budgets, six device target-pair rows in three training families
and one clock row in one training family qualify. Held-out families have none.
Eight meaningful budget-pair rows reduce to two physical contexts in two
training families; validation has no meaningful budget winner transitions.
These counts are overlapping observations, not independent samples.

Threshold sensitivity is explicit in `receipts_v2/coverage_verification.json`:
at 1%, validation has two device target-pair rows in Viterbi; at 5% and 10%,
validation has none. The primary threshold stays 5%. The expanded
`receipts_v2/full_audit.json` includes resource-level physical deltas and
winner sensitivity. Zero-resource Viterbi/U200/5 ns ambiguity is excluded from
ADP ordering, with original measurements retained.

## Implementation validation

The v2 evaluator handles the dictionary-based difference arm, covers contexts
with no meaningful pairs, and selects checkpoints by family-macro feasible
selection failure plus regret. It reports candidate-level feasibility
confusion, measured-feasibility oracle filtering, physical prediction errors,
exact matched physical delta errors, ranking correlation, and memory ablations.
Budget-invariant predictions are asserted in every evaluation. CPU caches stay
on CPU during model execution.

Optimizer/model/RNG/sample-position states are atomic and protocol-bound.
The CPU interruption/resume regression exactly matches uninterrupted training.
Completed runs are rejected on resume; completed arms are retained when
resuming a partially completed multi-arm run. Parameters, gradient norms,
training sample identities, family coverage and peak GPU allocation are recorded.
The same fixed family-balanced schedule is shared by all controls.

See `PROTOCOL_V2.md` for frozen optimization, selection, tie and ablation rules.
The 1,500-update budget is an initial controlled budget, not evidence of
convergence. The no-graph control shares analytical compiler-derived candidate
descriptors, so its comparison isolates learned graph memory.

## Required scientific answers

1. **Usable kernels:** 40 of 54 non-Kalman kernels, because all registered
   candidate actions have authenticated measured values and source-legal tuples.
2. **Recover excluded kernels:** none proven recoverable under the present
   authenticated contract; unknown values are not defaulted.
3. **Device/clock reversals:** sparse meaningful changes exist in training;
   the held-out split has none at the primary 5% threshold.
4. **Improvement from 37 kernels:** not established until matched training and
   held-out evaluation complete. The historical 14-kernel run is an unmatched
   experiment; a direct causal cohort-size claim would require a matched
   14-kernel v2 sensitivity control.
5. **Graph versus independently trained no-graph:** pending measured results.
6. **Additional hardware-message benefit:** pending measured results.
7. **Action-difference benefit:** pending measured results.
8. **Budget response:** physical invariance and budget-sensitive feasibility
   are tested; learned meaningful held-out winner adaptation is unidentifiable
   on this split, which has zero qualifying transitions.
9. **Alignment useful:** not established by gradients, nonzero memory or
   attention. Held-out ablation regret and controlled perturbations are required.
10. **Bottleneck:** target-preference coverage is an established limitation;
    learning/encoder/comparison bottlenecks remain unresolved without results.

Current classification: **IMPLEMENTATION_VALIDATED_GPU_EVALUATION_PENDING**.
No V1/V4 production memory, RC3 release, Stage-2/3 or synthesis controller is
changed. No new synthesis or Kalman test evaluation is authorized or performed.
