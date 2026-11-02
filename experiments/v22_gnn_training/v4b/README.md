# V4b: full-cohort action-aligned preference learning

V4b is an isolated experiment. It does not change V1 production memory, V4
checkpoints, Stage-2/Stage-3, or the frozen MLIR V2.2 RC3 graph release.

## Scientific controls

The planned controls share the same V4 tensor schema, candidate assignments,
physical labels, and ADP preference construction:

| Arm | Structural input | Candidate decision path |
|---|---|---|
| Candidate/target control | No learned structural-memory encoder; common candidate and analytical-demand inputs plus physical target | Capacity-matched candidate/demand MLP |
| Compiler V4 | V4 `SmallGraphEncoder`, hardware-message edges removed | Pointwise QoR and budget-aware preference heads |
| Target-FiLM V4 | Compiler memory modulated by authenticated physical profile/capability and clock; zero-initialized gate | Same pointwise QoR and preference heads |
| Hardware V4 | Same encoder and parameterization, hardware edges retained | Same pointwise and preference heads |
| Compiler V4 difference | Compiler-only V4 encoder | Antisymmetric action-difference attention plus pointwise QoR |

Graph-based arms preserve two `TransformerConv` layers, width 64, two heads,
128D action memories, and 64 absolute action slots. The encoder API accepts only
the program/physical-target graph. Candidate directives, candidate demand,
requested budgets, objective identities, QoR labels, and preference labels do
not enter structural memory. Graphs must be tensorized at full physical device
capacity; the physical QoR head has no budget argument. Feasibility is computed
after prediction against `budget_fraction * physical_capacity`.

The five-arm comparison comprises the no-learned-memory candidate/target
control, compiler-only V4, target-FiLM V4, hardware-message V4, and compiler
V4 with action-difference comparison. The comparison head's candidate-specific action vectors combine an action
memory with that candidate's directive tuple and analytical demand. Attention
is computed over the absolute slots whose candidate representations differ.
Its signed projection is antisymmetric: swapping candidates negates the score,
and identical candidates yield exactly zero. Pool ranking uses deterministic
net wins; cycles are possible and ties use stable candidate hashes.

The hybrid loss is training-normalized physical-QoR SmoothL1 plus a fixed 0.1
pairwise logistic preference term. The term treats feasible candidates as
preferred to infeasible ones, compares ADP only among feasible candidates, and
excludes near ties under the established 5% convention. This is comparative
supervision, not DPO.

## Data contract and audit

Use the authenticated seed-123 family split and RC3 release. The strict
complete-assignment cohort is all eligible training kernels and eligible
validation kernels; Kalman is excluded. Candidate pools default to 16 per
kernel/device/clock context, selected by target support and directive Hamming
diversity without inspecting QoR. Target-response coverage uses exact complete
assignment-hash intersections from the entire eligible pool, not just the
selected 16. Budget views reuse the same physical measurements and are
correlated decision contexts.

Excluded actions with no measured APL/CSV column remain unknown. V4b does not
replace their values with disabled pragmas or preprocessing defaults. The
Viterbi/U200/5 ns zero-area ambiguity is retained in source data and excluded
from ADP winner ordering when it can determine the winner.

The authenticated full-cohort audit is complete. It reconstructs 55 original
kernels (50 train, four validation, one test), 150,026 aggregated design
points and 104,147 strict-V4 eligible rows before the test reservation. The
eligible non-test data comprise 88,466 training and 8,976 validation rows
across 37 training kernels/17 families and three validation kernels/two
families. The remaining 14 non-Kalman kernels have one or more registered
actions without measured APL/CSV fields. Their values remain unknown; no
authenticated alternate value mapping or fixed-state proof was found. They are
not imputed. The manifest and per-action exclusions are in `receipts/`.

The full exact-assignment target audit examined 360 kernel/target-pair rows.
Device comparisons were paired for 104/120 rows (3,605 overlapping exact
assignment pairs); clock comparisons were paired for 209/240 rows (9,733
overlapping assignment pairs). At the primary 5% winner-margin rule, only
two device-pair contexts at full budget and zero clock-pair contexts had a
meaningful ADP winner reversal. Across all five budgets, reversals appeared
in six device-pair contexts spanning three training families and one clock
context in one training family. Validation had no meaningful device or clock
winner reversal. These overlaps are correlated and do not establish broad
target-adaptive generalization.

The five-budget audit found 98 winner changes in 2,400 budget-pair rows, but
only eight rows met the 5% meaningful-margin rule; those eight rows reduce to
two unique kernel/device/clock contexts in two families. The 16-candidate
QoR-blind manifest covers 240 physical target contexts (222 train, 18
validation) with 3,664 selected observations; context pools contain 7–16
assignments. Preference pairs are audited separately by context/budget, with
both-infeasible and near-tie pairs excluded and feasible-over-infeasible pairs
retained.

Full receipts and SHA256 identities:

- `receipts/eligibility.json`
- `receipts/preference_cohort_manifest.json`
- `receipts/preference_pair_audit.json`
- `receipts/preference_target_coverage.json`

The full exact-assignment audit uses all eligible assignments. It is distinct
from the 16-candidate QoR-blind pool manifest used for a future matched model
comparison.

Example CPU audit command (use a new output directory):

```bash
PYTHONPATH=. python -m experiments.v22_gnn_training.v4b.audit \
  --release-dir /path/to/authenticated-rc3-release \
  --authenticated-audit experiments/v22_gnn_training/data-audit.json \
  --output-dir /durable/path/v4b-audit-<unique-id> \
  --candidates-per-context 16
```

The output contains eligibility, pair-identity, candidate-pool, and exact
target/budget coverage receipts. Do not use Kalman for training,
normalization, validation, candidate selection, or model selection.

## Current execution status

The V4b encoder controls, zero-gated Target-FiLM adapter, antisymmetric
action-difference comparator,
budget-invariant physical-QoR head, feasibility calculation, pair construction
and audit utilities are implemented with focused CPU tests. A CUDA-only
five-arm matched runner is provided as `train_probe.py`; it refuses to start without an
available GPU and loads only the manifest's 37 train/three validation kernels.
The local sandbox could not communicate with the NVIDIA driver and its pinned
PyTorch process reported CUDA unavailable. Direct host inspection after the
user requested V100S found it idle and CUDA
available (the A2 was occupied by an unrelated Ray worker). A one-update
five-arm smoke then spent 31 minutes in CPU preparation before reaching any
GPU operation and was gracefully stopped at the 30-minute cap. Inspection
identified that `PreparedData.candidate_inputs()` re-enriched the same
authenticated graph separately for each assignment. V4b now caches one
overlay per kernel and candidate vector/demand per complete assignment across
physical targets; CPU regressions and one authentic cross-clock example check
exact equality against the existing path. This optimized path has not yet been
timed on the full cohort.
The smoke is preserved at
`/home/elvouvali/mailohls-gnn-runs/v4b-fullcohort-v100s-20261102-smoke01`
with no checkpoint. The audit and smoke do not constitute a learning result;
the previous V4 metrics remain historical and are not V4b outcomes.

The default target is the isolated A2; there is no automatic fallback. For
the current user-authorized V100S attempt, select it explicitly by its verified
UUID after confirming that no process is using it. The script checks
`nvidia-smi`, device name/UUID, active compute processes, free memory,
utilization, and UUID-based `CUDA_VISIBLE_DEVICES`; it refuses to share either
GPU. The historical V4 protocol's V100S reservation remains in force outside
this explicit V4b request.

```bash
CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb \
SALT_HLS_GPU_UUID=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb \
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.v4b.train_probe \
  --release-dir /path/to/authenticated-rc3-release \
  --authenticated-audit experiments/v22_gnn_training/data-audit.json \
  --manifest experiments/v22_gnn_training/v4b/receipts/preference_cohort_manifest.json \
  --output /durable/path/v4b-run-<unique-id> \
  --gpu-name V100S --gpu-uuid GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb \
  --updates 1 --validate-every 1 --seed 123
```

After that smoke succeeds, repeat with a different new output path and
`--updates 1500 --validate-every 250`. The smoke is not a model result and
must not be merged into the primary training comparison.

The run receipt will use a single family-balanced pair trace shared by all
five arms. It selects checkpoints using validation feasible-selection failure
plus feasible top-1 ADP regret, with preference accuracy only as a tie-break.
The current runner is an experimental initial implementation; pairwise
selection uses deterministic net wins and can be nontransitive. Candidate
memory/pathway ablations and Stage-2 integration remain separate follow-up
work. A complete GPU evaluation is required before scientific benefit can be
claimed.

The action-difference design follows the high-level comparative-learning idea
in [CompareXplore](https://github.com/NVlabs/CompareXplore): combine pointwise
physical prediction with pairwise comparison focused on candidate differences.
V4b changes the comparison unit to SALT's source-authenticated, absolute action
slots and enforces candidate-independent memory and pair antisymmetry. No
CompareXplore code is copied.

## Stage-2 boundary

Even a successful V4b measured-pool result would not establish that its memory
works with production V1-trained cross-attention. No V4b memory is exported or
integrated. A later Stage-2 experiment must retrain/adapt the cross-attention
under the `[64,128]` action-aligned, target-conditioned memory contract and
compare it against V1 memory and prompt-only conditioning.

## Design decisions and current limitations

| Decision | Alternatives considered | Reason and scientific effect | Validation |
|---|---|---|---|
| Keep the RC3 graph and V4 encoder frozen | Add resource node types/relations or increase depth | Historical full-cohort eligibility and V4 audits do not establish an encoder defect; holding this fixed isolates memory conditioning and candidate comparison | Existing graph/tensor tests plus V4b output shape and edge ablation tests |
| Use 37/3 strict complete-assignment train/validation kernels | Impute missing registered actions or mix recovered subsets | Fourteen exclusions contain unknown action states; defaults would fabricate assignments. The fixed cohort maximizes the already authenticated full assignments without cross-family leakage | Full eligibility receipt; no test kernels in V4b records |
| Use a support-first Hamming-diverse 16-candidate pool | Select validation winners or require all six targets | QoR-independent selection avoids winner leakage and exact target intersections remain separately auditable; contexts with fewer than 16 examples remain smaller | Frozen manifest hashes and deterministic selection test |
| Separate physical QoR from requested feasibility budget | Preserve budget-conditioned physical predictions | The budget is a deployment limit, not a synthesis input. Resource feasibility is computed after QoR prediction from predicted counts and physical capacities | Full-capacity graph guard and exact QoR budget-invariance test |
| Add zero-gated profile FiLM as a simple target-memory arm | Add a richer FPGA graph or device-ID embedding | Profile capabilities and knownness masks exist in the authenticated registry; FiLM tests whether these physical facts can modulate action memory without a larger graph or ID shortcut | Zero-gate equality and finite-gate-gradient tests; full run not yet executed |
| Prefer complete-configuration ADP pairs as the learning target | Make absolute QoR the primary objective or train independent directive labels | The research goal is directive choice. Pair labels respect feasibility and near ties; physical QoR remains an auxiliary hybrid loss | Pair direction, feasibility, tie and ambiguous-zero tests |
| Compare five controls under one pair trace | Compare unmatched historical checkpoints | Candidate/target control, compiler, target-FiLM, hardware-message and action-difference arms isolate candidate-only, compiler-memory, modulation, relation-message and comparison-head effects | Matched initial graph parameter and deterministic pair-trace tests |
| Keep a no-learned-memory control capacity-near the graph models | Treat it as graph-free | It still shares the authenticated analytical candidate-demand descriptors and action masks, but has no learned GNN memory; it measures incremental learned-memory value beyond those common features | Parameter counts and shared-input contract; interpretation is explicitly limited |
| Defer Stage-2 and richer FPGA graph | Export V4b memory or add more topology | No V4b selection result exists and validation has no meaningful device/clock winner reversal. Neither downstream value nor additional topology is supported yet | Stage-2 contract remains read-only; no production artifact is modified |

The target-FiLM profile feature vector reuses the existing authenticated
capability-vector utility, including per-field knownness and requested clock;
no bare device-ID embedding is introduced. The target-FiLM gate starts at
zero, so its initial action memory exactly matches the compiler encoder. The
candidate/target control does use the common analytical demand descriptors,
which are compiler-derived; it is best interpreted as **no learned graph
memory**, not an input representation with no compiler information.

The audit shows the data contain some device and budget preference changes,
but almost all meaningful reversals are in training families. The validation
set cannot demonstrate target-conditioned winner adaptation. Therefore a
complete V4b GPU result on this split can compare generic held-out-family
selection, but it cannot substantiate broad device/clock preference transfer.
A later target-conditioning claim needs additional family-disjoint validation
measurements containing meaningful matched winner changes.
