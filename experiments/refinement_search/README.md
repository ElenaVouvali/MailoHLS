# Budgeted refinement with HLS feedback

One-pass SALT predicts a complete directive configuration with the frozen LLM,
objective adapter and compiler memory. This optional experiment starts from that
stored prediction and queries synthesis before choosing each next edit. It does
not run the LLM or GNN, change checkpoints, or replace standard inference.

The [authenticated R11 results](RESULTS.md) answer the current scientific question:
measured refinement helps GAN and covariance, but structural and hybrid
superiority is unproven. All 144 logical queries and the three final winners'
existing C-simulation passes were reauthenticated on 2026-11-01. No additional
synthesis, simulation, training or policy retuning was needed for that assessment.
The supplementary stricter audit below qualifies historical relation labels.

The search engine has two modules: `policies.py` selects candidates from source
facts and one policy's observed history; `controller.py` validates, queries,
records and summarizes them. `benchmark.py` seals and schedules controlled
comparisons using that same engine. Complete assignment handling, compiler
verification, source materialization, Vitis execution, ADP and measurement
authentication come from the existing [`refinement`](../refinement/README.md)
implementation. Historical code and prepared experiments remain unchanged.

## Policies fixed before measurement

| CLI name | Rule |
| --- | --- |
| `random` (P0) | Seeded random order over the legal single-action neighborhood of the current incumbent. |
| `coordinate` (P1) | Evaluate the next balanced single-action proposal; immediately regenerate around a measured improvement. |
| `compose` (P2) | Probe up to four single-action siblings at one reference. As soon as two distinct actions improve over that measured reference, propose their composition. |
| `structural` (P3) | Propose two changes on a verified compiler pair around the incumbent, even without beneficial individual results. |
| `hybrid` (P4) | Use composition and sibling probes, reserving logical queries 4, 8, 12 and 16 for verified structural exploration when available. Fall back to singles, then verified pairs when singles are exhausted. |

All methods measure the same original SALT configuration first and only install
a strictly better, feasible design. Ties keep the earlier incumbent. Failed and
infeasible queries consume budget without replacing it. An exhausted neighborhood
ends the search; a missing replay record is an unavailable observation, not a
synthesis failure or a fabricated objective.

Composition needs a common measured reference, so its four-probe sweep differs
from P1's immediate reanchoring. Its incumbent still updates immediately. A
composition is explicitly `reference+A+B`, with two previously observed distinct
action edits; it is not relabeled sequential coordinate search. Each current
action tuple must remain compatible with that reference or its observed edit.
Stale combinations and unrelated rollbacks are rejected. Joint effects are
measured, never assumed from the constituent gains.

Numerical proposals consider the nearest smaller and larger enabled value and a
disabled alternative. Directions alternate within numerical families, and
UNROLL, partition factor, PIPE, partition type, dimension and whole-tuple edits
are visited in round-robin order with seeded tie-breaking. This is a bounded
local neighborhood, not enumeration of every legal assignment. It never prefers
larger factors on the assumption of better QoR. Structural proposals prioritize
verified direct loop–array relationships and UNROLL/partition-factor edits,
using up to two alternatives per endpoint and round-robin pair exploration.
Conservative shared-helper relationships do not qualify. Graph disconnection
does not establish semantic independence.

New tuples follow the explicit TCAD policy: powers of two at most the resolved
size, plus all divisors at least two, with no independent 64/1024 cap. Array
factors depend on the selected dimension. Compiler-proven static loop counts,
inactive absolute action IDs and recorded baseline-only exceptions are reused
unchanged. Exceptions preserve historical designs; they cannot introduce new
illegal tuples. Each proposal records its domain classification and values newly
admitted compared with the historical policy.

## Logical budget and hidden physical cache

A **logical evaluation** reveals one chosen configuration's outcome to one
policy. The initial configuration costs one. A cached physical report still
costs one logical evaluation. A **physical synthesis** launches Vitis; exact
authenticated historical or newly shared results can avoid that launch.

The controller durably reserves a logical query **before** looking up its result.
Policies receive no cache signatures, historical winners, paper QoR, or other
policies' histories. Shared physical caching affects runtime, not available
policy knowledge. Compare best ADP at equal logical budgets, not physical jobs.
Physical costs depend on execution order; they are reported separately.

The initial validated request is ADP/ZCU104, `xczu7ev-ffvc1156-2-e`, 10 ns, full
BRAM/DSP/FF/LUT capacities and genuine Vitis HLS 2021.1 headers. Other objectives
or deployment contexts are rejected. Area is mean percentage utilization across
the four resources; ADP is area times latency cycles. Every resource budget is
checked separately. All reused and prospective results must match source,
metadata, materialized directives, Tcl, part, clock, tool/header identity and
report provenance. The historical GAN floating-point stub measurements cannot
be mixed with genuine Vitis-header results.

## Commands from the repository root

Use Python **3.8–3.11**, including the normal `mailohls-llm-v2` Python 3.10
environment locally and Python 3.8 on Coroni. Preparation also needs CPU
torch/networkx for the existing compiler verifier; run/resume and summarize need
the shared source modules but no model weights or GPU. Python 3.12+ changed
floating-point `sum` rounding: the historical verifier's exact numerical checks
can reject otherwise identical reports by the last bit. Keep its original
arithmetic environment rather than weakening authentication or rewriting results.
Strict replay also compares auxiliary proposal annotations. The completed Coroni
run used Python 3.8.0 and glibc 2.27; another host's `log2` can differ by one
floating-point bit even when the chosen configuration is identical. Use the
recorded execution host for byte-exact prospective trace replay and summaries.
Source/QoR authentication and local CPU tests do not relax their checks.

```bash
# Uses the existing authenticated local experiment and all five policies.
# This is also the default mode; it never launches synthesis.
python -B -m experiments.refinement_search.controller --mode prepare \
  --output-dir experiments/refinement/results/search/replay \
  --execution replay --kernels GAN covariance Kalman \
  --max-evaluations 8 --seed 123

python -B -m experiments.refinement_search.controller --mode replay \
  --output-dir experiments/refinement/results/search/replay

python -B -m experiments.refinement_search.controller --mode summarize \
  --output-dir experiments/refinement/results/search/replay

# Prepare a separate prospective pilot. No HLS runs yet.
python -B -m experiments.refinement_search.controller --mode prepare \
  --output-dir experiments/refinement/results/search/pilot \
  --execution synthesis --max-evaluations 8

# On Coroni, first select one kernel/policy and allow only three logical queries.
python3.8 -B -m experiments.refinement_search.controller --mode run \
  --output-dir experiments/refinement/results/search/pilot \
  --kernels GAN --policies coordinate --limit 3 \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

# Resume that search; --limit bounds additional queries, not its total budget.
python3.8 -B -m experiments.refinement_search.controller --mode run \
  --output-dir experiments/refinement/results/search/pilot \
  --kernels GAN --policies coordinate \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

# Explicitly opt into the full prepared batch only after the smoke test.
python3.8 -B -m experiments.refinement_search.controller --mode run \
  --output-dir experiments/refinement/results/search/pilot --allow-batch \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls
```

`--input-dir` selects the existing authenticated refinement experiment (default
`experiments/refinement/results/local_refinement`). Repeat `--measurements PATH`
to add compatible archives during preparation. Use a new output directory to
change policies, seed, code, context or budget. `--max-evaluations 16` enables the
larger comparison. `--timeout` bounds each Vitis job. To validate actual physical
execution instead of cache lookup, prepare a separate one-policy smoke directory
with `--fresh-synthesis`; this setting is sealed and forces a fresh job for each
new query. It does not repeat a completed query on resume.

A fresh Git checkout does not contain ignored measurement archives. First run
`python -m experiments.refinement.experiment --mode prepare` to create the source
and compiler preparation described in its README, or supply an existing sealed
archive with `--input-dir`. Authenticated measurements are optional for a
prospective run; replay without them will correctly report missing coverage.

Transfer the prepared output **at the same repository-relative path**, both
refinement packages, and the common modules listed in
`shared/manifest.json`'s `code_sha256` to Coroni. Preserve snapshot and proof
subdirectories, not just CSV files. A minimal bundle needs no checkpoints or
training datasets. Code/input hashes are checked before every run. Locks prevent
concurrent execution of one policy; a global lock protects shared physical
receipts. Physical reservations are durable. A reserved job without a final
`result.json` stops for log inspection instead of silently launching twice.

## Outputs and interpretation

All output is ignored under `experiments/refinement/results/search/`. One frozen
shared directory contains snapshots, verified relations and hidden authenticated
proofs. Each `KERNEL/POLICY/` has five main files:

- `manifest.json`: sealed source/context/code/policy/seed/budget.
- `search_trace.jsonl`: paid proposals, reference edits, incumbent edits,
  rationale, relation evidence, resource/latency/ADP outcomes and incumbent updates.
- `measurements.jsonl`: only that policy's revealed measurements and reuse costs.
- `summary.csv`: budget points 1, 2, 4, 8, 16 and separate logical/physical costs.
- `report.md`: observed trajectory and scientific limitations.

Shared prospective projects and durable physical attempt receipts are stored
under `physical/`; root `summary.csv` and `report.md` combine the isolated runs.
The controller recomputes policy decisions from journals on resume, checking that
each outcome follows its paid query. Completed physical results and saved
logical observations survive interrupted trace updates without duplicate jobs.

Replay deliberately continues after missing observations, retaining the best
known incumbent. If a budget prefix contains an unavailable outcome, its fair
`best_ADP` is blank; `best_observed_ADP` is only a partial observation. Such runs
validate the controller but cannot rank policies. Genuine neighborhood exhaustion
can carry forward the best result with an explicit early-termination label.
Paused or budget-limited runs do not extrapolate to larger budgets.

Report single-action, composition and structural improvements separately, along
with TCAD-expanded values. A joint improvement does not automatically establish
a compiler-guidance benefit. The small local neighborhoods, fixed deployment,
single seed and unreplicated HLS reports limit generalization. No functional
simulation/equivalence or post-route timing is established by these synthesis
runs. Controller overhead records per-query selection, authentication and
persistence time, excluding physical HLS runtime; preparation, report generation
and crash downtime are outside that counter.

Coordinate refinement is a conceptual baseline informed by
[AutoDSE](https://github.com/UCLA-VAST/AutoDSE); this experiment does not reproduce
its bottleneck analysis. [MPM-LLM4DSE](https://github.com/wslcccc/MPM-LLM4DSE) and
[Prometheus](https://github.com/UCLA-VAST/Prometheus) are future comparison points,
not components implemented here. No novelty claim follows from adding feedback.
[HARP](https://github.com/UCLA-VAST/HARP) and
[NLP-DSE](https://web.cs.ucla.edu/~pouget/papers/nlp_dse.pdf) motivate future
representation/ranking and analytical screening work; neither is integrated in
this controller. The supplied [Reflex-HLS DOI](https://doi.org/10.1145/3849869)
could not be fetched here. Its requested planning/materialization/feedback
principles are methodological guidance, not a claim of reproducing that system.

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_refinement_search.py tests/test_refinement.py tests/test_directive_domains.py
```

Tests use synthetic observations and a CPU test executable through the existing
HLS runner. They never require Vitis or expose hidden measurements to policies.

## Controlled matched-budget benchmark

The benchmark compares the five unchanged policies at 1, 2, 4 and 8 logical
evaluations. `composition` is accepted as an alias for the engine's `compose`.
One additional `matched` control explores legal pairs without a verified compiler
relationship. It matches structural templates in action types, changed field
families/counts and domain novelty, then minimizes the difference in log2 factor
magnitude. Magnitudes can differ; every score is recorded. These actions are
called `no_verified_relation`, never functionally independent.

The main three-kernel/one-seed pilot has at most **120 logical calls**. The matched
control has a separately declared maximum of **24**. The scheduler visits one
query per context in round-robin order. Each context still uses the original
controller, has its own budget/history, and cannot inspect the cache before its
paid query. All physical reuse is authenticated. A reserved job is recovered by
its owning engine, so interruption cannot turn a physical run into a free reuse.

```bash
# Freeze the protocol before synthesis; adopt the existing original R1 pilot.
python -B -m experiments.refinement_search.benchmark --mode prepare \
  --output-dir experiments/refinement/results/search/benchmark_r11 \
  --search-dir experiments/refinement/results/search/pilot \
  --execution synthesis --seeds 123 --max-evaluations 8 \
  --import-search experiments/refinement/results/search/smoke

# Compatible Vitis host: small wrapper smoke first (includes cached observations).
python3.8 -B -m experiments.refinement_search.benchmark --mode run \
  --output-dir experiments/refinement/results/search/benchmark_r11 \
  --kernels GAN --policies coordinate --limit 4 \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

# Explicit bounded batch. Repeat the same command to resume.
python3.8 -B -m experiments.refinement_search.benchmark --mode run \
  --output-dir experiments/refinement/results/search/benchmark_r11 --allow-batch \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

# Original execution host: strictly reconstruct histories and rebuild tables.
python3.8 -B -m experiments.refinement_search.benchmark --mode summarize \
  --output-dir experiments/refinement/results/search/benchmark_r11

# Prepare a separate replay protocol; no Vitis is launched.
python -B -m experiments.refinement_search.benchmark --mode prepare \
  --output-dir experiments/refinement/results/search/benchmark_replay \
  --execution replay --import-search experiments/refinement/results/search/smoke
python -B -m experiments.refinement_search.benchmark --mode replay \
  --output-dir experiments/refinement/results/search/benchmark_replay

python -B -m pytest -q -p no:cacheprovider \
  tests/test_refinement_benchmark.py tests/test_refinement_search.py \
  tests/test_refinement.py tests/test_directive_domains.py
```

Omit `--search-dir` for a fresh controlled experiment. `--seeds 123 456` prepares
additional predeclared seeds; seeds are never selected from final QoR. Changing
protocol settings or source code requires a new output directory.
`--groups main` or `--groups control` selects the declared comparison group;
`--limit` bounds additional logical queries across the selected contexts.

The original R1 manifests hashed every Python file in this directory. Adding
the benchmark would otherwise invalidate them. Its scoped adapter instead hashes
the **actual unchanged original engine files** plus all original compiler/domain
dependencies, with exactly the original file set. It separately seals the new
benchmark file in `protocol.json`. It does not trust expected hashes in place of
actual hashes or relax source/report authentication. Use the benchmark entry
point to adopt/resume those original queues. The original engine Python files
and historical manifests are not edited. New controlled preparations should also
use this entry point, keeping the wrapper and engine provenance separate.

The five main output files are `protocol.json`, `search_traces.jsonl`,
`measurements.jsonl`, `summary.csv` and `report.md`. Per-policy journals remain in
their original prepared locations. The protocol records those paths, source and
graph hashes, original engine hashes, tool identity, Git/worktree state, initial
journal hashes, scheduling/matching rules and separate accounting limits. PNG/PDF
curves use measured ADP divided by the original baseline; a cross-kernel mean is
shown only with complete kernel coverage. Missing replay prefixes stay blank.
Early exhaustion carries results forward explicitly. No larger-budget value is
extrapolated from a paused run.

The completed run includes `execution_environment.json` and a sealed
`optimization_curve_data.json`, exported after strict replay on Coroni and bound
to the exact trace/measurement hashes. This allows portable plotting with local
matplotlib without changing the frozen execution code or rounding its annotations:

```bash
python -B - <<'PY'
from experiments.refinement_search import benchmark as b, controller as ctl
from experiments.refinement import configurations as c
out = b.DEFAULT_OUTPUT
protocol = b.load_protocol(out)
data = ctl._read_sealed(out / 'optimization_curve_data.json')
assert data['protocol_seal'] == protocol['seal']
for path, digest in data['journal_sha256'].items():
    assert c.sha256(c.repo_path(path)) == digest
b._plot(out, protocol, data['curves'])
PY
```

For new runs, `--mode summarize --plot` works when matplotlib is installed in the
same numerical environment used for execution. In the completed audit, all 144
candidate choices and all source/QoR measurements also verified locally; only
three auxiliary distance annotations differed by one bit. No tolerance was added
to measured QoR or candidate legality. Additional authenticated historical cache
proofs are recorded under `historical_cache/`; cache availability can change
physical cost, while every queried result still costs one logical evaluation.

The report contrasts coordinate/compose, structural/matched and compose/hybrid.
The first contrast includes the sibling-sweep difference, so it is not a pure
composition-toggle ablation. Explicit composition increments are also recorded.
Compose is the hybrid's exploitation-only counterpart. Existing balanced factor
ordering is retained, with initial pool directions sealed and actual query
directions counted. It does not favor only increases; lower-first resets can
still bias short trajectories. None of these rules are retuned after synthesis.
Positive gains below **0.5%** are flagged as small and physically unvalidated;
the threshold is a reporting convention, not a significance test.

### Functional checks outside the synthesis-selection budget

Each aggregate measurement has `validation_status=not_tested` until a matching
executed receipt exists. Once complete results identify finalists, prepare one
original-source versus final-design C-simulation comparison per kernel:

```bash
python -B -m experiments.refinement_search.benchmark --mode validate \
  --output-dir experiments/refinement/results/search/benchmark_r11

# Explicitly execute the prepared checks on the Vitis host.
python3.8 -B -m experiments.refinement_search.benchmark --mode validate \
  --output-dir experiments/refinement/results/search/benchmark_r11 --execute-csim \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls
```

Generated testbenches use identical zero, alternating and seed-123 bounded inputs
for the original source and candidate. GAN checks all outputs, covariance checks
outputs and its mutated input arrays, and Kalman checks all packed outputs.
Tolerances are `1e-6 + 1e-6*abs(reference)` for float outputs and `1e-9 +
1e-9*abs(reference)` for doubles; nonfinite results fail. Genuine vendor headers
are required and simplified analysis stubs are excluded. Logs, source/testbench
hashes, output hashes, return codes, comparisons and runtime are recorded under
`functional/`, independently of selection-time evaluation costs.
`csim_pass` means agreement on those finite inputs; it does not verify generated
RTL or establish post-route improvement. No equivalence checker is integrated,
so `equivalence_verified` is never assigned by this benchmark.

## Authenticated R11 assessment

The scientific comparisons are H1: refinement versus the initial SALT assignment;
H2: structural versus matched; H3: coordinate versus compose; H4: compose versus
hybrid. Incumbent monotonicity follows from best-feasible selection and is not
evidence of a superior policy. Each context has the same original assignment,
legal space, target, tool/header identities and eight-query budget. Query one is
the baseline, and cached, failed or infeasible queries still consume a query.

For kernel `k`, policy `p` and paid prefix `b`, define `J*(k,p,b)` as the minimum
ADP among successful feasible observations in the prefix, then
`N(k,p,b) = J*(k,p,b) / ADP(original SALT,k)` and improvement `100*(1-N)`.
The area/ADP formulas and four independent resource checks above are unchanged.
Normalization is undefined for unavailable/infeasible baselines. Missing replay
observations invalidate prefixes; pauses never extrapolate. Explicit completely
observed neighborhood exhaustion may carry an incumbent forward.

The separate [`experiments/refinement_analysis.py`](../refinement_analysis.py)
strictly replays the paid journals, checks measured baseline equality and aggregate
exports, reconstructs normalized curves/costs, computes only valid common-reference
factorials and authenticates executed functional checks. It never launches
synthesis or simulation. Its optional live-tool check uses only the existing
version/identity probe. Its location is outside the directories globbed into the
original core hash contract; frozen engine and benchmark files remain unchanged.

Restore the ignored evidence roots from
`/home/bonus24/mailohls_search_20261026_NqbK0V/` on Coroni at the recorded relative
paths: `benchmark_r11`, `pilot`, `smoke`, including all copied reuse proofs.
The optional covariance single-edit control also needs `local_refinement`.
[RESULTS.md](RESULTS.md) records recovery locations and source/tool hashes.
On the original Python/glibc environment, select a fresh assessment file outside
the frozen archive:

```bash
python3.8 -B -m experiments.refinement_analysis \
  --input-dir experiments/refinement/results/search/benchmark_r11 \
  --output /tmp/r11-assessment-fresh.json \
  --live-tool /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls
```

Portable verification checks the committed Coroni assessment seal and restored
input hashes without claiming exact foreign-libc policy replay:

```bash
python -B -m experiments.refinement_analysis \
  --verify-export experiments/refinement_search/evidence/r11/assessment.json \
  --compiler-output /tmp/r11-compiler-diagnostic-fresh.json \
  --tables-dir /tmp/r11-tables --plot
```

Compiler diagnostics need CPU torch/networkx; plotting needs matplotlib. No model
or GPU is used. The existing plot exporter reads the original sealed curve data.
CSV exports preserve raw/normalized ADP, actual budgets, candidate identities,
reference/baseline changes, matching distances, domain sizes and physical costs.

The optional historical single-edit control is authenticated using
`experiments.refinement.experiment.read_measurements` on `local_refinement`,
selecting covariance signature
`ccf8badf9fca12b41c2237797e636099c9c5c32a3e9dd9048d41f92a7a61769b`.
All 33 archived observations authenticate; the selected single-edit result has
identical comparison context and hardware metrics to the R11 structural winner.
Its separate sealed receipt is retained with the assessment exports.

### Actual conclusions and evidence limitations

R11 completed 120 main-policy plus 24 control calls: 86 new syntheses, 58 reuses,
99 distinct assignments, 137 feasible observations and seven BRAM-infeasible
observations. No query is pending or unavailable. All selected winners passed the
original finite C-simulation checks. The measured decision is
`REFINEMENT_BENEFICIAL_BUT_STRUCTURAL_GAIN_UNPROVEN`.

Best ADP improvements are GAN 42.694281% (matched), covariance 28.754456%
(compose), Kalman 0.141797% (matched). Structural wins one of three matched
contrasts. Hybrid wins GAN modestly, loses substantially on covariance, and ties
Kalman. Only 0.453627 baseline percentage points of covariance's gain came from
its explicit composition. Complete curves and physical interpretation are in
[RESULTS.md](RESULTS.md); the three-kernel/one-seed result is not universal.

All 21 paid matched controls have zero recorded magnitude distance, but endpoint
legal action-tuple counts differ in 17 cases. Equal action kinds/field changes
and novelty do not imply equal local opportunity or causal isolation.

The separate diagnostic reexecutes the historical graph verifier, then the
existing `strict_callsite_and_access_v1` auditor on the same snapshots. Nine GAN
array–loop relations become unresolved when call summaries are excluded as
concrete-access proof. These include all GAN structural/hybrid pairs. Seven
Kalman pairs instead gain direct evidence under the native write-orientation
correction; three nonimproving historical matched queries use those pairs.
The control means no relation verified by R11's verifier, never semantic
independence. No historical eligibility, relation, trace or QoR is rewritten.
A stronger verifier requires a new sealed experiment.

Original relation facts reproduce. Dependence-edge list iteration may vary with
Python hash randomization; the supplementary diagnostic compares full edge
multisets including multiplicity. This does not alter exact original policy
replay or relax source/QoR authentication.

The assessment verifies original source, materialized candidate, testbench/plan,
header/tool identity, completion logs, return codes, output hashes, lengths and
finite values, then reruns the binary-output comparator. Four prior design
checks cover the three winners and an earlier GAN finalist, with zero error and
mismatches. No new Vitis job is needed. Finite C simulation establishes neither
exhaustive equivalence nor RTL correctness.

Per-query overhead combines selection, authentication and persistence and excludes
HLS runtime. Separate subphase timers and full end-to-end elapsed time were not
recorded and remain unavailable. Physical counts depend on cache and order. The
recorded-unique-candidate runtime sum is only a retrospective cold-cache workload
proxy using historical runtimes, not a new physical-speed comparison.

## Design Decisions and Rationale

| Decision | Alternatives considered | Rationale | Scientific consequence | Validation |
| --- | --- | --- | --- | --- |
| Authenticate existing R11 before new execution | Rerun benchmark; add an algorithm | All required evidence exists | Zero new synthesis; historical measurements remain authoritative | Exact Coroni replay: 18 contexts / 144 queries |
| Separate reporter; preserve frozen source | Modify sealed summarizer | Hash contracts protect historical decisions | No candidate-selection change | Frozen hashes; original 108-test baseline |
| Equal logical budgets | Equal realized physical counts | Cache/order changes execution cost without changing paid knowledge | Physical cost is secondary | Existing accounting/cache-isolation tests |
| Preserve matching; expose domain sizes | Retune matching after results | Avoid post hoc optimization and expose confounding | Zero distance is not causal isolation | Exact replay and cardinality exports |
| Separate stronger-verifier diagnostic | Relabel or repair R11 eligibility | Call-summary evidence limits historical labels | A changed verifier needs a new experiment | Existing strict auditor; edge-fact/multiplicity tamper tests |
| Common-reference feasible factorials only | Combine unrelated states; add single gains | ADP is not additive | Explicit composition gain separated from search ordering | New context/feasibility regressions |
| Preserve ordering and domain policy | Retune from seed-123 winners | Avoid post hoc selection | Direction/expansion effects remain limitations | Actual reference-edit and domain audit |
| Reauthenticate executed C simulations | Repeat identical completed checks | Matching passes exist for every winner | Functional evidence without extra runtime | Source/testbench/log/binary checks; original mismatch/nonfinite/dimension tests |
| Leave missing costs/prefixes undefined | Estimate timers; carry paused results | Avoid invented precision/coverage | Incomplete curves cannot rank policies | Baseline/prefix/exhaustion/aggregate regressions |
| No extra seeds or learned selector now | Two full seeds, LLM agent, RL/ranker | Current bounded conclusion supported; evidence quality needs attention first | Three-kernel/one-seed limitation remains | Complete gates; resource preflight; no model loading |

### Implementation change log

The implementation commit is recorded below after validation. The later
documentation commit is discoverable with `git log --follow` for this file;
a commit cannot embed its own content-derived SHA.

| Commit SHA | Modified files | Category / reason | Scientific contract affected | Tests | Experiment / evidence |
| --- | --- | --- | --- | --- | --- |
| `b9687e59dce518c9735e18e4ee59b7d09425946b` | `experiments/refinement_analysis.py`, `tests/test_refinement_analysis.py`, `evidence/r11/*` | Reporting improvement and verifier-sensitivity evaluation; versioned reproducible assessment and explicit evidence-quality analysis were missing | No search-policy, source, metric, budget, historical protocol or functional threshold change | Before: 108 passed. After: 143 passed / 1 prerequisite skip, including 14 new tests | R11 and authenticated historical covariance single-edit control; no new synthesis or C simulation |

The new reporter adds cross-policy/export consistency checks, normalized coverage
semantics, common-reference composition calculations, actual cost and matching
diagnostics, functional evidence verification and compact exports. Alternatives
were changing the frozen summarizer or relying on ignored reports alone. The
separate module preserves comparability and rejects changed code, inputs, seals,
QoR and functional evidence. This is a reporting extension plus supplementary
evaluation, not a search-policy change or new synthesis experiment.

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_refinement_benchmark.py tests/test_refinement_search.py \
  tests/test_refinement.py tests/test_directive_domains.py \
  tests/test_refinement_analysis.py tests/test_structural_audit.py
git diff --check
```

The existing optional saved-memory/encoder-artifact test skips when its local
research tensors are absent. No threshold was changed to obtain passing tests.

### Related work and scope

[AutoDSE](https://arxiv.org/html/2009.14381v2) uses bottleneck-guided coordinate
optimization, parameter ordering and space partitioning. R11's coordinate policy
is a conceptual comparator, not a reproduction of that full system.
[MPM-LLM4DSE](https://arxiv.org/html/2601.04801v1) combines multimodal QoR prediction
and LLM optimization; R11 uses deterministic selection and actual HLS feedback.
[HARP](https://github.com/UCLA-VAST/HARP) studies hierarchical GNN representations
and surrogate-assisted exploration. [NLP-DSE](https://escholarship.org/content/qt072909j7/qt072909j7_noSplash_23f734e0aee37780cda8b8403d29c93c.pdf)
uses analytical performance/resource models and constrained pragma optimization.
These are conceptual comparisons; none is reproduced under R11's protocol.

The supplied characterization of [Reflex-HLS](https://doi.org/10.1145/3849869)
describes a training-free Planner–Actor–Critic feedback loop. Publisher full text
was inaccessible during this assessment, so no independently verified detailed
reproduction or numerical comparison is claimed. Feedback alone is not novel.
The possible contribution is whether verified action relationships improve
strict-budget refinement from an LLM starting point; R11 does not establish that
general advantage.
