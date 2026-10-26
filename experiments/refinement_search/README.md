# Budgeted refinement with HLS feedback

One-pass SALT predicts a complete directive configuration with the frozen LLM,
objective adapter and compiler memory. This optional experiment starts from that
stored prediction and queries synthesis before choosing each next edit. It does
not run the LLM or GNN, change checkpoints, or replace standard inference.

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
