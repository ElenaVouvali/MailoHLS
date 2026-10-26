# V2.1 structure-guided refinement validation

This experiment asks whether corrected compiler relationships and a small
structural ordering rule help choose legal directive edits. It imports the
existing refinement controller, TCAD domains, configuration validator, synthesis
runner and authenticated measurement cache. It does not run SALT, a GNN or a new
synthesis engine. Production V1 graphs and checkpoints remain unchanged.

Compiler facts, proposal rationale and measured hardware quality are distinct:
V2.1 describes structure, TCAD supplies legal values, and Vitis measures quality.
An absent proven relation does not establish independence.

## Corpus

From the repository root, in the project Python environment:

```bash
python -B -m GNN_branch.generate_mlir_v2_corpus
```

The driver reads `ALL_KERNEL` with AST instead of importing training config,
resolves sources using `Data/ApplicationInformation.csv`, and checks external
workloads under `evaluation_set/`. Identical source/metadata/header snapshots
collapse while their aliases and known split memberships remain recorded.
Different action metadata or headers preserve different snapshots. The
index-only Stage-1/2 split is sealed but is not guessed into kernel membership;
saved kernel-name split manifests supply that information where available.

The existing native builder is invoked with **`--revision 2.1`**. The schema must
be `salt-hls-mlir-structural-v2.1`. Exact source, metadata, local header, frontend,
analysis-extension and builder hashes are checked before reuse. Every completed
graph passes existing validation, exact compact reconstruction, all typed
relationship queries, derived views and loop-semantic query comparisons.
Source-only labels do not become active directive slots. Unknown targets are
not called compiler-eliminated objects.

Defaults use `~/tools/Polygeist/build-mailohls-assertions` and
`~/.mlir-python311/bin/python`; override with `--build` and `--native-python`.
Use `--only GAN covariance serrano-kalman-filter` for a subset. Rerunning resumes
and revalidates completed records. `--retry-blocked` retries failures; changed
code/input/tool contracts require a fresh `--output-dir`. `--reuse-dir` can
point to a previous canonical directory to avoid recompiling authenticated
artifacts. Each failure is isolated, with a log and exact builder command.

Outputs under the ignored `GNN_branch/MLIR_graphs/v21_corpus/` include the corpus
manifest, canonical/compact graphs, raw IR, generation and coverage CSVs. No
V1 graph or sealed historical comparison is overwritten. Serialized compact
files retain reconstruction evidence and need not be smaller than canonical
files even when their topology is smaller.

## Frozen four-policy comparison

| Arm / CLI name | What changes |
|---|---|
| P0 `coordinate` | Existing balanced greedy single-action search. |
| P1 `structural` | Existing V1-derived verified pair policy. |
| P2 `v21_relations` | Replace pair eligibility with proven V2.1 access, hierarchy and dependence evidence. |
| P3 `v21_priority` | Use V2.1 compiler context for action ordering, with pair exploration at logical evaluations 4 and 8. |

P2 preserves P1's shared-pair ordering bucket, seeded tie, endpoint alternatives
and pair round-robin. Actual V2.1 direct/through-call evidence is retained,
independently of that ordering bucket. Newly eligible pairs use the same ordering
rule. P2 does not change candidate values or require successful individual edits.

P3 uses a lexicographic rule, without a weighted QoR score: compiler-known
repeated work first; no **proven** carrying dependence first; larger proven trip
count; more local/nested arithmetic operations; shallower nesting; seeded tie.
Unknown dependence is not independence, and dependence does not prohibit edits.
Arithmetic inside helpers is not counted as caller-local arithmetic. Families
still rotate and each action retains the existing nearest lower/upper/disable
value order. Pair slots fall back to singles when no eligible pair remains.
This is an exploratory heuristic, not a predicted hardware improvement.

All arms start from the complete original SALT ADP assignment. Unchanged
historical exceptions and compiler-inactive actions retain the existing
contracts. Kalman's 13-action synthesis metadata is explicitly mapped to its
11-active-action compiler metadata; L7/L8 remain frozen. Exact compiler trip
counts must agree between arms, otherwise preparation blocks the comparison.

Prepare without launching HLS:

```bash
python -B -m experiments.v21_refinement.experiment --mode prepare \
  --corpus GNN_branch/MLIR_graphs/v21_corpus \
  --output-dir experiments/refinement/results/v21_refinement/replay \
  --cache-search experiments/refinement/results/search/pilot \
  --cache-search experiments/refinement/results/search/benchmark_r11/control_seed_123 \
  --cache-search experiments/refinement/results/search/smoke
python -B -m experiments.v21_refinement.experiment --mode replay \
  --output-dir experiments/refinement/results/v21_refinement/replay
```

Before reading measurement archives, preparation seals source/graph identities,
policy code/rules, seed 123, budget, complete initial candidate pools, and
compiler-grounded action reports. Historical known edits are inspected only in
a retrospective report after this seal. The default neighborhood proposes
nearest values; a previously useful factor can therefore be absent from its
initial pool. Policies reanchor using only their own revealed measurements.

Physical caches remain hidden until a paid query. Baseline, cache reveals and
failed evaluations each cost one logical evaluation. The pilot budgets are
1, 2, 4 and 8; histories are isolated. Missing replay observations remain
unavailable and make the corresponding performance curve incomplete. Do not
rank policies from incomplete replay. These three kernels are previously studied
development diagnostics, not independent held-out confirmation.

For prospective execution prepare a **different** output with
`--execution synthesis`, the same inputs and cache roots. Then smoke-test:

```bash
python -B -m experiments.v21_refinement.experiment --mode run \
  --output-dir experiments/refinement/results/v21_refinement/prospective \
  --kernels GAN --policies v21_priority --limit 3 \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls
```

After preflight, tests and smoke validation, run/resume the bounded pilot:

```bash
python -B -m experiments.v21_refinement.experiment --mode run \
  --output-dir experiments/refinement/results/v21_refinement/prospective \
  --allow-batch --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls
python -B -m experiments.v21_refinement.experiment --mode summarize \
  --output-dir experiments/refinement/results/v21_refinement/prospective
```

The identical run command resumes durable logical queries and physical attempts.
`--limit` bounds additional queries per selected policy/kernel, not globally.
The maximum pilot is 96 logical evaluations across three kernels/four policies,
including baselines. Physical jobs may be fewer through authenticated reuse.
Preparation must occur on the compiler/project host; the sealed directory,
required code and matching physical proof caches can then be transferred to
Coroni preserving repository-relative paths. No model weights are required.
Vitis must be the authenticated 2021.1 build with genuine arithmetic headers.

## Outputs and interpretation

`protocol.json` is immutable. The directory also contains `corpus_manifest.json`,
`graph_coverage.csv`, `relation_coverage.csv`, `action_point_reports.jsonl`,
`candidate_proposals.jsonl`, `candidate_comparison.csv`, `search_traces.jsonl`,
`synthesis_results.jsonl`, `refinement_summary.csv`, and
`V21_REFINEMENT_REPORT.md`. Per-policy engine receipts remain under `search/`.
Matched controls are prepared using the existing profile matcher as a separate
diagnostic pool; they are not counted as evaluated or free policy observations.

ADP is latency cycles times the mean utilization percentage of BRAM/DSP/FF/LUT,
normalized by full-device capacity. Every resource limit is checked separately.
The fixed context is ADP/ZCU104/10 ns/full capacity/Vitis 2021.1. Failed or
infeasible designs never replace a feasible incumbent. Estimated HLS clock is
not post-route timing. Candidate functional status is `not_tested` unless
separate functional validation has actually run. Analysis stubs do not establish
bit-accurate vendor arithmetic. Draft source annotations are not reviewed ground
truth. Structural correctness alone cannot establish learned predictive utility.

CPU/compiler-backed validation:

```bash
python -B -m pytest -q tests/test_v21_refinement.py \
  tests/test_mlir_graph_v2.py tests/test_structural_audit.py \
  tests/test_refinement.py tests/test_refinement_search.py \
  tests/test_refinement_benchmark.py tests/test_directive_domains.py
```

The `--input-dir` default is the authenticated initial experiment under
`experiments/refinement/results/local_refinement/`. On a fresh clone, prepare
that input using the existing [refinement workflow](../refinement/README.md),
or supply a compatible archived directory. Paper QoR is not substituted for
an authenticated vendor-header baseline.

On the execution host used for this validation, the sealed pilot can be resumed:

```bash
cd /home/bonus24/mailohls_v21_validation_20261026
/usr/bin/python3.8 -B -m experiments.v21_refinement.experiment --mode run \
  --output-dir experiments/refinement/results/v21_refinement/evaluation \
  --allow-batch --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls
```

When relocating an existing physical cache, preserve its referenced proof
closure, including `benchmark_r11/historical_cache/reused/`, as well as its
`physical/` journal. A cache journal alone is insufficient authentication.
Physical evaluation identity uses the unchanged original R1 engine hashes;
the new policy code, V2.1 graphs, rules and queues are sealed separately. Thus a
proposal-only wrapper does not invalidate identical hardware receipts, while
all source, metadata, header, materialized directive, Tcl, tool and XML checks
remain active.

For byte-exact resume and summary of the completed Coroni traces, use the same
`/usr/bin/python3.8` execution environment. One Covariance coordinate proposal's
reported logarithmic distance differs by one ULP on the local Python/libm build;
all 96 selected configuration signatures and authenticated hardware results
agree. The existing strict controller intentionally rejects that presentation
mismatch locally. Its checks have not been relaxed. Canonical numeric-distance
serialization should be fixed in a separately frozen future controller protocol;
it does not require regenerating graphs or repeating HLS measurements.
