# Local HLS refinement

This optional experiment tests legal changes around a stored SALT prediction.
It compares single-action edits, compiler-verified pairs, and matched pairs
without a verified relationship. For each pair it measures the baseline, each
single edit, and both edits together. It does not run the LLM, retrain the GNN,
or implement an iterative optimizer.

## Code map

| File | Responsibility |
| --- | --- |
| `experiment.py` | One CLI: prepare, reuse measurements, run/resume, and summarize. |
| `configurations.py` | Complete assignments, source-domain checks, deterministic selection, corpus audit and factorial analysis. |
| `relations.py` | Authenticate MLIR graphs, verify relationships and preserve sparse action identities. |
| `synthesis.py` | Materialize sources, invoke Vitis, parse reports, compute ADP and authenticate evidence. |

Directive policies are shared with model workflows in
[`LLM_branch/common/directive_domains.py`](../../LLM_branch/common/directive_domains.py).
The original training, inference and GNN entry points remain the standard
framework interfaces described in the root README.

## Prepare and execute

From the repository root, using the normal MailoHLS Python environment:

```bash
# Preparation never launches HLS.
python -m experiments.refinement.experiment --mode prepare \
  --kernels GAN covariance Kalman --seed 123

# Optional: import compatible measured results instead of synthesizing again.
# Use a NEW output directory when changing preparation settings.
python -m experiments.refinement.experiment --mode prepare \
  --output-dir experiments/refinement/results/reused_measurements \
  --reuse /path/to/existing/measured/experiment

# Explicit synthesis on a Vitis host. Repeating this command resumes.
python -m experiments.refinement.experiment --mode run \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

python -m experiments.refinement.experiment --mode summarize
```

The default output is ignored `experiments/refinement/results/local_refinement/`.
Use `--output-dir` consistently across commands. `--limit N` bounds new jobs in
one invocation; `--timeout` sets seconds per job. `--retry-failed` explicitly
allows another attempt. Successful authenticated jobs are never repeated.
Durable reservations, including retries and imported charged costs, enforce a
48-evaluation cap. Baseline failures block further jobs for that kernel.
Charged costs from multiple archives are added conservatively; use the latest
aggregate archive when histories overlap.

The initial deployment is ADP, ZCU104 `xczu7ev-ffvc1156-2-e`, requested clock
10 ns, full resource limits, Vitis HLS 2021.1. Different source, header, tool,
clock, target or synthesis settings cannot be mixed in a comparison.

### Sources and compiler graphs

Baselines come from complete ADP `predicted-directives` rows in
`paper_plots/SALT_experiments.jsonl`. Sources are under `evaluation_set/`;
Kalman maps explicitly to `serrano-kalman-filter`. `--config` accepts source,
metadata and top-function overrides; `--baselines` overrides the prediction file.

Supply authenticated `GAN.gexf`, `covariance.gexf` and `Kalman.gexf` with
`--graphs-dir`. The default ignored cache is `results/compiler_graphs/`.
The tracked Kalman graph is also supported. When a graph is missing, preparation
tries the pinned native pipeline documented in [`toolchain/README.md`](../../toolchain/README.md):

```bash
POLYGEIST_BUILD=/path/to/locked/build MLIR_PYTHON=/path/to/python3.11 \
python -m experiments.refinement.experiment --mode prepare \
  --output-dir experiments/refinement/results/prepared_with_compiler
```

`--polygeist-build`, `--mlir-python` and `--graph-timeout` provide CLI overrides.
Unavailable compiler evidence is reported as blocked; no relationships are
invented. Sparse inactive sites retain their original values and are not edited.
For Coroni, prepare locally and transfer the package, shared dependencies and
sealed output at the same repository-relative paths. Running and summarizing
require Python 3.8+; preparation additionally needs CPU torch/networkx and, when
regenerating graphs, the native MLIR toolchain. No model weights are needed.

## Domain and selection rules

The `tcad` policy uses powers of two no greater than n plus integer divisors of n
at least two. There are no separate 64/1024 caps. Array factors are checked
against the selected dimension; partial decoding keeps only legal completions.
Pipeline and partial unroll may coexist. Disabled and complete partitions keep
the framework's existing tuple encodings.

Loop sizes must be compiler-proven static trip counts for new unroll proposals.
Approximate metadata bounds are not substituted for unknown counts. Unchanged
historical values can have explicit baseline-only exceptions; edits cannot use
those exceptions to introduce invalid tuples. Existing checkpoint behavior is
available through `--domain-policy legacy`, without a second implementation tree.

Selection is deterministic and independent of measured QoR. It chooses diverse
single-action edits and up to two verified pairs per kernel, preferring loop
unroll plus accessed-array partition factor. Numeric edits prefer the nearest
supported increase, falling back to decreases. A control matches action types,
field counts, factor magnitude and policy novelty as closely as available.
No verified relationship does not establish semantic independence.

All pair endpoints change, all configurations are complete, and canonical
signatures deduplicate shared cells. The manifest records the considered pool,
selection reasons, relation evidence, compiler count reconciliation and hashes.
Unknown/shared-helper relations never qualify as verified structural pairs.

## Outputs and interpretation

The working output has seven main files:

- `manifest.json`: sealed inputs, code hashes, domains, considered candidates,
  relation evidence, context and budget.
- `candidates.jsonl`: frozen complete assignments.
- `measurements.jsonl`: authenticated QoR, statuses and evidence paths.
- `attempts.jsonl`: durable synthesis accounting.
- `summary.csv`: best feasible results per kernel/neighborhood.
- `factorials.csv`: all cells and attribution metrics.
- `report.md`: measured results and limitations.

Source/graph snapshots, reused evidence and HLS attempts live in dedicated
subdirectories. Large projects, logs and generated results stay out of Git.
`--mode audit-domains` writes a separate `domain_audit.json` in a fresh output
and audits the existing SFT corpus against supplied metadata. `--sft-dataset`
changes the corpus; `--domain-audit` includes an existing audit in preparation.
Corpus coverage is not a compiler trip-count audit of every training kernel.

```bash
python -m experiments.refinement.experiment --mode audit-domains \
  --output-dir experiments/refinement/results/domain_audit
```

Area is the mean BRAM/DSP/FF/LUT utilization against full device capacities;
ADP is latency cycles times that mean. Every resource limit is checked
individually. Failed, missing, infeasible or incomparable cells cannot be winners.
HLS estimated timing is reported separately from resource feasibility.

For minimized ADP, interaction is `J(AB)-J(A)-J(B)+J(0)` and joint-over-single gain
is `[min(J(0),J(A),J(B))-J(AB)]/J(0)`. A favorable interaction alone does not mean
ADP improved. Small, unreplicated comparisons and imperfect controls cannot
establish general structural superiority. Synthesis does not prove functional
equivalence; functional simulation status is recorded explicitly.

Measurement import reads both plain and historical prefixed artifact filenames
and rechecks sources, generated source, Tcl, tool/header identities, logs and
XML. Historical selector code is not required to authenticate physical results.
Exact earlier source is retained in Git history and an ignored local source
archive; earlier measured directories are preserved, not rewritten. Changed
code or frozen inputs require a new preparation directory.

## Tests

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_directive_domains.py tests/test_refinement.py
```

Tests use synthetic vendor reports and a test-only executable; they need neither
Vitis nor a GPU. Ordinary prepare/audit/test commands never launch synthesis.

## Independent refinement research branch

Develop feedback/search changes on `research/refinement-mechanism`. Stable
`experiments/refinement`, `refinement_search` and `v21_refinement` remain on main.
Reuse their shared configuration, materialization, cache and report utilities;
use modules/configurations for policy variants rather than new remote branches.
Stage-1/GNN proposals may become optional inputs, but no new algorithm or
synthesis run is introduced by this branch setup. Preserve objective/feasibility
definitions and measured-report identities when comparing policies.
