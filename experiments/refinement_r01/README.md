# R0.1 — Refinement attribution and relation validation

This package audits the existing R0 evidence without changing R0 Python files,
its code-hash contract, production inference/training, checkpoints, or compiler
analyses. It loads no LLM or GNN weights and performs no iterative optimization.

The experiment uses GAN, ADP, ZCU104 (`xczu7ev-ffvc1156-2-e`), 10 ns, full
device resource capacity, and Vitis HLS 2021.1 with genuine vendor AP headers.
The original R0 parser, source materializer, semantic checks, HLS runner, report
parser, normalized-resource area, and resource feasibility are imported directly.

## Experiments

- A: ARRAY_F_L2 5 to 6, preserving cyclic type/dimension 1.
- B: PIPE_L11 0 to 1, preserving UNROLL_L11=5.
- Primary cells: d0, dA, dB, dAB.
- Verified-direct cells: d0, dC, dB, dCB, where C changes ARRAY_T_L3 from
  cyclic to block. This arm requires verified compiler evidence for L3–L11.
- Matched control: the same loop B and cyclic-to-block array-type edit at an
  array lacking an evidenced relationship with L11. Selection uses evidence,
  action/field edit counts and domain classification, never candidate QoR.
  Array extent and element type may differ; no independence claim is made.
- One optional native factor-4 single-edit comparator is included only because
  4 belongs to both the native source domain and the existing R0 local policy.
  No arbitrary new factor proposals are introduced.

Each factorial uses complete baseline-derived configurations and computes
`I = J(joint) - J(left) - J(right) + J(baseline)` for ADP. Negative interaction
is favorable nonadditivity, not proof of a physical mechanism. Missing,
unverified, infeasible, failed or mismatched-context cells make it unavailable.

Whole assignments are classified as native/extended, and changed values are
classified separately. A historical baseline can already be extended. In that
case a native edit does not make the full configuration native; factor 6 being
outside the native proposal domain is not by itself proof of domain dominance.

## Recovery and provenance

Pass `--r0-output` to the actual R0 output directory. Results may have been
archived outside the repository. The original directory is read-only: manifests,
candidate pools/assignments, results, source snapshots, metadata, headers,
receipts, Tcl, completion logs and XML are authenticated. Exact materialized
source and header-copy/substitution policies are checked. Missing provenance
is labeled `historical_unverified`, never used in factorial calculations.

Authenticated evidence is copied into R0.1's own directory, so later execution
does not need to alter or relocate the original R0 experiment. If artifacts are
unavailable, the exact published directives and repository source are used to
prepare new measurements; absent structural evidence blocks the relevant arms.
`--fresh-measurements` disables reuse for a new, consistent evaluation context.

Baseline comparison checks matching archived GAN source, inserted pragmas,
other dependency headers, AP numeric implementation, tool version, top, part,
clock, Tcl settings and report metrics. Original paper data is never rewritten.

## Relation verification

The wrapper preserves the original relation tensor. It checks operations in a
loop's own function/region against concrete compiler memory-root access edges.
Exact source-action anchors are required; slots are never renumbered.

- `direct_verified`: a concrete root access at an operation/call site in the
  loop's own region, with validated source/action/root binding.
- `indirect_verified`: the actual operand at this call, its position, called
  function identity and matching formal binding establish the buffer identity,
  with verified callee access. An explicit call-site binding is also supported.
  Shared formal nodes or a root passed somewhere else do not suffice.
- `conservative_unknown`: only shared helper reachability or unresolved binding
  supports the legacy relationship. Uncertain dependences are not upgraded.
- `no_evidenced_relation`: no qualifying access was established. This does not
  prove semantic independence.

The current graph may lack call-site-specific binding metadata for some helper
cases. Those stay uncertain; the wrapper does not invent mappings or modify the
compiler schema. The shared-operator, different-buffer failure mode has a
regression fixture in `tests/test_refinement_r01.py`.

## Commands from the repository root

Use a CPU Python environment with NetworkX and PyTorch for audit/preparation.
Running/summarizing a prepared queue needs only the standard library; synthesis
also needs the verified Vitis installation. No GPU is needed.

```bash
R0_OUTPUT=/path/to/authentic/r0_local_headroom

# Audit only; no synthesis. The subsequent prepare can use the same directory.
python -m experiments.refinement_r01.r01_attribution \
  --mode audit --r0-output "$R0_OUTPUT"

python -m experiments.refinement_r01.r01_attribution \
  --mode prepare --r0-output "$R0_OUTPUT"

# Explicit execution. The missing A-only cell has first priority when possible.
python -m experiments.refinement_r01.r01_attribution \
  --mode run --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

# Repeat run to resume; success records are verified and not duplicated.
python -m experiments.refinement_r01.r01_attribution --mode summarize

python -B -m pytest -q -p no:cacheprovider \
  tests/test_refinement_r01.py tests/test_refinement_r0.py \
  GNN_branch/tests/test_mlir_top_selection.py
```

The default output is `experiments/refinement/results/r01_attribution/`, already
ignored by the original R0 `.gitignore`. Use `--output-dir` consistently for a
different run. The manifest and prepared evidence are immutable; choose a new
directory after implementation/input/context changes.

New synthesis attempts are durably reserved before invocation, with a hard
limit of 12 across all calls and retries. `--limit N` bounds one invocation;
`--retry-failed` explicitly spends additional attempts. Interrupted reservations
are not blindly repeated; completed durable result files can be recovered.
A file lock prevents simultaneous execution. Tool/header changes block mixing
new measurements with reused cells. No test or preparation launches real HLS.

Required outputs are R01_MANIFEST.json, R01_CANDIDATES.jsonl,
R01_MEASUREMENTS.jsonl, R01_RELATIONS.jsonl, R01_FACTORIAL.csv and R01_REPORT.md.
Additional records preserve authentication, baseline comparison, interactions,
tool identity and actual synthesis attempts. Reports distinguish unavailable
measurements from numerical outcomes and show both individual and joint effects.
