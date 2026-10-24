# R0 compiler-grounded local-refinement mechanism

R0 is a controlled headroom harness around stored SALT directive predictions.
It preserves the existing model, checkpoints, training, and inference paths.
The default command prepares candidates; synthesis requires an explicit run.
Generated measurements, reports, source copies, projects, and run-specific remote
scripts are not kept in the repository.

## Current scope

- Baseline: the exact `objective=ADP`, `stage=predicted-directives` row from
  `paper_plots/SALT_experiments.jsonl` for each requested kernel.
- Independent candidate: change exactly one `Lk` optimization action.
- Structural candidate: change exactly two distinct, compiler-related actions.
  An action can contain PIPE/UNROLL or the joint array type/factor/dimension
  tuple, so two actions can change more than two serialized fields.
- Four candidates per arm, with a complete frozen assignment per design.
  Every neighbor starts from the original baseline. There is no iterative loop.
- Objective: ADP only. The initial stored prediction is ADP-oriented, and
  synthesized candidates are ranked by ADP. Candidate enumeration does not
  invoke an LLM, choose an adapter, or use candidate QoR for selection.

The harness is not connected to prompt-driven SALT inference. Its summary ranks
neighbors within each arm and may report a worse neighbor; it does not update
or deploy an incumbent configuration automatically.

Two-action edits are an experimental control, not a restriction imposed by the
relation representation. Extending to connected groups of actions would require
an explicit edit-size policy and bounded proposal/synthesis budgets. Efficient
relation storage does not remove the combinatorial cost of editing many actions.
A comparison with unrelated groups of the same size is needed to isolate the
benefit of structural guidance from edit size.

A future objective-conditioned proposer can reuse the appropriate existing
adapter plus the objective/target context, incumbent configuration, and selected
structural group. The existing adapters were trained for one-pass prediction;
refinement input/decoding support and validation would still be needed. Synthesis
must evaluate the user's actual objective and enforce resource limits. This
objective-conditioned proposal path is not implemented by R0. Consult
`FINAL_ARTIFACTS.json` for checkpoint availability: Stage-2 ADP is production;
AREA and LATENCY Stage-2 paths are pending rather than interchangeable production
adapters.

## Source and compiler contracts

| Kernel | Source | Metadata | Top |
| --- | --- | --- | --- |
| covariance | `evaluation_set/covariance/kernel.cpp` | sibling `kernel_info.txt` | `kernel_covariance` |
| GAN | `evaluation_set/GAN/network.cpp` | `evaluation_set/GAN/code_to_memory_outputs/meta/kernel_info.txt` | `forward_propagation` |
| Kalman | `evaluation_set/serrano-kalman-filter/kernel.cpp` | sibling `kernel_info.txt` | `krnl_KALMAN` |

The existing source-action parser, template builder, semantic tuple validators,
and native MLIR action-relation projector are reused. Self edges, inactive
slots, unknown action identities, duplicates, and no-op assignments are excluded.
Action and directive-field edit distances are recorded separately.

The factor whitelist is waived only for this harness, as requested by the
experiment owner. Published baseline values are preserved exactly. Per factor
field, proposals are `{0, published_factor, 2, factor-1, factor+1}`; new enabled
factors are at least two and bounded by source trip counts/array extents. All
non-factor domains, complete-assignment checks, and array tuple semantics remain
active. Production SALT domains are unchanged.

Only provenance-bearing native MLIR graphs are accepted. Exact source/metadata
hashes, sparse absolute `Lk` alignment, and supported relation bits are validated.
Relations retain parent/child loops, loop-array relationships, and proven
inter-action dependences. Unknown dependence analyses are not treated as proven.
The unchanged projector helpers are loaded without the exporter's CLI/model
side effects; no dense attention or LLVM/HARP fallback is substituted.

Graph provenance and exact action anchors establish artifact identity, not the
precision of every projected pair. Helper-function reachability is currently
context-insensitive: shared C++ operator/helper nodes can introduce loop-array
relationships without a direct access at that particular call site. Treat these
projected relationships as potentially over-approximated until call-site buffer
bindings are validated. This limitation must be addressed before interpreting
them as precise direct coupling or training a refinement policy from them.

Kalman's tracked graph uses authenticated compiler-active metadata that is an
exact subset of the synthesis action contract. Its omitted L7/L8 sites remain
in complete assignments but cannot be edited in either arm. Missing graphs are
regenerated with the existing native Polygeist/MLIR pipeline when available;
otherwise structural preparation reports an explicit blocker.

GAN's C++ selector is `_Z19forward_propagationPfS_`, passed through
`--compiler-symbol`. Source-level top identity remains `forward_propagation`.
No source declarations or linkage are rewritten. The selector is recorded in
graph provenance, and exact action-location validation remains required.

## Mechanism files

| File | Responsibility |
| --- | --- |
| `r0_local_headroom.py` | Prepare/run/summarize CLI, frozen experiment context, kernel/source configuration |
| `r0_candidates.py` | Complete assignments, legality, canonical hashes, neighborhoods and deterministic selection |
| `r0_relations.py` | Native compiler regeneration, authenticated relation projection and evidence |
| `r0_synthesis.py` | Source materialization, vendor headers, Tcl, isolated/resumable HLS execution and XML validation |
| `r0_metrics.py` | Mean-resource area, ADP, independent resource limits and feasible ranking |
| `r0_common.py` | Paths, content hashes, deterministic JSON and atomic output helpers |
| `__init__.py` | Package setup |

The companion regression tests are `tests/test_refinement_r0.py` and
`GNN_branch/tests/test_mlir_top_selection.py`. They are executable safeguards,
not saved experiment measurements.

## Usage

Use Python 3.10+ with CPU-capable PyTorch, NetworkX, and pytest for preparation
and tests. Running/summarizing a prepared queue uses the standard library and
supports Python 3.8+. No GPU or model weights are needed.

From the repository root:

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_refinement_r0.py GNN_branch/tests/test_mlir_top_selection.py

# Preparation never launches synthesis.
python -m experiments.refinement.r0_local_headroom \
  --mode prepare --kernels covariance GAN Kalman --seed 123

# Explicit three-job smoke test on a Vitis HLS 2021.1 host.
python -m experiments.refinement.r0_local_headroom \
  --mode run --smoke covariance \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

# Run the remaining frozen queue; repeat to resume without duplicate jobs.
python -m experiments.refinement.r0_local_headroom \
  --mode run --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls

python -m experiments.refinement.r0_local_headroom --mode summarize
```

Use `--output-dir` consistently across commands for a separate run. Preparation
refuses to overwrite a directory containing synthesis records. `--limit N`
bounds new attempts; `--retry-failed` explicitly authorizes additional attempts.
`--timeout` controls each synthesis timeout. Failed and partial reports remain
unrankable; missing measurements never become numerical predictions.

For compiler preparation, follow `toolchain/README.md` and set `POLYGEIST_BUILD`
and `MLIR_PYTHON`, or pass `--polygeist-build` and `--mlir-python`. `--config`
accepts per-kernel source, metadata, top, graph, and compiler-symbol overrides.
Mismatching graphs are blocked rather than repaired. The graph generator also
supports direct `--compiler-symbol EXACT_SYMBOL` use; see
`docs/mlir-pipeline.md`.

## Synthesis and objective contracts

Deployment is fixed to ZCU104 / `xczu7ev-ffvc1156-2-e`, 10 ns, Vitis HLS 2021.1,
and full device capacity. The Tcl preserves `set_top`, `add_files`, Vivado flow,
`set_part`, `create_clock`, `csynth_design`, and the existing explicit partition
and auto-pipeline settings. Disabled pragmas are removed safely; complete
partitions omit the encoded zero factor. Inputs are never overwritten.

Repository `ap_fixed.h`/`ap_int.h` files are compiler-analysis stubs. HLS jobs
exclude them and require verified genuine vendor headers; other dependencies
are copied exactly. This distinction limits comparisons with historical
measurements made under a different header environment.

For capacities BRAM_18K=624, DSP=1728, FF=460800, LUT=230400:

```text
utilization_r = 100 * measured_resource_r / device_capacity_r
area = mean(utilization_BRAM, utilization_DSP, utilization_FF, utilization_LUT)
ADP = worst_case_latency_cycles * area
```

Every resource must independently satisfy its budget. Timing is reported
separately. XML top, part, version, flow, clock, units, latency, and all resource
counts are checked. A new same-flow baseline is required unless identical
historical provenance is verified. Paper values are preserved as references.

## Generated outputs

Future runs create `manifest.json`, candidate pools/queue, synthesis records,
`summary.csv`, and `R0_REPORT.md` under `results/r0_local_headroom/`, plus source,
graph, materialization, validation, and isolated HLS attempt directories.
The `results/` tree is ignored by Git. No previous run outputs or measured
result tables are included in this mechanism-only repository state.
