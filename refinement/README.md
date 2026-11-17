# Tested LLVM-action-grounded refinement

**The newer November 17 v2 prototype is now published in
[`v2/`](v2/README.md).** Use `python3 refinement/v2/cg_bdr.py` for v2.
This page and `refinement/cg_bdr.py` retain the earlier six-kernel version and
its exact evidence. Their sealed bundle formats and latency objectives differ.

CG-BDR is an optional post-inference refinement step for the original MailoHLS
LLVM/ProGraML pipeline. It freezes the initial prediction, synthesizes it, maps
explicit Vitis diagnostics to source actions and LLVM graph anchors, and tests
bounded repairs. It retains the best measured resource-feasible design whose
HLS estimated clock is within the requested period. It does not update model
weights or replace Stage-3 DPO.

`cg_bdr.py` is byte-for-byte the controller used in Coroni's completed
`cg_bdr_six_v02` campaign on 16 November 2026. Its SHA-256 is
`c5a7c7077c24adde96b32ee5eee34afdc0d2faeb6698b34151abb27815c56b32`.
The bundle schema remains `cg-bdr-v0.1-research`, as it was in the measured runs.
The separate `cg_bdr_v2_20261117` semantic prototype is preserved under `v2/`,
alongside this controller and its results.

## Measured scope

The verified experiment uses ZCU104 (`xczu7ev-ffvc1156-2-e`) at 10 ns and full
device resource budgets. The maximum budget is **four total evaluations,
including the initial configuration**; the controller can abstain earlier.
There were eleven actual evaluations across six kernels.

| Kernel | Charged HLS calls | Initial worst cycles | Selected worst cycles | Refinement speedup |
|---|---:|---:|---:|---:|
| Covariance | 2 | 375,643 | 375,643 | 1.00× |
| Jacobi-2D | 1 | 1,494,521 | 1,494,521 | 1.00× |
| SYR2K | 2 | 549,771 | 549,771 | 1.00× |
| TRMM | 2 | 154,261 | 154,261 | 1.00× |
| GAN | 3 | 3,133 | 1,389 | 2.26× |
| Kalman | 1 | 536,502 | 536,502 | 1.00× |

Only GAN selects changed directives. It changes `layer_1_out` (L2) from block
partitioning at factor 5 to cyclic at factor 15, through two measured candidate
steps. The other five kernels retain their exact initial assignment; tried
inferior repairs are retained in the evidence. These outcomes do not establish
universal improvement or the incremental benefit of LLVM structure over a matched
feedback-only control.

`evidence_plot_rows.jsonl` is the exact eighteen-row plotting snapshot that
motivated this branch update. The controller selects by **Worst-caseLatency**.
The plot's `latency` field uses **Best-caseLatency** for that same selected
candidate: Covariance reports 59,643 and SYR2K 511,371. Both native bounds are
retained in the XML and `selected_latency_bounds.jsonl`; they are not interchangeable.
The `LATENCY_refinement` label names the treatment, including unchanged outcomes.

## Mechanism and limits

- Preparation seals the template, project headers, action metadata, LLVM GEXF,
  initial directives and controller hash. It makes no HLS calls. Analysis-only
  AP/header stubs are excluded from synthesis inputs.
- Diagnostics identify reported memory-port pressure and source-line/loop
  attribution. Array repairs additionally require direct source accesses and
  pragma → array-scope → representative LLVM memory-operation links in the
  same function. This restricted provenance is not alias or dependence proof.
- The tested changes include correct loop-body parsing around placeholder
  braces, exact attribution of Vitis `_V_...` array names, combined banking-style
  and factor proposals, and memory-port concurrency repair for loop-only spaces.
  Achieved-II scheduling messages can be parsed separately for inspection.
- Each tool call is reserved in `pending.json` before execution. Failures consume
  budget. An interrupted reservation blocks further runs; preserve it for manual
  investigation rather than deleting it for a free retry. Use a single runner
  per bundle: this historical controller does not implement concurrent writers.
- The runner diagnoses the measured incumbent, skips already tried signatures,
  and abstains when no supported repair remains. `--explore-on-no-diagnostic`
  enables a weaker exploratory policy; it was not used in these measured runs.
- The tested controller supports only the exact ZCU104 part. Its capacity table
  must not be reused for U200 or U50. The separate U200 few-shot results are not
  measurements of this controller.
- Eligibility compares the HLS estimated period with the requested period,
  not the uncertainty-adjusted scheduling period. The selected Covariance,
  SYR2K and TRMM estimates exceed the stricter 7.3 ns scheduling budget at 10 ns.
  These results establish neither routed timing closure nor functional correctness.
  No new synthesis, co-simulation or place-and-route was run for this branch update.

## Offline checks and measured evidence

From the repository root, create a separate environment. The core runner and
evidence audit use Python 3.8+ and the standard library; preparation/tests also
need NetworkX. The original LLVM graph generator needs its own historical environment.

```bash
python3 -m venv /tmp/mailohls-refinement-env
/tmp/mailohls-refinement-env/bin/python -m pip install -r refinement/requirements-test.txt
/tmp/mailohls-refinement-env/bin/python -m unittest discover -s refinement/tests -v
python3 refinement/audit_evidence.py
```

The audit verifies all eleven native executions: controller and input hashes,
candidate source materialization, configuration signatures, Tcl/log/report hashes,
Vitis completion markers, exact top/part/clock identity, capacities, native metrics,
selected incumbents and the corresponding plotting rows. It makes no HLS calls
and does not modify the evidence. Compressed GEXF snapshots are verified against
the original uncompressed hashes. Synthetic tests are kept separate and must not
be cited as hardware measurements.

The archived bundles contain their exact manifests, states, summaries, source
templates/headers and the native top XML, source, Tcl, log and receipt for every
evaluation. Redundant generated RTL and subfunction reports are omitted. The
original publication audit and both latency bounds are retained alongside them.

Optionally restore the raw graphs into a **fresh** directory:

```bash
python3 refinement/audit_evidence.py --extract-to /tmp/mailohls-cg-bdr-evidence
python3 refinement/cg_bdr.py preview \
  --bundle /tmp/mailohls-cg-bdr-evidence/GAN_LATENCY_10ns_4eval
```

`preview` is read-only. The archived states are completed experiments, not fresh
budgets. Prepare a new bundle for new measurements; never delete their state.

## Prepare a new run from an existing prediction

Use a structured directive prediction with an exact kernel/device/clock/budget
context. The selector does not perform inference or qualify imported latency.
For a complete offline example using the supplied source/graph snapshot:

```bash
python3 refinement/select_initial.py \
  --jsonl refinement/evidence_plot_rows.jsonl --kernel GAN \
  --part xczu7ev-ffvc1156-2-e --clock 10 --budget-percent 100 \
  --out /tmp/gan-initial.json

python3 refinement/cg_bdr.py prepare \
  --kernel GAN \
  --source-dir /tmp/mailohls-cg-bdr-evidence/GAN_LATENCY_10ns_4eval/inputs \
  --template /tmp/mailohls-cg-bdr-evidence/GAN_LATENCY_10ns_4eval/inputs/template.cpp \
  --kernel-info /tmp/mailohls-cg-bdr-evidence/GAN_LATENCY_10ns_4eval/inputs/kernel_info.txt \
  --graph /tmp/mailohls-cg-bdr-evidence/GAN_LATENCY_10ns_4eval/inputs/llvm_graph.gexf \
  --initial /tmp/gan-initial.json --part xczu7ev-ffvc1156-2-e \
  --clock 10 --budget-percent 100 --max-evaluations 4 \
  --bundle /tmp/gan-refinement-new
```

For your own inference output, select its `LATENCY` row rather than the example
snapshot, and supply the matching source, metadata, headers and genuine LLVM
graph. Current model weights, source arithmetic and action spaces are not changed
by refinement. Keep the same settings across compared runs:
`config_compile -pipeline_loops 0` and
`config_array_partition -complete_threshold 0 -throughput_driven off`.

On a machine with Vitis HLS 2021.1 and genuine vendor headers, run one evaluation:

```bash
source /opt/Xilinx/Vitis_HLS/2021.1/settings64.sh
python3 refinement/cg_bdr.py run --bundle /tmp/gan-refinement-new \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls \
  --limit 1 --timeout 7200 --min-free-gb 20
python3 refinement/cg_bdr.py preview --bundle /tmp/gan-refinement-new
python3 refinement/cg_bdr.py summary --bundle /tmp/gan-refinement-new
```

Repeat the same `run --limit 1` as needed within the sealed four-evaluation cap.
`summary` writes derived `summary.json`; the state, receipts and native reports
remain the measurement authority. For Coroni, transfer the new bundle and the
unchanged `cg_bdr.py`, then run using Python 3.8+ on that server.

## Generate another LLVM graph

`build_llvm_graph.py` invokes this branch's `GNN_branch/graph_gen_deterministic.py`
with an isolated work root. It patches the historical hardcoded root in memory
without editing the original generator. Use the working LLVM-10/ProGraML
environment with `programl`, `torch`, `networkx`, `klepto` and its historical
dependencies. This graph-generation environment is separate from Vitis synthesis.

```bash
PYTHONHASHSEED=0 python3 refinement/build_llvm_graph.py \
  --llvm-repo "$PWD" --name GAN \
  --original /path/to/pragma-free/network.cpp \
  --placeholder /path/to/GAN_placeholders.cpp \
  --kernel-info /path/to/kernel_info.txt \
  --headers-dir /path/to/project-headers \
  --work-root /tmp/gan-llvm-graph-build \
  --out-graph /tmp/gan-llvm.gexf
```

Source/action correspondence must match the initial prediction. Do not substitute
an MLIR graph or enable partial graph coverage as a shortcut for paper evaluation.
Graph generation is provided for future preparation; it was not rerun to publish
the existing measured snapshots.
