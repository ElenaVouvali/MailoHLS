# CG-BDR v2 refinement prototype

This is the exact November 17, 2026 refinement prototype from Coroni's
`/home/bonus24/cg_bdr_v2_20261117`, published on the `llvm-ir` branch.
It is an optional post-inference step; model weights, Stage-3 DPO and the original
LLVM/ProGraML graph generator remain unchanged.

| Module | SHA-256 |
|---|---|
| `cg_bdr.py` | `01e3a18aac4aa92531c1b54893f2b242a015b1c3c9af7acb0d6bc4e2211988ff` |
| `semantic_v2.py` | `a8e2f4b59957eefa258695f17ab4e7ed57b07a411a10101f04dcad277f103a81` |

Both modules are unchanged from the server snapshot. Preparation seals both code
hashes into each manifest; v2 rejects earlier v0.1 bundles. The earlier
[six-kernel controller and evidence](../README.md) remain available separately.

## What changes in v2

- `semantic_v2.py` derives labeled source-loop containment, direct statements
  excluding nested labeled-loop regions, possible accumulation clues and shared
  source-array access hints. LLVM/ProGraML anchors establish action existence.
  These facts do not prove LLVM LoopInfo correspondence, alias freedom, true
  recurrences, independent lanes or physical banking behavior.
- Diagnostics separately retain the explicitly scheduled loop and the loop owning
  the source-line access. The runner parses achieved/target II from Vitis logs
  and records an attributed action/constraint graph for every evaluation.
- Bank/access repair requires source and LLVM array evidence. Loop-only port
  relief requires an observed achieved II greater than target II. Warnings with
  no measured stall do not alone justify that repair.
- Parent/child pipeline relocation requires source-supported hierarchy, an
  observed parent II stall and a child without suspicious source accumulation.
  It is a hypothesis to test, not a proof that a dependence was removed.
- Resource/timing relief reduces replication only in a measured infeasible
  context. An optional `--enable-one-scout` policy allows at most one explicitly
  exploratory probe, distinct from diagnosed repairs.
- Candidate selection now minimizes numeric HLS **Best-caseLatency**. Native
  worst-case bounds are retained as `latency_worst_cycles` and in summaries.
  The earlier controller selected by worst-case latency. V2 also requires an
  affirmative estimated-clock feasibility result; unknown timing is ineligible.

The four-evaluation maximum still includes the initial design. Failed tool calls
consume budget, interrupted reservations remain in `pending.json`, repeated
candidate signatures are excluded, and unsupported repairs cause abstention.
Run only one writer per bundle; the frozen controller does not implement locking
for simultaneous invocations. Do not delete a reservation to obtain a free retry.

## Recorded outcomes

The snapshot contains **nine native evaluation records across five kernels**,
all at ZCU104 (`xczu7ev-ffvc1156-2-e`), 10 ns and full resource budgets.
All five selected incumbents are their original configurations.
The five initial native XML reports match the earlier archive byte for byte;
these nine records must not be counted as nine independently fresh v2 syntheses.
The snapshot includes four additional candidate records.

| Kernel | Recorded evaluations | Initial best cycles | Selected best cycles | Selected worst cycles | Speedup |
|---|---:|---:|---:|---:|---:|
| Covariance | 2 | 59,643 | 59,643 | 375,643 | 1.00× |
| Jacobi-2D | 1 | 1,494,521 | 1,494,521 | 1,494,521 | 1.00× |
| SYR2K | 2 | 511,371 | 511,371 | 549,771 | 1.00× |
| TRMM | 2 | 154,261 | 154,261 | 154,261 | 1.00× |
| Kalman | 2 | 536,502 | 536,502 | 536,502 | 1.00× |

Kalman's pipeline-relocation candidate worsened best-case latency to 798,640
cycles. SYR2K's port-relief candidate reached 516,171 cycles, TRMM's reached
155,611, and Covariance's retained 59,643. Inferior and tied candidates remain in
the evidence; none are presented as speedups.

**There is no v2 GAN run in this snapshot.** The earlier GAN improvement from
3,133 to 1,389 cycles belongs to the six-kernel version, not this v2 campaign.
These are recorded partial trajectories with one or two evaluations and no pending
reservation; their state stop flags are unset. The snapshot does not establish
that every possible repair or the four-call budget was exhausted.

Best-case bounds do not establish full-workload or worst-case execution time.
Eligibility uses HLS estimated timing at the requested 10 ns, not post-route timing
or the stricter uncertainty-adjusted scheduling budget. The selected Covariance,
SYR2K and TRMM estimates exceed the 7.3 ns uncertainty-adjusted budget. No functional
equivalence, co-simulation or place-and-route result is claimed here. Publishing
this snapshot does not alter the existing plotting tables or launch new synthesis.

## Validate offline

From the repository root:

```bash
python3 -m venv /tmp/mailohls-refinement-env
/tmp/mailohls-refinement-env/bin/python -m pip install -r refinement/requirements-test.txt
/tmp/mailohls-refinement-env/bin/python -m unittest discover -s refinement/v2/tests -v
/tmp/mailohls-refinement-env/bin/python -m unittest discover -s refinement/tests -v
python3 refinement/v2/audit_evidence.py
python3 refinement/audit_evidence.py
```

The v2 audit verifies both code hashes, the packed evidence inventory, original
uncompressed input hashes, source-derived semantics, directive signatures,
materialized sources, native HLS report/log/Tcl hashes, part/clock/capacities,
best/worst metrics, scheduling-II records, diagnostic attribution, derived
constraint graphs and selected incumbents. It makes no HLS calls and does not
modify stored results. Synthetic compiler fixtures are tests, not hardware evidence.

`evidence/` retains each recorded manifest, state and summary, exact input template
and compressed LLVM graph, and each evaluation's candidate, native top XML, Tcl,
log and receipt. Redundant generated RTL and subfunction reports are omitted.
`recorded_results.json` is the derived audit summary.

## Prepare and run a fresh v2 bundle

Python 3.8+ is required. The runner, semantic analyzer and audit use the standard
library. Preparation additionally needs NetworkX. Generating new LLVM graphs
requires the original LLVM-10/ProGraML environment described in the
[graph preparation instructions](../README.md#generate-another-llvm-graph).
Only ZCU104 is supported by the frozen capacity table; U200/U50 few-shot results
are separate experiments.

First verify and restore the recorded graphs into a **fresh** directory:

```bash
python3 refinement/v2/audit_evidence.py --extract-to /tmp/mailohls-v2-evidence
python3 refinement/v2/cg_bdr.py preview \
  --bundle /tmp/mailohls-v2-evidence/Kalman_LATENCY_10ns_4eval
```

`preview` is read-only. For a new run, use your new structured model prediction
and its matching source/template/metadata/LLVM graph. This complete example
replays the original Kalman prediction with the archived source snapshot:

```bash
python3 refinement/select_initial.py \
  --jsonl refinement/evidence_plot_rows.jsonl --kernel Kalman \
  --part xczu7ev-ffvc1156-2-e --clock 10 --budget-percent 100 \
  --out /tmp/kalman-v2-initial.json

python3 refinement/v2/cg_bdr.py prepare \
  --kernel Kalman \
  --source-dir /tmp/mailohls-v2-evidence/Kalman_LATENCY_10ns_4eval/inputs \
  --template /tmp/mailohls-v2-evidence/Kalman_LATENCY_10ns_4eval/inputs/template.cpp \
  --kernel-info /tmp/mailohls-v2-evidence/Kalman_LATENCY_10ns_4eval/inputs/kernel_info.txt \
  --graph /tmp/mailohls-v2-evidence/Kalman_LATENCY_10ns_4eval/inputs/llvm_graph.gexf \
  --initial /tmp/kalman-v2-initial.json --part xczu7ev-ffvc1156-2-e \
  --clock 10 --budget-percent 100 --max-evaluations 4 \
  --bundle /tmp/kalman-refinement-v2-new
```

Preparation makes no HLS calls. On the Vitis HLS 2021.1 server with genuine vendor
headers, evaluate one candidate at a time:

```bash
source /opt/Xilinx/Vitis_HLS/2021.1/settings64.sh
python3 refinement/v2/cg_bdr.py run --bundle /tmp/kalman-refinement-v2-new \
  --vitis-hls /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls \
  --limit 1 --timeout 7200 --min-free-gb 20
python3 refinement/v2/cg_bdr.py preview --bundle /tmp/kalman-refinement-v2-new
python3 refinement/v2/cg_bdr.py summary --bundle /tmp/kalman-refinement-v2-new
```

Transfer **both** unchanged `cg_bdr.py` and `semantic_v2.py` together with the
prepared bundle. Use a new sealed bundle rather than resuming a v0.1 experiment.
The four-call budget is enforced across repeated `run --limit 1` invocations.
For an intentionally exploratory protocol, pass `--enable-one-scout` consistently
to `run` and `preview` and report that policy explicitly.
