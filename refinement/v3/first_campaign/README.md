# V3 first campaign: measured status on 18 November 2026

This is a partial campaign for ZCU104 (`xczu7ev-ffvc1156-2-e`), 10 ns,
Vitis HLS 2021.1 and 100% resource budgets. The executable V3 modules are unchanged
from the frozen checkpoint. Publication updates documentation and adds evidence;
it does not implement the proposed improvements below or launch synthesis.
The cross-version test's publication allowlist additionally permits the root
version links and Git evidence-file rules; executable modules remain byte-identical.

## Accounting and results

Six archived native baselines were verified and reused. Only Jacobi evaluation 2
is a newly invoked V3 synthesis. There are **7 charged evaluations, 6 baseline
reuses, 1 new HLS call, 0 synthesis failures and 0 pending reservations**.
The four-evaluation maximum includes the reused baseline; it was not exhausted.

| Kernel | Baseline best cycles | New candidate best cycles | Selected best cycles | Selected speedup | Current strict-policy decision |
|---|---:|---:|---:|---:|---|
| GAN | 3,133 | Not synthesized | 3,133 | 1.000x | Propose existing-site banking repair |
| jacobi-2d | 1,494,521 | 1,679,961 | 1,494,521 | 1.000x | Retain baseline; L4 proposal remains available |
| Kalman | 536,502 | Not synthesized | 536,502 | 1.000x | Abstain |
| trmm-opt | 154,261 | Not synthesized | 154,261 | 1.000x | Abstain |
| covariance | 59,643 | Not synthesized | 59,643 | 1.000x | Abstain |
| syr2k | 511,371 | Not synthesized | 511,371 | 1.000x | Abstain |

Speedup is baseline best-case cycles divided by selected best-case cycles.
Baseline-only entries do not establish that a new candidate was tested and tied.
No V3 latency gain is demonstrated. Budget curves that carry an incumbent forward
do not represent additional tool invocations. FEWSHOT/U200 experiment-table gains
and historical legacy/V2 gains are separate experiments.

## What Jacobi established

The candidate changed only L2 unroll factor **32 to 16**, leaving L4 at 32.
The L3 scheduling variants fell from 32 to 16, but achieved II stayed at 3.
L5 remained at 32 variants and II=3. Native XML reports:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Best and worst cycles | 1,494,521 | 1,679,961 |
| Nested L2 latency | 18,680 | 23,316 |
| Nested L4 latency | 18,680 | 18,680 |
| FF | 41,009 | 31,153 |
| LUT | 33,602 | 25,820 |
| DSP | 25 | 25 |
| Estimated clock, ns | 7.042 | 7.042 |

The edit lost parallelism without II relief: candidate speedup **0.889616x**,
latency **+12.40799%**, FF **-24.03375%**, LUT **-23.15934%**.
The nested-loop delta explains the whole-kernel delta exactly:
`(23316 - 18680) * 40 = 185440` extra cycles. Selection retained the baseline.

## Proposals, abstentions and discussion points

- **GAN:** propose `layer_1_out` dimension 1, block/5 to cyclic/10. L5 achieved
  II=6 with explicit port and recurrence warnings. This is the strongest available
  banking hypothesis, not a measured gain. The log also shows partial inner-loop
  unrolling followed by complete unrolling for parent pipelining; requested factor
  10 alone is not a complete model of effective concurrency.
- **Kalman:** L3 achieves II=3 with a carried-dependence diagnostic. Pipeline
  relocation is disabled without a dependence-safety proof. BRAM is 604/624
  (96.79%); the baseline remains resource-eligible. The reported L10 latency is
  405,307 cycles, so warning presence alone does not establish criticality.
- **TRMM:** preserve all ten L2 final IIs (52..61), not one exact value. The
  proportional-II proxy for unroll 2 to 1 gives no credible throughput gain.
- **Covariance:** reported IIs are 50, 80 and 50. There is no editable array site
  for `data`; triangular L6 bounds are unsupported by the literal-trip parser.
  Parent-pipeline-induced complete unrolling further limits directive-only models.
- **SYR2K:** unroll 2 to 1 and even optimistic II 2 to 1 yield the same work
  proxy, 80. Abstention avoids equating lower II with higher throughput.

Concrete limitations for a future checkpoint: reconstruct compiler-effective
replication; read nested XML loop latencies recursively (the inherited parser
currently retains immediate children only); distinguish observed port constraints
from predicted II relief; improve whole-kernel impact ranking; and use failed
interventions across equivalent loop families. No such fixes are included here.
There is no matched feedback-only synthesis establishing added graph value.
HLS estimates are not routed timing closure; the new Tcl runs `csynth_design`,
without functional co-simulation or place and route.

## Evidence and read-only verification

- [results.csv](results.csv): compact six-kernel accounting.
- [evidence.json](evidence.json): verified summaries, current strict previews,
  source bundle paths, state hashes, scheduling evidence and native nested-loop values.
- [Jacobi bundle](jacobi-2d_LATENCY_10ns_4eval/): original manifest, seal, inputs,
  prediction, state, both receipts, exact candidates/Tcl/logs/native top XML and
  the new candidate's readable top report. Redundant RTL/subfunction output is omitted.
- [CHECKSUMS.json](CHECKSUMS.json): publication file inventory and SHA-256 hashes.

The local collected Jacobi state matched Coroni byte for byte at publication
(`f3ce9218ab0e9ac4a5a2aa00ded4ece0882b0e8e2536a5f351a81458c5471bf4`).
Other five kernels' baseline origins remain in the existing legacy/V2 archives;
they can be reconstructed with the frozen runner's offline preparation workflow.
Machine-specific original paths in receipts are provenance, not portable commands.

From the repository root, with the dependencies in `refinement/requirements-test.txt`:

```bash
python3 -B -m unittest discover -s refinement/v3/tests -v
python3 -B -m unittest discover -s refinement/v2/tests -v
python3 -B -m unittest discover -s refinement/tests -v
python3 -B refinement/v3/audit_evidence.py --bundles \
  refinement/v3/first_campaign/jacobi-2d_LATENCY_10ns_4eval
python3 -B refinement/v3/cg_bdr.py summary --bundle \
  refinement/v3/first_campaign/jacobi-2d_LATENCY_10ns_4eval
```

These commands audit archived measurements or use labelled mock tools; they do
not invoke real Vitis. Use a separate fresh bundle for new experiments. The
[`replay/` directory](../replay/) preserves the earlier offline checkpoint and
its baseline-only proposals; it predates the new Jacobi candidate.
