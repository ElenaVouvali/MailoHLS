> Revision3 added twelve exact-count Coroni measurements. The same three configurations are measured at both clocks in two families: no meaningful clock winner reversal, plus one unusually tight Viterbi resource boundary. V4/Stage2/DPO change Viterbi PIPE_L6 from1 to0 for U200 under an identical preceding prefix, but full-device budgets differ and physical usefulness is unverified. This is transmission/decision sensitivity, not adaptation benefit.
>
> Current evidence: [Revision-3 results](RESULTS_V3.md); [technical architecture](FINAL_SALT_HLS_ARCHITECTURE.md).

# Target-conditioning data audit

[All per-contrast rows](../../experiments/final_architecture_20261109/results/target_contrasts.csv) and [complete sets/counts](../../experiments/final_architecture_20261109/results/target_contrasts.json) derive from97,442 authenticated non-test complete measurements across40 kernels. There are no conflicting duplicate physical labels in this reconstruction. No test labels were loaded.

The audit intersects complete assignment hashes before classifying response. It records original target-specific pool sizes, feasible proxy sets, common support,1% acceptable sets and both-direction transfer regret. A reversal needs at least5% loss from reusing the opposite winner in both directions. Budget changes reuse physical measurements. Device views explicitly distinguish full fractions from common absolute ZCU limits; these are correlated views.

Counts: `{"train": {"budget": {"feasibility_boundary_proxy": 74, "no_change": 525, "unsupported": 67}, "clock": {"latency_reversal": 3, "near_tie": 1, "no_change": 130, "unsupported": 14}, "device": {"latency_reversal": 2, "near_tie": 2, "no_change": 188, "unsupported": 30}}, "val": {"budget": {"feasibility_boundary_proxy": 11, "no_change": 43}, "clock": {"no_change": 10, "unsupported": 2}, "device": {"no_change": 16, "unsupported": 2}}}`.

Training has3 clock-reversal rows and2 device-reversal views. The device views describe the same physical contrast, not two independent examples. Reversals involve lavaMD and spcl_example families. The spcl contrast has an extreme latency ratio (>36,000); authenticate original report units before using it as training emphasis. These are latency-specific candidate-bank signals, not qualified generated hardware improvements. Resource feasibility uses percentage-derived counts; rounded zero observations and timing uncertainty prevent exact feasibility claims.

Validation contains no latency reversal on shared pools:10 supported clock contrasts and16 device views retain an acceptable common winner. Two clock and two device views lack support. Budget views show11 proxy boundaries,43 invariants. These are3 kernels in2 families, not independent budget examples. No timing-qualified context is available from these tables. A changed feasible proxy set is insufficient proof of physical resource binding.

Conclusion: input variation exists, but validation lacks trustworthy switch coverage needed to rank target-memory architectures. Retain invariant contexts. Confirm one development switch with unrounded resource reports and finite latency before escalating. Missing metrics are unknown; unchanged directives may be correct on invariant contexts.
