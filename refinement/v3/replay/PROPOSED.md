# Offline v3.0 proposals — no candidate synthesis

All six baseline imports passed native source, headers, graph, command, top, part, clock, capacities and HLS-version verification. Each bundle has **1/4 charged, 0 actual new Vitis calls, 1 verified reused evaluation, 0 failures**. Only the initial native measurement is present. Final selected values below carry that initial incumbent forward; no repair gain is claimed.

| Kernel | Proposed first strict repair | Qualitative hypothesis / abstention reason |
|---|---|---|
| GAN | L2 `layer_1_out`: block/5 → cyclic/10, dim 1 | Unit-stride unrolled accesses, explicit port warning and measured scheduled L5 II 6>1; bank distribution may help. General banking rule, no kernel-specific patch; new benefit unknown. |
| jacobi-2d | L2 unroll 32→16; L4 32→16 is the separate second eligible candidate | L3/L5 each have 32 observed rows at II=3, target=1. Verified parents L2/L4 replicate offending A/B accesses. Only one parent changes at a time. Lower port demand may help, but lost lanes may outweigh II relief. Baseline L2/L4=32 remains frozen. |
| Kalman | Abstain | Recurrence warning at L3 does not provide a positive dependence-safety proof for pipeline relocation. Current native resource/timing eligibility holds, so BRAM pressure alone is not a latency repair. |
| trmm-opt | Abstain | L2 2→1: ten different final IIs 52..61, median 56.5. A proportional reduction (about 61→31) trades away lanes. No automatic latency win; exact criticality unknown. |
| covariance | Abstain | No editable banking site. Local L6 reduction lacks a corresponding literal source trip count/end-to-end benefit; source metadata bound is not substituted. Parent scheduling warnings do not authorize unrelated child edits. |
| syr2k | Abstain | L4 2→1 with optimistic II 2→1 gives the same noncalibrated work proxy (80): no optimistic throughput win. |

For Jacobi, the parent-group/child-II scenario proxy is old=9, optimistic=6, proportional=12. These values are deliberately **not** kernel cycles, exact loop latencies or calibrated forecasts. GAN keeps all requested parallel lanes; physical banking and actual gains remain synthesis hypotheses.

| Kernel | Initial = selected best / worst cycles | Estimated ns | BRAM / DSP / FF / LUT | Stop / next step |
|---|---:|---:|---|---|
| GAN | 3133 / 3133 | 6.947 | 81 / 80 / 6507 / 16752 | Await a fresh typed approval |
| jacobi-2d | 1494521 / 1494521 | 7.042 | 0 / 25 / 41009 / 33602 | Await a fresh typed approval |
| Kalman | 536502 / 536502 | 7.3 | 604 / 4 / 7062 / 19074 | Abstain: no sufficiently supported untried latency repair under this policy |
| trmm-opt | 154261 / 154261 | 7.819 | 0 / 28 / 237223 / 156025 | Abstain: no sufficiently supported untried latency repair under this policy |
| covariance | 59643 / 375643 | 7.672 | 0 / 28 / 48163 / 20006 | Abstain: no sufficiently supported untried latency repair under this policy |
| syr2k | 511371 / 549771 | 7.475 | 0 / 50 / 55655 / 35129 | Abstain: no sufficiently supported untried latency repair under this policy |

Selected best-case improvement is **0%** for each imported-only v3 trajectory. Budget 1 is observed; budgets 2–4 carry the incumbent forward without claiming those slots were executed. HLS estimates above 7.3 ns retain separate schedule-margin warnings. Routed closure is unknown.

| Kernel | Full frozen template/source SHA-256 | Selected full configuration signature |
|---|---|---|
| GAN | `85a487965653782892e98777d3f992b641f898b85fe83671c616d4441b1edc61` | `f5597279ea5641c2457dfc7e2d007e5e85e135c497e9a10f16493b74d56534e3` |
| jacobi-2d | `d467158400f2377660b4ffff9567ebeb762032912a8594908e37a444d8151162` | `58461ccbe78dea1a503970a04c1fdc27bb46eed478d7a1a7392d9f8a466de87d` |
| Kalman | `b9decff29b96c13d5cba3197e802eef0a2ffdfb6badfa57a3aab347e8c1d4e33` | `f4c353224f85e646ab640f9552a89faaf5eaadcfe74f0767495fdfbbe349b074` |
| trmm-opt | `a8df1ab7c31e885b1d327491b6d6083346f81b9eaafd68f35276899979a2f5ac` | `1fb5a8cfea8aa33f758e2a072ea99b11ae9611b9c5f4bdb2e58ef4226d3e464f` |
| covariance | `2586e28622dfb0d83073457db13fa2fde8fcc5726bb73b2b8f73ba83b7f5c9f4` | `56ea84fdb9ef7563482c0c75d0a4e84609afe983db41e9fef853327184cde4ad` |
| syr2k | `29aa8c17999424f63cd167442567e5ee8c52a6f9cc438e0a7f715e020cc96665` | `65ddd87f503ed241a7ef9fd516aab6bd12199038f1a2c2ee26277c9978f5b60c` |

Full proposed candidate signatures, diffs, grouped diagnostics, all scheduler variants/log-line provenance, rank breakdowns and rejections are in `preview.json.gz`. `report.json.gz` includes resources, counters, trajectories, best-case incumbent curves at budgets 1–4 and separately labelled historical v0.1/v2 comparisons. `audit.json.gz` verifies all 11 legacy, 9 v2 and six copied v3 baseline receipts; these are not independent fresh syntheses.

Validation: 19 v3 tests, 15 frozen v2 tests and 16 frozen legacy tests passed (50 total). V3 mock process fixtures are explicitly synthetic and never selected as native incumbents. Python 3.8 compiled all new modules and ran previews, dry-run pilot, report and cross-version audits. Preparation/tests used liono Python 3.7 with NetworkX. No SSH or real Vitis process was invoked.

Reproduction: `bash refinement/v3/replay/commands.sh /tmp/cg-bdr-v3-reproduced` from the repository root. The recorded preparation path was `/tmp/cg_bdr_v3_v30_checkpoint`; paths and runtimes can differ in a new replay. Use `gzip -dc refinement/v3/replay/preview.json.gz` to inspect the complete dry-run log.

Experimental gate remains pending: inspect Jacobi and GAN first; each actual call needs its own full typed token. No matched feedback-only synthesis or general-optimizer claim is supported yet. Preserve this checkpoint before any permitted second pass.
