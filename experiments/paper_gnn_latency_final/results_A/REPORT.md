# Arm A: executed latency-GNN development metrics

Selected checkpoint: `627e52fae6d0613e346c6ab5e7e0047d511a02af1806600ee67f2b45041f8075`. Selected epoch: **3** (888 optimizer steps). Full run: **11 epochs / 3,256 optimizer steps** on Liono’s V100S. The selected checkpoint, last checkpoint, optimizer/RNG states and loss/gradient receipts are preserved in the run directory.

| Protocol | Unique observed configurations | Kendall tau-b | Feasible Regret@1 (%) | Log2-latency MAE | Resource-feasible selection (%) |
|---|---:|---:|---:|---:|---:|
| zcu104_10ns | 2,362 | 0.135680 | 38.5033 | 2.075502 | 100.00 |
| all_measured_contexts | 13,119 | 0.129632 | 68.6231 | 2.255203 | 100.00 |

Both protocols use the shared evaluator and the actual saved predictions of the selected checkpoint. The ZCU104/10 ns slice is separate from the all-context protocol. Training uses all measured contexts; this slice is not a reproduction of historical single-target-only training.

| Protocol | Kernel | Kendall tau-b | Regret@1 (%) | Log2-latency MAE | Resource-feasible selection (%) |
|---|---|---:|---:|---:|---:|
| zcu104_10ns | machsuite-sort-radix | 0.259286 | 16.3284 | 3.119593 | 100.00 |
| zcu104_10ns | machsuite-viterbi | 0.256459 | 0.0208 | 2.338318 | 100.00 |
| zcu104_10ns | rodinia_pathfinder_0_baseline_0 | -0.200388 | 196.2725 | 1.103647 | 100.00 |
| zcu104_10ns | rodinia_pathfinder_4_doublebuffer_0 | -0.017022 | 2.0489 | 0.433540 | 100.00 |
| all_measured_contexts | machsuite-sort-radix | 0.255741 | 27.4760 | 3.506122 | 100.00 |
| all_measured_contexts | machsuite-viterbi | 0.243795 | 51.6728 | 2.502250 | 100.00 |
| all_measured_contexts | rodinia_pathfinder_0_baseline_0 | -0.248736 | 249.7152 | 1.273813 | 100.00 |
| all_measured_contexts | rodinia_pathfinder_4_doublebuffer_0 | 0.027457 | 3.7262 | 0.240662 | 100.00 |

Aggregation averages two budget views within each physical context, contexts within kernels, the two Pathfinder kernels within their family, then the three independent families equally. This is not an unweighted average of the four kernel rows.

Sort-radix is evaluated on its ten authenticated observed actions, with L6 explicitly ignored as requested. Its missing eleventh directive is not reconstructed. The full corrected development pool has 13,119 observed configurations; the ZCU104/10 ns pool has 2,362.

Regret is selected measured latency / best measured-feasible latency - 1, shown as a percentage. Selection uses predicted latency and predicted resource requirements; measured resource feasibility is checked afterward. The candidate pool is not first restricted by oracle feasibility. Latency targets and predictions use log2 milliseconds; MAE is in log2 units.

Conditional regret coverage is 48/48 budget pools for all contexts and 8/8 for ZCU104/10 ns. Neither protocol has a pool with no measured-feasible candidate or an undefined tau. Raw resource percentages are rounded measurements, not exact integer resource counts. Exact budget-boundary equality is treated as unknown; the unknown cases were not selected. Timing qualification remains unknown.

A’s resource-selection success does not establish good latency ranking: Pathfinder baseline has negative tau-b and large regret. No freeze upgrade is concluded from A alone; the matched B and historical comparisons remain pending. Downstream LLM improvement is unverified.

Full per-context CSVs and shared-evaluator JSON reports accompany this file. The source-action bank passed the maintained XAttn forward/backward checks with fresh fusion weights; complete Stage-2 training was not launched.
