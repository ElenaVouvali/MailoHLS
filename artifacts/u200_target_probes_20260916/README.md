U200 target-conditioning probes. Six independent Vitis HLS 2021.1 runs on
coroni using the established GAN user flow. The root `TA_experiments.jsonl` was
unchanged during probing; after report verification, its three 3.33 ns U200
rows were replaced with the higher-parallelism results.

All probes retain their original U200 device, clock, objective, and budget
percentage. Higher-parallelism probes at 3.33 ns copy the current 5 ns U200
directives. Lower-parallelism tight probes copy the respective ZCU104 tight
directives, including the newly synthesized ZCU104 bank-3/unroll-6 choice at
3.33 ns.

| U200 case | Selection | L2/L3 | L6/L7; L9/L11 | Cycles | Estimated period | BRAM/DSP/FF/LUT |
| --- | --- | --- | --- | ---: | ---: | --- |
| 3.33 ns, any budget | current | 15/10 | 30/30; 10/10 | 3479 | 2.470 ns | 84/250/42882/29221 |
| 3.33 ns, any budget | 5 ns directives | 15/25 | 30/30; 15/25 | 3477 | 2.470 ns | 84/250/43348/30355 |
| 3.33 ns, tight | ZCU104 bank-3/unroll-6 | 3/3 | 6/6; 3/3 | 5882 | 2.782 ns | 84/250/40073/24088 |
| 5 ns, tight | current | 15/25 | 30/30; 15/25 | 2921 | 3.518 ns | 84/250/34172/32164 |
| 5 ns, tight | ZCU104 tight | 5/5 | 10/10; 5/5 | 4907 | 3.675 ns | 84/250/30952/25964 |
| 10 ns, tight | current | 15/25 | 30/30; 15/25 | 1967 | 7.097 ns | 84/250/33158/39316 |
| 10 ns, tight | ZCU104 tight | 5/5 | 10/10; 5/5 | 3951 | 7.231 ns | 84/250/29781/32770 |

Under a strict latency-only objective with a 20% per-resource cap, the
3.33 ns higher-parallelism candidate is feasible and beats the current row by
two cycles (0.057%) for tight, medium, and full. It adds 466 FFs and 1134 LUTs
without changing BRAM or DSP. The other six current U200 rows should be kept:
forced lower parallelism in tight cases roughly doubles cycle latency while
all current designs already use less than 20% of each device resource.

If a resource-aware tie break is desired, the current 3.33 ns choice may be
preferable because it gives up only two cycles to save FFs and LUTs. Identical
directives across budgets are consistent with nonbinding 20%, 50%, and 100%
caps on U200; they are not evidence by themselves that budget conditioning
failed. Budget labels and percentages are metadata in this flow and were not
enforced by Vitis HLS. These are HLS synthesis estimates, not implemented
hardware measurements. The downloaded higher-parallelism full report and log
show an II violation and ignored partial unroll directives on L9 and L11.

Inputs, outputs, and the copied runner are in this directory. XML reports and
logs for each of the three adopted 3.33 ns candidates are in their respective
`higher_parallel_3p33_*` directories. All six rows include remote run paths
and report hashes.
