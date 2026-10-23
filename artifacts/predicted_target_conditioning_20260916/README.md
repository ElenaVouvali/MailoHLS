# Predicted target-conditioning synthesis

Four predicted directive configurations were synthesized independently with
Vitis HLS 2021.1 on `coroni.microlab.ntua.gr`. After report verification, the
corresponding four rows (1, 10, 11, and 12) in the repository-root
`TA_experiments.jsonl` were replaced with these results.

The ZCU104 3.33 ns tight prediction uses L2/L3 partition factors 3/3,
L6/L7 unroll factors 6/6, and L9/L11 unroll factors 3/3. The three U200
3.33 ns predictions use L2/L3 factors 15/15, L6/L7 factors 30/30, and
L9/L11 factors 15/15. Other directives match their respective original rows.

| Device | Clock | Budget | Cycles | Requested-period latency | Estimated period | BRAM/DSP/FF/LUT |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| ZCU104 | 3.33 ns | tight (20%) | 6318 | 21.039 us | 2.969 ns | 114/250/45824/27066 |
| U200 | 3.33 ns | tight (20%) | 3478 | 11.582 us | 2.470 ns | 84/250/43039/29604 |
| U200 | 3.33 ns | medium (50%) | 3478 | 11.582 us | 2.470 ns | 84/250/43039/29604 |
| U200 | 3.33 ns | full (100%) | 3478 | 11.582 us | 2.470 ns | 84/250/43039/29604 |

All four HLS timing estimates are below the requested 3.33 ns period. The
ZCU104 result uses 18.27% of BRAM_18K, 14.47% of DSPs, 9.94% of FFs, and
11.75% of LUTs, so it satisfies the component-wise 20% cap according to HLS
estimates. Each U200 result is below 4% for every reported resource and thus
satisfies all three budget levels.

Compared with the corresponding original rows, the ZCU104 prediction changes
8626 cycles/4.678 ns estimated period to 6318 cycles/2.969 ns. Each U200
prediction changes 3479 to 3478 cycles, retains the 2.470 ns estimate, and
adds 157 FFs and 383 LUTs. These comparisons validate the predicted
configurations; they were not used to select or alter the predictions.

The logs report pipeline-II violations in all four cases. Vitis also reports
ignored partial unroll directives on L6/L9/L11 for the ZCU104 case and L9/L11
for the U200 cases because their parent loops are pipelined. Requested
directive changes therefore do not always map literally to implemented
parallelism. Metrics are HLS synthesis estimates rather than post-route
measurements.

`TA_predictions.input.jsonl` contains the four immutable prediction inputs,
and `TA_predictions.results.jsonl` contains their outputs with remote run
paths and report hashes. Downloaded XML reports and logs are under `reports/`.
