GAN clock-direction probes for ZCU104 at a requested 3.33 ns clock.

The first three candidates copy each case's 5 ns directives while retaining
its 3.33 ns target and budget label. Two subsequent tight-budget candidates
interpolate between the original 3.33 ns tight directives and the 5 ns tight
directives. The repository-root JSONL is unchanged. Runs use the existing
user-validated Vitis HLS 2021.1 flow on coroni.

| Budget and candidate | L2/L3 | L6/L7 | L9/L11 | Cycles | Estimated period | BRAM/DSP/FF/LUT |
| --- | --- | --- | --- | ---: | ---: | --- |
| tight, original | 2/2 | 4/4 | 2/2 | 8626 | 4.678 ns | 114/248/44640/27601 |
| tight, 5 ns directives | 5/5 | 10/10 | 5/5 | 5895 | 3.418 ns | 114/250/45499/28587 |
| tight, bank 3/unroll 6 | 3/3 | 6/6 | 3/3 | 6318 | 2.969 ns | 114/250/45824/27066 |
| tight, bank 4/unroll 8 | 4/4 | 8/8 | 4/4 | 8165 | 4.678 ns | 114/248/45265/29012 |
| medium/full, original | 15/10 | 30/30 | 10/10 | 4990 | 3.115 ns | 114/250/48593/30715 |
| medium/full, 5 ns directives | 15/25 | 30/30 | 15/25 | 4988 | 3.115 ns | 114/250/49539/31325 |

The bank-3/unroll-6 tight candidate is the only useful replacement found:
it improves cycles by 26.8% and HLS estimated timing meets 3.33 ns. It costs
1184 more FFs and two more DSPs than the original tight case, while using 535
fewer LUTs and the same BRAM. Budget labels are selection metadata, so actual
feasibility against a numerical tight resource cap remains unverified.

The medium/full candidate saves only two cycles (0.04%) while adding 946 FFs
and 610 LUTs. The original medium/full selection is preferable on the measured
tradeoff. Vitis HLS reported ignored partial unroll directives on L6, L9, and
L11 and a pipeline II violation for the bank-3 candidate; requested directives
therefore do not all map literally to implemented parallelism. Timing and
resources are HLS synthesis estimates, not post-implementation measurements.

The first-batch inputs and outputs are in this directory; intermediate tight
inputs, outputs, the best candidate's XML report, and synthesis log are in
`intermediate_tight/`. Remote run directories and report hashes are recorded
in each result row.
