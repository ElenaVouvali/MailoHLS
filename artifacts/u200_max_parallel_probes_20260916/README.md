U200 full-layer parallelism probes at each requested clock. Three independent
Vitis HLS 2021.1 runs used the existing GAN user flow on coroni. Root JSONL
was unchanged during these probes.

For each full-budget case, L2/L9 were raised from 15 to 30 and L3/L11 from
10 or 25 to 50. L6/L7 were already fully unrolled at factor 30, their loop
length. Every candidate remains below the 20% per-resource U200 budget, but
none offers a meaningful latency improvement.

| Clock | Current cycles/LUT | Full-factor cycles/LUT | Estimated period, full-factor |
| --- | ---: | ---: | ---: |
| 3.33 ns | 3479 / 29221 | 3608 / 83448 | 2.470 ns |
| 5 ns | 2921 / 32164 | 3002 / 85257 | 3.632 ns |
| 10 ns | 1967 / 39316 | 1966 / 92409 | 7.097 ns |

The 10 ns one-cycle improvement costs 53,093 LUTs. The shorter-clock full
factor choices are both slower and much larger. Device capacity alone does
not imply that these directives improve GAN latency; scheduling dependencies
and memory structure can limit the benefit of requested parallelism. Inputs,
results, source hashes, and remote report paths are recorded here.
