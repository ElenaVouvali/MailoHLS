# GAN target-aware synthesis

Each of the 18 rows in `TA_experiments.input.jsonl` receives an independent
Vitis HLS 2021.1 `csynth_design` run on coroni. The remote run directory is
`/home/bonus24/ta_gan_18_20260906`. Three jobs run concurrently.

The kernel comes from `/home/bonus24/Application/GAN`: `network.cpp`,
`network.h`, `weight_definitions.h`, and `tanh.h`. Existing active HLS pragmas
are replaced with the row's directives using labels L1–L11. The application's
`ap_fixed.h` is a software stub and is intentionally excluded; synthesis uses
the Vitis vendor fixed-point headers, matching the recent GAN target-response
experiments on this server. The source and Tcl hashes are recorded per row.

Targets are `xczu7ev-ffvc1156-2-e` (ZCU104) and `xcu200-fsgd2104-2-e` (U200).
Clock periods are passed exactly as supplied: 3.33, 5.0, and 10.0 ns.
The flow target is `vivado`; automatic array partitioning and automatic loop
pipelining use the existing experiment settings in `run_ta.py`.

`latency` is worst-case latency in cycles. `latency_ns` multiplies those cycles
by the requested clock period. `resources.bram` counts BRAM_18K units; DSP,
FF, LUT, and URAM are counts from `AreaEstimates/Resources` in the top-level
`forward_propagation_csynth.xml` report. Estimated clock period, clock
uncertainty, best-case latency, and invocation intervals are also retained.
These are HLS estimates, without RTL co-simulation or place-and-route.

Budget labels are preserved as selection metadata. No numerical tight or
medium budgets were supplied, so no allocation limits or budget-feasibility
claims are added. The inferred ZCU104 / 10 ns / full row retains its original
inference annotation and is synthesized independently.

Vitis may ignore partial inner-loop unroll factors when the parent is
pipelined, and may fail to achieve requested initiation intervals. The final
JSONL records these outcomes under `synthesis`; the input directives remain
unchanged. Zero unroll factors retain the existing source convention and
produce the expected ignored-zero-factor warning.

`collect_ta.py` checks all 18 downloaded reports against their result records,
verifies source/report hashes, checks that original configuration metadata
is unchanged, and writes `TA_experiments.enriched.jsonl`. Logs, sources, Tcl
files, and synthesis reports are retained under `reports/`.
