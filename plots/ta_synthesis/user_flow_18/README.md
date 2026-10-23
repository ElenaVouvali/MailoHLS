# GAN synthesis using the user-validated flow

These 18 independent runs replace the earlier vendor-header results.
They use Vitis HLS 2021.1, the requested FPGA part and clock, and the
application-local `/home/bonus24/Application/GAN/ap_fixed.h`, matching
 the baseline confirmed by the user (299204 cycles at ZCU104 / 10 ns).

The Tcl flow uses `add_files network.cpp` without extra include flags,
`-flow_target vivado`, `config_array_partition -complete_threshold 0
-throughput_driven off`, and `config_compile -pipeline_loops 0`.
Existing active source HLS pragmas are replaced with each row's directives.
All application headers are copied, including the local fixed-point header.
Each row receives a separate synthesis, even when its directives duplicate
another row's. Budget labels remain selection metadata.

Remote runs: `/home/bonus24/ta_gan_18_user_flow_20260906`.
`TA_experiments.before.jsonl` is the full previous-results backup.
`TA_experiments.input.jsonl` contains the configurations for this batch.
`reports/` contains downloaded source snapshots, logs, and synthesis reports.
`collect_ta.py` verifies all report metrics and source hashes before staging
`TA_experiments.enriched.jsonl` for replacement of the repository-root file.

Latency is worst-case cycles; BRAM counts BRAM_18K units. Metrics are HLS
synthesis estimates. Requested directives are preserved even if the compiler
reports unmet pipeline constraints or overrides partial unroll factors.
