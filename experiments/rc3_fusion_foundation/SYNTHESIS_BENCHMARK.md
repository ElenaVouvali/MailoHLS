# Latency-focused initialized GEMV synthesis benchmark

## Source and candidates

The benchmark uses the full-depth study's compiler-verified `reduce32` diagnostic source, not the historical GEMV. Its local arrays are initialized from explicit inputs, and every prefix reduction result is stored to the output. The prior numerical check reported maximum absolute error `0.0001220703125` against the reference implementation. It has source SHA-256 `0bf12183719b41fec7b50dd458848e991964454a89467577616f4460480caece`; its RC3 graph and MLIR hashes are recorded in the frozen bundle manifest. This is a new diagnostic kernel, not a relabeling of the old benchmark.

The complete, source/tool-validated candidate pool contains six assignments:

1. V4 Stage-1 R1 assignment.
2. V4 Stage-1 ZCU104/3.33 ns/25% assignment from the recorded target grid.
3. One-field L1 partition dimension alternative.
4. One-field L2 partition factor alternative.
5. One-field L5 pipeline-II alternative.
6. One-field L5 unroll alternative.

Each assignment has all 13 directive fields and preserves complete action assignments. The LLM outputs were originally predicted on the authenticated original source. Applying them to `reduce32` is a controlled, same-schema diagnostic, not a fresh prediction on the variant.

Four physical contexts are frozen: ZCU104 at 10 and 3.33 ns; XCU200 at 10 and 3.33 ns. This gives 24 unique physical jobs. Each result can be checked post hoc against both 100% and 25% resource budgets; these budget views are not duplicated as synthesis jobs because budgets do not alter the physical implementation request. All target capacities, part IDs, materialized source hashes, directive maps, and validation receipts are in `artifacts/rc3_fusion_foundation/v1_20261107/synthesis_bundle_v2/manifest.json`.

## Execution results (bonus24)

The frozen batch ran on `coroni`/`bonus24` using Vitis HLS 2021.1, build 3247384, with the authenticated runner. All **24/24 jobs completed successfully**, with parsed integer resource reports. The compact local receipts and XML are in ignored `artifacts/rc3_fusion_foundation/v1_20261107/synthesis_results/`; the batch `report.json` SHA-256 is `eff8002d75e6657198cf0e12281922305297c66e694558c5fcd891657ada299e`. The remote complete temporary project tree remains on bonus24 under `/tmp/rc3_fusion_foundation_dd54fe3/synthesis_results`.

Every candidate fit both the 100% and 25% resource views on all four contexts. Thus the 25% requested budget was not binding in this candidate pool and did not change its measured-pool winner. For example, the maximum observed ZCU104 use was 4 BRAM, 8 DSP, 1,800 FF, and 1,876 LUT, well below the 25% capacities 156/432/115,200/57,600. XCU200 maxima were 2/8/1,442/1,743 versus 1,080/1,710/591,120/295,560.

| Device / requested clock | Best pool by cycles (cycles; target-clock projection; HLS estimated clock) | Best pool also meeting HLS clock estimate (cycles; target-clock projection; estimated clock) | 25% resource-feasible |
|---|---|---|---|
| ZCU104 / 10 ns | `alt_l5_pipeline1`: 5,539; 55.39 µs; 9.788 ns | Same | 6/6 |
| ZCU104 / 3.33 ns | `alt_l5_pipeline1`: 10,787; 35.92 µs; 4.584 ns (misses target) | `stage1_zcu104_3p33_25pct`: 17,075; 56.86 µs; 2.790 ns | 6/6 |
| XCU200 / 10 ns | `alt_l5_pipeline1`: 3,427; 34.27 µs; 12.610 ns (misses target) | `stage1_zcu104_3p33_25pct`: 6,323; 63.23 µs; 6.635 ns | 6/6 |
| XCU200 / 3.33 ns | `alt_l5_pipeline1`: 6,595; 21.96 µs; 4.166 ns (misses target) | `stage1_zcu104_3p33_25pct`: 11,187; 37.25 µs; 2.633 ns | 6/6 |

The time column is only the cycle count multiplied by the requested clock; it is a target-clock projection, not evidence that the design met timing. The cycle-only measured-pool regret of `alt_l5_pipeline1` is zero in all four contexts. Once candidates whose HLS estimated clock misses the request are excluded, the pool winner changes in three contexts. These are small-pool HLS estimates, not a global optimum or post-route result. The L5 pipeline alternative is resource-feasible but not estimated to meet the requested clock in 3/4 contexts. No general conclusion about the model's quality follows from this deliberately small candidate set.

Pipeline XMLs were emitted for the L5 pipeline alternative. The two report names `VITIS_LOOP_15_5` and `VITIS_LOOP_15_51` report II values 6/6 at ZCU104 10 ns, 14/14 at ZCU104 3.33 ns, 2/2 at XCU200 10 ns, and 8/8 at XCU200 3.33 ns. They are preserved with their exact Vitis-generated names; we do not relabel them as action-level II without a verified compiler-to-report mapping.

The runner explicitly marks `functional_validation=false`: this is `csynth_design`, not C/RTL simulation, implementation, or post-route timing. The initialized source was numerically checked during its prior compiler-variant preparation (maximum absolute error 0.0001220703125), but that is separate from these synthesis receipts.

To reproduce analysis from saved receipts, run `python -m experiments.rc3_fusion_foundation.summarize_synthesis --bundle artifacts/rc3_fusion_foundation/v1_20261107/synthesis_bundle_v2 --results artifacts/rc3_fusion_foundation/v1_20261107/synthesis_results --out artifacts/rc3_fusion_foundation/v1_20261107/synthesis_summary.json`. The summary reports resource-budget feasibility, regret within this measured pool, estimated-clock compliance, and pipeline II/interval XML data. Missing or malformed resource reports remain unknown/ineligible.
