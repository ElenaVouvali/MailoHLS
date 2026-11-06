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

## Execution status

No HLS synthesis was run. `command -v vitis_hls` returned exit status 1, and `/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls` does not exist. No latency, resource, II, or clock result is claimed. The bundle includes the authenticated existing Vitis 2021.1 batch runner (runner SHA-256 is in `execution_status.json`) and checks its version and vendor headers before any job begins.

To execute later, install or expose the already-approved Vitis 2021.1 environment with genuine vendor headers, then run the exact command saved in `synthesis_bundle_v2/execution_status.json`. It writes results to the sibling `synthesis_results` directory. The batch runner records completed-cycle latency, requested-clock time estimate, BRAM/DSP/FF/LUT counts, full-device fit, and HLS estimated clock; it marks functional validation false. Then run `python -m experiments.rc3_fusion_foundation.summarize_synthesis --bundle <bundle> --results <synthesis_results> --out <summary.json>` to calculate both budget views without duplicate jobs, latency regret among feasible measured pool candidates, and loop II/interval values from pipeline XML reports where present. HLS estimates are not post-route timing.

When measurements exist, compute feasible-pool latency regret only among successfully synthesized candidates whose exact resource reports fit the requested budget. Compare target/budget-specific best candidates within this small measured pool, not against a global optimum. Do not interpret a missing or malformed resource report as feasibility. Until jobs run, synthesis status, feasibility, latency, regret, and budget-binding are all **unknown**.
