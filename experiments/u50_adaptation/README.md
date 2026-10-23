# U50 Stage-1 device adaptation

This is a controlled Stage-1 experiment. The parent is the frozen latency
Stage-1 adapter in `mailohls_runs/stage1_final_final_latency_s123`; Stage 2 and
Stage 3 are deliberately excluded. The final GAN, JPEG, Kalman, and covariance
measurements never enter calibration or checkpoint selection.

`prepare.py` validates the 16/4/1 family split, copies source provenance, and
selects the three ZCU104/10 ns Pareto winners. A GPU run must generate one
zero-shot U50/10 ns/PARETO_LATENCY completion per kernel. Resolve any identical
directive signatures with one legal local neighbor, then run
`assemble_raw_csv.py` with exactly 84 measured rows. Only after that should
`Preprocessing.data_preprocess`, `Preprocessing.create_jsonl`, and
`Preprocessing.build_family_split` be run.

For the 20 train/validation kernels, Coroni synthesis is driven by
`run_u50_synthesis_coroni.py`. It consumes an 80-line manifest containing four
canonical target completions per kernel, materializes the placeholders under
`/home/bonus24/Application`, and writes resumable results to
`synthesis_results.jsonl`. Convert successful results with
`results_to_raw_records.py`, then assemble the GNOSIS CSVs. The internal
`spcl_example_03` four-point test can be run in a separate manifest later.

The four exact U50 capacities are recorded in `u50_device_specs.json` and are
from a real U50 HLS report: BRAM_18K 2688, DSP 5952, FF 1743360, LUT 871680.

The training command is intentionally not launched here: this workspace has no
working NVIDIA driver. Run it on the GPU host with seeds 123, 456, and 789,
then select only by adaptation-validation macro accuracy. The composed adapter
loader and `mailohls-device-adapt-trainables-v1` contract are tested locally in
`tests/test_device_adaptation.py`.
