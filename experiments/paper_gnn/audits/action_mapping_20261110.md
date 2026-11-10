# Source-action mapping audit

No mappings changed. Missing applied states have not been assumed to be no-directive. Locked-test measurement rows were not parsed.

| Split | Kernel | Unmapped actions | Rows excluded by current guard |
|---|---|---|---:|
| val | machsuite-sort-radix | L6 | 4,035 |
| train | rodinia-knn-1-tiling | L8 | 3,390 |
| train | rodinia-knn-2-pipeline | L8 | 3,745 |
| train | rodinia-knn-3-unroll | L9 | 4,258 |
| train | rodinia-knn-4-doublebuffer | L11 | 3,951 |
| train | rodinia-knn-5-coalescing | L12 | 4,327 |
| train | rodinia_dilate_2_pipeline_0 | L5, L6, L12 | 4,322 |
| train | rodinia_dilate_3_pipeline_0 | L5, L6, L12, L25, L27, L29, L31, L33, L35 | 2,058 |
| train | rodinia_lavaMD_1_tiling_0 | L8 | 3,481 |
| train | rodinia_lavaMD_1_tiling_1 | L8 | 3,481 |
| train | rodinia_lavaMD_2_pipeline_0 | L8 | 3,291 |
| train | rodinia_lud_1_tiling_0 | L20, L21 | 1,941 |
| train | spcl_example_03 | L3, L4 | 2,378 |
| train | spcl_example_05 | L5 | 1,221 |

## Structural inventory versus measurement inventory

26 of 26 unmapped source actions are absent from their historical src_info.json loop inventories. 0 mapped loops are absent from those inventories.
This supports an inventory mismatch when mapped loops are retained but source-only loops are absent; it does not by itself prove a per-design fixed directive state.

## Cohort impact

- train: 13 of 50 kernels and 41,844 rows excluded; 17 of 18 families remain.
- val: 1 of 4 kernels and 4,035 rows excluded; 2 of 3 families remain.

## Additional header/provenance issues

- rodinia_cfd_flux_0_baseline_0: stale mapping entries [['Array_1', 'L1']]; undeclared labels require review. Active unmapped columns: [].
- rodinia_lavaMD_2_pipeline_0 (train): mapped Array_1 -> L3 absent from raw header.
- rodinia_lavaMD_2_pipeline_0 (train): mapped Array_1 -> L3 absent from preprocessed header.
- serrano-kalman-filter (test): mapped Array_5 -> L7, Array_6 -> L8 absent from preprocessed header.
- serrano-kalman-filter: manifest hash checks {'raw_csv_matches_manifest': False, 'processed_csv_matches_manifest': True}; no test measurement rows were parsed.

The JSON report records source locations, file hashes, header gaps, stale/duplicate mappings and active unmapped CSV columns.

Upstream JSONL creation defaults missing action columns to no-directive values; those defaults do not prove the historical synthesis configuration. Recover an authenticated fixed-state/generator trace before reinstating a kernel.
