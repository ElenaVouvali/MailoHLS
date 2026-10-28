# Latest MLIR and MLIR + FPGA hardware graphs

**One place to review and download the current graph collection.** Each kernel folder contains raw MLIR, both compiler-graph views, action registrations, a readable hardware-evidence excerpt, and every full hardware-enriched graph.

- **63 authenticated V2.2 RC3 snapshots**, 126 original canonical/compact graphs.
- **1,512 full MLIR + hardware graphs**: 63 snapshots × ZCU104/U200/U50 × 3.33/5/6.67/10 ns × canonical/compact.
- Full physical resource availability in the published contexts. Arbitrary clocks and restricted budgets remain supported by the existing composition CLI.
- Corrected development **[Kalman](kernels/serrano-kalman-filter-96b4d608969f/)** and latest **[Hotspot pipeline](kernels/rodinia-hotspot-2-pipeline-c448680ca9be/)** / **[Hotspot unroll](kernels/rodinia-hotspot-3-unroll-7c11a20afd22/)** are easy to find here. Historical Kalman alternatives are explicitly marked below.
- The fresh Hotspot native MLIR and four independently regenerated compiler graphs are retained inside the corresponding kernel folders, alongside the primary authenticated compilation context.

RC3 is technically frozen; **independent human semantic adjudication remains pending**. Hardware edges express documented capabilities and possible compatibility, not measured resource bindings or QoR. Production GNN checkpoints still consume V1 tensors. No synthesis, training, or QoR inference was performed for this catalog.

## Review and download

Open a kernel below, inspect its raw MLIR/action registrations/hardware evidence in GitHub, then download a complete graph for the desired target and clock. Complete graph JSON is gzip-compressed to keep duplicate compiler facts from inflating the repository; no custom graph format or Git LFS is needed. The underlying schemas remain `salt-hls-mlir-structural-v2.2` and `salt-hls-target-enriched-v22-v1`.

[Architecture and evidence guide](GUIDE.md) · [Target specifications](profiles.json) · [Manifest and graph identities](manifest.json) · [Checksums](SHA256SUMS) · [Validation report](validation/catalog.json) · [Sealed source/header/metadata snapshots](sources/)

From a checkout of this branch, with Python 3.10+:

```bash
# Check every published file against SHA256SUMS (standard library only).
python scripts/graphs/publish_catalog.py verify
# Extract all graphs and raw MLIR for one kernel into a fresh directory.
python scripts/graphs/publish_catalog.py unpack \
  --kernel serrano-kalman-filter-96b4d608969f --output /tmp/kalman-graphs
# Validate a complete enriched graph using the existing independent validator.
python -m GNN_branch.target_graph_v22 validate \
  --graph /tmp/kalman-graphs/hardware/zcu104/10ns_full/canonical.json
```

Unpack/verify use only the standard library. Graph validation/build require the repository's existing NetworkX dependency. Reading these graphs does not require Polygeist, MLIR Python bindings, Torch, a GPU, or Vitis.

To reproduce the complete catalog from the immutable release, use a fresh output directory and `python scripts/graphs/publish_catalog.py build --output /tmp/rebuilt-graph-catalog --workers 8`. Full-corpus validation independently rederives every hardware overlay, restores every program graph, compares canonical/compact supported semantics, checks repeated serialization, and verifies gzip round trips. Runtime receipts can vary between builds; graph bytes and gzip headers are deterministic for the same implementation, inputs and runtime.

## Kernel index

Counts below are original canonical nodes/edges, added hardware nodes/edges per context, and compiler-matched/registered actions. All 63 archived snapshots are retained; this count is not a claim of 63 distinct development kernels. The release comprises 55 development snapshots and eight external snapshots.

| Kernel / source snapshot | Compiler nodes / edges | Added nodes / edges | Actions | Selection note |
| --- | ---: | ---: | ---: | --- |
| [GAN-afa1bdfea894](kernels/GAN-afa1bdfea894/) | 1,536 / 2,772 | 5 / 205 | 11/11 | RC3 snapshot |
| [chstone-aes-883c223fddcf](kernels/chstone-aes-883c223fddcf/) | 4,204 / 53,788 | 5 / 83 | 6/6 | RC3 snapshot |
| [covariance-e3ce197df78b](kernels/covariance-e3ce197df78b/) | 142 / 457 | 5 / 51 | 7/7 | RC3 snapshot |
| [jacobi-2d-038157dbee01](kernels/jacobi-2d-038157dbee01/) | 109 / 255 | 5 / 49 | 5/5 | RC3 snapshot |
| [machsuite-gemm-blocked-87bab6d97f84](kernels/machsuite-gemm-blocked-87bab6d97f84/) | 78 / 128 | 5 / 33 | 5/5 | RC3 snapshot |
| [machsuite-gemm-ncubed-8692fbcbe7b2](kernels/machsuite-gemm-ncubed-8692fbcbe7b2/) | 67 / 106 | 5 / 23 | 3/3 | RC3 snapshot |
| [machsuite-md-knn-6b5a601c3bcb](kernels/machsuite-md-knn-6b5a601c3bcb/) | 176 / 375 | 5 / 55 | 2/2 | RC3 snapshot |
| [machsuite-sort-radix-7bc2717b6aac](kernels/machsuite-sort-radix-7bc2717b6aac/) | 355 / 626 | 5 / 75 | 11/11 | RC3 snapshot |
| [machsuite-spmv-ellpack-01a6b91f7611](kernels/machsuite-spmv-ellpack-01a6b91f7611/) | 74 / 122 | 5 / 18 | 2/2 | RC3 snapshot |
| [machsuite-stencil2d-624ee88fef89](kernels/machsuite-stencil2d-624ee88fef89/) | 76 / 124 | 5 / 28 | 4/4 | RC3 snapshot |
| [machsuite-stencil3d-af57e7d7cc9e](kernels/machsuite-stencil3d-af57e7d7cc9e/) | 154 / 533 | 5 / 41 | 9/9 | RC3 snapshot |
| [machsuite-viterbi-3b87b550b5de](kernels/machsuite-viterbi-3b87b550b5de/) | 288 / 630 | 5 / 99 | 8/8 | RC3 snapshot |
| [rodinia-backprop-0-baseline-back-d6cf9e8ef8ef](kernels/rodinia-backprop-0-baseline-back-d6cf9e8ef8ef/) | 103 / 197 | 5 / 24 | 2/2 | RC3 snapshot |
| [rodinia-backprop-0-baseline-forward-b0198e97c7a5](kernels/rodinia-backprop-0-baseline-forward-b0198e97c7a5/) | 219 / 426 | 5 / 60 | 6/6 | RC3 snapshot |
| [rodinia-backprop-1-tiling-back-160f006353fe](kernels/rodinia-backprop-1-tiling-back-160f006353fe/) | 426 / 929 | 5 / 100 | 10/10 | RC3 snapshot |
| [rodinia-hotspot-0-baseline-180968d0d9c8](kernels/rodinia-hotspot-0-baseline-180968d0d9c8/) | 683 / 1,544 | 5 / 240 | 3/3 | RC3 snapshot |
| [rodinia-hotspot-1-tiling-43864ef92829](kernels/rodinia-hotspot-1-tiling-43864ef92829/) | 578 / 1,258 | 5 / 145 | 8/8 | RC3 snapshot |
| [rodinia-hotspot-2-pipeline-c448680ca9be](kernels/rodinia-hotspot-2-pipeline-c448680ca9be/) | 695 / 1,505 | 5 / 164 | 14/19 | RC3 snapshot |
| [rodinia-hotspot-3-unroll-7c11a20afd22](kernels/rodinia-hotspot-3-unroll-7c11a20afd22/) | 695 / 1,505 | 5 / 164 | 14/19 | RC3 snapshot |
| [rodinia-kmeans-0-baseline-1f98cb95928e](kernels/rodinia-kmeans-0-baseline-1f98cb95928e/) | 84 / 150 | 5 / 25 | 3/3 | RC3 snapshot |
| [rodinia-kmeans-1-tiling-e8b72c3cc074](kernels/rodinia-kmeans-1-tiling-e8b72c3cc074/) | 280 / 450 | 5 / 99 | 12/12 | RC3 snapshot |
| [rodinia-kmeans-2-pipeline-c43d7535b7bf](kernels/rodinia-kmeans-2-pipeline-c43d7535b7bf/) | 280 / 450 | 5 / 99 | 12/12 | RC3 snapshot |
| [rodinia-kmeans-3-unroll-37698c1d6d86](kernels/rodinia-kmeans-3-unroll-37698c1d6d86/) | 280 / 450 | 5 / 99 | 12/12 | RC3 snapshot |
| [rodinia-knn-0-baseline-8ed65714afcc](kernels/rodinia-knn-0-baseline-8ed65714afcc/) | 62 / 100 | 5 / 20 | 2/2 | RC3 snapshot |
| [rodinia-knn-1-tiling-4abb3ffa4a5a](kernels/rodinia-knn-1-tiling-4abb3ffa4a5a/) | 225 / 361 | 5 / 76 | 9/9 | RC3 snapshot |
| [rodinia-knn-2-pipeline-f55616be8090](kernels/rodinia-knn-2-pipeline-f55616be8090/) | 225 / 361 | 5 / 76 | 9/9 | RC3 snapshot |
| [rodinia-knn-3-unroll-136c568e829a](kernels/rodinia-knn-3-unroll-136c568e829a/) | 232 / 372 | 5 / 85 | 10/10 | RC3 snapshot |
| [rodinia-knn-4-doublebuffer-f619da7770fe](kernels/rodinia-knn-4-doublebuffer-f619da7770fe/) | 395 / 672 | 5 / 101 | 12/12 | RC3 snapshot |
| [rodinia-knn-5-coalescing-b1e6b44c0ca5](kernels/rodinia-knn-5-coalescing-b1e6b44c0ca5/) | 435 / 771 | 5 / 116 | 13/13 | RC3 snapshot |
| [rodinia_cfd_flux_0_baseline_0-cbef52486b81](kernels/rodinia_cfd_flux_0_baseline_0-cbef52486b81/) | 1,107 / 3,625 | 5 / 354 | 2/2 | RC3 snapshot |
| [rodinia_cfd_step_factor_0_baseline_0-00cb85945c08](kernels/rodinia_cfd_step_factor_0_baseline_0-00cb85945c08/) | 331 / 700 | 5 / 60 | 1/1 | RC3 snapshot |
| [rodinia_cfd_step_factor_1_tiling_0-84093e267618](kernels/rodinia_cfd_step_factor_1_tiling_0-84093e267618/) | 441 / 917 | 5 / 78 | 4/4 | RC3 snapshot |
| [rodinia_cfd_step_factor_2_pipeline_0-8344119d5e5c](kernels/rodinia_cfd_step_factor_2_pipeline_0-8344119d5e5c/) | 445 / 915 | 5 / 87 | 7/7 | RC3 snapshot |
| [rodinia_cfd_step_factor_3_unroll_0-2292435ed81c](kernels/rodinia_cfd_step_factor_3_unroll_0-2292435ed81c/) | 674 / 1,972 | 5 / 96 | 8/8 | RC3 snapshot |
| [rodinia_cfd_step_factor_4_doublebuffer_0-3e7f8581709b](kernels/rodinia_cfd_step_factor_4_doublebuffer_0-3e7f8581709b/) | 1,132 / 2,805 | 5 / 181 | 18/18 | RC3 snapshot |
| [rodinia_cfd_step_factor_5_coalescing_0-1f19b5a03b65](kernels/rodinia_cfd_step_factor_5_coalescing_0-1f19b5a03b65/) | 1,038 / 2,588 | 5 / 181 | 18/18 | RC3 snapshot |
| [rodinia_dilate_0_baseline_0-21285821af18](kernels/rodinia_dilate_0_baseline_0-21285821af18/) | 130 / 322 | 5 / 38 | 5/5 | RC3 snapshot |
| [rodinia_dilate_1_tiling_0-84e35a273a31](kernels/rodinia_dilate_1_tiling_0-84e35a273a31/) | 315 / 619 | 5 / 99 | 12/12 | RC3 snapshot |
| [rodinia_dilate_2_pipeline_0-e7e749c3647c](kernels/rodinia_dilate_2_pipeline_0-e7e749c3647c/) | 447 / 880 | 5 / 157 | 20/20 | RC3 snapshot |
| [rodinia_dilate_3_pipeline_0-fc753e2f66aa](kernels/rodinia_dilate_3_pipeline_0-fc753e2f66aa/) | 593 / 1,242 | 5 / 242 | 35/35 | RC3 snapshot |
| [rodinia_lavaMD_0_baseline-37d49978e024](kernels/rodinia_lavaMD_0_baseline-37d49978e024/) | 415 / 2,562 | 5 / 138 | 5/5 | RC3 snapshot |
| [rodinia_lavaMD_1_tiling_0-9ed448bdff8c](kernels/rodinia_lavaMD_1_tiling_0-9ed448bdff8c/) | 541 / 2,570 | 5 / 211 | 14/14 | RC3 snapshot |
| [rodinia_lavaMD_1_tiling_1-eef05ed1fd64](kernels/rodinia_lavaMD_1_tiling_1-eef05ed1fd64/) | 545 / 2,577 | 5 / 211 | 14/14 | RC3 snapshot |
| [rodinia_lavaMD_2_pipeline_0-1b047179e0db](kernels/rodinia_lavaMD_2_pipeline_0-1b047179e0db/) | 541 / 2,570 | 5 / 211 | 14/14 | RC3 snapshot |
| [rodinia_lc_gicov_0_baseline_0-7b584fc86efb](kernels/rodinia_lc_gicov_0_baseline_0-7b584fc86efb/) | 370 / 1,429 | 5 / 116 | 11/11 | RC3 snapshot |
| [rodinia_lc_mgvf_0_baseline_0-8471f7950b77](kernels/rodinia_lc_mgvf_0_baseline_0-8471f7950b77/) | 355 / 724 | 5 / 82 | 3/3 | RC3 snapshot |
| [rodinia_lud_1_tiling_0-e35f0c4cd5d4](kernels/rodinia_lud_1_tiling_0-e35f0c4cd5d4/) | 1,479 / 2,790 | 5 / 424 | 46/46 | RC3 snapshot |
| [rodinia_pathfinder_0_baseline_0-5c80e31b25d0](kernels/rodinia_pathfinder_0_baseline_0-5c80e31b25d0/) | 221 / 512 | 5 / 43 | 4/4 | RC3 snapshot |
| [rodinia_pathfinder_4_doublebuffer_0-fcf143fb1d72](kernels/rodinia_pathfinder_4_doublebuffer_0-fcf143fb1d72/) | 466 / 1,024 | 5 / 81 | 8/8 | RC3 snapshot |
| [rodinia_streamcluster_0_baseline_0-fdd0bde6ff51](kernels/rodinia_streamcluster_0_baseline_0-fdd0bde6ff51/) | 166 / 328 | 5 / 28 | 2/2 | RC3 snapshot |
| [rodinia_streamcluster_1_tiling_0-099b9bf4ffc7](kernels/rodinia_streamcluster_1_tiling_0-099b9bf4ffc7/) | 762 / 1,466 | 5 / 116 | 14/14 | RC3 snapshot |
| [rodinia_streamcluster_2_pipeline_1-750e9c568d81](kernels/rodinia_streamcluster_2_pipeline_1-750e9c568d81/) | 762 / 1,466 | 5 / 116 | 14/14 | RC3 snapshot |
| [rodinia_streamcluster_3_doublebuffer_0-a5ac863bc37a](kernels/rodinia_streamcluster_3_doublebuffer_0-a5ac863bc37a/) | 999 / 1,893 | 5 / 144 | 18/18 | RC3 snapshot |
| [rodinia_streamcluster_4_coalescing_0-7f6373e7ee31](kernels/rodinia_streamcluster_4_coalescing_0-7f6373e7ee31/) | 1,107 / 2,023 | 5 / 144 | 18/18 | RC3 snapshot |
| [serrano-kalman-filter-41c8975cabb3](kernels/serrano-kalman-filter-41c8975cabb3/) | 541 / 1,006 | 5 / 123 | 11/11 | Historical development Kalman (11 registrations) |
| [serrano-kalman-filter-96b4d608969f](kernels/serrano-kalman-filter-96b4d608969f/) | 541 / 1,006 | 5 / 123 | 11/13 | Recommended development Kalman (13 registrations) |
| [serrano-kalman-filter-b90908f83f76](kernels/serrano-kalman-filter-b90908f83f76/) | 541 / 1,006 | 5 / 123 | 11/13 | External Kalman snapshot |
| [spcl_example_00-56c3d6e6a0e6](kernels/spcl_example_00-56c3d6e6a0e6/) | 47 / 76 | 5 / 15 | 1/1 | RC3 snapshot |
| [spcl_example_01-103dd1c12a83](kernels/spcl_example_01-103dd1c12a83/) | 69 / 117 | 5 / 15 | 1/1 | RC3 snapshot |
| [spcl_example_03-fdea84258953](kernels/spcl_example_03-fdea84258953/) | 95 / 162 | 5 / 54 | 6/6 | RC3 snapshot |
| [spcl_example_05-a6fc1fe5de8c](kernels/spcl_example_05-a6fc1fe5de8c/) | 91 / 153 | 5 / 48 | 5/5 | RC3 snapshot |
| [syr2k-ffe2543033f5](kernels/syr2k-ffe2543033f5/) | 101 / 202 | 5 / 38 | 4/4 | RC3 snapshot |
| [trmm-opt-7d58f595d68f](kernels/trmm-opt-7d58f595d68f/) | 70 / 119 | 5 / 25 | 3/3 | RC3 snapshot |

## Provenance and limitations

The compiler graph files decompress to the exact authenticated RC3 bytes. The archive, its historical manifest and checksums remain unchanged. The manifest records their hashes, profile identity and overlay source hashes. All compiler queries use the current repository implementation, including the incomplete-memory-root-set correction; archived helper code is not substituted.

The recent Hotspot native regeneration established unchanged compiler facts. All 19 actions are registered, but L1/L2/L3/L4/L6 still have no established compiler target. Corrected development Kalman registers 13 actions, with L7/L8 unresolved. These remain unresolved here; slots are not renumbered and targets are not guessed. See [Hotspot regeneration evidence](validation/hotspot-native-comparison.json) and [Kalman receipt](validation/kalman.json).

Physical capacities and requested deployment constraints remain separate. Published full-resource contexts assume no platform reservation; actual shell/platform availability is unknown. Hardware mapping, achievable II, scheduling, bank conflicts and timing closure are not established. A new V2.2 heterogeneous GNN encoder, training and Stage-2 integration remain future work.
