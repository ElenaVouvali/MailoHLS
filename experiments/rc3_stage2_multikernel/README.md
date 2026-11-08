# Multi-kernel Stage-2 C pilot

This is a bounded, family-disjoint Stage-2 experiment using the production V4 multi-reference Stage-1 target and frozen V4b compiler-only encoder. The GNN remains a frozen feature extractor (its ADP/QoR supervision is not latency-qualified). Stage-2 training and inference reuse the production trainer, production constrained decoder, and existing masked cross-attention implementation.

## Current evidence and limits

See [COVERAGE.md](COVERAGE.md) for the dataset and label audit and
[RESULTS.md](RESULTS.md) for the live run status and resume command.

The authenticated V4 train.jsonl SHA-256 is 92ffd967652caa5e56a305625b72e03a9defa9c75c78a36cee09c259d10d0eba; it contains 2,578 rows (2,524 train, 54 validation), no test rows. The RC3 archive SHA-256 is feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73, with 61 canonical graphs. All 40 V4 kernels have a graph and exact clean-source hash match; two training kernels have incomplete compiler matching and are excluded. This leaves 35 train kernels/17 families and all 3 validation kernels/2 disjoint families. The deterministic pilot uses one fully mapped representative kernel per training family, all four authenticated multi-reference exposures, and all 54 validation rows. This is 1,164 training examples and 54 validation examples; test data is not accessed.

All 2,458 retained train/validation complete assignments pass current source-derived TCAD whole-action tuple validation. Curated rows pass their requested BRAM/DSP/FF/LUT budgets using the recorded utilization percentages; these percentages are rounded. The curated file does not contain per-candidate achieved/estimated HLS clock, and it is not a complete candidate bank. Therefore the dataset supports a latency-focused multi-reference directive SFT pilot but does not certify timing-qualified winners or global constrained optima. No budget-binding claim is made solely from the filtered rows.

The frozen V4b encoder is ADP/QoR-trained, not latency-selected. Memory C is static per source; target context reaches cross-attention only through the frozen hardware-conditioned LLM query.

## Reproduction

1. Audit and prepare the exact family-balanced data:
   python -m experiments.rc3_stage2_multikernel.audit
2. Build Memory C using an environment with the repository's authenticated PyG dependencies:
   PYTHONPATH=$PWD LD_LIBRARY_PATH=<torch-and-cuda-library-path> python -m experiments.rc3_stage2_multikernel.build_memory
3. Run the one-layer smoke only after assigned V100S preflight (the wrapper
   refuses to use a busy GPU):
   python -m experiments.rc3_stage2_multikernel.train --stage smoke --execute
4. Inspect smoke metrics, zero-gate initialization, and validation behavior. Then, as a separate decision:
   python -m experiments.rc3_stage2_multikernel.train --stage pilot --execute

By default, the training wrapper refuses to start if the required V100S UUID is missing, any V100S compute process is active, free memory is below 24 GiB, or utilization exceeds 10%. The user explicitly authorized sharing with the current Ray workers; in that mode, use `--allow-gpu-sharing`, which permits only Ray worker processes on the required V100S and requires at least 12 GiB free and below 95% utilization. It never terminates or reconfigures those jobs. Shared execution can run slower and can still fail if Ray expands its allocation. The wrapper uses the existing trainer with frozen Stage-1 LoRA/special embeddings, frozen GNN-derived memories, fresh attention weights, action-normalized CE, exact constrained target serialization, and family-disjoint validation. Variable evidence-key counts are padded only within each training batch; false masks prevent any access to padding while preserving absolute action slots.

Large data, graph, memory, checkpoint, and run artifacts are under ignored artifacts/rc3_stage2_multikernel/ or /home/elvouvali/mailohls_runs/rc3_stage2_multikernel/. The temporary C memory format is versioned and backward-compatible with action-only memories.
