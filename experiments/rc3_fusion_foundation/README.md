# RC3 Structural-Fusion Foundation

This package builds and checks a small, frozen compiler-memory interface for later latency-focused Stage-2 experiments. It does not train Stage-1, the GNN, or cross-attention and does not alter the production V1 Stage-2.

## Authenticated inputs

The isolated branch `research/rc3-fusion-foundation` is based on GNN commit `172e05a69650a666c323955ea78cb501fd9364c2`. At execution, the GitHub heads checked for this work matched: GNN `172e05a69650a666c323955ea78cb501fd9364c2`, audit `bb4bc8f8261e33f7a229d92f30e36a0a53f986a7`, Stage-1 `63afcfc7a54e573a544678e728cdb7d99d865981`, full-depth study `97ed75ed3f41ad80247d739888d9ee488881872c`.

The V4b GNN checkpoint is `compiler_selected.pt`, SHA-256 `ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162`, selected step 1500. It was trained with physical-QoR/ADP-oriented supervision and is used here only as a frozen encoder for interface diagnostics; it is not latency-qualified. Its two-layer compiler-only encoder generates the exact same action vectors for A, B, and C.

The canonical RC3 graph is the 66-node/106-edge graph `42fe84cb8b8c2f5665dc016fcfb6a969f93b848a3407c7343100b6b4bcabae21`, file SHA-256 `7d295d791f8438eeda62a6b230ab80c8d3a051ecda43b822c7a3827ec0cb43a6`. Three already compiler-built, action-mapped full-depth diagnostics (initialized reduction, independent-iteration, and reduction rename control) are also encoded. The independent variant changes the computation and is not claimed equivalent.

## Implemented

- `rc3_action_relations.py`: versioned directed routes derived from compiler query evidence and provenance, not a copied GEXF traversal or action-pair lookup table.
- `structural_memory_v1.py`: three frozen-memory views: exact action slots A, action vectors plus RC3 eligibility B, and the same action vectors plus selected typed node/edge evidence C.
- `build_memory.py`: CPU-only artifact builder using the authenticated RC3 tensorizer and frozen V4b checkpoint.
- `test_rc3_memory.py`: tests the derived routing, absolute slots, selective evidence, padding, and the production `MaskedCrossAttention` masking behavior on synthetic input.
- `prepare_synthesis.py`: validates and freezes a small initialized-GEMV Vitis job bundle; it never invokes synthesis.

## Reproduction

Run from the isolated GNN-based worktree. The known CPU environment is `/tmp/mailohls-v22-gnn-env-20261030/bin/python` (PyTorch 2.6.0 and PyG 2.6.1); no GPU is required.

```bash
PYTHONPATH="$PWD" /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.rc3_fusion_foundation.build_memory
PYTHONPATH="$PWD" /tmp/mailohls-v22-gnn-env-20261030/bin/python -m pytest experiments/rc3_fusion_foundation/test_rc3_memory.py -q
PYTHONPATH="$PWD:/tmp/MailoHLS-stage1-refinement" /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.rc3_fusion_foundation.prepare_synthesis --out artifacts/rc3_fusion_foundation/v1_20261107/synthesis_bundle_reproduction
PYTHONPATH="$PWD" /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.rc3_fusion_foundation.export_manifests
```

Generated tensor files and frozen synthesis inputs live under ignored `artifacts/rc3_fusion_foundation/v1_20261107/`; their hashes and source identities are recorded in `manifests/provenance.json`. The build fails on graph/checkpoint mismatch, incomplete absolute action masks, or evidence overflow; it never silently truncates evidence. Capacity is 128 records, chosen to hold the largest current diagnostic graph (82 records).

## Boundaries

Memory A/B/C interfaces are CPU-tested read-only adapters. They do not establish that C contains useful information, that V4b is the appropriate latency encoder, or that any attention layer helps. No V100S is occupied. Vitis HLS 2021.1 was not available at the recorded executable path and was not run; the prepared jobs are not QoR results. See `GEMV_MEMORY_COVERAGE.md`, `MEMORY_INTERFACE.md`, `SYNTHESIS_BENCHMARK.md`, and `NEXT_ABLATIONS.md`.
