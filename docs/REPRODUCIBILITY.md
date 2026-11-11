# Reproducing the selected research control

Start with [component status](ARCHITECTURE_EVIDENCE.md) and
[current checks](CONSOLIDATION_STATUS.md). The source foundation is
`3bd533b5c489a0ad7487c32d0fccfcdaf6657180`. Production selections remain in
[FINAL_ARTIFACTS.json](../FINAL_ARTIFACTS.json); the new latency study has not
replaced them. Preserve `LLM_branch/`, `GNN_branch/`, `src/` and saved paths.

## Inspect without weights or hardware

Python's standard library and Git suffice for the evidence and tracked-tree audit:

```bash
python scripts/maintenance/audit_final_architecture.py --output /tmp/salt-audit-new.json
python scripts/artifacts.py verify --artifact salt-hls-graphs-v22-20261027-rc3
```

The audit refuses an existing output file. It verifies byte identity against
source Git blobs, containing refs, copied local-receipt hashes and maintained
local documentation paths. Its inventory includes tracked counts, largest files
and source references for cleanup candidates; it authorizes no deletion. Copied
reports preserve original relative links; use their exact source commits in
[SOURCE_MAP.tsv](evidence/SOURCE_MAP.tsv) to follow those links.

See the [GEMV walkthrough](../examples/README.md) for source/action/graph inspection.
Artifact recovery and raw-data access are described in the existing
[artifact guide](reproducibility.md). Frozen graphs can be inspected without
rebuilding the compiler. Regeneration requires the exact
[Polygeist lock and ordered patches](../toolchain/README.md).

## CPU contract checks

The existing audited research stack is Python 3.10, Torch 2.6.0+cu124,
Transformers 4.51.0, PEFT 0.18.1 and PyG 2.6.1. This differs from the historical
root requirements file; that file is not claimed to be a freshly verified lock.
The current host uses the existing `mailohls-llm-v2` conda environment and the
existing `/tmp/mailohls-pyg-audit-site` dependency overlay. The exact package
receipt is linked from [current checks](CONSOLIDATION_STATUS.md). That overlay
is local infrastructure, not a portable dependency distribution.

With the matching dependencies installed, explicitly hide CUDA and prevent model
downloads. These selected checks do not launch training or HLS:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
python -m pytest -p no:cacheprovider -q \
  LLM_branch/tests/test_structural_xattn.py \
  LLM_branch/tests/test_sparse_memory_slots.py \
  LLM_branch/tests/test_source_directive_domains.py \
  LLM_branch/tests/test_inference_schema.py \
  LLM_branch/tests/test_stage2_production_contract.py \
  LLM_branch/tests/test_stage3_dpo_contract.py \
  tests/test_directive_domains.py tests/test_paper_memory_identity.py \
  tests/test_mlir_graph_v22.py tests/test_refinement.py \
  tests/test_refinement_search.py experiments/final_architecture_v3/tests
```

Native compiler tests require the pinned external frontend and bindings; record
any skips or failures. Do not replace these with a GPU/HLS experiment sweep.

## Real checkpoint gate and host responsibilities

Before an inference release, authenticate the exact base-model revision,
tokenizer, learned token deltas, adapter, source/headers, action-domain registry,
memory schema/checkpoint/normalization, reference masks and request. Decode one
complete assignment, then validate every source action and conditional field.
The current consolidation has **not performed this gate**. Existing historical
generation receipts and synthetic test models do not substitute for it.

Liono holds GNN runtime artifacts; the completed local study's paths and hashes
are recorded in [LOCAL_SOURCE_MAP.json](evidence/LOCAL_SOURCE_MAP.json). Davinci
holds the completed fused-policy training/generation lineage; follow the exact
[research protocol](../experiments/final_architecture_v3/README.md). Inspect
current ownership before allocating GPUs. No new jobs are needed for this audit.

All new measured HLS feedback belongs on `bonus24@coroni.microlab.ntua.gr`,
with authenticated Vitis HLS 2021.1 and exact source/header/top/part/clock/flow
identity. `python -m LLM_branch.inference.optimize --help` describes the opt-in
controller. Refinement disabled makes zero HLS calls. Keep estimated timing,
cosimulation and post-route results separate. Never restart a canceled ledger
with its charged attempts erased.

## Publication gates

No new root license or citation record is supplied without verified rights,
authors and version. Review [third-party provenance](third-party-provenance.json)
and retain [HARP's notice](../HARP-rep/LICENSE). A reproducible supervisor checkout
also requires the external model/data access plan and preserved source commits.
Keep `main`, `llvm-ir`, `mlir-target` and all research refs until the separate
archive/dependency review is signed off. Publishing, default-branch cutover and
branch retirement are separate operations from this local documentation audit.
