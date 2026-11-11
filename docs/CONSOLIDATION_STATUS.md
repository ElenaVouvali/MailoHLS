# Local consolidation status — 11 November 2026

Branch `final-architecture` already existed at foundation
`3bd533b5c489a0ad7487c32d0fccfcdaf6657180`. Existing README/design-note/evidence
edits were preserved and extended. This is the documentation/evidence and
dry-run inventory phase. No tracked-tree removal or package refactor is authorized
by this report.

Ignore rules now also cover future wandb/TensorBoard logs, FPGA tool cache
directories and explicitly named local Vitis build roots. These rules do not
untrack existing evidence or remove host artifacts.

The selected CPU contract suite passed **218 tests, zero skips, zero failures**
in 73.62 seconds with CUDA hidden and Hugging Face offline. It covers domains,
source registration, sparse slots, memory identity, graph semantics, fusion,
decoder legality, DPO loss/masks and refinement. These are software checks;
synthetic fixtures are not synthesis or real-checkpoint evidence. The exact
[environment receipt](evidence/receipts/consolidation_environment.json) records
the existing Python 3.10.20 stack. See [reproduction](REPRODUCIBILITY.md).
The compact [validation receipt](evidence/receipts/consolidation_validation.json)
records test counts and the local JUnit report SHA-256.

The real Stage-1 adapter and domain registry match the historical authenticated
SHA-256 values, and the saved tokenizer passes the maintained loader's token-ID
mapping check ([asset receipt](evidence/receipts/stage1_asset_preflight.json)).
A new complete real decode is **not performed**. The current host GPU preflight found both devices
occupied by Ray workers (A2: 13,666 MiB; V100S: 29,702 MiB). No workload was
stopped and no GPU was allocated for this consolidation. Historical generation
receipts and CPU tests do not close that release gate.

The local GNN ranking continuation has completed. Its completion receipt
authenticates the report and decision, and its selected checkpoint SHA matches
the local file. The result is `FREEZE_NOT_YET_JUSTIFIED`; no model was promoted.
See [local provenance](evidence/LOCAL_SOURCE_MAP.json). Liono's default tmux
socket reports no running server; process visibility in the sandbox is limited.
Davinci and Coroni process/scheduler status were not inspected, so no global
job-idleness or branch-retirement claim is made.

A read-only `git ls-remote --heads origin` confirmed **17 remote branches**;
`final-architecture` is currently local. `llvm-ir` and `mlir-target` match their
recorded historical SHAs. The recorded remote inventory is preserved in
[remote heads](evidence/receipts/remote_heads_20261111.tsv). No fetch, pruning,
push, default-branch change, branch deletion, training or synthesis was performed.

The [read-only audit](../scripts/maintenance/audit_final_architecture.py) verifies
Git evidence hashes/reachability, local receipt hashes, maintained links and a
tracked-tree dependency inventory. Cleanup candidates remain retained pending
signed-off source/dependency and artifact-preservation review. No bundle was
created and no runtime artifact was deleted. Root licensing, software citation
authorship/version and portable environment reproduction remain review gates;
existing third-party notices remain intact.

The [readable inventory](CONSOLIDATION_INVENTORY.md) lists directory counts,
concrete dependencies and all 17 branch dispositions. Its machine counterpart
is the [compact inventory receipt](evidence/receipts/consolidation_inventory.json).

The historical [six-branch consolidation report](REPOSITORY_CONSOLIDATION_REPORT.md)
describes an earlier snapshot. Its branch counts and main-authority statements
do not describe this new research consolidation. Current component qualification
is in [the evidence matrix](ARCHITECTURE_EVIDENCE.md).
