# MailoHLS framework development status

**Status date:** 2026-10-31  
**Published production baseline:** `main` at `d488681f68e58ab1b1e666f6911f5257d6462dc2`  
**Last recorded target-aware GNN training status (research branch `bf93db9`):** implementation and preflight are documented; full paired training has not run and no V2.2 trained checkpoint exists.

This page separates code published on `main` from research implementations on
their own GitHub branches. A published branch is not thereby merged into `main`
or promoted to production.

## What is production today

The supported end-to-end SALT-HLS pipeline uses the existing V1 GNN and
structural-memory contract with the selected Stage-1/2/3 checkpoints recorded in
[`FINAL_ARTIFACTS.json`](../FINAL_ARTIFACTS.json). Those weights and contracts
have not been replaced by the V2.2 work. Current evaluation and training entry
points remain under `scripts/` and `LLM_branch/`.

The latest compiler-derived graph release on `main` is MLIR V2.2 RC3: 63
authenticated snapshots, canonical and compact forms, and recorded checksums.
It is technically frozen, but independent human semantic adjudication is still
pending. The release is documented in the
[graph guide](structural-graph-v22.md),
[release review](V22_FINAL_RELEASE_REVIEW.md), and
[manifest](graph-release-v22-rc3.json).

The FPGA overlay, action-workload analysis and semantic tensor encoder are
research extensions. They do not alter historical RC3 graphs and do not make
old V1 tensors or checkpoints compatible with the newer representation. No
measured QoR improvement or cross-device generalization claim has been
established for these extensions.

## Development map

| Work | GitHub branch / location | Current status |
|---|---|---|
| Production SALT-HLS and frozen V2.2 RC3 release | [`main`](https://github.com/ElenaVouvali/MailoHLS/tree/main) | Available; production inference remains V1. |
| FPGA target overlay and profiles | [`research/target-aware-mlir-v22`](https://github.com/ElenaVouvali/MailoHLS/tree/research/target-aware-mlir-v22) | Validated composition; compatibility is not synthesis binding. |
| Heterogeneous tensor readiness pilot | [`research/target-graph-readiness`](https://github.com/ElenaVouvali/MailoHLS/tree/research/target-graph-readiness) | Conditional readiness result; no evidence of improved QoR/generalization. |
| Static and candidate action-workload features | [`research/v22-action-hardware-demand`](https://github.com/ElenaVouvali/MailoHLS/tree/research/v22-action-hardware-demand) | Analytical descriptors, not actual synthesis costs. |
| Semantic tensor schema v3 | [`research/v22-semantic-tensor-enrichment`](https://github.com/ElenaVouvali/MailoHLS/tree/research/v22-semantic-tensor-enrichment) | Exposes supported affine, dependence, view, SSA and call evidence. |
| Corpus semantic-coverage refinement | [`research/v22-semantic-coverage-repair`](https://github.com/ElenaVouvali/MailoHLS/tree/research/v22-semantic-coverage-repair) | Coverage audit/refinement; RC3 archive unchanged. |
| First supervised V2.2 GNN implementation | [`research/v22-target-aware-gnn-training`](https://github.com/ElenaVouvali/MailoHLS/tree/research/v22-target-aware-gnn-training) | Training code and data audit; earlier run produced no checkpoints. |
| Resumable V2.2 paired-training protocol | [`research/v22-gnn-resumable-20261030`](https://github.com/ElenaVouvali/MailoHLS/tree/research/v22-gnn-resumable-20261030) | Preflight and safe launcher documented; last recorded full-training attempt blocked by GPU contention. |
| Hotspot V2.2 graph export | [`research/hotspot-v22-graph-export-20261028`](https://github.com/ElenaVouvali/MailoHLS/tree/research/hotspot-v22-graph-export-20261028) | Two kernels, 48 target-enriched variants, checksums and reproduction package. |
| Stage-1 decision-learning recovery | [`research/stage1-directive-learning-repair`](https://github.com/ElenaVouvali/MailoHLS/tree/research/stage1-directive-learning-repair) | Tokenizer correction and diagnostics; selected checkpoints unchanged. |
| Stage-1 measured-quality assessment | [`research/stage1-measured-quality`](https://github.com/ElenaVouvali/MailoHLS/tree/research/stage1-measured-quality) | Validation-only assessment; not a promoted model. |
| Historical baselines | [`mlir-target`](https://github.com/ElenaVouvali/MailoHLS/tree/mlir-target), [`llvm-ir`](https://github.com/ElenaVouvali/MailoHLS/tree/llvm-ir) | Preserved historical branches, not modified by these experiments. |

The full 1,512-graph catalog is available as a
[checksum-verified Release](https://github.com/ElenaVouvali/MailoHLS/releases/tag/archive/v22-graph-catalog-20261029).
Its historical branch remains preserved; no branch retirement was performed.
The focused Hotspot export is a separate smaller artifact and does not replace
the frozen corpus release.

## Target-aware GNN: exact current point

The training implementation has a source-authenticated audit covering 55
GNΩSIS kernels and 104,147 eligible measured configurations (88,466 train,
8,976 validation, 6,705 test). It retains measured values for registered but
compiler-unresolved actions with unresolved masks. It excludes records where
the action column is not measured and source-domain-invalid tuples rather than
inventing labels. The held-out test partition contains Kalman only.

The numerical-conditioning and hardware-message-passing arms, their QoR and
ranking objectives, candidate-demand inputs, validation checkpoint-selection
rule, and detached/resumable launch tooling are implemented on research
branches. Authenticated data checks, focused tests, A2 forward/backward memory
profiles, and measured-candidate gradient checks are documented in the
[training audit](https://github.com/ElenaVouvali/MailoHLS/blob/research/v22-gnn-resumable-20261030/experiments/v22_gnn_training/AUDIT.md)
and [results record](https://github.com/ElenaVouvali/MailoHLS/blob/research/v22-gnn-resumable-20261030/experiments/v22_gnn_training/RESULTS.md).

The paired interruption/resume smoke and full training were **not launched**:
the recorded preflight found the A2 and V100S occupied by unrelated processes, and the safe launcher
refused to share either GPU. There are no Arm A or Arm B checkpoints, no
validation-selected model and no held-out QoR result. Consequently, there is no
evidence yet that hardware-message passing improves over numerical target
conditioning. U50 and 6.67 ns remain future evaluation contexts, and the
single-kernel Kalman test cannot establish broad unseen-family performance.

## What a researcher can do next

1. Review the published research branches above; the resumable training branch
   contains the paired V2.2 experiment protocol.
2. Follow its `experiments/v22_gnn_training/RUN_PROTOCOL.md` to rerun the safe
   preflight and launch when an approved GPU is genuinely available.
3. Report validation-selected and held-out results before making QoR or
   generalization claims.
4. Treat Stage-2 consumption of target-context-specific 128D memories as a
   separate future integration; it has not been trained or wired into the
   production cross-attention path.

No branch retirement, merge, checkpoint promotion or dataset-version change is
implied by this status page.

This maintenance branch reuses the earlier consolidation base `7c6844f`. Later
semantic/training branches in the map remain separate. GPU availability and run
outputs were not re-probed during repository maintenance; status above is the
last committed research receipt, not a live process monitor.
