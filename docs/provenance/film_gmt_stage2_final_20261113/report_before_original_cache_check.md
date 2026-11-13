# Final FiLM / historical GMT Stage-2 preparation

**NOT READY for the requested recovered 49-train + 4-validation cohort.**
**READY for an explicitly selected, independently authenticated original 31+2 fallback.**
No full 450-update Stage-2 campaign was launched. The selected V4 parent, Arm B
step-600 residual LoRA/FiLM, historical epoch-9 GNN, raw measurements, and split
are unchanged. All work is isolated on `research/film-gmt-stage2-final-20261113`.

## Expanded-cohort blockers

Canonical source-level eligibility remains 54: 49 train, four validation, one locked
test. Qualification stopped **before aggregation, Pareto selection or serialization**;
no partial recovered bank was published. The frozen V4 registry lacks 27 RHS keys
for the following nine kernels. The historical graphs also lack the corresponding
complete action inventories / metadata identities. Source hashes themselves match.

| Kernel | Missing action IDs | Fold |
|---|---|---|
| `rodinia_cfd_flux_0_baseline_0` | L1 | train |
| `rodinia_cfd_step_factor_1_tiling_0` | L5 | train |
| `rodinia_cfd_step_factor_2_pipeline_0` | L5, L9 | train |
| `rodinia_cfd_step_factor_3_unroll_0` | L9 | train |
| `rodinia_lavaMD_0_baseline` | L1 | train |
| `rodinia_lavaMD_1_tiling_0` | L1 | train |
| `rodinia_lavaMD_1_tiling_1` | L1 | train |
| `rodinia_lavaMD_2_pipeline_0` | L1 | train |
| `rodinia_pathfinder_0_baseline_0` | L1, L2 | val |

These are contract / compiler-graph blockers, not unknown historical directive
states. No old registry entries, source actions, graph identities or hashes were
changed to bypass them. Completing this cohort requires a separately declared
source-authenticated Stage-2 domain contract and authentic historical graph
regeneration/qualification. No partial-cohort override was used.

All 53 non-test sources received a static **historical** graph audit: 44 pass
source/action/generator checks; nine need regeneration. Only the complete original
31+2 cohort received numerical GNN/GMT qualification. RC3 compatibility is a
different release contract and is not used to qualify this historical encoder.
The measured-source coalescing KNN version remains separately namespaced under
the existing recovery provenance; it was not substituted, fitted or exported.
Locked-test memory export was omitted; no locked-test QoR rows were parsed.

## Qualified fallback and sampling

The new immutable fallback directory contains `train_sft.jsonl`, `train_pairs.jsonl`,
`val_sft.jsonl`, `val_pairs.jsonl`, the byte-identical selected Stage-1 manifest,
and a separate versioned `stage2_manifest.json`. It is explicitly CSV/APL complete;
it does not pretend to contain recovered sidecar data. The recovery join API is
tested to join complete historical states before configuration grouping; unknown,
incomplete and locked-test assignments fail closed. The expanded-bank loader has
a distinct schema, full-cohort/source/contract/hash gates and preserves original
Stage-1 authentication.

| Quantity | Training | Validation |
|---|---:|---:|
| Supervised kernels / families | 31 / 15 | 2 / 2 |
| SFT references | 3,276 | 216 |
| Budget contexts in bank | 1,237 | 72 |
| Device/clock targets in bank | 181 | 12 |
| Preference examples / complete two-context groups | 498 / 249 | 28 / 14 |

The old 112-context cap and `steps // 3` ceiling are removed. At 450 updates and
four SFT microbatches/update, **all 1,237 training contexts receive one or two
visits**. Every preference group (128 budget, 121 clock) is visited with **both**
target contexts; its two ranking losses are averaged to preserve objective scale.
Twenty-four fixed validation SFT references cover all 12 device/clock targets,
both validation kernels and 20 distinct budget contexts; eight complete preference
groups are used. This is prospective schedule coverage, not a claim that training
has happened. The five missing Cartesian training targets were already absent
from the original authenticated bank: CFD baseline U200/10 ns and four SPCL-01
targets; no new outcomes are invented to fill them.

Exact prompt/RHS token audit covers every SFT reference and both paired alternatives:
maximum 6,660 tokens, below 7,168, with no truncation. The GPU smoke uses the longest
training SFT example (CFD coalescing, 6,659 tokens). Full budget remains 450 updates.

## Historical GNN and GMT receipts

The exact epoch-9 four-layer GNN uses its captured 403-node / 82-edge tensorizer,
training artifacts and normalization; no GNN retraining or normalization refit.
All original 64 x 128 action anchors are preserved bit-identically in the exported
memory. Appended summaries preserve memory, compute/dependence and control roles;
every active action has genuine compiler evidence and valid relational routing.
No empty, action-only, truncated or mismatched-source fallback is accepted.

One `HistoricalRoleGMT` run completed **600 auxiliary updates**; internal selection
chose **update 330**. The model fit 27 original training kernels; the four KMeans
variants were a family-disjoint internal development group within Stage-2 training.
The two Stage-2 validation families did not participate in GMT fit/selection or
historical GNN gradient fitting. Historical GNN checkpoint selection did use its
held-out validation ranking, and this inherited selection exposure is explicit.
Frozen normalization has `training-kernels-only` fit policy.

Evidence cap is now **16,384**, with the strict overflow rejection retained. The
two largest qualified CFD cases contain 14,477 and 14,281 scoped evidence records;
both pass without truncation. Initial source-size preflight also checked CFD flux,
which correctly failed its historical action-metadata identity. No stored graph
was rewritten to hide that failure.

| Kernel | Fold | Active actions | Evidence records | Memory rows (64 anchors included) |
|---|---|---:|---:|---:|
| `machsuite-gemm-blocked` | train | 5 | 390 | 79 |
| `machsuite-gemm-ncubed` | train | 3 | 183 | 73 |
| `machsuite-md-knn` | train | 2 | 463 | 70 |
| `machsuite-spmv-ellpack` | train | 2 | 146 | 70 |
| `machsuite-stencil2d` | train | 4 | 287 | 76 |
| `machsuite-stencil3d` | train | 9 | 902 | 91 |
| `machsuite-viterbi` | val | 8 | 1195 | 88 |
| `rodinia-backprop-0-baseline-back` | train | 2 | 245 | 70 |
| `rodinia-backprop-0-baseline-forward` | train | 6 | 621 | 82 |
| `rodinia-backprop-1-tiling-back` | train | 10 | 998 | 94 |
| `rodinia-hotspot-0-baseline` | train | 3 | 4564 | 73 |
| `rodinia-hotspot-1-tiling` | train | 8 | 3404 | 88 |
| `rodinia-hotspot-2-pipeline` | train | 19 | 5613 | 121 |
| `rodinia-hotspot-3-unroll` | train | 19 | 5613 | 121 |
| `rodinia-kmeans-0-baseline` | train | 3 | 277 | 73 |
| `rodinia-kmeans-1-tiling` | train | 12 | 1013 | 100 |
| `rodinia-kmeans-2-pipeline` | train | 12 | 1013 | 100 |
| `rodinia-kmeans-3-unroll` | train | 12 | 1013 | 100 |
| `rodinia-knn-0-baseline` | train | 2 | 120 | 70 |
| `rodinia_cfd_step_factor_0_baseline_0` | train | 1 | 492 | 67 |
| `rodinia_cfd_step_factor_4_doublebuffer_0` | train | 18 | 14281 | 118 |
| `rodinia_cfd_step_factor_5_coalescing_0` | train | 18 | 14477 | 118 |
| `rodinia_dilate_0_baseline_0` | train | 5 | 1530 | 79 |
| `rodinia_dilate_1_tiling_0` | train | 12 | 2835 | 100 |
| `rodinia_lc_gicov_0_baseline_0` | train | 11 | 5379 | 97 |
| `rodinia_lc_mgvf_0_baseline_0` | train | 3 | 2165 | 73 |
| `rodinia_pathfinder_4_doublebuffer_0` | val | 8 | 1704 | 88 |
| `rodinia_streamcluster_0_baseline_0` | train | 2 | 239 | 70 |
| `rodinia_streamcluster_1_tiling_0` | train | 14 | 1830 | 106 |
| `rodinia_streamcluster_2_pipeline_1` | train | 14 | 1830 | 106 |
| `rodinia_streamcluster_3_doublebuffer_0` | train | 18 | 2704 | 118 |
| `rodinia_streamcluster_4_coalescing_0` | train | 18 | 2751 | 118 |
| `spcl_example_01` | train | 1 | 64 | 67 |

## Validation and engineering changes

CPU tests: **74 passed, 19 subtests passed**; one historical GMT test also passes.
Canonical/cache integrity and both recovery verifiers pass. All original raw CSV,
source/header, mapping, split and checkpoint identities are unchanged.
Four exact `graph.log` extracts referenced by recovery commit `2e72d26` had been
omitted by the global `*.log` ignore. They were restored from the existing local
recovery worktree by their already stored hashes, and a scoped ignore exception
now makes fresh checkouts complete. No stored receipt/hash was regenerated.

The first composite smoke exposed an eval-mode backward diagnostic that disabled
checkpointing on the long CFD example; its failed output is retained. The fixed
probe uses checkpointed training mode. A subsequent successful BF16-emulation
run exposed PyTorch's permissive V100 BF16 detection; capability gating now enforces
native **FP16** on Volta, and the final native-FP16 smoke passes. These are execution
fixes, not architecture experiments. Only standalone smoke updates occurred, with
no trained Stage-2 checkpoint promoted or full campaign started.

Four structural layers remain at 8/16/24/32, with zero initial gates and the original
60-update gate-opening curriculum. V4, residual LoRA, FiLM and historical memories
remain frozen; only 8,683,524 structural attention/gate parameters train. Action-
normalized RHS SFT, target-aware ranking and structural ON/OFF diagnostics remain.

Native-FP16 zero-gate parity absolute delta: **0.0**. All four gate gradients are finite/nonzero. Opened-branch projection gradients span 1.8e-05 to 0.0102. At the declared diagnostic gate `tanh(g)=0.05`, mean absolute ON/OFF score delta is 0.0009505; this is a pathway test, not learned QoR improvement.

| Layer | Actual post-rounding residual RMS / hidden RMS |
|---|---:|
| 8 | 0.01029053 |
| 16 | 0.00449685 |
| 24 | 0.00191902 |
| 32 | 0.00080484 |

## Paths and manual commands

These paths select the **qualified original fallback**, not a recovered 49+4 bank.

```bash
ROOT=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final
PARENT=/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run
FILM_CP=/home/elvouvali/MailoHLS/worktrees/gnosis-stage1-film-ab/artifacts/gnosis_stage1_ab/stage1_ab_pilot_20261112_retry2/B/checkpoint-0600
GNOSIS_BANK=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/artifacts/film_gmt_stage2_final_20261113_v1/fallback_bank
GMT_MEMORY=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory/memory_B
GPU_UUID=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
ENTRY=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/experiments/film_gmt_stage2_final/train.py
RUN_ROOT=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/artifacts/film_gmt_stage2_final_20261113_v1/campaign
```

`GPU_UUID` is a device identifier. `campaign` has no training output yet.

```bash
COHORT=original-film-31+2 MODE=audit bash /home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/experiments/film_gmt_stage2_final/commands.sh
COHORT=original-film-31+2 MODE=smoke bash /home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/experiments/film_gmt_stage2_final/commands.sh
COHORT=original-film-31+2 MODE=train bash /home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/experiments/film_gmt_stage2_final/commands.sh
```

Outputs must be fresh; resume requires an authenticated saved optimizer checkpoint.
The final `MODE=train` command is supplied for manual use and was **not executed**.
Do not set `ALLOW_EXPLICIT_SUBSET=1` (the launcher rejects it).

## SHA-256 identities and recommendation

- Dataset four-file identity: `958da104bb3ad1a20f4d4cae3c1bc9464e6ac1c3b1a9e2ad3144a4b7c339a9cf`
- Versioned Stage-2 dataset manifest: `f4eb34e35c64879c11dfa9ccf4cb74979a218a93f899739025ee935c2fbc0d60`
- Original selected Stage-1 bank manifest: `2b5745ab029a32f3f0088367b80a7478212249912a7ee8e7b890fdaffc2cbf0c`
- GMT memory manifest: `e7410298eb39887cad074d9d0578fe4df745434346b9f7c9a2a6d1c843fb05f8`
- Complete memory identity: `f15b7201ca5b3280b20b405d5f43da2dc8cc5c13bce8213a2a407d6337d9aeeb`
- V4_parent: `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`
- FiLM_residual: `6628fe1f268f93ce760428ca0c50b46dbcbfe55e00d45fba97a20142e89b2f80`
- FiLM: `d2697977c97c467c7f433b134dff3031c4e842f3b8f2cf9f0c57613c1ecf46ae`
- historical_GNN: `34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623`
- selected_GMT: `1d4cdf3071375349b94809062d05d30ad3e5c981bb6e2885b6f99c9e94398c12`

Complete per-file checkpoint, graph, source, action, memory and runtime hashes,
selection receipts and failed/successful validation output are in
[`docs/provenance/film_gmt_stage2_final_20261113`](provenance/film_gmt_stage2_final_20261113/).
Existing FiLM/GNN/GMT and paper results were not overwritten. No push was made.

**Choose the explicitly qualified original 31+2 cohort for a manual campaign, or
keep the expanded campaign stopped until the nine domain/graph contracts are
resolved. Do not label this fallback as the requested 49+4 integration.**
