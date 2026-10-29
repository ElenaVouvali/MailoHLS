# Completed Stage-1 recovery experiments — 2026-10-28

The tokenizer defect is repaired. Four V100 training controls and all planned validation comparisons completed. **No experimental adapter qualifies as a production replacement.** The original checkpoint's adapter, training contract and tokenizer hashes remain unchanged.

See [measured results JSON](results/20261028.json), [method and reproduction guide](README.md), and the local detailed artifacts under `artifacts/diagnostics/stage1_recovery_20261028`.

## What changed

Inference loads the saved tokenizer and validates each special token's identity. Reconstructing a tokenizer now uses the same token order as training. Swapping ZCU104/U200 IDs is rejected even when vocabulary size and the unordered ID set match. New training contracts record an explicit token-to-ID map. The repaired reconstruction's entire vocabulary matches the saved checkpoint vocabulary.

The existing full-training launcher accepts optional learning-rate overrides while retaining its original defaults and overwrite guard. Experimental loss/search changes remain separate from the production trainer. They do not change the backbone architecture, existing checkpoints, prompts, legal directive domains, GNN, or Stage-2 artifacts.

## Learning-mechanism control

Each control starts from the same parent tensor state and the same 20 selected training examples. The initial teacher-forced value accuracy is 53.22%. Values count as correct only if all their ordinary tokens are correct; delimiters are excluded from this metric.

| Training control | Attempted / actual updates | Final value accuracy | Completely correct examples |
|---|---:|---:|---:|
| Original weighted CE, original rates | 120 / 120 | 68.42% | 5 / 20 |
| Original weighted CE, stronger rates | 200 / 198 | **97.08%** | **16 / 20** |
| Value-only, equal-site CE | 200 / 198 | 94.74% | 13 / 20 |
| Value-only CE + sampled joint-action loss | 200 / 196 | 90.06% | 11 / 20 |

The stronger setting changes both learning rates and training duration relative to the 120-step control. The three 200-step controls share rates and attempted update budgets. Skipped steps were recorded through Accelerate; all completed adapters have changed trainable hashes and unchanged parent file hashes.

This demonstrates that the architecture, trainable token rows and optimizer can learn non-default directive values. It does not prove that the original run had a specific optimizer defect or that a particular learning rate is optimal. Original optimization logs remain unavailable. The value-only and joint objectives did not improve memorization over standard CE in this control; the joint run also cost substantially more compute.

## Validation results

Eight cases cover all four validation kernels and three families. Each kernel contributes a same-device, same-clock pair with different budgets and reference targets. All cases use 10 ns; the pairs include both known devices. None uses Kalman or test-split targets.

Core balanced accuracy averages per-value recall over PIPELINE, UNROLL, array type and array factor at decision sites. Dimension is reported in the detailed results but excluded from this core score because this pilot's two nonforced dimension fields have only one reference value. Complete-action accuracy requires every field in an action to match. A correct budget transition requires both endpoints of every changed reference field in a pair to match.

| Adapter | Core balanced, field / joint decoding | Complete actions, field / joint decoding | Correct budget pairs |
|---|---:|---:|---:|
| Legal constant baseline | 34.82% | 8.06% | 0 / 4 |
| Original checkpoint | 33.38% / 34.82% | 9.68% / 9.68% | 0 / 4 |
| Stronger standard CE | 32.43% / **35.05%** | **24.19%** / 20.97% | 0 / 4 |
| Value-only CE | 33.52% / 31.01% | 14.52% / 12.90% | 0 / 4 |
| Joint-action training | 33.76% / 32.87% | 17.74% / 20.97% | 0 / 4 |

Every method has 0/8 exact reference-design matches. The stronger and alternative-loss adapters change predictions on one budget pair each, but those changes are not correct complete reference transitions. Output diversity alone therefore does not pass the gate. All 64 requested validation predictions completed: four adapters × eight contexts × two decoding methods. Validation took 45.1 minutes on the V100.

The original checkpoint's field decoder still selects zero for every one of the 48 pilot UNROLL fields. Stronger CE produces several nonzero factors and improves PIPELINE balanced accuracy, while worsening array-type/factor recall. Its 0.23 percentage-point core gain under joint decoding is too small and the pilot too narrow to establish generalization improvement.

Separately, re-scoring all 62 **stored** original validation outputs gives 30.00% five-kind decision-balanced accuracy versus 31.39% for the legal constant baseline. Those stored-case statistics are a different evaluation set and aggregation from the eight-case core score above; they should not be compared directly across tables.

## Interpretation and next experiment

The evidence narrows the issue to learning the conditional optimization policy and generalizing it, rather than a decoder that prohibits positive directives or an architecture that cannot learn values at all. The tokenizer issue was real, but correcting it alone leaves the original Kalman argmax directives unchanged, as established in the earlier audit.

The tiny training set deliberately tests memorization: 16 kernels from 13 training families, mostly U200 at 3.33 ns, no 5 ns examples, and limited dimension coverage. Its adapters must not be substituted for a fully trained model. These results do not establish that either new objective would fail with full training; they establish that neither is currently a demonstrated repair.

The justified next larger study is full-data CE optimization with adequate coverage and learning-rate/schedule comparisons, retaining all device/clock conditions and complex kernels. Evaluate against the legal constant baseline, nontrivial per-value scores, whole actions and budget transitions across the full validation set. The guide contains a fresh-output full-data recipe with rate overrides and an explicit compaction-policy setting. **That larger job was prepared, not launched by these diagnostic controls.** Kalman must remain outside hyperparameter selection; it has already been inspected diagnostically and cannot be called a fresh blind test.

No adapter was promoted, no gold directives were manually inserted into inference, and no values were forced merely to increase diversity. No HLS synthesis or new QoR measurements were performed. Agreement with measured directive references is not a measurement of latency/resource improvement.

## Verification and preservation

- CPU suite: **127 passed**, one CUDA-only test skipped in the sandbox.
- That CUDA FP16 test separately passed on the V100: **1 passed**.
- Launcher syntax, unchanged default rates, explicit rate overrides, and experiment/log overwrite protection were checked without launching full training.
- All four training runs share selected-example SHA-256 `df321329eb5a37c9276ec5c291bbda1643a75ce274feaf5456d1a3499616cfa0` and the same initial trainable tensor hash.
- The reproducible preparation command authenticates the dataset/split and matches all 62 saved validation references. Its train/validation exports match the earlier audit byte for byte.
- Preserved parent adapter SHA-256: `a15a0af1022370b08ee1e64cb9934a12e9f1c7ef1b96aaaf4ccfc406f4e0cc86`.
- Preserved parent contract SHA-256: `d2c63fdaf32199d5de3625dcd4a9358c34a79e48f6761f16948ca7a2c7120dd8`.
- Preserved parent tokenizer SHA-256: `eef84d1f080adcb66a40c467931055a8b95e3d37fd6534c7340aa6c63b966891`.

The first two controls' inherited metadata was normalized to the explicit experimental schema, with backups and hashes, without changing their weights. Experimental model cards label their limited purpose. No original graph archive, structural memory, measurement, checkpoint, main branch or unrelated worktree change was overwritten.

## Full-data recovery implementation and supervision audit — 2026-10-29

The new [full-data evidence summary](results/full_data_20261029.json) records the authenticated audit and new checks. This section does not revise the historical 20-example results above.

The historical selection was reproduced exactly: 2,187 training contexts, 905 distinct selected measured rows, 772 distinct program/directive assignments, 50 kernels and 18 families. Validation has 383 selected contexts from 87 measured rows and four kernels. All 62 historical saved selection references match. No test labels informed this audit, weighting or validation.

Training target contrasts include 6,560 changed budget pairs, 1,491 changed fixed-budget clock pairs and 117 changed full-budget device pairs. These overlapping pairs are not independent measurements. All 6,560 changed budget pairs have measured cross-budget infeasibility; 1,695 contexts participate and 492 do not. A different feasible measured assignment is within 1% latency in 1,862/2,187 training contexts. Exact winner labels therefore often express weak measured preferences rather than a robust hardware-policy boundary. The original labels and compaction are preserved for the initial controlled recovery.

Production now offers opt-in `decision` and `decision_balanced` loss policies alongside unchanged default `legacy`. The new objective normalizes value-only CE by complete action and excludes forced values/delimiters. Optional training-only bounded frequency and budget-contrast emphasis remain disabled in the first recovery. Stage 2 is unchanged.

Executed validation:

- CPU: `python -B -m pytest -q LLM_branch/tests tests/test_directive_domains.py` — **164 passed, one CUDA smoke skipped** in the sandbox.
- V100: `python -B -m pytest -q LLM_branch/tests/test_stage1_fp16_smoke.py` — **one passed**, separately.
- Token preflight: **2,187/2,187 examples under each of three policies**, identical target IDs, no supervised input-special tokens, no code truncation; 480–6,656 tokens.
- Fresh rank-8 DeepSeek production smoke: **two attempted and two successful updates**, finite nonzero LoRA and input-token gradients, finite validation CE, adapter/tokenizer/checkpoint saved. This is an interface check, not a recovery result.

The first smoke attempt was stopped during redundant serialization preprocessing. The successful model smoke used bounded pure-function serialization caches already used by reconstruction. Its outer Bash command returned 127 after Python completed because the launcher was edited while Bash was waiting, shifting the trailing read position. The model finished both updates and all saves; no optimizer failure occurred. The launcher was corrected, syntax/stub tested, and frozen before the full run. Both attempts and logs are preserved locally.

A **single optimized legacy-CE full-data run has been launched** in `mailohls_runs/stage1_full_data_ce_recovery_s123`, using fresh LoRA, LR `1e-4`/`2e-5`, historical invariant compaction, seed 123 and unchanged QLoRA architecture. The sealed `recovery_protocol.json` records input hashes and implementation hashes. Transformers 4.51.0 plans **546** steps; the conservative one-epoch early-stopping guard remains **274**. Actual updates, AMP skips and scheduler progress are recorded independently. Full-data learning and promotion are not yet established while this run is in progress.

On the **same 62 stored reference cases**, the new all-fields complete-action reanalysis finds historical adapter 8.32% versus legal constant 6.56%; core nonforced balanced recall is 25.00% versus 26.74%. Both have zero exact complete designs and zero correct complete budget pairs. These are not the eight-case metrics above, and neither set substitutes for full 383-context evaluation.

The reference adapter SHA256 remains `a15a0af1022370b08ee1e64cb9934a12e9f1c7ef1b96aaaf4ccfc406f4e0cc86`; its contract and tokenizer hashes also remain unchanged. No full-data decision-aware, balanced, contrast-weighted, joint-action or DPO variant has been launched.

Two additional defensive tests verified optional weighting/accounting: changed winners that both fit both budgets are not emphasized, and resuming an older checkpoint archives later abandoned audit receipts instead of double-counting their updates. Both were reproduced as failing cases before correction. These optional/resume safeguards do not change the fresh, contrast-disabled CE run; its exact earlier implementation bytes are archived under its `source_snapshot/`.

The final isolation check also omits Stage-1-specific metadata entirely from Stage-2 contracts, rather than adding a null field that would change a historical resume hash. This does not change the Stage-1 branch or the running CE optimization. Full validation uses a fixed evaluation seed and records dataset/selection hashes; the `finalize --full-run` command requires completed root artifacts and matching reference/candidate lineage before reporting a recovery.

Fresh reproductions can now set `STAGE1_RECOVERY_PREPARED` to authenticate the complete selected train/validation multisets and automatically seal the protocol/source snapshot before model loading. A CPU integration check accepted all 2,187/383 authenticated contexts regardless of row ordering, while a changed-label test failed closed. This replaces the manual protocol-sealing step used to launch the already running initial experiment; it does not restart or alter that experiment.

### Live execution versus this recorded audit

The section above is a launch-time record, not a completed full-data model result. The single full run is executing independently of the documentation. Its exact initial code is in its `source_snapshot/`; a mechanical AST comparison confirmed unchanged loss construction, dataset, Trainer, selection callback, decision metrics and contrast-query definitions across the later optional/metadata safeguards.

For this execution, monitor `/tmp/stage1-full-data-ce-recovery.log` and `mailohls_runs/stage1_full_data_ce_recovery_s123/postprocess.status`. A waiting, detached postprocessor checks successful training completion and code hashes, then calls the existing full-validation evaluator for the reference and selected checkpoint and the read-only full-run finalizer. It launches no additional training variant. Its outputs will be:

- `artifacts/diagnostics/stage1_full_data_recovery_validation/report.json` — matched full-validation predictions and metrics.
- `mailohls_runs/stage1_full_data_ce_recovery_s123/full_run_report.json` — authenticated optimizer/schedule/checkpoint and promotion assessment.

`WAITING_FOR_TRAINING` is not a successful learning result. A failure or source-code mismatch stops postprocessing; no checkpoint is promoted. Generated artifacts remain ignored by Git. The final classification must be taken from completed measurements, not the launch-time snapshot.

## Assessment follow-up — 2026-10-29 19:58:52 UTC

The full-data run is **still executing**; this is a progress receipt, not its final result. The host GPU process is PID `2893110` on the assigned V100S, using about 18.8 GiB at 99% GPU utilization when checked. The audit records 88 successful updates out of the Trainer's 546-step plan, zero AMP-skipped updates, finite recorded embedding/LoRA gradient norms, and scheduler epoch 88. The postprocessor remains `WAITING_FOR_TRAINING`. The 274-step one-effective-epoch guard has not been reached. The final adapter and all-383-context assessment therefore remain pending.

The frozen protocol SHA-256 is `2d8fae40dedcf141268f6190f94073492eb3eecd244e9640b4691503d13895da`. Its selected context counts (2,187 train / 383 validation), dataset, split, source-sidecar, dataset-manifest and training-code snapshot hashes were rechecked successfully. The contract confirms a fresh adapter from the pinned DeepSeek backbone and `legacy` supervision with the stronger learning rates; this run does not test decision-aware CE. The historical reference adapter remains unchanged.

At the latest follow-up, 2026-10-29 20:03:15 UTC, it had advanced to **100/546** successful updates, still with zero skipped updates; selection files remained at steps 0 and 50 and the postprocessor was still waiting. Progress is reported from the append-only update audit and training log, not inferred from adapter files.

At 20:06:12 UTC, the host process was still alive on the assigned V100S at 100% utilization, with **100/546** successful updates, zero skipped updates, finite recorded gradient norms, teacher-forced validation loss `0.53197` at step 100, and postprocessing still waiting. The step-100 constrained-selection receipt had not yet been written at that instant; the best selected checkpoint remained step 0. This state is explicitly time-stamped because it is live and will change.

The latest periodic decision selection is step 50; the best-selection record is still step 0. The step-50 key has lower primary core nonforced balanced recall than step 0, although its complete-action component is higher; both show zero correct complete budget transitions so far. This is an interim selection trajectory, not a completed validation comparison or a promotion decision.

On the same 62 periodic cases, the current intermediate checkpoint metrics are:

| Snapshot | Core nonforced balanced recall | Complete loop actions | Complete array actions | All complete actions | Positive PIPE recall | Positive UNROLL recall | Exact designs | Correct changed budget pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fresh adapter, step 0 | 26.16% | 0.00% | 10.57% | 2.84% | 100.00% | 12.92% | 0/62 | 0/67 |
| Step 50 | 19.32% | 24.85% | 0.81% | 18.38% | 54.48% | 2.25% | 0/62 | 0/67 |
| Training-derived constant | 26.74% | 3.29% | 15.45% | 6.56% | 0.00% | 0.00% | 0/62 | 0/67 |

All outputs in this table were legal under the source domains. Step 50 improves complete loop decisions on this small bank but loses array decisions, has lower core-balanced recall than both alternatives, and has no correct budget transition. The lexicographic predeclared selector therefore retains step 0; this is not evidence that a trained checkpoint is ready for Stage 2. The step-50 output has an exact measured-design match for **0/62** cases, leaving its measured QoR unknown throughout; this low coverage cannot be repaired by calculating regret on unmatched outputs. At 20:00:15 UTC the optimizer audit had advanced to **92/546** successful updates, still with zero skips; the job remained active.

Teacher-forced validation CE on that bank decreased from **1.1767** at step 0 to **0.5481** at step 50, while the predeclared core decision score decreased. This is why the final assessment uses constrained complete outputs and transitions rather than CE alone.

To validate the new measured-candidate evaluator before the full report exists, the step-50 saved predictions were checked against the authenticated 62-case periodic bank. They matched **0/62** raw measured complete designs, so their measured QoR/regret is unknown for every case. This subset is deliberately not accepted by the final full-validation CLI, which requires all 383 unique selected validation contexts. It is not used as the final model assessment.

The read-only measured-candidate evaluator was added on a separate worktree/branch. It reuses the validation split and measured rows only, checks exact source/device/clock/full-assignment identity, uses the production fractional resource-feasibility rule, and emits unknown QoR for unmatched outputs. CPU tests cover context mismatches, repeated measured rows under distinct budget requests, feasibility, regret, duplicate receipts, held-out row exclusion and overlapping target-pair accounting. Focused evaluation/recovery/domain tests pass **58 tests**; the complete `LLM_branch/tests` plus directive-domain suite passes **172 tests, one CUDA-only skip**. The earlier separate V100 FP16 smoke remains **1 passed**. The command and metric boundaries are documented in the canonical [README](README.md). No second training run was launched.

The authenticated validation split contains **13,011** raw measurement rows; the exact kernel/source/device/clock/directive identities were unique across all 13,011 in the validation-pool audit, so no duplicate aggregation was needed for this dataset. The evaluator still rejects conflicting duplicates if a future authenticated input contains them.
