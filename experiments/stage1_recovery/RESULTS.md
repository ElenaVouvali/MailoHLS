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

The first two controls' inherited metadata was normalized to the explicit experimental schema, with backups and hashes, without changing their weights. Experimental model cards label their limited purpose. Recovery changes did not overwrite the original graph archives, structural memories, measurements, checkpoints, main branch or unrelated worktree changes.

A concurrent commit, `a9cb717`, updated Kalman's registrations/CSV and graph selection. It was preserved and is separate from this recovery work. The experiments use the authenticated frozen JSONL and source sidecar, whose hashes remain tied to the parent contract; Kalman is excluded from training and validation. Historical Kalman inference artifacts therefore describe their recorded source/action revision, not an automatic rebuild using the newer registrations.
