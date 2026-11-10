> Final status: canceled remaining work at user request; no jobs queued. [FINAL_STATUS.md](FINAL_STATUS.md) is authoritative.

# FiLM diagnostics v2

Parent branch head at start: `e485c0a3f1038908dd53ea91593f8fcd00d581db`.
Historical checkpoints/results remain immutable. These diagnostics make no optimizer updates and no HLS calls. Locked-test outcomes are not read.

## Implemented commands and artifacts

- `test_alignment.py`: actual V4 tokenizer; five directive kinds, numeric/multi-token factors and array modes;11 cases passed. Uses the existing action-SFT `_value_positions` boundary verifier, rejecting merged value/delimiter tokens.
- `run_development`: actual B48,54 validation pairs, same residual adapter for OFF/value-only/legacy masking. Scores each completion under its own prefix using the existing action-normalized value NLL. Includes a real-LM one-position cache comparison, parent-only adapter intervention, frozen tensor verification and FP16 survival traces. `--limit 5 --training-precision-control` is an inference-only replica of the trainer's k-bit preparation casts, not another learned architecture.
- `audit_bank`: broader frozen non-test bank, complete-assignment identities, matched pair contexts, conservative +/-1 percentage-point resource intervals. Current canonical eligibility and restoration metadata are inspected at their separate pinned commit. Restored source/action cohorts are excluded from old-bank qualification until reconstructed; no old measurement is relabeled as a restored configuration. Aggregate matched counts are a candidate-level audit, not proof of source/tool legality for every pair. Do not train from this audit directly.
- `infer --film-mode on/off --mask-policy value/legacy`: native production decoder, unchanged candidate domains and mean suffix scoring. New default masking is value-only; historical output files remain unchanged.
- `infer --paired-trace FILE`: same-prefix OFF/value/legacy head scores using the identical real backbone hidden state. FiLM acts only before the head, so backbone states do not depend on the intervention. Full-length head evaluation preserves its matrix shape; counterfactual replay is checked against the native score at tolerance1e-5.
- `infer --replay-trace FILE`: exact prefix/candidate/context cache; uses only the requested condition's score. A new prefix is a cache miss and receives a real forward. This executes natural native constrained decoding without treating divergent trajectories as identical-prefix comparisons.

The implemented continuation was canceled at the user request after10 corrected ON requests; OFF and legacy natural decoding did not execute. The historical-mask natural replay uses the same paired score cache where exact prefixes match. Existing GNN training on V100S is not interrupted. Davinci V100/A30 were occupied when checked.

```bash
python -m experiments.stage1_target_film.diagnostics_v2.run_development \
 --parent V4 --checkpoint B48 --pairs AUTHENTICATED_PAIRS \
 --out NEW_DIRECTORY --gpu-uuid VERIFIED_IDLE_UUID
python -m experiments.stage1_target_film.diagnostics_v2.audit_bank \
 --bank AUTHENTICATED_NON_TEST_BANK --out NEW_DIRECTORY
python -m experiments.stage1_target_film.infer --parent V4 --adaptation B48 \
 --prepared SEALED_GAN --out NEW_DIRECTORY --gpu-uuid VERIFIED_IDLE_UUID \
 --film-mode on --mask-policy value --paired-trace NEW_TRACE
```

Large raw diagnostics reside at `artifacts/stage1_target_film/diagnostics_v2/` on Liono. GPU receipts include model/file hashes. The existing B48 gate is approximately-1.8586086e-6. This is a measured parameter, not evidence that its contribution is useful or useless. Final reports require completed receipts; partial evaluation is not a completed ablation.

## Retraining gate

No retraining has been launched. Robust percentage-proxy evidence alone can justify at most `EXPLORATORY_RETRAIN_ONLY` under the user's policy, which does not authorize an automatic expensive run. Restored measurement columns require fresh source/action identity qualification. GAN diversity never enters this decision. Missing independent validation switches, unknown resource reporting precision and unresolved source legality prevent a qualified retraining recommendation.
