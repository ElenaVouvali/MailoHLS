# Stage-1 residual target LoRA and action-aware FiLM

This isolated research branch is `research/stage1-target-film-20261110`, based on `58c098936c14cfb021154884d5ad75cdd5fcf068`. **Investigation stopped at the user request.** No experiment is currently queued. Checkpoints/results are preserved; no production promotion or new HLS synthesis occurred.

## Actual implementation

Frozen V4 latency parent SHA256 `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`, identical to original Stage-1 final315. Its backbone, original objective LoRA, learned tokens, tokenizer and domains remain active/frozen. The residual reuses PEFT/device-adaptation integrity mechanisms, rank8/alpha16, q/k/v/o projections. No GNN, XAttn, DPO or refinement is involved.

A trains only residual target LoRA. B trains the same LoRA plus one shared final-hidden-state FiLM module:15 hardware/context features, five directive kinds (PIPE/UNROLL/ARRAY_T/ARRAY_F/ARRAY_D),8-wide kind embedding,32-wide MLP bottleneck, bounded gamma/beta and tanh gate. Absolute Lk indices are not semantic features. Capacities come from the canonical prompt contract. QoR/winning directives never condition FiLM.

Both use action-normalized nonforced RHS SFT +0.1 same-request logistic rank (temperature1), complete measured assignments and independent causal prefixes. Near-optimal reference sampling is preserved at the sampled-pair level; this pilot uses one authenticated preferred completion per request, not every near-optimal reference. AdamW2e-5, effective batch2, seed123, identical initial residual weights and schedule. Initialization equality is authenticated by SHA256 `dae1f1f1204781c0fca5e2ad4f805a4227cb6a360178ea5421ca75793497a06c`.

## What actually completed

|Arm|Successful training updates|Selected checkpoint|Original GAN inference|
|---|---:|---|---|
|A: residual LoRA|58, interrupted on a skipped mixed-precision update|32|18/18,4 configurations|
|B: LoRA + FiLM|64|48|18/18,5 configurations|

Both selected validation family-macro pair accuracies are58.3333%. This is2-family preference validation, not synthesized quality. Unequal completed update budgets prevent a final matched-budget superiority claim. Original GAN A/B differ on9/18 requests (mean1.0556 fields/25; max4). The external GAN was never used for checkpoint selection. [Complete original directives and hashes](results_20261110/DIRECTIVES.md).

## Final diagnostics v2

[Final preserved status](diagnostics_v2/FINAL_STATUS.md) is authoritative over earlier intermediate descriptions.

- Fixed causal inference mask: value tokens only; newline/formatting logits receive no FiLM. Native mean RHS-plus-newline candidate score is unchanged. Historical predictions remain immutable; explicit `--mask-policy legacy` retains the old inference policy.
- Completed54-pair B48 FiLM ON/OFF/historical-mask ablation shows identical target scores/orderings:28/54 micro,58.3333% family macro.
- Gate1.86e-6 magnitude, strong FP16 attenuation; max vocabulary-logit effect0.0078125 without target-score changes in this cohort. Five-pair training-preparation precision control reveals a larger score-path difference than FiLM's effect.
- Full corrected GAN ablation was canceled:10/18 ON,0/18 OFF,0/18 matched-legacy natural outputs.1049 paired candidate traces have no mean-score change. Unrun requests are unknown, not claimed unchanged.
- Recovered conditional-ranking and proxy-budget candidate banks are versioned and hashed. Canonical restored measurements were checked for compatibility, not silently substituted. No locked-test outcomes were read.
- Verdict **EXPLORATORY_RETRAIN_ONLY**; no new training. Retain V4 conservatively; no physical-QoR improvement or architecture freeze is claimed.

See [implementation audit](diagnostics_v2/IMPLEMENTATION_AUDIT.md), [FiLM ablation](diagnostics_v2/FILM_ABLATION.md), [data audit](diagnostics_v2/TARGET_CONTRAST_AUDIT.md), [decision](diagnostics_v2/RETRAINING_DECISION.md) and [persistent artifact hashes](diagnostics_v2/ARTIFACT_MANIFEST.json).

## Reproduction interfaces

```bash
python -m experiments.stage1_target_film.audit --pairs ORIGINAL_PAIRS --out NEW_AUDIT.json
python -m experiments.stage1_target_film.train --parent V4 --pairs ORIGINAL_PAIRS \
  --out NEW_RUN --gpu-uuid VERIFIED_IDLE_UUID --steps 64 --arm B
python -m experiments.stage1_target_film.infer --parent V4 --adaptation B48 \
  --prepared SEALED_GAN --out NEW_OUTPUT --gpu-uuid VERIFIED_IDLE_UUID \
  --film-mode on --mask-policy value
python -m experiments.stage1_target_film.diagnostics_v2.run_development \
  --parent V4 --checkpoint B48 --pairs ORIGINAL_PAIRS --out NEW_DIRECTORY \
  --gpu-uuid VERIFIED_IDLE_UUID
```

These are supported commands, not instructions to restart canceled work automatically. Standard decoder and source/tool policies are retained. The new variable-target bundle has its own schema; it is not relabeled as the historical single-device bundle. `--resume` restores saved LoRA/FiLM/optimizer/scaler/RNG; partial journals are preserved. GPUs must pass idle-process/memory checks. A2 inference explicitly uses NF4/FP16 to match the V100 precision convention. No process belonging to another experiment was stopped.

Weights, model caches, raw input banks and long logs remain in persistent ignored storage. Git contains code, compact measured reports and SHA manifests; cloning this branch does not copy those weights or raw data. Production registries and historical checkpoints were not overwritten.
