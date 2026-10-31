# Stage-1 directive-learning recovery

This experiment follows the [checkpoint collapse audit](../../artifacts/diagnostics/stage1_latency_collapse_20261027/README.md). It fixes inference token identity and tests whether poor directive predictions come from failed optimization, weak decision supervision, or generalization.

The original `mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1` remains immutable. No HLS, QoR measurement, GNN or Stage-2 training runs here. All new adapters are experimental controls, trained on **20 training examples**, and must not replace the production checkpoint.

## Correctness repair

Inference loads the saved adapter tokenizer. For old contracts that store only a sorted set of IDs, it reconstructs expected identities using the pinned base tokenizer and the contract's recorded ordered token strings. It compares every token's ID, vocabulary size, and the declared ID set. New training contracts also record an explicit `special_token_to_id` mapping. The default reconstruction order now matches training.

Regression tests reject a swapped device-token mapping even when the vocabulary size and unordered ID set match. They cover saved-tokenizer loading, missing-tokenizer reconstruction, explicit mappings, and corrupt contracts. The checkpoint's own tokenizer files are unchanged.

## Controlled training

`run.py` reuses the production RHS pack, data collator, trainable-parameter checks and `LengthGroupedTrainer`. It starts each experiment from the same parent adapter and seed. Only LoRA and special input-token embeddings train; the backbone, standard embeddings and output head stay frozen. Training indices are checked against the original family split. Kalman/test data is never used.

The deterministic 20-example selection favors diverse families and directive values, allows at most two examples per kernel, and fits complete sources and targets without truncation. Selection comes from the authenticated training-winner reconstruction. The selected example files and parent trainable hashes permit checking that controls start from identical data and weights.

In the recorded run this is 16 kernels from 13 families: 16 U200 and four ZCU104 examples, with 18 examples at 3.33 ns and two at 10 ns. The control is not stratified across all targets and has no 5 ns examples. Its purpose is checking learning capacity; target-generalization claims require the full dataset.

| Control | Loss | Steps | LoRA / token LR |
|---|---|---:|---|
| `ce_control` | Original weighted RHS CE, including newline | 120 | 1.5e-5 / 5e-6 |
| `ce_lr_control` | Original weighted RHS CE, including newline | 200 | 1e-4 / 2e-5 |
| `decision_ce` | Equal site weight on value tokens; delimiter masked | 200 | 1e-4 / 2e-5 |
| `joint_choice` | Value-only CE plus sampled complete-action softmax | 200 | 1e-4 / 2e-5 |

The first versus second comparison changes **both** optimization rate and number of updates; it cannot isolate learning rate alone. The three 200-step runs use matched examples, initialization, rate and update-attempt budget. All runs use batch 1, no accumulation, constant LR, no warmup, inherited adapter dropout and 0.01 LoRA weight decay. These are memorization controls, not reproductions of the original full-corpus training schedule.

The decision objectives also replace the parent's per-kind weights with uniform total weight per decision site. Thus the CE versus decision-CE comparison tests a combined value-only/equal-site policy; it does not isolate newline removal alone.

`optimizer_audit.json` records gradient norms and whether each optimizer step was skipped. `report.json` stores initial/final adapter tensor hashes, parent file hashes, accuracy, settings and elapsed time. Teacher-forced value accuracy requires all ordinary tokens in each nonforced value to be correct, excluding newline; it is not free-generation accuracy. `train_greedy.json` separately records generated directives on four selected training cases.

## Decision objective and search

`decisions.py` enumerates complete actions through the repository's **existing** candidate-domain and tuple-semantics functions. Sparse labels remain unchanged. PIPELINE and UNROLL may both be positive. The dataset's exclusive encoding is not added as a universal legality constraint.

For a loop, a candidate contains `(PIPELINE, UNROLL)`. For an array it contains `(type, factor, dimension)`. Syntax, anchors and newlines are supplied as context. The candidate score is the **sum of value-token log probabilities**, excluding the supplied syntax and delimiters.

In `joint_choice`, each update selects one action from the example, scores its reference choice and up to three randomly sampled legal alternatives, and adds sampled-softmax loss to the value-only CE. Earlier actions use gold prefixes. This is an approximate contrastive objective, not an exhaustive softmax over the whole design. Sampling introduces additional dropout/negative noise; the experiment does not establish a uniquely best objective.

Alternatives are contrastive rivals to the selected reference, not synthesis-proven inferior implementations. Whole-action scores include all serialized value tokens, including deterministic companion values; auxiliary CE still masks forced fields. Branching-only scoring is a possible further ablation. No unmeasured alternative is assigned a fabricated QoR value.

Joint inference exhaustively scores legal choices **within each action**, batching four choices at a time, then greedily proceeds to the next action. It is not globally optimal design search, beam search across all actions, or synthesis evaluation. Length preferences remain possible under sum scoring; the candidate-choice objective tests learning complete semantic alternatives instead of treating a scoring change alone as a repair.

`balanced-decision-ce` is also implemented as an optional ablation. It clips inverse-square-root within-kind value weights to [0.5, 2], using only the selected training examples. It is not part of the four primary controls above.

Experimental adapters use `mailohls-stage1-recovery-adapter-v1` / `stage1_experimental`, retaining a minimal backbone/tokenizer contract and parent lineage. They intentionally do not satisfy the production inference/training contract. The experimental evaluator loads them explicitly. The first two controls were run while this runner was being developed; their inherited contract metadata is subsequently normalized to the explicit experimental schema without changing adapter weights.

## Validation pilot and acceptance

`evaluate.py` selects two resource-budget cases with different reference targets for each of the four validation kernels. Each pair has the same device and clock; preferred devices alternate across kernels. All eight row indices must belong to the validation split. The cases are chosen for examining budget sensitivity, not as a random or statistically powerful test set.

Each adapter is compared using conventional fieldwise mean scoring and complete-action value-only scoring. A constant baseline is derived **only from all reconstructed training labels**, with semantically forced values filled from the existing domains.

Reported metrics include per-kind/per-value recall, prediction distributions, balanced value accuracy, exact-design accuracy, complete-action accuracy, and exact correctness of reference-changed fields across each budget pair. Forced fields are reported separately from decision-only scores. Merely changing outputs between budgets does not count as a correct transition.

A pilot passes its gate only if it exceeds the constant baseline on decision-balanced and complete-action accuracy and predicts at least one complete changed-field budget transition correctly. This gate is a research diagnostic, not a synthesis or deployment acceptance criterion. Even a passing eight-case pilot requires a full training/validation experiment before promotion. Kalman is excluded from all tuning and pilot selection.

The summary also reports a **core decision-balanced score** over PIPELINE, UNROLL, array type and array factor. The two nonforced dimension fields in this pilot both have the same reference value, so including dimension can inflate the five-kind aggregate. `finalize.py` requires a core-score gain as well as the recorded pilot gate before listing a full-training follow-up candidate. It never promotes a 20-example adapter to production.

## Reproduction

Use the existing `mailohls-llm-v2` environment: torch 2.6.0+cu124, transformers 4.51.0, PEFT 0.18.1 and bitsandbytes 0.49.2. The pinned DeepSeek backbone must already be cached. No package installation or compiler is required. From repository root:

```bash
export MAILOHLS_PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
```

Reconstruct authenticated training/validation winners with the tracked CPU preparation command. The trainer/evaluator also check or create this cache automatically:

```bash
"$MAILOHLS_PYTHON" -m experiments.stage1_recovery.prepare
```

Use a fresh output root; run directories refuse overwrites:

```bash
mkdir -p artifacts/diagnostics/stage1_recovery_reproduction
"$MAILOHLS_PYTHON" -m experiments.stage1_recovery.run \
  --output artifacts/diagnostics/stage1_recovery_reproduction/ce_control \
  --mode ce --steps 120 --examples 20
"$MAILOHLS_PYTHON" -m experiments.stage1_recovery.run \
  --output artifacts/diagnostics/stage1_recovery_reproduction/ce_lr_control \
  --mode ce --steps 200 --examples 20 --lr-lora 1e-4 --lr-embed 2e-5
bash scripts/experiments/stage1_recovery_controls.sh \
  artifacts/diagnostics/stage1_recovery_reproduction
"$MAILOHLS_PYTHON" -m experiments.stage1_recovery.finalize \
  --root artifacts/diagnostics/stage1_recovery_reproduction
"$MAILOHLS_PYTHON" -m pytest -q LLM_branch/tests
```

Preparation authenticates the dataset and split, reproduces the parent's kind weights and all 62 saved validation cases, and writes checksummed exports under `artifacts/diagnostics/stage1_recovery_data`. Subsequent runs verify the cache's parent-contract identity, export hashes and recorded input hashes. No untracked diagnostic script is required for reproduction.

## Results

See [RESULTS.md](RESULTS.md) and the [compact measured results](results/20261028.json). Actual run files are under [artifacts/diagnostics/stage1_recovery_20261028](../../artifacts/diagnostics/stage1_recovery_20261028). The verification report records completed runs, skipped optimizer updates, training versus validation results and remaining limitations. No QoR improvement is claimed from directive agreement or output diversity.

## A later full-data ablation

The production launcher now accepts `STAGE1_LR_LORA` and `STAGE1_LR_EMBED`; their defaults remain 1.5e-5 and 5e-6. A syntax check and a stubbed invocation verified both defaults and overrides without launching training. Output-directory overwrite protection remains intact.

The following is a **prepared follow-up recipe**, not one of the executed 20-example controls. It retrains a fresh adapter on the full selected training corpus. Setting the invariant-duplicate cap to -1 matches the audited parent's compaction policy; the current launcher's default cap of 1 is a different data-policy ablation.

```bash
OBJECTIVE=PARETO_LATENCY \
OUT=mailohls_runs/stage1_recovery_full_latency_s123_lr1e4 \
STAGE1_BUDGET_INVARIANT_MAX_DUPLICATES=-1 \
STAGE1_LR_LORA=1e-4 STAGE1_LR_EMBED=2e-5 \
bash scripts/training/train_stage1.sh
```

Activate the Python environment first. This larger job includes the production validation callbacks and may take hours on the V100. The stronger rate is an experimental setting, not a validated full-data optimum. A replacement checkpoint must outperform the legal baseline on nontrivial fields and budget transitions on the full validation set before any deployment or Stage-2 reuse. The pilot does not justify adopting a 20-example adapter as the replacement.
