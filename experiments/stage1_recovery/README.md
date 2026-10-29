# Stage-1 directive learning and full-data recovery

Stage 1 must learn **complete directive decisions conditioned on a program and an optimization request**. Producing frequent values is insufficient. This directory is the canonical description of the data construction, historical failure, controlled recovery objectives, validation and reproduction procedure. [RESULTS.md](RESULTS.md) separates completed measurements from unexecuted recipes.

The historical latency checkpoint remains immutable:
`mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1`.
New runs use fresh directories. Nothing here promotes an adapter automatically, changes Stage 2, loads structural memory, or generates HLS measurements.

## Model and output

The pinned backbone is `deepseek-ai/deepseek-coder-6.7b-base`, revision
`ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912`. Training uses 4-bit NF4 with double quantization, rank-8 attention LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`; alpha 16), and trainable deltas for special **input** token embeddings. Ordinary backbone parameters and output embeddings stay frozen. The production trainable-parameter validator rejects additional trainable groups. No GNN or cross-attention participates in Stage 1.

The prompt supplies source/action markers, objective, FPGA capacities, available resource budgets and requested clock. The response retains the existing deterministic syntax, for example:

```text
<L3>
auto{_PIPE_L3} = 1
auto{_UNROLL_L3} = 2
```

Array actions serialize type, factor and dimension. Absolute `Lk` identities and special-token ordering are unchanged. PIPE and UNROLL are separate legal choices: both may be positive when their source domains allow it. Their mutual exclusion in these selected training labels is an observation, not a legality rule.

Teacher forcing supplies earlier reference values while calculating the next-token loss. Validation instead uses the existing constrained autoregressive decoder: earlier predicted values become context, and errors can propagate. Teacher-forced value accuracy and generated complete-action accuracy therefore measure different behavior.

## Authenticated data construction

The existing pipeline is reused:

```text
GNΩSIS measured rows → fixed family split → shared resource-budget requests
  → objective-selected winners → training-only budget compaction
  → source/target serialization → legal-domain supervision → QLoRA
```

The reconstruction authenticates dataset bytes, source sidecar, split membership, historical contract and exported selections. It reproduces historical kind weights and all 62 saved selection cases. Test membership is checked for disjointness; test targets are not selected, exported or summarized.

The latency audit reproduced:

| Quantity | Training | Validation, reported separately |
|---|---:|---:|
| Raw measured rows in split | 130,310 | 13,011 |
| Selected request contexts after training compaction | 2,187 | 383 |
| Distinct selected measured-row IDs | 905 | 87 |
| Distinct complete assignments, keyed by program | 772 | 78 |
| Kernels | 50 | 4 |
| Families | 18 | 3 |

The complete dataset has 150,026 rows. The held-out Kalman family has 6,705 rows; its labels are unused. The training split contains 117,905 distinct program/directive assignments before winner selection. Measurements repeated across target contexts and repeated budget winners are not independent synthesis observations. The 2,187 examples contain 1,282 repeated winner contexts beyond their 905 selected measured rows.

For this first recovery, preserve the historical latency task: measured ZCU104/U200 devices, 3.33/5/10 ns clocks, 16 generated budgets per case, minimum feasible candidate count 3, candidate pool 24, `top_k=1`, training target cap 4, **invariant cap disabled**. The current launcher default invariant cap is 1; explicitly setting `STAGE1_BUDGET_INVARIANT_MAX_DUPLICATES=-1` is essential for historical reproduction. Tight and full/loosest budgets and distinct selected targets remain represented. Invariant contexts remain training examples. Validation is not compacted.

The frozen historical source-domain registry is passed explicitly. Current source metadata can change independently; rebuilding a registry from today's metadata is a separate experiment. This recovery does not change domains, data bytes, family membership, winning labels or source serialization to suit a loss function.

Dataset SHA256:
`afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`.
Canonical split-membership SHA256:
`143f187f6867afd853014be4ed2de3cb35a1ff635995c8459d74763227b2b341`.
The preparation manifest also records file-byte hashes, including the source templates and split file.

### What the labels teach

The selected training contexts contain 14,788 complete loop-action occurrences:
969 `(PIPE=0, UNROLL=0)` (6.55%), 6,203 with positive PIPE and 7,616 with positive UNROLL. None has both positive in this selection, although the legal domain permits it. Complete array tuples, per-kind values, forced versus nonforced choices and device/clock budget ranges are recorded in `supervision_audit.json`. These occurrence counts include repeated contexts; balancing separately deduplicates measured row IDs.

An informative target contrast is an **observed change in the selected complete assignment** between comparable requests. The audit compares:

| Changed constraint | Changed pairs | Invariant pairs | Conditions held fixed |
|---|---:|---:|---|
| Budget | 6,560 | 2,374 | Program, device, clock, objective |
| Clock | 1,491 | 380 | Program, device, absolute resource limits, objective |
| Device | 117 | 32 | Program, clock, objective; full capacities on both devices |

Thus `N_examples=2187` and `N_informative_target_contrasts=8168` under these definitions. Pair counts overlap and are not independent samples. Device comparisons at unmatched partial budgets are excluded. The machine-readable audit preserves endpoint indices and changed fields.

All 6,560 changed budget pairs additionally have a measured feasibility boundary: at least one selected design does not fit the opposite budget. This involves 1,695 training contexts; 492 remain outside these changed-budget groups. Optional contrast emphasis requires this witness and rejects mere winner changes when both designs fit both budgets.

A critical limitation: in **1,862/2,187** training contexts, a different feasible measured assignment lies within 1% latency of the selected winner. The audit records the feasible candidate count and relative latency gap for every context. These are latency gaps, not repeated-measurement confidence intervals; winner changes are not statistically established hardware-policy boundaries. No labels are invented to increase contrast counts. Exact agreement with a nearly tied winner should not be mistaken for measured QoR improvement.

## Original failure and previous controls

Saved historical predictions collapse toward PIPE=0, UNROLL=0 and cyclic partition factor 2; no positive UNROLL appears in the audited saved outputs. On those outputs, complete designs and correct complete budget transitions are zero. Fixing tokenizer-ID reconstruction did not repair this policy. The checkpoint tokenizer and weights themselves were preserved.

Four **20-example** controls established memorization capacity, not generalization:

| Control | Teacher-forced value accuracy |
|---|---:|
| Historical weighted CE, 120 attempted updates | 68.42% |
| Stronger CE, 200 attempted updates | 97.08% |
| Value-only field CE, 200 attempted updates | 94.74% |
| Value-only CE plus sampled joint-action loss | 90.06% |

The stronger controls changed rates and training duration together. Actual successful updates were respectively 120, 198, 198 and 196. None solved a complete transition on the eight-case validation pilot. That pilot, the historical 62 cases and the full 383 selected validation contexts are different evaluation sets; do not compare their percentages as if the cases were identical. `run.py` preserves these historical controls, including their **field-normalized** experimental loss. The new production decision objective below is **action-normalized**. No joint-action loss is added to production.

## Configurable Stage-1 supervision

`--stage1_supervision_policy` selects `legacy` (default), `decision` or `decision_balanced`. Stage-2 use of the new policies is rejected. The new decision modes currently require specified-clock, no-EOS supervision; extending automatic-clock loss is outside this controlled comparison.

### Historical RHS CE (`legacy`)

For nonforced field `j`, the existing pack tokenizes **RHS plus newline together**. Its supervised tokens have total mass `value_loss_weight × kind_weight[j]`, divided equally among those tokens. Fixed directive syntax and input-only special tokens already have zero loss. Newlines inside a nonforced RHS pack **are supervised**. The trainer takes the token-weighted per-example CE mean, followed by its historical sample-weighted batch reduction. This path retains identical IDs, labels and weights when `legacy` is selected explicitly or omitted.

Writing `S_j` for those RHS-plus-newline tokens and `c_j` for the historical kind weight:

\[
 L_{legacy}=\frac{\sum_j c_j\,|S_j|^{-1}\sum_{t\in S_j} CE_t}{\sum_j c_j}.
\]

This is an equal-field-mass objective with kind reweighting, rather than unweighted CE on the whole response. The repair must not incorrectly describe historical training as supervising the fixed schema.

### Decision CE (`decision`)

All context IDs remain identical. The encoder identifies value tokens **inside the original RHS-plus-newline tokenization**. Offset boundaries are used when available; the checked fallback requires exact value/newline segmentation. A token crossing the value/delimiter boundary fails validation. Fixed syntax, delimiters, forced fields and input-only tokens receive zero decision weight.

For action `a`, nonforced fields `J_a`, and value-token positions `V_j`:

\[
 L_j=|V_j|^{-1}\sum_{t\in V_j}CE_t,\qquad
 L_a=|J_a|^{-1}\sum_{j\in J_a} L_j,\qquad
 L_{decision}=|A|^{-1}\sum_{a\in A}L_a.
\]

Only actions containing a nonforced field enter `A`. A long factor and a short factor have equal field mass; a loop and an array have equal action mass despite different serialized field counts. Forced values still appear as context and must be correctly emitted by inference. This remains autoregressive CE through the same model and Trainer, without a classifier head.

### Conservative value balancing (`decision_balanced`)

Count nonforced values by directive kind using **distinct selected training measured-row IDs**, excluding repeated budget requests. For each kind, use the mean observed class count as the reference and take inverse-square-root frequency. A bounded monotone rescaling keeps the count-weighted mean at one, within configurable `[0.5,2.0]` bounds. Unseen values receive neutral weight one; no fictitious observations are added. The balancing counts and weights are stored in the training contract.

Multiply decision field masses by these weights and renormalize the example's total token mass to one. This optional extension can alter relative action mass; the unbalanced mode implements the exact equal-action formula above. It avoids an accidental global learning-rate change. Validation and test frequencies never fit training weights. Historical directive-kind weights remain recorded for compatibility but do not multiply the new action-normalized objective.

[Cui et al.](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html) motivate handling imbalance and repeated information. Our bounded inverse-square-root directive weighting is a modeling choice, **not** their effective-number formula or a demonstrated SALT improvement.

## Optional target-contrast emphasis

`--stage1_contrast_weight` defaults to **1**, meaning disabled. Values up to 2 modestly emphasize selected training requests participating in changed budget-target pairs with measured cross-budget infeasibility. Invariants retain positive weight; target labels and the sampler are unchanged. Weights are normalized to training mean one. Repeated contexts remain contexts, not additional HLS measurements. Clock/device emphasis is not enabled.

With batch size one, historical division by the batch sum cancels sample weights. Therefore this explicit option uses the existing sample-weight inputs with a **mean of training-normalized weighted losses**. It is isolated from legacy/default and Stage-2 behavior. Initial recovery leaves contrast weighting disabled, so objective/optimization and sampling effects remain separable. Near-tie caveats still apply.

## Optimization and checkpoint selection

Keep the original QLoRA architecture, dropout 0.10, paged AdamW8bit, LoRA weight decay 0.01, embedding weight decay zero, gradient checkpointing, clipping norm 1, batch size 1 and accumulation 8. Historical launcher defaults remain LoRA LR `1.5e-5` and embedding LR `5e-6`. The optimized-CE experiment explicitly uses `1e-4` and `2e-5`, with cosine scheduling, 5% warmup and two epochs. These rates are a starting point, not a demonstrated full-data optimum.

Setting `STAGE1_RECOVERY_PREPARED` authenticates the complete selected train/validation context multisets against reconstruction and seals code/input hashes and a source snapshot before loading model weights. A selection mismatch fails before training.

Fresh recovery starts from the pinned backbone and **fresh LoRA**, not the collapsed adapter. Existing-adapter initialization remains explicitly opt-in via production arguments. Output paths refuse overwrites. The reference checkpoint is never a recovery output path.

For 2,187 examples, covering one epoch needs `ceil(2187/8)=274` update opportunities. The Stage-1 early-stopping guard remains 274. Installed Transformers 4.51.0 computes epoch-based `max_steps` using **floor**, giving 546 total planned steps for two epochs, rather than 548. The existing scheduler is retained. `schedule_audit.json` records the actual Trainer schedule; `optimizer_audit.json` records AMP-skipped versus successful updates, scheduler position, group rates and periodic gradient norms. A saved best step alone does not reveal when training ended. On resume, audit history beyond the restored checkpoint is archived and excluded from the retained trajectory; abandoned physical work remains visible in that receipt.

Optional decision diagnostics extend the existing `StageValSelectionCallback`. They report:

- Nonforced value accuracy, core macro-balanced recall, per-value recall and positive PIPE/UNROLL recall (exact value and binary positivity separately).
- Complete loop, array and overall action accuracy; exact complete-design accuracy; forced-field accuracy separately.
- Kernel/family/device/clock/budget strata, source-domain legality, correct changed endpoints and complete transitions, and correct invariant predictions.
- A legal constant baseline fitted from **training** modes and evaluated on the exact same cases. Wrong but equal predictions do not count as correct invariance.

With diagnostics explicitly enabled, checkpoint selection is lexicographic: core nonforced balanced recall, complete-action accuracy, correct complete budget pairs, then minimum-kernel decision accuracy. A log-probability margin is only a final tie-breaker and cannot reset early-stopping patience. Historical default selection remains unchanged. Periodic selection can retain the historical 62-case bank; `--stage1_full_validation` explicitly selects all 383 contexts instead. The separate evaluation command below reports the full selected split on identical cases for the historical and candidate adapters.

Promotion is report-only: exceed both historical and legal-constant complete-action accuracy, recover meaningful nondefault recall, correctly solve authenticated target transitions, preserve syntax/domain legality, and inspect device/clock groups for collapse. Case-identity hashes must agree. A lower CE, more diverse values or arbitrary budget responsiveness does not pass this gate. Domain legality also does not prove that an unmeasured generated configuration meets physical resource budgets; no new synthesis validation is performed here.

## Reproduction

Use the existing environment; do not upgrade libraries. Verified versions are torch 2.6.0+cu124, Transformers 4.51.0, PEFT 0.18.1 and bitsandbytes 0.49.2. The authenticated compact dataset, source sidecars, split, historical domain registry/tokenizer and pinned backbone must be available locally. Generated data and adapters remain ignored.

```bash
export PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PREPARED=artifacts/diagnostics/stage1_full_data_recovery_data
"$PYTHON" -B -m experiments.stage1_recovery.prepare --output "$PREPARED"
"$PYTHON" -B -m experiments.stage1_recovery.prepare --output "$PREPARED" --audit
"$PYTHON" -B -m experiments.stage1_recovery.prepare --output "$PREPARED" --validate-tokens
"$PYTHON" -B -m pytest -q LLM_branch/tests tests/test_directive_domains.py
```

The token preflight checks every training example under all three policies without loading model weights. In this run, all three passed 2,187/2,187; sample lengths were 480–6,656, with no code truncation. Source/token IDs stay identical across policies. Detailed audit outputs are under `$PREPARED`.

Set shared experimental controls explicitly:

```bash
export CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
export OBJECTIVE=PARETO_LATENCY
export STAGE1_LR_LORA=1e-4 STAGE1_LR_EMBED=2e-5
export STAGE1_BUDGET_INVARIANT_MAX_DUPLICATES=-1
export STAGE1_DOMAIN_REGISTRY=mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json
export STAGE1_RECOVERY_PREPARED="$PREPARED"
export STAGE1_DECISION_DIAGNOSTICS=1
export STAGE1_SUPERVISION_POLICY=legacy STAGE1_CONTRAST_WEIGHT=1
```

GPU optimizer smoke and fresh-adapter production smoke (use new output names):

```bash
"$PYTHON" -B -m pytest -q LLM_branch/tests/test_stage1_fp16_smoke.py
STAGE1_MAX_STEPS=2 STAGE1_EVAL_ON_START=0 \
  OUT=mailohls_runs/stage1_full_data_ce_smoke_reproduce_s123 \
  bash scripts/training/train_stage1.sh
```

One full-data optimized standard-CE recovery:

```bash
OUT=mailohls_runs/stage1_full_data_ce_recovery_s123 \
  bash scripts/training/train_stage1.sh
```

For a **later**, separately controlled objective comparison, select `STAGE1_SUPERVISION_POLICY=decision` and a fresh output directory; keep the same data, schedule and diagnostics. `decision_balanced` and `STAGE1_CONTRAST_WEIGHT=1.25` are explicit additional variables, not part of the initial run. `STAGE1_FULL_VALIDATION=1` enables all-context periodic selection and must be held identical between compared runs.

CPU reanalysis of existing saved selection predictions:

```bash
"$PYTHON" -B -m experiments.stage1_recovery.evaluate --prepared "$PREPARED" \
  --saved-metrics mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1/best_selection_metrics.json \
  --output artifacts/diagnostics/stage1_reference_review_reproduce
```

Full validation of the reference and candidate on all 383 selected requests, using the production decoder (not test labels):

```bash
"$PYTHON" -B -m experiments.stage1_recovery.evaluate --prepared "$PREPARED" \
  --full-validation \
  --adapters mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1 \
             mailohls_runs/stage1_full_data_ce_recovery_s123/best_custom_stage1 \
  --output artifacts/diagnostics/stage1_full_data_recovery_validation
```

After training and full validation, create a read-only assessment (no checkpoint replacement):

```bash
"$PYTHON" -B -m experiments.stage1_recovery.finalize \
  --full-run mailohls_runs/stage1_full_data_ce_recovery_s123 \
  --prepared "$PREPARED" \
  --full-validation-report artifacts/diagnostics/stage1_full_data_recovery_validation/report.json
```

Without the full validation argument, this command authenticates provenance and summarizes completed training and the saved 62-case comparison. It explicitly reports incomplete training if only intermediate checkpoints exist. The full run's exact implementation bytes are retained in `source_snapshot/`; this preserves reproduction even when later optional/reporting safeguards are added.

The historical eight-case evaluation and 20-example controls remain available through `evaluate.py` without `--full-validation` and `scripts/experiments/stage1_recovery_controls.sh`; their results remain separate.

## Evidence boundaries and deferred work

See [RESULTS.md](RESULTS.md) for newly executed tests, smoke/training status and authenticated receipts. The full-data implementation does not establish recovery until a completed adapter beats the reference and legal baseline on meaningful generated decisions and correct target transitions.

[LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) justify parameter-efficient adaptation, not our winner labels or objective. [Transformers Trainer](https://huggingface.co/docs/transformers/v4.51.3/main_classes/trainer) supports custom loss and checkpoint callbacks; this project keeps its installed 4.51.0 implementation and existing Trainer. Masked causal supervision is next-token prediction with ignored labels, not a new action classifier. The [HF completion-only explanation](https://huggingface.co/docs/trl/v0.23.1/sft_trainer#train-on-completion-only) illustrates the masking principle; TRL is not installed or used here. Our field/action masks are implemented directly in the existing Transformers trainer.

Joint-action ranking loss, [DPO](https://arxiv.org/abs/2305.18290), device-specific adapters/MoE, continuous target-conditioned residual adapters and Stage-2 structural integration are deferred. First establish full-data optimization and correct held-out-family target responses. Preference training would additionally require reliable measured alternatives and a defensible treatment of near ties; device-specialized architectures require evidence of an unresolved device-specific deficit. No such additional model is implemented here.
