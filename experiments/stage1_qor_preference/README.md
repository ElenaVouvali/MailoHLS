# Stage-1 QoR-aware preference refinement (experimental)

This package refines the **V4 multi-reference PARETO_LATENCY** adapter using
measured feasible-configuration preferences. It does not change production
registries, model architecture, tokenizer, GNN, cross-attention or inference.
The experiment branch is `research/stage1-qor-preference`, starting from
`63afcfc7a54e573a544678e728cdb7d99d865981`. It is not a production promotion.

## Authenticated inputs and coverage

The initial adapter is exclusively
`/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run`,
SHA-256 `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`;
contract SHA-256 `1c7a2034dba45030e51525fd415367e6fcf276cceddd89eb1132edfd66840671`.
The pinned DeepSeek-Coder-6.7B revision, saved tokenizer bytes, rank-8 attention
LoRA, trained special-token deltas, objective, source IDs and domains are retained.

`prepare.py` reads the existing sealed `/tmp/stage1_quality_sft_v1` banks; their
eligibility-manifest SHA-256 matches the original V4 training manifest. It reuses
the existing candidate identities, feasibility rules, source/action validators,
source-target serializer and GNΩSIS family split. The parent dataset is hashed;
only selected train/validation measurement rows are decoded for identity checks.
Test membership is inspected for exclusion, but test measurements are not decoded.

For each identical source/device/measured-clock/objective/budget request, preferred
configurations are within 1% of the best feasible admissible measured latency;
rejections are feasible and at least 10% slower than their preferred configuration.
All assignments remain complete, measured designs. Current V2 tool-policy checks
are applied to pair eligibility, separately from the unchanged legacy inference
candidate policy. Missing/invalid QoR and illegal assignments are excluded; source
observability exclusions and duplicate/conflict handling are inherited from the
authenticated bank construction. Dataset latency units are retained; no conversion
to cycles or new QoR labels is invented.

The audit found 631 eligible train requests and 9,859,730 pair combinations before
capping. One deterministic pair per request and eight pairs per family seal a
**136-pair pilot across 17 train families**, balanced over both devices, three
clocks and three budgets. Validation seals **54 pairs across two disjoint
families** (18 Viterbi, 36 Pathfinder). All 18 device/clock/budget strata occur in
the training sample. Full counts, gap quantiles, exclusions and hashes are in the
generated `manifest.json`. The broad latency gaps make this a pilot, not evidence
of fine discrimination among near-optimal configurations.

The preliminary sampling audit was retained separately before fixing sorted-device
truncation in the cap. No model was trained on that preliminary sample, and no
model outcomes were used to choose the corrected balanced sample.

## Objective and masks

For each pair, `objective.py` implements:

```text
DPO = -logsigmoid(0.1 * [(logpi(chosen)-logpi(rejected))
                       -(logref(chosen)-logref(rejected))])
loss = existing_action_normalized_SFT(chosen) + 0.2 * DPO
```

`logpi`/`logref` are **sums**, not action averages, of causal log-probabilities
over all nonforced directive VALUE tokens. The unchanged production
`SFTDataset(..., supervision_policy='decision')` supplies the labels and weights.
Prompt, anchors, fixed field text, newlines, EOS and semantic singletons are
excluded. Preferred/rejected packs are built independently with their own correct
preceding values; exact common source/target prompt tokens and absence of source
truncation are asserted. Token counts and encoding/mask hashes are recorded.

The SFT term alone uses the existing equal-action/equal-nonforced-field/value-token
weights. No action normalization is applied to the DPO sequence sums. All selected
logits, log-softmax, log-probability sums and loss arithmetic are float32.

The reference is the exact starting V4 checkpoint, including trained LoRA and
special-token deltas. Scores are cached **before optimizer creation**, tied to
checkpoint/tokenizer/domain/pair/mask hashes. Cache and policy use the same kbit
preparation, FP16 autocast and scoring function. One quantized backbone is used;
there is no second 6.7B reference model. Dropout is disabled for deterministic
preference training/scoring, without changing the saved LoRA configuration.

## Predeclared training protocol

Seed 123; NF4 double quantization/FP16 on V100S UUID
`GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb`; gradient checkpointing; AdamW;
learning rate 1e-5 for LoRA and special-token deltas versus V4's 5e-5;
accumulation 4; gradient clip 1.0; two-update warmup and bounded cosine decay.

After four successful updates, the fixed training probe must have finite gradients,
positive policy-minus-reference margin improvement, SFT NLL no greater than
`start*1.15+0.1`, and a successful adapter roundtrip. If it passes, training continues
to **34 successful updates**, one capped pass. No broad sweep, validation-driven
checkpoint search, GEMV-based tuning or test selection is performed. AMP skips are
not silently counted as updates. Both intended gradient groups are checked.

Pilot and final adapters are saved separately under
`/home/elvouvali/mailohls_runs/stage1_qor_dpo_latency_v1/`; original V4 bytes remain
unchanged. Roundtrip uses another small adapter on the same backbone, not a second
base model. Contracts preserve the Stage-1 interface and add explicit preference
lineage, masks, cache identity and experimental status.

## Evaluation boundaries

- Offline ranking uses family-disjoint measured validation pairs, with raw summed
  policy margins, SFT anchor NLL/value-token accuracy and family/target breakdowns.
- The original frozen 54 inference requests and authenticated V4 legacy outputs
  are reused. The refined checkpoint uses the same production greedy decoder and
  mean RHS-plus-newline scoring. The existing paired evaluator reports complete
  actions, source/V2 legality, target-response counts and exact-match QoR; unmatched
  configurations remain unknown, not hardware failures.
- Physical comparison is limited to two source/target/budget contexts already
  declared in the prior six-request validation subset, fixed before refined
  decoding: at most four deduplicated configurations, 300 seconds per new job.
  Existing exact completed receipts are preferred. Shared unrelated HLS jobs are
  not interrupted; a blocked or incomplete physical comparison does not qualify
  the checkpoint or establish hardware improvement.
- GEMV reuses the four frozen requests and the existing tracing/replay utility.
  Its V4 outputs remain immutable. Both policies' controlled comparisons use the
  **same authentic V4 R1 preceding assignments**. Representation comparisons are
  within-checkpoint target contrasts; cross-checkpoint vectors are not treated as
  a common learned basis. GEMV is a held-out descriptive probe, never a selector.

## Recorded execution status

The bounded 34-update V100S training run completed and the final adapter was
saved and roundtrip checked. Offline held-out preference accuracy changed from
19/54 to 18/54; this pilot did **not** improve validation preference ranking.
The trained adapter remains an experimental candidate, without evidence of
hardware superiority or qualification for Stage-2 research use. Mechanical
Stage-2 contract compatibility passed separately.

Following the requested scope reduction, subsequent inference ran **only the
four GEMV requests**. All four were complete, passed the separate source/tool
V2 audit, and passed hook-on/off behavioral equivalence. The 54-request native
inference and new synthesis comparison were not executed. The training launcher
reported a shell EOF after Python had completed and saved the model; the exact
wrapper receipt is retained with the successful training results.

See [GEMV_RESULTS.md](GEMV_RESULTS.md) for the mechanism and findings, and
[results/v1_20261106/](results/v1_20261106/) for committed compact evidence.
Raw hidden tensors, pair datasets and weights remain external; their SHA-256
and paths are in `results/v1_20261106/external_artifacts.json`.

For GEMV alone, use the environment below and run
`python -m experiments.stage1_qor_preference.evaluate gemv`.
`run_eval.sh` is the broader evaluation launcher and was not run after the scope
reduction. Preserve existing outputs when reproducing.

## Reproduction

Use `/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python` with repository
`PYTHONPATH`, offline Hugging Face caches and the exact V100S UUID. GPU commands
require approved host execution because the restricted sandbox hides device nodes.

```bash
cd /tmp/MailoHLS-stage1-qor-preference
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.stage1_qor_preference.prepare
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python -m pytest -q \
  experiments/stage1_qor_preference/test_preference.py
bash experiments/stage1_qor_preference/run_train.sh
bash experiments/stage1_qor_preference/run_eval.sh
```

Existing pair manifests and model directories are protected against overwrite.
These commands describe the original run; choose new explicit artifact/model
locations for another experiment rather than replacing the sealed evidence.
Large pairs, caches and outputs are ignored under
`artifacts/stage1_qor_preference/v1_20261106/`; trained weights remain outside Git.
The user authorized publishing this experiment branch after the GEMV run.

## Future Stage-2 initialization

Mechanical compatibility is distinct from qualification. Rehash the new Stage-1
adapter/contract and verify the original backbone revision, tokenizer mapping,
objective and directive registry. Use the existing Stage-2 trainer with
`--init_adapter_dir /home/elvouvali/mailohls_runs/stage1_qor_dpo_latency_v1/run`,
`--objective PARETO_LATENCY`, `--top_k 1`, `--lr_lora 0`, `--lr_embed 0`, the
authenticated family split and a separately validated compatible memory manifest.
Omit `--disable_structural_memory`. Initialize **new** structural attention/gates
at the declared placement/schedule; do not provide historical structural weights
through `--init_structural_xattn_from`. The existing trainer freezes Stage-1,
records parent hashes and checks the structural trainable contract. Verify those
hashes stay unchanged during Stage-2 training. This task does not run Stage-2 or
establish that V4b memory is interchangeable with production memory.
