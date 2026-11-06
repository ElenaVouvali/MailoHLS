# GEMV V100S diagnostic: QoR-preference Stage-1

This note records only the four frozen GEMV Stage-1 requests for the trained
V4-based QoR-preference adapter. It is an inference and representation probe,
not an HLS QoR evaluation.

Compact committed evidence and figures are available in [results/v1_20261106/gemv/](results/v1_20261106/gemv/). Raw tensors remain external with hashes in [external_artifacts.json](results/v1_20261106/external_artifacts.json).

## Execution identity

- Checkpoint: `/home/elvouvali/mailohls_runs/stage1_qor_dpo_latency_v1/run`
- Adapter SHA-256: `b440d06aeb7aac85ae01018cf2caa421505ec81a717052c8163afbd95eff6057`
- Training contract SHA-256: `b55cc6e046312731c1913864946258fbf67c588d0f0ebfb691459b42d452bfec`
- Device: Tesla V100S-PCIE-32GB, UUID `GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb`
- Decoder: repository Stage-1 entrypoint, `legacy` candidate policy, greedy, one sample, mean candidate score.
- Requests: authenticated GEMV source and metadata; ZCU104 10 ns/full, ZCU104 10 ns/20%, ZCU104 3.33 ns/full, U200 10 ns/full.
- Validation: four complete assignments; all four pass the separate current source/tool V2 audit. Full-decode hook-on/off equivalence passed for every request.
- Raw output directory: `artifacts/stage1_qor_preference/v1_20261106/gemv/`.

## Predicted directives

Array partition columns are `(type, factor, dimension)`; loop columns are
`(pipeline II, unroll factor)`. Bold values differ from R1.

| Action | R1: ZCU104 10 ns 100% | R2: ZCU104 10 ns 20% | R3: ZCU104 3.33 ns 100% | R4: U200 10 ns 100% |
|---|---|---|---|---|
| L1 — A partition | block, 2, 1 | block, 2, **2** | block, 2, 1 | block, 2, 1 |
| L2 — x partition | cyclic, 2, 1 | cyclic, **4**, 1 | cyclic, 2, 1 | cyclic, 2, 1 |
| L3 — y partition | complete, 0, 1 | complete, 0, 1 | complete, 0, 1 | complete, 0, 1 |
| L4 — outer loop | 0, 2 | 0, 2 | 0, 2 | 0, 2 |
| L5 — inner loop | 0, 2 | **1, 0** | 0, 2 | 0, 2 |

The only changed target here is the available resource budget in R2. Identical
outputs for R1/R3/R4 are observations, not evidence of failed adaptation or
target-optimality. The V4 SFT baseline had a different R1 L3 choice
(`cyclic, 4, 1`); the refined model predicts `complete, 0, 1` for L3 in all
four requests. Other baseline/refined differences are recorded in
`artifacts/stage1_qor_preference/v1_20261106/gemv_comparison.json`.

## What the captured hidden states mean

The decoder serializes one directive field at a time. Immediately before
scoring a field's right-hand side, the prompt plus all preceding generated
fields forms a causal prefix. The hidden vector at that prefix's final token is
the decision-context representation. In the standard model path, the final
decoder state is normalized, projected through the LM head, and converted to
token log-probabilities. The existing constrained decoder scores only its
admissible RHS candidates (mean token log-probability here) and chooses the
highest-scoring candidate. Earlier choices are then appended to the prefix for
later fields.

The vectors are therefore the contexts used by the scoring computation; they
are not standalone explanations or causal attributions for a directive. A
vector's cosine or movement does not say which semantic feature caused a
candidate to win. The candidate score table is the direct evidence for the
ranking at each decision.

For all four requests, the trace saved float32 vectors at layers 8, 16, 24 and
32, at both `post_self_attention_residual` and `post_decoder_residual`. The
five `<SRC_Lk>` source-marker positions are captured on each natural prompt;
all 13 directive-decision prefix positions are captured for both natural and
controlled prefixes. Each selected vector has shape `[4096]`; the compact NPZ has shape
`[992, 4096]`. `hidden_index.json` gives request, natural/controlled prefix,
action, field, exact token index/ID/text, layer, boundary and source module.
The decoder's final selected candidate scores are in `official.jsonl` under
each site's `site_debug`; token/state comparisons are in `pairwise.csv`.

## Most informative decisions

Candidate scores below are the existing decoder's per-field mean log-prob
scores; margins are top score minus runner-up, not hardware-quality estimates.

| Field | R1 top candidates and margin | R2 top candidates and margin | Choice |
|---|---|---|---|
| L1 A dimension | `1` −0.33315 vs `2` −0.38094; margin 0.04779 | `2` −0.31848 vs `1` −0.39606; margin 0.07759 | 1 → 2 |
| L5 pipeline II | `0` −0.28137 vs `1` −0.42236; margin 0.14099 | `1` −0.28845 vs `0` −0.41312; margin 0.12467 | 0 → 1 |
| L5 unroll | `2` −0.32573 vs `4` −0.67123; margin 0.34550 | `0` −0.01126 vs `16` −3.98859; margin 3.97733 | 2 → 0 |

R1/R3/R4 have the same selected values in these fields. For the full 13-field
set, natural-prefix candidate choices change in 4 of the 39 target contrasts;
the complete candidate domains are identical across requests.

## Hidden-state comparisons

On the **natural prefix**, L5 unroll is the largest observed movement: at layer
32 `post_decoder_residual`, R1 vs R2 has cosine `0.87883`, relative L2 change
`0.53307`, and norms `787.08` vs `878.80`. At layer 24 the same site reaches
relative L2 `0.75449` and cosine `0.71179`. This coincides with a changed
candidate preference, but R2 also has different earlier generated fields, so
this natural-path difference mixes target context with autoregressive prefix
changes.

In the **controlled-prefix** replay, all previous fields are fixed to the
authentic V4 R1 assignment while each request retains its own hardware prompt.
At L5 unroll, the R1/R2 preference remains `2` in both requests; its ranking
changes but does not reverse. At L5 pipeline, `0` changes to `1` for R2. Thus
the natural L5 unroll flip is not reproduced when the earlier directive prefix
is held fixed; it is associated with the complete natural target-conditioned
trajectory, rather than demonstrated to be a direct effect of the R2 target
alone.

For L1 A dimension, which is early in the sequence, R1/R2 still changes `1`
to `2` with the R1 prefix frozen. Its layer-32 post-decoder state has cosine
`0.99399` and relative L2 movement `0.12698`. This provides a cleaner
example of budget-conditioned candidate movement. For comparison, L1
partition factor remains `2` in all requests; its layer-32 R1/R2 state has
cosine `0.99959` and relative L2 `0.03207`.

Across decision states, mean relative L2 movement is `0.1232` for natural R1/R2,
`0.0420` for R1/R3, and `0.0589` for R1/R4. With controlled prefixes, the means
are `0.0370`, `0.0430`, and `0.0601`, respectively. These aggregates include
all 13 field sites and all four inspected layers at the post-self-attention and
post-layer boundaries. They describe representation sensitivity only; they
are not evidence that the decisions improve latency or satisfy physical
constraints.

## Reproduce and inspect

The run used the already-existing GEMV launcher, prompt builder, constrained
decoder, tracing and analysis code; it did not retrain a model or change its
weights. On the assigned V100S, from the experiment worktree:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb \
/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
-m experiments.stage1_qor_preference.evaluate gemv
```

Key evidence files:

- Exact prompt/request/checkpoint provenance: `gemv/manifest.json`, `gemv/prompts/`, `gemv/launch.json`.
- Official assignments and candidate scores: `gemv/official.jsonl`.
- Raw hidden vectors and token metadata: `gemv/hidden_states.npz`, `gemv/hidden_index.json`.
- Pairwise representation metrics: `gemv/pairwise.csv`, `gemv/final_decision_states.csv`.
- Field ranking/margins: `gemv/candidate_comparisons.csv`.
- Source/tool and instrumentation checks: `gemv/validation.json`, `gemv/launch_status.json`.
- Figures: `gemv/figures/` (PNG and SVG).

No synthesis was run. The authenticated source has uninitialized local input
arrays and returns only `y[0]`; these predictions do not establish numerical
correctness, physical QoR, resource feasibility or latency improvement.
