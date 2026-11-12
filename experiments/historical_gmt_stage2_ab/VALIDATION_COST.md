# Review of Stage-2 A/B validation cost

The Liono A2 smoke launch has an expensive **initial selection evaluation**.
Its `0/2` progress bar counts optimizer updates, not candidate-scoring work.
This report documents the executing configuration; it does not change or stop
the running experiment. The driver automatically starts the full run if both
smoke arms succeed. The full comparison was not started at inspection.

## Executed observations and reproducible workload count

The ordinary 36-example validation-loss pass finished in approximately 95 s.
At 2026-11-12 11:24:20 UTC, the A2 was at 99% GPU utilization, using 5,787 MiB,
and the Python process was active. No current traceback or OOM appeared in its
log. These observations support active computation; they are not per-case
progress measurements or proof against every possible stall.

The subsequent selection callback evaluates **17 cases** from Viterbi and
Pathfinder, each with 17 or 20 directive fields. For every case it performs:

1. Free-running constrained directive generation by complete RHS scoring.
2. A separate reference-prefix candidate-scoring pass for ranking diagnostics.

A CPU reconstruction counts **1,405 LLM forwards for the reference-prefix pass
alone**, plus uncounted free-running forwards whose number depends on earlier
model-selected directives. The raw count is in
[validation_workload.json](validation_workload.json). This is a workload count,
not a measured inference-time breakdown.

The current structural scorer forces candidate batch size to **one**. Every
non-forced candidate reruns the full source and preceding directive prefix
through DeepSeek-Coder; this path does not reuse a prefix KV cache. The callback
builds all case results before printing and saving its report. Therefore a long,
GPU-active period without new log output is expected. The small GMT warm-start
has already finished and is not responsible for this long evaluation phase.

Relevant maintained code is in
[train_SFT_xattn_new.py](../../LLM_branch/train/train_SFT_xattn_new.py):
`score_rhs_candidate_suffix`, `score_teacher_forced_rhs_candidates`,
`constrained_decode_rhs_by_candidate_scoring`, and
`StageValSelectionCallback.on_evaluate`.

Reproduce the count without using a GPU or loading an LLM:

```bash
CUDA_VISIBLE_DEVICES='' /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  experiments/historical_gmt_stage2_ab/inspect_validation_cost.py \
  --run artifacts/historical_gmt_stage2_ab/historical_gmt_ab_liono_a2_smoke_retry2_20261112
```

## Settings to inspect before a new full run

These are existing trainer controls, not changes applied to the live run.
The wrapper currently hard-codes some of them; review its trainer invocation
before editing a copy and using fresh run IDs.

| Setting | Executing configuration | Consequence / possible adjustment |
| --- | --- | --- |
| Initial evaluation | `--eval_on_start` defaults true | Runs selection before update 1. `--no-eval_on_start` skips this baseline in a new run; record that omission. |
| Smoke evaluations | Steps 0, 1 and 2, both arms | Six full selection evaluations for only four total optimizer updates. A smaller identical smoke subset can test integration at lower cost. |
| Full evaluations | Start, every 50 steps, and final | A 300-update arm schedules seven selection evaluations, before any early stop. Four arms schedule 28 in total. |
| Validation cases | `--selection_cases_per_kernel_device 4`, `--selection_num_val_kernels 0` | Produces 17 cases here. Reducing cases changes the selection protocol and can omit target/budget transitions; use identical cases across arms/seeds and retain a full final evaluation. |
| Early stopping | Trainer default patience 3 | Arms may stop after different update counts. The wrapper's 300 updates are a cap, not a guaranteed matched realized budget. For a strictly matched fixed-budget run, explicitly use `--early_stopping_patience 0` for both arms. |
| Context contract | `MAX_LENGTH=7168` | Required by the authenticated V4 parent. The original 4096 launch was rejected before training. Actual longest source/answer row is 3811 tokens; do not bypass the parent check. |
| Candidate batching | Effective size 1 with structural memory | Raising `--selection_candidate_batch_size` alone does not accelerate this path. Structural batching requires implementation and score/routing parity checks. |
| Prefix caching | No reuse in current scorer | A future optimization must preserve complete RHS/newline scores, causal positions and action-specific structural masks. Simply enabling `use_cache` does not implement safe reuse. |

Per-case progress logging would improve visibility without changing the scores.
It is not yet added to the running callback. No credible full-run ETA is available
until a complete selection evaluation and training throughput are measured.

## Scientific scope remains bounded

The comparison freezes the compiler-trained GMT tokenizer during Stage-2; only
cross-attention and attention gates are trained through the directive objective.
It does not demonstrate joint GMT/LLM alignment or improved physical QoR.
Changing evaluation frequency, selection cases or stopping rules requires a new
recorded protocol applied equally to A and B. Preserve the authenticated GNN,
V4 parent, memory identities, family separation and legal causal scoring.
