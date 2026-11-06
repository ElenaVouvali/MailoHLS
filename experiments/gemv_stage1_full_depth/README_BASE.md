GEMV V4 multi-reference Stage-1 diagnostics
========================================

**Completed:** four official requests, natural/controlled prefix scoring, 992
selected 4096D vectors, validation and figures. Start with the
[technical report](GEMV_STAGE1_V4_DEEP_DIVE.md) and
[seven-slide presentation notes](PRESENTATION_NOTES.md).

Worktree: `/tmp/MailoHLS-stage1-refinement`, branch `research/stage1-refinement`,
base commit `3b7fd7cd39eb052b5661e09ad3562b98a5412036`. Unrelated work was
preserved. This package and the completed run's evidence form one focused update.

Only checkpoint
`/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run`
and V100S UUID `GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb` are supported.

The initial GPU failure was **restricted-sandbox device isolation**. The host
driver is healthy. GPU execution needs approved access outside that sandbox;
ordinary host-terminal execution exposes the GPU. No driver repair, service
restart, software installation or GPU substitution was required.

The [completed run](../../artifacts/gemv_framework_deep_dive/v4_20261106/)
contains the recorded evidence, including the
[final-state figure](../../artifacts/gemv_framework_deep_dive/v4_20261106/figures/fig8_final_decision_states.svg)
and [raw selected vectors](../../artifacts/gemv_framework_deep_dive/v4_20261106/hidden_states.npz):

- `official.jsonl`, per-request official JSON, exact prompts/token IDs.
- `candidate_scores.json`, `validation.json` and full decoder provenance.
- `hidden_states.npz` (float32 [992,4096]), `hidden_index.json`.
- `pairwise.csv`, `candidate_comparisons.csv`, `sensitivity_summary.json` and
  `findings.md`.
- `final_decision_states.csv` and `figures/fig8_final_decision_states.{png,svg}`:
  the final-residual summary discussed in report section 6.4.
- `figures/`: all PNG/SVG files; `figure_receipt.json` records hashes.
- `launch.json`, `launch_status.json`, `gpu_access_resolution.json`, checkpoint,
  compiler, code, Stage-2 and test receipts.

`environment.json` describes **sandboxed preparation**, not the successful GPU
process. Consult `launch.json` and `gpu_access_resolution.json` for execution.
The earlier `v4_20261105` directory preserves the blocked preparation, rejected
objective-only request schema, and archived blocked reports. The corrected
manifest adds the loader-supported `obj_mode` field; all prompt text/token IDs
are unchanged. `inputs/` contains exact authenticated source/metadata copies.
`results/` retains concise reviewable receipts and numerical summaries.

The completed run's figures, numerical outputs and selected hidden-state tensors
are included in this update under the exact artifact directory above. Model
checkpoint weights are not included. The earlier blocked-run directory is kept
locally; its archived drafts and temporary review patch are not publication data.

Regenerate all figures and numerical summaries **without loading the model**:

```bash
cd /tmp/MailoHLS-stage1-refinement
MPLCONFIGDIR=/tmp/gemv-matplotlib PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_framework_deep_dive.analyze
```

To reproduce inference, prepare a **new** directory. Run the GPU command from an
ordinary host terminal, or with approved access outside the restricted sandbox:

```bash
cd /tmp/MailoHLS-stage1-refinement
CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_framework_deep_dive.prepare --out /tmp/gemv-v4-repeat
bash experiments/gemv_framework_deep_dive/resume.sh /tmp/gemv-v4-repeat
```

The launcher invokes `LLM_branch.inference.eval_stage1_stage2_stage3.main` with
`--stage stage1`. It temporarily wraps loading and completed prediction in the
same process. Production source files and decoding semantics remain unchanged.
It verifies the idle GPU UUID/name/mapping, checkpoint hashes and frozen requests,
then performs the authentic decode, hook-equivalence decode, and selected replay.
Controlled scoring uses the original candidate filter/scorer/selector and freezes
previous values to fresh R1. No structural modules are initialized.

The launcher refuses to overwrite existing official outputs or tensors. Preserve
partial evidence and choose another new directory after investigating a failure.

Focused CPU checks (small random Llama fixtures, not V4 CPU inference):

```bash
MPLCONFIGDIR=/tmp/gemv-matplotlib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python -m pytest -q \
  experiments/gemv_framework_deep_dive/test_diagnostics.py \
  LLM_branch/tests/test_checkpoint_tokenizer.py \
  LLM_branch/tests/test_source_tool_policy.py
```

No new dependencies, training, synthesis, driver changes or checkpoint writes.
