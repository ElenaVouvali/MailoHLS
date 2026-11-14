# Reproducible compiler-guided routing ablation (Liono)

This package does not train anything in ChatGPT or access the Liono filesystem. It installs two **isolated generated source files**, verified against the exact public MailoHLS Git blobs from `research/stage2-evaluated-20261113`. It does **not** edit original trainer or inference code.

**Design.** Compiler-guided is the pre-existing immutable checkpoint-0300 baseline. Two new Stage-2 checkpoints start from the exact Stage-1 Arm-B FiLM checkpoint-0600, frozen V4 LoRA/FiLM/GNN/GMT, identical 31+2 bank, 450-step cosine training **schedule**, and stop just after saving checkpoint-0300. Only the per-action XAttn mask changes:

- `self_only`: action `Lk` sees only original anchor slot `k` (one of the first 64 slots), not GMT evidence slots.
- `full_memory`: action `Lk` sees all **valid** original action anchors and all valid typed GMT slots, excluding padding.
- `compiler_relational`: original compiler-derived action-to-memory relation (unchanged baseline).

In all three, textual tokens before an `Lk` route receive no structural memory and cross-attention is applied at the same supervised RHS positions. Different attention masks do not alter model weights/parameter counts. The extra two variants are **trained** with the selected mask; they are not OOD inference interventions.

## Step 1 — stage package on Liono

Copy this directory to e.g. `~/routing_ablation_package/`, keeping the Python and Bash files together (`setup_routing_ablation.py`, `run_routing_ablation.sh`, `check_compare.py`, `mask_audit.py`). Use the same local Python environment as the 2026-11-13 Stage-2 campaign.

## Step 2 — explicit paths and preflight

```bash
export ROOT=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final
export PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
export GPU_UUID=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb

# IMPORTANT: Set these three using actual Liono artifacts; do not guess.
export BASELINE_RUN=/ACTUAL/PATH/TO/original_stage2_run_with_protocol_json_and_checkpoint_0300
export KALMAN_SOURCES_ROOT=/ACTUAL/PATH/TO/eval_sources_compat9
export KALMAN_MEMORY_DIR=/ACTUAL/PATH/TO/eval_gmt_memory_B_compat9

# Inspect first, without GPU use:
$PYTHON ~/routing_ablation_package/setup_routing_ablation.py --repo "$ROOT"
$PYTHON ~/routing_ablation_package/check_compare.py --baseline-run "$BASELINE_RUN"
```

The source installer **fails** if the public upstream Git blobs differ from the exact audited versions. The compact inference script belongs to `research/stage2-evaluated-20261113`, not the earlier `film-gmt-stage2-final` branch. If it is absent locally, fetch the evaluated branch first with `git -C "$ROOT" fetch origin research/stage2-evaluated-20261113` and rerun setup. This is intentional; inspect any mismatch rather than removing the check.

The campaign uses exactly the native `film_gmt_stage2_final` launcher arguments for the original 31+2 fallback (same dataset, optimizer, seed 123, grad accumulation 4, LR warmup/cosine, Gate opening 60, same validation selection); no hidden regeneration/curation of data.

## Step 3 — launch campaign

```bash
nohup bash ~/routing_ablation_package/run_routing_ablation.sh \
  > "$ROOT/routing_ablation_20261114_controller.log" 2>&1 < /dev/null &

tail -f "$ROOT/routing_ablation_20261114_controller.log"
```

The campaign first authenticates the immutable baseline and audits actual compiler mask cardinalities, then runs audit/smoke/train for self-only and full-memory **sequentially on the Liono V100S**, checks checkpoint and protocol comparability, then performs three Kalman inference runs at ZCU104 / 10 ns / 100% resources. It creates separate logs, receipts, predictions and a Markdown report under `$ROOT/artifacts/routing_ablation_20261114/`.

The campaign **refuses ambiguous output states**. If a training process is interrupted, do not delete outputs blindly; inspect the log and stored `checkpoint-0150` or `checkpoint-0300`. Use the trainer's authenticated resume mechanism only after checking the exact protocol. This launcher deliberately does not auto-resume or overwrite results.

## Important limitations

1. This exact-source installer is designed from the **public GitHub versions**, not the private running Liono file tree. It needs a real Liono preflight and the generated GPU smoke before results can be trusted.
2. The original baseline must be an authenticated **checkpoint 300 from the same 450-step schedule and frozen FiLM/GMT bank**. The comparator stops if dataset, memory, parents, loss, selection or hyperparameters differ.
3. The script **only generates inference predictions**, not Vitis HLS latency/resource measurements. Three different directive lists are not sufficient for a claim of QoR improvement. Synthesize under the identical target and constraints before reporting the comparison as an HLS QoR result.
4. One Kalman configuration is a compact architectural case study, not enough by itself for a general claim of superiority. Retain independent family-level held-out validation metrics and expand to other unseen kernels if possible.
