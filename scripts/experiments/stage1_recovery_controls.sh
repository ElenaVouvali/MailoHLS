#!/usr/bin/env bash
set -euo pipefail
# Run after the two CE controls. Outputs are separate, immutable run directories.
# Set MAILOHLS_PYTHON to the environment's Python and CUDA_VISIBLE_DEVICES to a GPU UUID.
ROOT="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
RECOVERY_PYTHON="${MAILOHLS_PYTHON:-python}"
RECOVERY_OUTPUT="${1:-artifacts/diagnostics/stage1_recovery_20261028}"
for RECOVERY_CONTROL in ce_control ce_lr_control; do
  if [[ ! -f "$RECOVERY_OUTPUT/$RECOVERY_CONTROL/report.json" ]]; then
    echo "Missing completed prerequisite: $RECOVERY_OUTPUT/$RECOVERY_CONTROL/report.json" >&2
    exit 2
  fi
done
for RECOVERY_RUN in decision_ce joint_choice validation; do
  if [[ -e "$RECOVERY_OUTPUT/$RECOVERY_RUN" || -e "$RECOVERY_OUTPUT/$RECOVERY_RUN.log" ]]; then
    echo "Refusing to overwrite existing experiment or log: $RECOVERY_OUTPUT/$RECOVERY_RUN" >&2
    exit 2
  fi
done
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$RECOVERY_PYTHON" -m experiments.stage1_recovery.run \
  --output "$RECOVERY_OUTPUT/decision_ce" --mode decision-ce \
  --steps 200 --examples 20 --lr-lora 1e-4 --lr-embed 2e-5 --eval-greedy 4 \
  > "$RECOVERY_OUTPUT/decision_ce.log" 2>&1
"$RECOVERY_PYTHON" -m experiments.stage1_recovery.run \
  --output "$RECOVERY_OUTPUT/joint_choice" --mode joint-choice \
  --steps 200 --examples 20 --lr-lora 1e-4 --lr-embed 2e-5 --eval-greedy 4 \
  > "$RECOVERY_OUTPUT/joint_choice.log" 2>&1
"$RECOVERY_PYTHON" -m experiments.stage1_recovery.evaluate --joint \
  --output "$RECOVERY_OUTPUT/validation" --adapters \
  mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1 \
  "$RECOVERY_OUTPUT/ce_lr_control/adapter" \
  "$RECOVERY_OUTPUT/decision_ce/adapter" \
  "$RECOVERY_OUTPUT/joint_choice/adapter" \
  > "$RECOVERY_OUTPUT/validation.log" 2>&1
