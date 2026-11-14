#!/usr/bin/env bash
# Intended for Liono, sequential exclusive V100S use. Run once after installation.
# Required inputs must be specified rather than guessed from arbitrary directories.
set -Eeuo pipefail
: "${ROOT:?Set ROOT to the checked-out film-gmt-stage2-final repository root}"
: "${BASELINE_RUN:?Set BASELINE_RUN to the existing original Stage-2 run dir containing checkpoint-0300 and protocol.json}"
: "${KALMAN_SOURCES_ROOT:?Set KALMAN_SOURCES_ROOT to the compat9 source root containing Kalman/kernel_info.txt}"
: "${KALMAN_MEMORY_DIR:?Set KALMAN_MEMORY_DIR to the frozen external Kalman GMT memory directory}"
ROOT="$(realpath "$ROOT")"
BASELINE_RUN="$(realpath "$BASELINE_RUN")"
KALMAN_SOURCES_ROOT="$(realpath "$KALMAN_SOURCES_ROOT")"
KALMAN_MEMORY_DIR="$(realpath "$KALMAN_MEMORY_DIR")"
PYTHON="${PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
GPU_UUID="${GPU_UUID:-GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb}"
PARENT="${PARENT:-/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run}"
FILM_CP="${FILM_CP:-/home/elvouvali/MailoHLS/worktrees/gnosis-stage1-film-ab/artifacts/gnosis_stage1_ab/stage1_ab_pilot_20261112_retry2/B/checkpoint-0600}"
GNOSIS_BANK="${GNOSIS_BANK:-$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_bank_original_anchors}"
GMT_MEMORY="${GMT_MEMORY:-$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/memory_B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/routing_ablation_20261114}"
TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.exclusive_campaign.lock"
flock -n 9 || { echo "Another routing campaign is running" >&2; exit 7; }
export CUDA_VISIBLE_DEVICES="$GPU_UUID" SALT_GPU_UUID="$GPU_UUID" PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
"$PYTHON" "$TOOL_DIR/setup_routing_ablation.py" --repo "$ROOT"
"$PYTHON" "$TOOL_DIR/check_compare.py" --baseline-run "$BASELINE_RUN"
"$PYTHON" "$TOOL_DIR/mask_audit.py" --memory-dir "$GMT_MEMORY" \
  --output "$OUTPUT_ROOT/mask_audit_training.json"
"$PYTHON" "$TOOL_DIR/mask_audit.py" --memory-dir "$KALMAN_MEMORY_DIR" \
  --output "$OUTPUT_ROOT/mask_audit_external.json"
for file in \
  "$PARENT/training_contract.json" "$FILM_CP/film.pt" \
  "$GNOSIS_BANK/manifest.json" "$GMT_MEMORY/memory_manifest.json" \
  "$KALMAN_SOURCES_ROOT/Kalman/kernel_info.txt" "$KALMAN_MEMORY_DIR/memory_manifest.json"; do
  [[ -f "$file" ]] || { echo "Missing required file: $file" >&2; exit 3; }
done
TRAIN="$ROOT/experiments/routing_ablation_20261114/train_routing.py"
INFER="$ROOT/experiments/routing_ablation_20261114/infer_routing.py"
mkdir -p "$OUTPUT_ROOT/predictions" "$OUTPUT_ROOT/logs"

common_train=(--repo "$ROOT" --parent "$PARENT" --film-checkpoint "$FILM_CP"
  --gnosis-bank "$GNOSIS_BANK" --gmt-memory "$GMT_MEMORY" --gpu-uuid "$GPU_UUID"
  --cohort original-film-31+2 --seed 123 --steps 450 --stop-at-step 300
  --contexts 0 --grad-accum 4 --max-length 7168 --lr-xattn 1e-4 --lr-gate 2e-4
  --gate-open-steps 60 --gate-open-lr 8e-4 --probe-sft 3
  --rank-budget .20 --rank-clock .05 --eval-interval 150
  --val-sft 24 --val-pair-groups 8 --min-train-families 10 --min-val-families 2)
for variant in self_only full_memory; do
  RUN="$OUTPUT_ROOT/$variant"
  if [[ ! -f "$RUN/TARGET_REACHED.json" ]]; then
    for mode in audit smoke train; do
      dest="$RUN/$mode"
      if [[ -e "$dest" ]]; then
        echo "Refusing ambiguous/rerun state: $dest" >&2
        exit 4
      fi
      echo "[START] $variant $mode"
      "$PYTHON" -u "$TRAIN" --routing-mode "$variant" --mode "$mode" \
        --output "$dest" "${common_train[@]}" >"$OUTPUT_ROOT/logs/${variant}_${mode}.log" 2>&1 || {
          tail -n 80 "$OUTPUT_ROOT/logs/${variant}_${mode}.log" >&2; exit 5;
        }
      echo "[DONE] $variant $mode"
    done
    [[ -f "$RUN/train/TARGET_REACHED.json" ]] || { echo "$variant did not produce checkpoint-0300" >&2; exit 5; }
    cp "$RUN/train/TARGET_REACHED.json" "$RUN/TARGET_REACHED.json"
  else
    [[ -f "$RUN/train/TARGET_REACHED.json" ]] || { echo "Invalid marker: $variant" >&2; exit 5; }
    echo "[VERIFIED SKIP] $variant training complete"
  fi
done

"$PYTHON" "$TOOL_DIR/check_compare.py" \
  --baseline-run "$BASELINE_RUN" --self-run "$OUTPUT_ROOT/self_only/train" \
  --full-run "$OUTPUT_ROOT/full_memory/train"

for variant in compiler_relational self_only full_memory; do
  case "$variant" in
    compiler_relational) RUN="$BASELINE_RUN" ;;
    *) RUN="$OUTPUT_ROOT/$variant/train" ;;
  esac
  output="$OUTPUT_ROOT/predictions/$variant.jsonl"
  if [[ -e "$output" ]]; then echo "Prediction exists: refusing overwrite $output" >&2; exit 6; fi
  echo "[INFER] $variant Kalman ZCU104 10ns 100%"
  "$PYTHON" -u "$INFER" --repo "$ROOT" --parent "$PARENT" \
    --film-checkpoint "$FILM_CP" --run-dir "$RUN" --routing-mode "$variant" \
    --checkpoint-step 300 --kernel Kalman --objective latency \
    --device zcu104 --clock-ns 10 --budget 100 \
    --sources-root "$KALMAN_SOURCES_ROOT" --memory-dir "$KALMAN_MEMORY_DIR" \
    --output "$output" --dry-run >"$OUTPUT_ROOT/logs/${variant}_dry_run.log" 2>&1 || {
        tail -n 100 "$OUTPUT_ROOT/logs/${variant}_dry_run.log" >&2; exit 8;
      }
  "$PYTHON" -u "$INFER" --repo "$ROOT" --parent "$PARENT" \
    --film-checkpoint "$FILM_CP" --run-dir "$RUN" --routing-mode "$variant" \
    --checkpoint-step 300 --kernel Kalman --objective latency \
    --device zcu104 --clock-ns 10 --budget 100 \
    --sources-root "$KALMAN_SOURCES_ROOT" --memory-dir "$KALMAN_MEMORY_DIR" \
    --output "$output" >"$OUTPUT_ROOT/logs/${variant}_inference.log" 2>&1 || {
        tail -n 100 "$OUTPUT_ROOT/logs/${variant}_inference.log" >&2; exit 8;
      }
done
"$PYTHON" "$TOOL_DIR/check_compare.py" \
  --baseline-run "$BASELINE_RUN" --self-run "$OUTPUT_ROOT/self_only/train" \
  --full-run "$OUTPUT_ROOT/full_memory/train" --predictions "$OUTPUT_ROOT/predictions"
echo "[DONE] Check $OUTPUT_ROOT/predictions/kalman_routing_comparison.md"
