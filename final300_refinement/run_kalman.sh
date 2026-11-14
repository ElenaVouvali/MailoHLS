#!/usr/bin/env bash
# Run ON LIONO. Exact checkpoint300 ZCU104 10ns full-budget Kalman pilot.
set -Eeuo pipefail
TASK="${1:-}"; [[ "$TASK" =~ ^(audit|plan|score-smoke|baseline|next3|rest4|report)$ ]] || {
  echo 'Usage: bash run_kalman.sh audit|plan|score-smoke|baseline|next3|rest4|report' >&2;exit 2;
}
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SALT_ROOT:-/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final}"
D="${SALT_DEPLOY_LIONO:-$ROOT/deployments/film-gmt-stage2-6ee1390-20261113}"
BASE="${SALT_BASELINE_KALMAN_JSONL:-$ROOT/artifacts/stage2_eval_9_20261113/final300_eval_20261113/predictions/zcu104_10ns_all9.jsonl}"
PY="${SALT_LIONO_PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
OUT="${SALT_REFINE_OUT:-$ROOT/artifacts/refinement_final300_kalman_20261114/model}"
for f in "$BASE" "$D/eval_sources_compat9/Kalman/kernel_info.txt" \
         "$D/eval_gmt_memory_B_compat9/memory_manifest.json" \
         "$D/runs/campaign_s123_v1/final/checkpoint-0300/receipt.json"; do
 [[ -f "$f" ]] || { echo "MISSING: $f" >&2;exit 3; }
done
COMMON=(--repo "$ROOT" --deployment "$D" --baseline-jsonl "$BASE" \
 --sources-root "$D/eval_sources_compat9" --memory-dir "$D/eval_gmt_memory_B_compat9" \
 --out "$OUT" --kernel Kalman --clock-ns 10 --budget-pct 100 \
 --max-evaluations 8 --policy model --python "$PY" \
 --scorer "$PKG/score_checkpoint300_liono.py")
case "$TASK" in
 audit) MODE=audit;LIMIT=1;;
 plan) MODE=plan;LIMIT=1;;
 score-smoke) MODE=score-smoke;LIMIT=1;;
 baseline) MODE=run;LIMIT=1;;
 next3) MODE=run;LIMIT=3;;
 rest4) MODE=run;LIMIT=4;;
 report) MODE=report;LIMIT=1;;
esac
cd "$ROOT"
exec "$PY" -u "$PKG/refine_final300.py" --mode "$MODE" --limit "$LIMIT" "${COMMON[@]}"
