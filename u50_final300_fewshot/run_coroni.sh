#!/usr/bin/env bash
# Run on Liono AFTER predict. Never run while another Coroni Vitis job is active.
set -Eeuo pipefail
SHOTS="${SHOTS:-8}"
SEED="${SEED:-123}"
ROOT="${ROOT:-/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final}"
D="${D:-$ROOT/deployments/film-gmt-stage2-6ee1390-20261113}"
OUT="${OUT:-$ROOT/artifacts/u50_fewshot_20261114}"
PKG="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
CORONI="${CORONI:-bonus24@coroni.microlab.ntua.gr}"
PRED="${PRED:-$OUT/predictions_shots${SHOTS}_seed${SEED}}"
TARGET="${HLS_ROOT:-$OUT/hls_shots${SHOTS}_seed${SEED}}"
REMOTE_TAG="${REMOTE_TAG:-full}"
[[ "$REMOTE_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Bad REMOTE_TAG" >&2; exit 2; }

# Safety: a coroni run needs explicit, manual choice to start HLS.
: "${CONFIRM_SYNTHESIS:?Export CONFIRM_SYNTHESIS=YES after inspecting prepared jobs}"
[[ "$CONFIRM_SYNTHESIS" == YES ]] || { echo 'Need CONFIRM_SYNTHESIS=YES' >&2; exit 2; }
for arm in zero_shot few_shot;do
  echo "[$arm] Preparing audited U50 HLS bundle"
  dir="$TARGET/$arm"
  mkdir -p "$dir"
  if [[ ! -f "$dir/bundle/manifest.json" ]];then
    "$PY" "$PKG/coroni_batch.py" prepare \
      --predictions "$PRED/$arm.jsonl" \
      --sources-root "$D/eval_sources_compat9" \
      --device u50 --clock-ns 10 --budget-percent 100 \
      --out "$dir/bundle"
  fi
  remote="salt_u50_fewshot_20261114_${REMOTE_TAG}_shots${SHOTS}_seed${SEED}_${arm}"
  ssh "$CORONI" "mkdir -p '$remote/bundle' '$remote/results'"
  rsync -a "$dir/bundle/" "$CORONI:$remote/bundle/"
  ssh -o ServerAliveInterval=60 "$CORONI" \
      "cd '$remote/bundle' && sha256sum -c SHA256SUMS >/dev/null && python3 coroni_batch.py run --bundle . --results ../results --tool /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls --timeout-seconds ${HLS_TIMEOUT:-3600}"
  mkdir -p "$dir/results"
  rsync -a "$CORONI:$remote/results/" "$dir/results/"
  "$PY" "$PKG/coroni_batch.py" collect --bundle "$dir/bundle" \
      --results "$dir/results" --summary "$dir/summary"
done
echo '[FINISHED] Check both collected summaries and then MODE=compare bash run_liono.sh'
