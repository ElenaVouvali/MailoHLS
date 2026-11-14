#!/usr/bin/env bash
# Usage: MODE=audit|train|predict|compare SHOTS=8 SEED=123 bash run_liono.sh
set -Eeuo pipefail
MODE="${MODE:-audit}"
SHOTS="${SHOTS:-8}"
SEED="${SEED:-123}"
ROOT="${ROOT:-/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final}"
D="${D:-$ROOT/deployments/film-gmt-stage2-6ee1390-20261113}"
PY="${PY:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
GPU_UUID="${GPU_UUID:-GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb}"
PARENT="${PARENT:-/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run}"
FILM_CP="${FILM_CP:-/home/elvouvali/MailoHLS/worktrees/gnosis-stage1-film-ab/artifacts/gnosis_stage1_ab/stage1_ab_pilot_20261112_retry2/B/checkpoint-0600}"
MEM="${MEM:-$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/memory_B}"
EVAL_MEM="${EVAL_MEM:-$D/eval_gmt_memory_B_compat9}"
EVAL_SRC="${EVAL_SRC:-$D/eval_sources_compat9}"
OUT="${OUT:-$ROOT/artifacts/u50_fewshot_20261114}"
ENTRY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/u50_fewshot.py"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
for f in "$ENTRY" "$D/runs/campaign_s123_v1/final/checkpoint-0300/receipt.json" \
         "$ROOT/mailohls_u50_adaptation/artifacts/u50_10ns_sft.jsonl" \
         "$MEM/memory_manifest.json";do test -f "$f" || { echo "Missing input: $f" >&2;exit 2; };done
ARGS=(--repo "$ROOT" --deployment "$D" --memory-dir "$MEM" --parent "$PARENT" --shots "$SHOTS" --seed "$SEED")
RUN="$OUT/shots${SHOTS}_seed${SEED}"
case "$MODE" in
 audit)
   mkdir -p "$OUT"
   "$PY" "$ENTRY" audit "${ARGS[@]}" --out "$OUT/audit_shots${SHOTS}_seed${SEED}_domainqualified.json"
   ;;
 train)
   export CUDA_VISIBLE_DEVICES="$GPU_UUID" PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
   mkdir -p "$OUT"
   flock -n "$ROOT/artifacts/.salt_final_v100s_global.lock" \
     "$PY" -u "$ENTRY" train "${ARGS[@]}" \
       --film-checkpoint "$FILM_CP" \
       --steps "${STEPS:-20}" --eval-interval "${EVAL_INTERVAL:-5}" \
       --lr "${LR:-0.02}" --rank-lambda "${RANK_LAMBDA:-0.1}" \
       --out "$RUN"
   ;;
 predict|predict-gan)
   export CUDA_VISIBLE_DEVICES="$GPU_UUID" PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
   dest="$OUT/predictions_shots${SHOTS}_seed${SEED}"
   if [[ "$MODE" == predict-gan ]];then dest="$OUT/smoke_gan_shots${SHOTS}_seed${SEED}";fi
   rest=();if [[ "$MODE" == predict-gan ]];then rest=(--test-kernels GAN);fi
   flock -n "$ROOT/artifacts/.salt_final_v100s_global.lock" \
     "$PY" -u "$ENTRY" predict "${ARGS[@]}" \
        --film-checkpoint "$FILM_CP" \
        --eval-memory-dir "$EVAL_MEM" --eval-sources-root "$EVAL_SRC" \
        --calibration "$RUN" --out "$dest" "${rest[@]}"
   ;;
 compare)
    "$PY" "$ENTRY" compare --results-root "$OUT/hls_shots${SHOTS}_seed${SEED}" \
        --out "$OUT/comparison_shots${SHOTS}_seed${SEED}.json"
    ;;
 *)echo "Unknown MODE $MODE" >&2;exit 2;;
esac
