#!/usr/bin/env bash
set -euo pipefail

# ================================================================================================
# Unseen Serrano Stage-3-only inference:
#   Stage 3 = Stage1 SFT LoRA + Stage3 HARP/xattn/DPO checkpoint
#
# Goal:
#   Produce exactly 3 final reranked predictions:
#     1. serrano-kalman-filter / PARETO_LATENCY_EXTREME
#     2. serrano-kalman-filter / PARETO_KNEE
#     3. serrano-kalman-filter / PARETO_AREA_EXTREME
#
# For each objective:
#   - pick one held-out Serrano case from objective-specific stage3 selected_debug/test_selected.jsonl
#   - generate NUM_SAMPLES candidates
#   - flatten unique candidates
#   - rerank with frozen GNN surrogate
#   - keep exactly 1 final reranked prediction
# ================================================================================================

GPU="${GPU:-0}"

ROOT="${ROOT:-/home/ubuntu/runs/unseen_serrano_n16_fixed}"
OUT="${OUT:-${ROOT}/serrano_n16_reranked_inference}"

SCRIPT="${SCRIPT:-/home/ubuntu/src/eval_stage1_stage2_stage3.py}"
GNN_RERANK_SCRIPT="${GNN_RERANK_SCRIPT:-/home/ubuntu/src/rerank_candidates_with_gnn.py}"

MODEL="${MODEL:-deepseek-ai/deepseek-coder-7b-base}"
MEMORY_DIR="${MEMORY_DIR:-/home/ubuntu/save/harp/memory_tokens}"
BANK_DATASET="${BANK_DATASET:-/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl}"

GNN_CKPT="${GNN_CKPT:-/home/ubuntu/logs/all_kernels_GNN_train/run1/val_model_state_dict_snapshot.pth}"

GEXF_DIR="${GEXF_DIR:-/home/ubuntu/harp/processed/extended-pseudo-block-connected-hierarchy}"
KERNEL_INFO_ROOT="${KERNEL_INFO_ROOT:-/home/ubuntu/harp}"

# Important: avoid RHS candidate-bank leakage from the held-out Serrano family.
EXCLUDE_FAMILIES="${EXCLUDE_FAMILIES:-serrano_kalman_filter}"

NUM_SAMPLES="${NUM_SAMPLES:-10}"
SAMPLE_SEED="${SAMPLE_SEED:-123}"

mkdir -p \
  "${OUT}/cases" \
  "${OUT}/verbose" \
  "${OUT}/candidates" \
  "${OUT}/reranked" \
  "${OUT}/logs"

SUMMARY_LOG="${OUT}/logs/_summary.log"
: > "${SUMMARY_LOG}"

echo "===================================================================================================="
echo "[PREFLIGHT] Unseen Serrano Stage-3-only reranked inference"
echo "[GPU]              ${GPU}"
echo "[ROOT]             ${ROOT}"
echo "[OUT]              ${OUT}"
echo "[SCRIPT]           ${SCRIPT}"
echo "[GNN_RERANK]       ${GNN_RERANK_SCRIPT}"
echo "[MODEL]            ${MODEL}"
echo "[MEMORY_DIR]       ${MEMORY_DIR}"
echo "[BANK_DATASET]     ${BANK_DATASET}"
echo "[EXCLUDE_FAMILIES] ${EXCLUDE_FAMILIES}"
echo "[GNN_CKPT]         ${GNN_CKPT}"
echo "[GEXF_DIR]         ${GEXF_DIR}"
echo "[KERNEL_INFO_ROOT] ${KERNEL_INFO_ROOT}"
echo "[NUM_SAMPLES]      ${NUM_SAMPLES}"
echo "===================================================================================================="

test -f "${SCRIPT}" || { echo "[ERROR] Missing eval script: ${SCRIPT}"; exit 1; }
test -f "${GNN_RERANK_SCRIPT}" || { echo "[ERROR] Missing reranker script: ${GNN_RERANK_SCRIPT}"; exit 1; }
test -f "${BANK_DATASET}" || { echo "[ERROR] Missing bank dataset: ${BANK_DATASET}"; exit 1; }
test -d "${MEMORY_DIR}" || { echo "[ERROR] Missing memory dir: ${MEMORY_DIR}"; exit 1; }
test -f "${GNN_CKPT}" || { echo "[ERROR] Missing GNN checkpoint: ${GNN_CKPT}"; exit 1; }
test -d "${GEXF_DIR}" || { echo "[ERROR] Missing GEXF dir: ${GEXF_DIR}"; exit 1; }
test -d "${KERNEL_INFO_ROOT}" || { echo "[ERROR] Missing kernel info root: ${KERNEL_INFO_ROOT}"; exit 1; }

if ! compgen -G "${MEMORY_DIR}/*.memory.pt" > /dev/null; then
  echo "[ERROR] No .memory.pt files found in ${MEMORY_DIR}"
  exit 1
fi

GOALS=(
  "PARETO_LATENCY_EXTREME:latency_extreme"
  "PARETO_KNEE:pareto_knee"
  "PARETO_AREA_EXTREME:area_extreme"
)

make_serrano_case() {
  local TAG="$1"
  local OBJECTIVE="$2"
  local OUT_JSONL="$3"

  local TEST_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/test_selected.jsonl"

  test -f "${TEST_SELECTED}" || {
    echo "[ERROR] Missing Stage3 selected test file for ${TAG}: ${TEST_SELECTED}"
    exit 1
  }

  python - "${TEST_SELECTED}" "${OBJECTIVE}" "${OUT_JSONL}" <<'PY'
import json, re, sys

inp, objective, out_path = sys.argv[1:4]

def norm(s):
    return re.sub(r"[-\s]+", "_", str(s).strip().lower())

def is_serrano(row):
    fam = norm(row.get("_family", ""))
    k = norm(row.get("kernel_name", ""))
    return fam == "serrano_kalman_filter" or "serrano" in k or "kalman" in k

best = None
best_key = None

with open(inp, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)

        if not is_serrano(row):
            continue

        key = (
            int(row.get("_rank_within_kernel", 10**9)),
            float(row.get("_score", 1e9)),
        )

        if best is None or key < best_key:
            best = row
            best_key = key

if best is None:
    raise SystemExit(f"[ERROR] No Serrano/Kalman row found in {inp}")

source_text = best.get("input") or best.get("code") or best.get("source_text")
if not source_text:
    raise SystemExit("[ERROR] selected Serrano row has no input/code/source_text")

if objective == "PARETO_LATENCY_EXTREME":
    w_lat, w_area = 1.0, 0.0
elif objective == "PARETO_AREA_EXTREME":
    w_lat, w_area = 0.0, 1.0
elif objective == "PARETO_KNEE":
    w_lat, w_area = 0.5, 0.5
else:
    raise SystemExit(f"[ERROR] Unknown objective: {objective}")

case = {
    "kernel_name": best["kernel_name"],
    "code": source_text,
    "obj_mode": objective,
    "objective": objective,
    "w_lat": w_lat,
    "w_area": w_area,

    "source_selected_debug": inp,
    "source_rank_within_kernel": best.get("_rank_within_kernel"),
    "source_score": best.get("_score"),
    "source_family": best.get("_family"),
}

with open(out_path, "w", encoding="utf-8") as g:
    g.write(json.dumps(case, ensure_ascii=False) + "\n")

print(f"[CASE] wrote -> {out_path}")
print(
    f"[CASE] kernel={case['kernel_name']} "
    f"objective={objective} "
    f"rank={case['source_rank_within_kernel']} "
    f"score={case['source_score']}"
)
PY
}

flatten_unique_candidates_for_gnn() {
  local IN_JSONL="$1"
  local OUT_JSONL="$2"

  python - "${IN_JSONL}" "${OUT_JSONL}" <<'PY'
import json, sys

inp, out = sys.argv[1:3]

n_in = 0
n_out = 0

with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        n_in += 1

        seen = set()

        for cand in row.get("candidates", []):
            if not cand.get("is_unique", True):
                continue

            pred = cand.get("canonical_prediction") or cand.get("prediction")
            if not pred:
                continue

            if pred in seen:
                continue

            seen.add(pred)

            out_row = {
                "kernel_name": row["kernel_name"],
                "obj_mode": row["obj_mode"],
                "w_lat": row.get("w_lat"),
                "w_area": row.get("w_area"),
                "sample_id": cand.get("sample_id", -1),
                "prediction": pred,
                "sequence_score": cand.get("sequence_score", 0.0),
                "sequence_sum_logprob": cand.get("sequence_sum_logprob", 0.0),
                "ablation_stage": "stage3",
            }

            g.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_out += 1

print(f"[FLATTEN] input rows={n_in}, candidate rows={n_out} -> {out}")

if n_in != 1:
    raise SystemExit(f"[ERROR] Expected exactly 1 eval row, got {n_in}")

if n_out == 0:
    raise SystemExit("[ERROR] Flattening produced zero candidate rows.")
PY
}

tag_reranked_output() {
  local IN_JSONL="$1"
  local OUT_JSONL="$2"
  local TAG="$3"

  python - "${IN_JSONL}" "${OUT_JSONL}" "${TAG}" <<'PY'
import json, sys

inp, out, tag = sys.argv[1:4]

n = 0

with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        row["ablation_stage"] = "stage3"
        row["objective_tag"] = tag

        g.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1

print(f"[TAG] wrote {n} row(s) -> {out}")

if n != 1:
    raise SystemExit(f"[ERROR] Expected exactly 1 reranked row for objective={tag}, got {n}")
PY
}

run_one_objective() {
  local OBJECTIVE="$1"
  local TAG="$2"

  local STAGE1_ADAPTER="${ROOT}/${TAG}_stage1/best_custom_stage1"
  local HARP_XATTN="${ROOT}/${TAG}_stage3/best_custom_stage3/harp_xattn.pt"

  if [[ ! -f "${HARP_XATTN}" ]]; then
    echo "[WARN] Missing best_custom_stage3 HARP for ${TAG}: ${HARP_XATTN}"
    echo "[WARN] Falling back to final Stage3 HARP checkpoint."
    HARP_XATTN="${ROOT}/${TAG}_stage3/harp_xattn.pt"
  fi

  test -d "${STAGE1_ADAPTER}" || {
    echo "[ERROR] Missing Stage1 adapter for ${TAG}: ${STAGE1_ADAPTER}"
    exit 1
  }

  test -f "${HARP_XATTN}" || {
    echo "[ERROR] Missing Stage3 HARP checkpoint for ${TAG}: ${HARP_XATTN}"
    exit 1
  }

  local CASES_JSONL="${OUT}/cases/${TAG}_serrano_case.jsonl"
  local PREDS_JSONL="${OUT}/verbose/stage3_${TAG}_serrano_verbose.jsonl"
  local CANDS_JSONL="${OUT}/candidates/stage3_${TAG}_serrano_candidates_for_gnn.jsonl"
  local RERANK_RAW_JSONL="${OUT}/reranked/stage3_${TAG}_serrano_reranked.raw.jsonl"
  local RERANK_TAGGED_JSONL="${OUT}/reranked/stage3_${TAG}_serrano_reranked.jsonl"
  local LOG_FILE="${OUT}/logs/stage3_${TAG}.log"

  make_serrano_case "${TAG}" "${OBJECTIVE}" "${CASES_JSONL}"

  echo
  echo "===================================================================================================="
  echo "[RUN] Stage3 Serrano objective=${OBJECTIVE} tag=${TAG}"
  echo "[ADAPTER] ${STAGE1_ADAPTER}"
  echo "[HARP]    ${HARP_XATTN}"
  echo "[CASES]   ${CASES_JSONL}"
  echo "[PREDS]   ${PREDS_JSONL}"
  echo "[LOG]     ${LOG_FILE}"
  echo "===================================================================================================="

  CUDA_VISIBLE_DEVICES="${GPU}" python -u "${SCRIPT}" \
    --stage stage3 \
    --model "${MODEL}" \
    --adapter_dir "${STAGE1_ADAPTER}" \
    --harp_xattn_path "${HARP_XATTN}" \
    --input_jsonl "${CASES_JSONL}" \
    --candidate_bank_dataset "${BANK_DATASET}" \
    --candidate_bank_exclude_families "${EXCLUDE_FAMILIES}" \
    --memory_dir "${MEMORY_DIR}" \
    --mem_dim -1 \
    --require_pragma_free_memory \
    --max_slots 64 \
    --every_n_layers 16 \
    --xattn_heads 4 \
    --xattn_dim_head 64 \
    --xattn_ff_mult 1 \
    --max_prompt_tokens 4096 \
    --score_reduction mean \
    --candidate_max_prefix_tokens 1536 \
    --candidate_keep_head_tokens 256 \
    --num_samples "${NUM_SAMPLES}" \
    --sample_temperature 0.8 \
    --sample_top_p 0.95 \
    --sample_top_k 8 \
    --sample_seed "${SAMPLE_SEED}" \
    --print_xattn_debug \
    --output_jsonl "${PREDS_JSONL}" \
    2>&1 | tee "${LOG_FILE}"

  flatten_unique_candidates_for_gnn "${PREDS_JSONL}" "${CANDS_JSONL}"

  CUDA_VISIBLE_DEVICES="${GPU}" python -u "${GNN_RERANK_SCRIPT}" \
    --candidates_jsonl "${CANDS_JSONL}" \
    --gnn_ckpt "${GNN_CKPT}" \
    --gexf_dir "${GEXF_DIR}" \
    --kernel_info_root "${KERNEL_INFO_ROOT}" \
    --objective_tag "${TAG}" \
    --output_jsonl "${RERANK_RAW_JSONL}" \
    --verbose \
    2>&1 | tee -a "${LOG_FILE}"

  tag_reranked_output "${RERANK_RAW_JSONL}" "${RERANK_TAGGED_JSONL}" "${TAG}"

  echo "[OK] Stage3 Serrano objective=${OBJECTIVE}" | tee -a "${SUMMARY_LOG}"
}

for item in "${GOALS[@]}"; do
  OBJECTIVE="${item%%:*}"
  TAG="${item##*:}"

  run_one_objective "${OBJECTIVE}" "${TAG}"
done

cat "${OUT}/verbose/"stage3_*_serrano_verbose.jsonl \
  > "${OUT}/serrano_stage3_all_objectives_verbose.jsonl"

cat "${OUT}/candidates/"stage3_*_serrano_candidates_for_gnn.jsonl \
  > "${OUT}/serrano_stage3_all_objectives_candidates_for_gnn.jsonl"

cat "${OUT}/reranked/"stage3_*_serrano_reranked.jsonl \
  > "${OUT}/serrano_stage3_all_objectives_reranked.jsonl"

echo
echo "===================================================================================================="
echo "[DONE] Unseen Serrano Stage-3-only reranked inference complete"
echo "[OUT] ${OUT}"
echo
echo "[COUNTS]"
wc -l "${OUT}/cases/"*_serrano_case.jsonl
wc -l "${OUT}/verbose/"stage3_*_serrano_verbose.jsonl
wc -l "${OUT}/candidates/"stage3_*_serrano_candidates_for_gnn.jsonl
wc -l "${OUT}/reranked/"stage3_*_serrano_reranked.jsonl
wc -l "${OUT}/serrano_stage3_all_objectives_verbose.jsonl"
wc -l "${OUT}/serrano_stage3_all_objectives_candidates_for_gnn.jsonl"
wc -l "${OUT}/serrano_stage3_all_objectives_reranked.jsonl"
echo "===================================================================================================="

echo
echo "[SUMMARY] Final Stage3 Serrano reranked predictions:"
python - "${OUT}/serrano_stage3_all_objectives_reranked.jsonl" <<'PY'
import json, sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

order = {
    "latency_extreme": 0,
    "pareto_knee": 1,
    "area_extreme": 2,
}

rows = sorted(rows, key=lambda r: order.get(r.get("objective_tag", ""), 99))

for r in rows:
    print(
        f"{r.get('objective_tag'):16s} "
        f"{r.get('kernel_name'):24s} "
        f"sample_id={str(r.get('sample_id')):>3s} "
        f"perf={float(r.get('gnn_pred_perf', float('nan'))):12.6f} "
        f"area={float(r.get('gnn_pred_area', float('nan'))):12.6f} "
        f"sig={r.get('aligned_design_sig')} "
        f"repairs={len(r.get('repair_notes', []))}"
    )

if len(rows) != 3:
    raise SystemExit(f"[ERROR] Expected exactly 3 final reranked rows, got {len(rows)}")

print("[OK] Final output has exactly 3 Stage3 reranked rows.")
PY

echo "===================================================================================================="
