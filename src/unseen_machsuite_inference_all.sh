#!/usr/bin/env bash
set -euo pipefail

# ================================================================================================
# Unseen MachSuite Stage-3 inference:
#
#   Trained setting:
#     - Train/val excluded all MachSuite families.
#     - Stage 1: objective-specific SFT LoRA
#     - Stage 2: objective-specific SFT + HARP/xattn
#     - Stage 3: objective-specific DPO over HARP/xattn/gates, LoRA frozen
#
#   Inference setting:
#     - Use Stage1 LoRA adapter
#     - Use Stage3 HARP/xattn checkpoint
#     - Use pragma-free structural memory
#     - Generate candidates for all 8 unseen MachSuite kernels
#     - Rerank with frozen GNN surrogate
#
# Expected final:
#   - 24 verbose rows:  3 objectives × 8 kernels
#   - 24 reranked rows: 3 objectives × 8 kernels
# ================================================================================================

GPU="${GPU:-0}"

ROOT="${ROOT:-/home/ubuntu/runs/unseen_machsuite_fixed}"
OUT="${OUT:-${ROOT}/machsuite_stage3_inference_all}"

SCRIPT="${SCRIPT:-/home/ubuntu/src/eval_stage1_stage2_stage3.py}"
GNN_RERANK_SCRIPT="${GNN_RERANK_SCRIPT:-/home/ubuntu/src/rerank_candidates_with_gnn.py}"

MODEL="${MODEL:-deepseek-ai/deepseek-coder-7b-base}"
MEMORY_DIR="${MEMORY_DIR:-/home/ubuntu/save/harp/memory_tokens}"
BANK_DATASET="${BANK_DATASET:-/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl}"

GNN_CKPT="${GNN_CKPT:-/home/ubuntu/logs/all_kernels_GNN_train/run1/val_model_state_dict_snapshot.pth}"

GEXF_DIR="${GEXF_DIR:-/home/ubuntu/harp/processed/extended-pseudo-block-connected-hierarchy}"
KERNEL_INFO_ROOT="${KERNEL_INFO_ROOT:-/home/ubuntu/harp}"

# Critical: avoid RHS-bank leakage from the held-out MachSuite suite.
EXCLUDE_FAMILIES="${EXCLUDE_FAMILIES:-machsuite_gemm;machsuite_md_knn;machsuite_sort_radix;machsuite_spmv_ellpack;machsuite_stencil2d;machsuite_stencil3d;machsuite_viterbi}"

# Use >1 sample because final prediction is selected by GNN reranking.
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

EXPECTED_KERNELS=(
  "machsuite-gemm-blocked"
  "machsuite-gemm-ncubed"
  "machsuite-md-knn"
  "machsuite-sort-radix"
  "machsuite-spmv-ellpack"
  "machsuite-stencil2d"
  "machsuite-stencil3d"
  "machsuite-viterbi"
)

EXPECTED_KERNELS_STR="$(IFS=';'; echo "${EXPECTED_KERNELS[*]}")"

GOALS=(
  "PARETO_LATENCY_EXTREME:latency_extreme"
  "PARETO_KNEE:pareto_knee"
  "PARETO_AREA_EXTREME:area_extreme"
)

echo "===================================================================================================="
echo "[PREFLIGHT] Unseen MachSuite Stage-3 inference"
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
echo "[EXPECTED_KERNELS] ${EXPECTED_KERNELS_STR}"
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

# --------------------------------------------------------------------------------------------------
# Check that memory packs exist for all expected MachSuite kernels.
# This checks file names only. eval_stage1_stage2_stage3.py will later verify shape/dim/pragma-free.
# --------------------------------------------------------------------------------------------------
python - "${MEMORY_DIR}" "${EXPECTED_KERNELS_STR}" <<'PY'
import os, re, glob, sys

memory_dir, expected_arg = sys.argv[1:3]

def norm(s):
    return re.sub(r"[-\s]+", "_", str(s).strip().lower())

expected = [x for x in expected_arg.split(";") if x.strip()]

available = {}
for p in glob.glob(os.path.join(memory_dir, "*.memory.pt")):
    base = os.path.basename(p).replace(".memory.pt", "")
    available[base] = p
    available[norm(base)] = p

missing = []
for k in expected:
    if k not in available and norm(k) not in available:
        missing.append(k)

if missing:
    print("[ERROR] Missing memory packs for:")
    for k in missing:
        print("  ", k)
    raise SystemExit(1)

print("[OK] Memory-pack filename sanity passed for all expected MachSuite kernels.")
PY

# --------------------------------------------------------------------------------------------------
# Build one inference case per MachSuite raw kernel for a given objective.
# Source rows are taken from objective-specific Stage-3 selected_debug/test_selected.jsonl.
# For each raw kernel, we keep the best selected row according to:
#   (_rank_within_kernel, _score, file order)
# --------------------------------------------------------------------------------------------------
make_machsuite_cases() {
  local TAG="$1"
  local OBJECTIVE="$2"
  local OUT_JSONL="$3"

  local TEST_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/test_selected.jsonl"

  test -f "${TEST_SELECTED}" || {
    echo "[ERROR] Missing test_selected for ${TAG}: ${TEST_SELECTED}"
    exit 1
  }

  python - "${TEST_SELECTED}" "${OBJECTIVE}" "${OUT_JSONL}" "${EXPECTED_KERNELS_STR}" <<'PY'
import json, re, sys
from collections import OrderedDict

inp, objective, out_path, expected_arg = sys.argv[1:5]

expected = [x for x in expected_arg.split(";") if x.strip()]

def norm(s):
    return re.sub(r"[-\s]+", "_", str(s).strip().lower())

expected_norm_to_name = OrderedDict((norm(k), k) for k in expected)

def family_id_from_kernel_name(name):
    s = norm(name).replace("-", "_")

    if s.startswith("machsuite_gemm"):
        return "machsuite_gemm"

    if s.startswith("machsuite_"):
        parts = s.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return s

    return s

machsuite_fams = {
    "machsuite_gemm",
    "machsuite_md_knn",
    "machsuite_sort_radix",
    "machsuite_spmv_ellpack",
    "machsuite_stencil2d",
    "machsuite_stencil3d",
    "machsuite_viterbi",
}

if objective == "PARETO_LATENCY_EXTREME":
    w_lat, w_area = 1.0, 0.0
elif objective == "PARETO_AREA_EXTREME":
    w_lat, w_area = 0.0, 1.0
elif objective == "PARETO_KNEE":
    w_lat, w_area = 0.5, 0.5
else:
    raise SystemExit(f"[ERROR] Unknown objective: {objective}")

best_by_kernel = {}
best_key_by_kernel = {}

with open(inp, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue

        row = json.loads(line)
        kernel = row.get("kernel_name", "")
        nk = norm(kernel)

        fam = norm(row.get("_family", "")) or family_id_from_kernel_name(kernel)

        is_expected_kernel = nk in expected_norm_to_name
        is_machsuite_family = fam in machsuite_fams

        if not is_expected_kernel and not is_machsuite_family:
            continue

        if nk not in expected_norm_to_name:
            continue

        source_text = row.get("input") or row.get("code") or row.get("source_text")
        if not source_text:
            raise SystemExit(f"[ERROR] Row for {kernel} has no input/code/source_text")

        key = (
            int(row.get("_rank_within_kernel", 10**9)),
            float(row.get("_score", 1e9)),
            idx,
        )

        if nk not in best_by_kernel or key < best_key_by_kernel[nk]:
            best_by_kernel[nk] = row
            best_key_by_kernel[nk] = key

missing = [raw for nk, raw in expected_norm_to_name.items() if nk not in best_by_kernel]
if missing:
    print(f"[ERROR] Missing expected MachSuite kernels in {inp}:")
    for k in missing:
        print("  ", k)
    raise SystemExit(1)

with open(out_path, "w", encoding="utf-8") as g:
    for nk, raw_name in expected_norm_to_name.items():
        row = best_by_kernel[nk]
        source_text = row.get("input") or row.get("code") or row.get("source_text")

        case = {
            "kernel_name": row["kernel_name"],
            "code": source_text,
            "obj_mode": objective,
            "objective": objective,
            "w_lat": w_lat,
            "w_area": w_area,

            "source_selected_debug": inp,
            "source_rank_within_kernel": row.get("_rank_within_kernel"),
            "source_score": row.get("_score"),
            "source_family": row.get("_family") or family_id_from_kernel_name(row["kernel_name"]),
        }

        g.write(json.dumps(case, ensure_ascii=False) + "\n")

print(f"[CASE] Wrote {len(expected)} MachSuite cases -> {out_path}")
for nk, raw_name in expected_norm_to_name.items():
    row = best_by_kernel[nk]
    print(
        f"[CASE] kernel={row['kernel_name']} "
        f"objective={objective} "
        f"rank={row.get('_rank_within_kernel')} "
        f"score={row.get('_score')} "
        f"family={row.get('_family') or family_id_from_kernel_name(row['kernel_name'])}"
    )
PY
}

# --------------------------------------------------------------------------------------------------
# Flatten eval output into reranker input.
# Reranker groups by kernel/objective, so running once per objective over all 8 kernels is valid.
# --------------------------------------------------------------------------------------------------
flatten_unique_candidates_for_gnn() {
  local IN_JSONL="$1"
  local OUT_JSONL="$2"

  python - "${IN_JSONL}" "${OUT_JSONL}" <<'PY'
import json, sys

inp, out = sys.argv[1:3]
n_rows = 0
n_out = 0

with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        n_rows += 1

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

print(f"[FLATTEN] input rows={n_rows}, candidate rows={n_out} -> {out}")

if n_rows != 8:
    raise SystemExit(f"[ERROR] Expected 8 verbose input rows, got {n_rows}")

if n_out == 0:
    raise SystemExit("[ERROR] Flattening produced zero candidate rows.")
PY
}

# --------------------------------------------------------------------------------------------------
# Add objective/stage metadata to reranked output and enforce 8 final rows per objective.
# --------------------------------------------------------------------------------------------------
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

print(f"[TAG] wrote {n} rows -> {out}")

if n != 8:
    raise SystemExit(f"[ERROR] Expected exactly 8 reranked rows for objective={tag}, got {n}")
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

  local CASES_JSONL="${OUT}/cases/${TAG}_machsuite_cases.jsonl"
  local PREDS_JSONL="${OUT}/verbose/stage3_${TAG}_machsuite_verbose.jsonl"
  local CANDS_JSONL="${OUT}/candidates/stage3_${TAG}_machsuite_candidates_for_gnn.jsonl"
  local RERANK_RAW_JSONL="${OUT}/reranked/stage3_${TAG}_machsuite_reranked.raw.jsonl"
  local RERANK_TAGGED_JSONL="${OUT}/reranked/stage3_${TAG}_machsuite_reranked.jsonl"
  local LOG_FILE="${OUT}/logs/stage3_${TAG}.log"

  make_machsuite_cases "${TAG}" "${OBJECTIVE}" "${CASES_JSONL}"

  echo
  echo "===================================================================================================="
  echo "[RUN] Stage3 objective=${OBJECTIVE} tag=${TAG}"
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
    --every_n_layers 8 \
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

  echo "[OK] Stage3 objective=${OBJECTIVE}" | tee -a "${SUMMARY_LOG}"
}

# --------------------------------------------------------------------------------------------------
# Main loop: 3 objectives × 8 kernels
# --------------------------------------------------------------------------------------------------
for item in "${GOALS[@]}"; do
  OBJECTIVE="${item%%:*}"
  TAG="${item##*:}"

  run_one_objective "${OBJECTIVE}" "${TAG}"
done

# --------------------------------------------------------------------------------------------------
# Concatenate final outputs
# --------------------------------------------------------------------------------------------------
cat "${OUT}/verbose/"stage3_*_machsuite_verbose.jsonl \
  > "${OUT}/machsuite_stage3_all_objectives_verbose.jsonl"

cat "${OUT}/candidates/"stage3_*_machsuite_candidates_for_gnn.jsonl \
  > "${OUT}/machsuite_stage3_all_objectives_candidates_for_gnn.jsonl"

cat "${OUT}/reranked/"stage3_*_machsuite_reranked.jsonl \
  > "${OUT}/machsuite_stage3_all_objectives_reranked.jsonl"

echo
echo "===================================================================================================="
echo "[DONE] Unseen MachSuite Stage-3 inference complete"
echo "[OUT] ${OUT}"
echo
echo "[COUNTS]"
wc -l "${OUT}/cases/"*_machsuite_cases.jsonl
wc -l "${OUT}/verbose/"stage3_*_machsuite_verbose.jsonl
wc -l "${OUT}/candidates/"stage3_*_machsuite_candidates_for_gnn.jsonl
wc -l "${OUT}/reranked/"stage3_*_machsuite_reranked.jsonl
wc -l "${OUT}/machsuite_stage3_all_objectives_verbose.jsonl"
wc -l "${OUT}/machsuite_stage3_all_objectives_candidates_for_gnn.jsonl"
wc -l "${OUT}/machsuite_stage3_all_objectives_reranked.jsonl"
echo "===================================================================================================="

echo
echo "[SUMMARY] Final reranked predictions:"
python - "${OUT}/machsuite_stage3_all_objectives_reranked.jsonl" <<'PY'
import json, sys

p = sys.argv[1]

with open(p, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

rows = sorted(rows, key=lambda r: (r.get("objective_tag", ""), r.get("kernel_name", "")))

for r in rows:
    print(
        f"{r.get('objective_tag'):16s} "
        f"{r.get('kernel_name'):32s} "
        f"sample_id={str(r.get('sample_id')):>2s} "
        f"perf={float(r.get('gnn_pred_perf', float('nan'))):10.4f} "
        f"area={float(r.get('gnn_pred_area', float('nan'))):10.4f} "
        f"sig={r.get('aligned_design_sig')} "
        f"repairs={len(r.get('repair_notes', []))}"
    )

if len(rows) != 24:
    raise SystemExit(f"[ERROR] Expected 24 final reranked rows, got {len(rows)}")

print("[OK] Final output has exactly 24 rows.")
PY

echo "===================================================================================================="
