#!/usr/bin/env bash
set -euo pipefail

# ================================================================================================
# Random-design-split Stage-3 inference for external unseen GAN application.
#
# Goal:
#   Produce exactly 3 final predictions:
#     - GAN / PARETO_LATENCY_EXTREME
#     - GAN / PARETO_KNEE
#     - GAN / PARETO_AREA_EXTREME
#
# Assumes you already ran:
#   PYTHONHASHSEED=0 python src/code_to_memory_pt.py \
#     --app-dir GAN \
#     --gnn-ckpt /home/ubuntu/logs/all_kernels_GNN_train/run1/val_model_state_dict_snapshot.pth \
#     --kernel-name GAN \
#     --overwrite \
#     --keep-stage
#
# ================================================================================================

GPU="${GPU:-0}"

RUN_ROOT="${RUN_ROOT:-/home/ubuntu/runs/random_design_split_fixed}"
OUT="${OUT:-${RUN_ROOT}/GAN_stage3_inference}"

SCRIPT="${SCRIPT:-/home/ubuntu/src/eval_stage1_stage2_stage3.py}"
GNN_RERANK_SCRIPT="${GNN_RERANK_SCRIPT:-/home/ubuntu/src/rerank_candidates_with_gnn.py}"

MODEL="${MODEL:-deepseek-ai/deepseek-coder-7b-base}"
DATASET="${DATASET:-/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl}"

# GAN application directory used by code_to_memory_pt.py.
APP_DIR="${APP_DIR:-/home/ubuntu/GAN}"
KERNEL_NAME="${KERNEL_NAME:-GAN}"

# Optional manual overrides.
GAN_CODE_FILE="${GAN_CODE_FILE:-}"
GAN_MEMORY_PT="${GAN_MEMORY_PT:-}"

# Frozen GNN reranker checkpoint.
GNN_CKPT="${GNN_CKPT:-/home/ubuntu/logs/all_kernels_GNN_train/run1/val_model_state_dict_snapshot.pth}"

# For external apps, these should usually point to the GAN staging/output directory produced by code_to_memory_pt.py.
GEXF_DIR="${GEXF_DIR:-${APP_DIR}}"
KERNEL_INFO_ROOT="${KERNEL_INFO_ROOT:-${APP_DIR}}"

# Candidate count before GNN reranking. Final output is still 1 per objective.
NUM_SAMPLES="${NUM_SAMPLES:-32}"
SAMPLE_SEED="${SAMPLE_SEED:-123}"

mkdir -p \
  "${OUT}/cases" \
  "${OUT}/verbose" \
  "${OUT}/candidates" \
  "${OUT}/reranked" \
  "${OUT}/logs" \
  "${OUT}/memory"

SUMMARY_LOG="${OUT}/logs/_summary.log"
: > "${SUMMARY_LOG}"

echo "===================================================================================================="
echo "[PREFLIGHT] Random-design-split Stage-3 GAN inference"
echo "[GPU]              ${GPU}"
echo "[RUN_ROOT]         ${RUN_ROOT}"
echo "[OUT]              ${OUT}"
echo "[SCRIPT]           ${SCRIPT}"
echo "[GNN_RERANK]       ${GNN_RERANK_SCRIPT}"
echo "[MODEL]            ${MODEL}"
echo "[DATASET]          ${DATASET}"
echo "[APP_DIR]          ${APP_DIR}"
echo "[KERNEL_NAME]      ${KERNEL_NAME}"
echo "[GNN_CKPT]         ${GNN_CKPT}"
echo "[GEXF_DIR]         ${GEXF_DIR}"
echo "[KERNEL_INFO_ROOT] ${KERNEL_INFO_ROOT}"
echo "[NUM_SAMPLES]      ${NUM_SAMPLES}"
echo "===================================================================================================="

test -f "${SCRIPT}" || { echo "[ERROR] Missing eval script: ${SCRIPT}"; exit 1; }
test -f "${GNN_RERANK_SCRIPT}" || { echo "[ERROR] Missing reranker script: ${GNN_RERANK_SCRIPT}"; exit 1; }
test -f "${DATASET}" || { echo "[ERROR] Missing dataset: ${DATASET}"; exit 1; }
test -d "${APP_DIR}" || { echo "[ERROR] Missing GAN app dir: ${APP_DIR}"; exit 1; }
test -f "${GNN_CKPT}" || { echo "[ERROR] Missing GNN checkpoint: ${GNN_CKPT}"; exit 1; }
test -d "${GEXF_DIR}" || { echo "[ERROR] Missing GEXF dir: ${GEXF_DIR}"; exit 1; }
test -d "${KERNEL_INFO_ROOT}" || { echo "[ERROR] Missing kernel info root: ${KERNEL_INFO_ROOT}"; exit 1; }

# --------------------------------------------------------------------------------------------------
# Locate the GAN.memory.pt produced by code_to_memory_pt.py.
# Copy it into OUT/memory/GAN.memory.pt so eval_stage1_stage2_stage3.py uses exactly this memory.
# --------------------------------------------------------------------------------------------------
python - "${APP_DIR}" "${OUT}/memory" "${KERNEL_NAME}" "${GAN_MEMORY_PT}" <<'PY'
import os, re, glob, shutil, sys

app_dir, out_mem_dir, kernel_name, manual = sys.argv[1:5]

def norm(s):
    return re.sub(r"[-\s]+", "_", str(s).strip().lower())

target_norm = norm(kernel_name)

if manual:
    if not os.path.isfile(manual):
        raise SystemExit(f"[ERROR] GAN_MEMORY_PT was set but does not exist: {manual}")
    chosen = manual
else:
    search_dirs = [
        app_dir,
        "/home/ubuntu/save/harp/memory_tokens",
    ]

    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            candidates.extend(glob.glob(os.path.join(d, "**", "*.memory.pt"), recursive=True))

    if not candidates:
        raise SystemExit("[ERROR] Could not find any *.memory.pt candidates.")

    scored = []
    for p in candidates:
        base = os.path.basename(p).replace(".memory.pt", "")
        nbase = norm(base)

        score = 0
        if nbase == target_norm:
            score += 1000
        if target_norm in nbase:
            score += 200
        if os.path.abspath(p).startswith(os.path.abspath(app_dir)):
            score += 100
        score += int(os.path.getmtime(p)) / 1e12

        scored.append((score, p))

    scored.sort(reverse=True)
    chosen = scored[0][1]

os.makedirs(out_mem_dir, exist_ok=True)
dst = os.path.join(out_mem_dir, f"{kernel_name}.memory.pt")
shutil.copy2(chosen, dst)

print(f"[MEMORY] selected: {chosen}")
print(f"[MEMORY] copied -> {dst}")
PY

MEMORY_DIR="${OUT}/memory"

# --------------------------------------------------------------------------------------------------
# Locate staged GAN source containing auto{..._Lk} placeholders.
# If GAN_CODE_FILE is provided, use it. Otherwise scan APP_DIR.
# The script also normalizes <SRC_Lk> markers back to /* Lk: */ because eval_stage1_stage2_stage3.py
# expects source labels before it converts them into <SRC_Lk> prompt tokens.
# --------------------------------------------------------------------------------------------------
make_gan_case() {
  local TAG="$1"
  local OBJECTIVE="$2"
  local OUT_JSONL="$3"

  python - "${APP_DIR}" "${GAN_CODE_FILE}" "${KERNEL_NAME}" "${OBJECTIVE}" "${OUT_JSONL}" <<'PY'
import os, re, json, glob, sys

app_dir, manual_code_file, kernel_name, objective, out_path = sys.argv[1:6]

SOURCE_LABEL_RE = re.compile(r"^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)", re.I)
AUTO_RE = re.compile(r"auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}", re.I)
SRC_TOKEN_RE = re.compile(r"<SRC_L(\d+)>", re.I)

def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def score_source(path, text):
    auto_count = len(AUTO_RE.findall(text))
    label_count = sum(1 for line in text.splitlines() if SOURCE_LABEL_RE.match(line.lstrip()))
    src_token_count = len(SRC_TOKEN_RE.findall(text))

    name = os.path.basename(path).lower()
    bonus = 0
    for key in ["staged", "stage", "placeholder", "placeholders", "labeled", "label"]:
        if key in name:
            bonus += 10

    return auto_count * 100 + label_count * 20 + src_token_count * 20 + bonus

if manual_code_file:
    if not os.path.isfile(manual_code_file):
        raise SystemExit(f"[ERROR] GAN_CODE_FILE was set but does not exist: {manual_code_file}")
    chosen = manual_code_file
    src = read_text(chosen)
else:
    exts = ["*.cpp", "*.cc", "*.c", "*.hpp", "*.h", "*.txt"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(app_dir, "**", ext), recursive=True))

    scored = []
    for p in paths:
        try:
            text = read_text(p)
        except Exception:
            continue
        if "auto{" not in text:
            continue
        s = score_source(p, text)
        if s > 0:
            scored.append((s, p, text))

    if not scored:
        raise SystemExit(
            "[ERROR] Could not find a staged GAN source containing auto{..._Lk} placeholders. "
            "Set GAN_CODE_FILE=/path/to/staged_source.cpp and rerun."
        )

    scored.sort(reverse=True, key=lambda x: x[0])
    _, chosen, src = scored[0]

# Normalize <SRC_Lk> to /* Lk: */ for eval script compatibility.
src = SRC_TOKEN_RE.sub(lambda m: f"/* L{m.group(1)}: */", src)

# Ensure every line containing auto{..._Lk} has a parseable source label.
fixed_lines = []
for line in src.splitlines():
    stripped = line.lstrip()
    if SOURCE_LABEL_RE.match(stripped):
        fixed_lines.append(line)
        continue

    m = AUTO_RE.search(line)
    if m:
        indent = line[:len(line) - len(stripped)]
        label = m.group(1).upper()
        fixed_lines.append(f"{indent}/* {label}: */ {stripped}")
    else:
        fixed_lines.append(line)

src = "\n".join(fixed_lines)

if objective == "PARETO_LATENCY_EXTREME":
    w_lat, w_area = 1.0, 0.0
elif objective == "PARETO_AREA_EXTREME":
    w_lat, w_area = 0.0, 1.0
elif objective == "PARETO_KNEE":
    w_lat, w_area = 0.5, 0.5
else:
    raise SystemExit(f"[ERROR] Unknown objective: {objective}")

auto_count = len(AUTO_RE.findall(src))
label_count = sum(1 for line in src.splitlines() if SOURCE_LABEL_RE.match(line.lstrip()))

if auto_count == 0:
    raise SystemExit("[ERROR] Selected GAN source has zero auto{..._Lk} placeholders after normalization.")

if label_count == 0:
    raise SystemExit("[ERROR] Selected GAN source has zero Lk labels after normalization.")

case = {
    "kernel_name": kernel_name,
    "code": src,
    "obj_mode": objective,
    "objective": objective,
    "w_lat": w_lat,
    "w_area": w_area,
    "source_code_file": chosen,
    "n_auto_placeholders": auto_count,
    "n_source_labels": label_count,
}

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as g:
    g.write(json.dumps(case, ensure_ascii=False) + "\n")

print(f"[CASE] wrote -> {out_path}")
print(f"[CASE] source={chosen}")
print(f"[CASE] kernel={kernel_name} objective={objective} auto_placeholders={auto_count} source_labels={label_count}")
PY
}

# --------------------------------------------------------------------------------------------------
# Flatten eval output into reranker input format.
# --------------------------------------------------------------------------------------------------
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
    raise SystemExit(f"[ERROR] Expected 1 eval row, got {n_in}")

if n_out == 0:
    raise SystemExit("[ERROR] Flattening produced zero candidate rows.")
PY
}

# --------------------------------------------------------------------------------------------------
# Tag reranked row and enforce exactly one selected design per objective.
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

print(f"[TAG] wrote {n} row(s) -> {out}")

if n != 1:
    raise SystemExit(f"[ERROR] Expected exactly 1 reranked row for objective={tag}, got {n}")
PY
}

run_one_objective() {
  local OBJECTIVE="$1"
  local TAG="$2"

  local STAGE1_ADAPTER="${RUN_ROOT}/${TAG}_stage1/best_custom_stage1"
  local HARP_XATTN="${RUN_ROOT}/${TAG}_stage3/best_custom_stage3/harp_xattn.pt"

  if [[ ! -f "${HARP_XATTN}" ]]; then
    echo "[WARN] Missing best_custom_stage3 HARP for ${TAG}: ${HARP_XATTN}"
    echo "[WARN] Falling back to final Stage3 HARP checkpoint."
    HARP_XATTN="${RUN_ROOT}/${TAG}_stage3/harp_xattn.pt"
  fi

  test -d "${STAGE1_ADAPTER}" || {
    echo "[ERROR] Missing Stage1 adapter for ${TAG}: ${STAGE1_ADAPTER}"
    exit 1
  }

  test -f "${HARP_XATTN}" || {
    echo "[ERROR] Missing Stage3 HARP checkpoint for ${TAG}: ${HARP_XATTN}"
    exit 1
  }

  local CASES_JSONL="${OUT}/cases/${TAG}_GAN_case.jsonl"
  local PREDS_JSONL="${OUT}/verbose/stage3_${TAG}_GAN_verbose.jsonl"
  local CANDS_JSONL="${OUT}/candidates/stage3_${TAG}_GAN_candidates_for_gnn.jsonl"
  local RERANK_RAW_JSONL="${OUT}/reranked/stage3_${TAG}_GAN_reranked.raw.jsonl"
  local RERANK_TAGGED_JSONL="${OUT}/reranked/stage3_${TAG}_GAN_reranked.jsonl"
  local LOG_FILE="${OUT}/logs/stage3_${TAG}.log"

  make_gan_case "${TAG}" "${OBJECTIVE}" "${CASES_JSONL}"

  echo
  echo "===================================================================================================="
  echo "[RUN] Stage3 GAN objective=${OBJECTIVE} tag=${TAG}"
  echo "[ADAPTER] ${STAGE1_ADAPTER}"
  echo "[HARP]    ${HARP_XATTN}"
  echo "[MEMORY]  ${MEMORY_DIR}/${KERNEL_NAME}.memory.pt"
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
    --candidate_bank_dataset "${DATASET}" \
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

  echo "[OK] Stage3 GAN objective=${OBJECTIVE}" | tee -a "${SUMMARY_LOG}"
}

GOALS=(
  "PARETO_LATENCY_EXTREME:latency_extreme"
  "PARETO_KNEE:pareto_knee"
  "PARETO_AREA_EXTREME:area_extreme"
)

for item in "${GOALS[@]}"; do
  OBJECTIVE="${item%%:*}"
  TAG="${item##*:}"

  run_one_objective "${OBJECTIVE}" "${TAG}"
done

cat "${OUT}/verbose/"stage3_*_GAN_verbose.jsonl \
  > "${OUT}/GAN_stage3_all_objectives_verbose.jsonl"

cat "${OUT}/candidates/"stage3_*_GAN_candidates_for_gnn.jsonl \
  > "${OUT}/GAN_stage3_all_objectives_candidates_for_gnn.jsonl"

cat "${OUT}/reranked/"stage3_*_GAN_reranked.jsonl \
  > "${OUT}/GAN_stage3_all_objectives_reranked.jsonl"

echo
echo "===================================================================================================="
echo "[DONE] Random-design-split Stage-3 GAN inference complete"
echo "[OUT] ${OUT}"
echo
echo "[COUNTS]"
wc -l "${OUT}/cases/"*_GAN_case.jsonl
wc -l "${OUT}/verbose/"stage3_*_GAN_verbose.jsonl
wc -l "${OUT}/candidates/"stage3_*_GAN_candidates_for_gnn.jsonl
wc -l "${OUT}/reranked/"stage3_*_GAN_reranked.jsonl
wc -l "${OUT}/GAN_stage3_all_objectives_verbose.jsonl"
wc -l "${OUT}/GAN_stage3_all_objectives_candidates_for_gnn.jsonl"
wc -l "${OUT}/GAN_stage3_all_objectives_reranked.jsonl"
echo "===================================================================================================="

echo
echo "[SUMMARY] Final GAN reranked predictions:"
python - "${OUT}/GAN_stage3_all_objectives_reranked.jsonl" <<'PY'
import json, sys

p = sys.argv[1]

with open(p, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

rows = sorted(rows, key=lambda r: r.get("objective_tag", ""))

for r in rows:
    print(
        f"{r.get('objective_tag'):16s} "
        f"{r.get('kernel_name'):12s} "
        f"sample_id={str(r.get('sample_id')):>2s} "
        f"perf={float(r.get('gnn_pred_perf', float('nan'))):10.4f} "
        f"area={float(r.get('gnn_pred_area', float('nan'))):10.4f} "
        f"sig={r.get('aligned_design_sig')} "
        f"repairs={len(r.get('repair_notes', []))}"
    )

if len(rows) != 3:
    raise SystemExit(f"[ERROR] Expected exactly 3 final reranked rows, got {len(rows)}")

print("[OK] Final output has exactly 3 rows.")
PY

echo "===================================================================================================="
