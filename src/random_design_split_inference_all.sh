ROOT=/home/ubuntu/runs/random_design_split_fixed
OUT=/home/ubuntu/runs/random_design_split_fixed/final_stage3_test_batch
SCRIPT=/home/ubuntu/src/eval_stage1_stage2_stage3.py
MODEL=deepseek-ai/deepseek-coder-7b-base
MEMORY_DIR=/home/ubuntu/save/harp/memory_tokens
BANK_DATASET=/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl

# your GNN reranker script / checkpoint
GNN_RERANK_SCRIPT=/home/ubuntu/src/rerank_candidates_with_gnn.py
GNN_CKPT=/home/ubuntu/logs/all_kernels_GNN_train/run1/val_model_state_dict_snapshot.pth

mkdir -p "${OUT}/cases" "${OUT}/preds" "${OUT}/rerank"

make_one_case_per_kernel () {
  local IN_JSONL="$1"
  local OUT_JSONL="$2"

  python - "$IN_JSONL" "$OUT_JSONL" <<'PY'
import json, sys

inp, out = sys.argv[1], sys.argv[2]
best = {}

with open(inp, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        k = row["kernel_name"]
        key = (
            int(row.get("_rank_within_kernel", 10**9)),
            float(row.get("_score", 1e9)),
        )
        if k not in best or key < best[k][0]:
            best[k] = (key, row)

rows = [best[k][1] for k in sorted(best.keys())]

with open(out, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows -> {out}")
PY
}

flatten_unique_candidates_for_gnn () {
  local IN_JSONL="$1"
  local OUT_JSONL="$2"

  python - "$IN_JSONL" "$OUT_JSONL" <<'PY'
import json, sys

inp, out = sys.argv[1], sys.argv[2]

with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)

        for cand in row["candidates"]:
            if not cand.get("is_unique", True):
                continue

            out_row = {
                "kernel_name": row["kernel_name"],
                "obj_mode": row["obj_mode"],
                "w_lat": row["w_lat"],
                "w_area": row["w_area"],
                "sample_id": cand["sample_id"],
                "prediction": cand["canonical_prediction"],
                "sequence_score": cand["sequence_score"],
                "sequence_sum_logprob": cand["sequence_sum_logprob"],
            }
            g.write(json.dumps(out_row, ensure_ascii=False) + "\n")

print(f"Wrote unique candidate file -> {out}")
PY
}

run_stage3_batch () {
  local TAG="$1"

  local STAGE3_DIR="${ROOT}/${TAG}_stage3/best_custom_stage3"
  local TEST_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/test_selected.jsonl"

  local CASES_JSONL="${OUT}/cases/${TAG}_one_per_kernel.jsonl"
  local PREDS_JSONL="${OUT}/preds/${TAG}_stage3_predictions_verbose.jsonl"
  local CANDS_FOR_GNN_JSONL="${OUT}/rerank/${TAG}_stage3_candidates_for_gnn.jsonl"
  local FINAL_RERANKED_JSONL="${OUT}/preds/${TAG}_stage3_predictions_reranked.jsonl"

  make_one_case_per_kernel "${TEST_SELECTED}" "${CASES_JSONL}"

  python -u "${SCRIPT}" \
    --stage stage3 \
    --model "${MODEL}" \
    --adapter_dir "${ROOT}/${TAG}_stage1/best_custom_stage1" \
    --harp_xattn_path "${STAGE3_DIR}/harp_xattn.pt" \
    --memory_dir "${MEMORY_DIR}" \
    --input_jsonl "${CASES_JSONL}" \
    --candidate_bank_dataset "${BANK_DATASET}" \
    --max_prompt_tokens 4096 \
    --score_reduction mean \
    --candidate_max_prefix_tokens 1536 \
    --candidate_keep_head_tokens 256 \
    --mem_dim -1 \
    --max_slots 64 \
    --every_n_layers 8 \
    --xattn_heads 4 \
    --xattn_dim_head 64 \
    --xattn_ff_mult 1 \
    --num_samples 3 \
    --sample_temperature 0.7 \
    --sample_top_p 0.9 \
    --sample_seed 123 \
    --output_jsonl "${PREDS_JSONL}"

  flatten_unique_candidates_for_gnn "${PREDS_JSONL}" "${CANDS_FOR_GNN_JSONL}"

  python -u "${GNN_RERANK_SCRIPT}" \
    --candidates_jsonl "${CANDS_FOR_GNN_JSONL}" \
    --gnn_ckpt "${GNN_CKPT}" \
    --objective_tag "${TAG}" \
    --output_jsonl "${FINAL_RERANKED_JSONL}"
}

run_stage3_batch latency_extreme
run_stage3_batch pareto_knee
run_stage3_batch area_extreme

cat \
  "${OUT}/preds/latency_extreme_stage3_predictions_reranked.jsonl" \
  "${OUT}/preds/pareto_knee_stage3_predictions_reranked.jsonl" \
  "${OUT}/preds/area_extreme_stage3_predictions_reranked.jsonl" \
  > "${OUT}/stage3_all_objectives_predictions_162_for_hls.jsonl"

wc -l "${OUT}/cases/"*_one_per_kernel.jsonl
wc -l "${OUT}/stage3_all_objectives_predictions_162_for_hls.jsonl"
