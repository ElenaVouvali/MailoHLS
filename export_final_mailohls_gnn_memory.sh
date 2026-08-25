#!/usr/bin/env bash
set -euo pipefail

# Export the selected final GNN as synthesis-free Stage-2 structural memory.
# Run from the MailoHLS repository root after the final GNN training succeeds.

EXP="${EXP:-Checkpoints/gnn_final_final_reference_delta_s123/run1}"
CKPT="${CKPT:-${EXP}/epoch_10_model_state_dict.pth}"
BASE_OUT="${BASE_OUT:-artifacts/gnn/final_final_reference_delta_epoch10_s123}"
CONTRACT="${EXP}/gnn_checkpoint_contract.json"
SIDECAR="${CKPT}.json"
SCHEMA="GNN_branch/MLIR_dataset/all_kernels/feature_schema.json"
PT_DIR="GNN_branch/MLIR_dataset/all_kernels/graphs"
GEXF_DIR="GNN_branch/MLIR_graphs"
STATIC_OUT="${BASE_OUT}/static_jkn"
LAYERWISE_OUT="${BASE_OUT}/layerwise"
MULTISCALE_OUT="${BASE_OUT}/multiscale_aligned"
STATS_OUT="${BASE_OUT}/multiscale_normalization_stats.json"

for path in "${CKPT}" "${CONTRACT}" "${SIDECAR}" "${SCHEMA}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${PT_DIR}" ]] || { echo "Missing ${PT_DIR}" >&2; exit 2; }
[[ -d "${GEXF_DIR}" ]] || { echo "Missing ${GEXF_DIR}" >&2; exit 2; }

for path in "${STATIC_OUT}" "${LAYERWISE_OUT}" "${MULTISCALE_OUT}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing memory output: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${BASE_OUT}"

python -u GNN_branch/build_structural_memory.py \
  --pt_path "${PT_DIR}" \
  --ckpt "${CKPT}" \
  --checkpoint_contract "${CONTRACT}" \
  --checkpoint_sidecar "${SIDECAR}" \
  --feature_schema "${SCHEMA}" \
  --gexf_dir "${GEXF_DIR}" \
  --out "${STATIC_OUT}" \
  --embedding_mode static_pre_npt \
  --layerwise_out "${LAYERWISE_OUT}"

# Production multiscale memory keeps the global/multihop JKN action state and
# a centered early-layer local component. Normalization is fitted ONLY on the
# authoritative training kernels; validation/test kernels do not fit statistics.
python -u GNN_branch/build_multiscale_memory.py \
  --jkn_dir "${LAYERWISE_OUT}/jkn" \
  --conv1_dir "${LAYERWISE_OUT}/conv_1" \
  --out "${MULTISCALE_OUT}" \
  --dataset_jsonl artifacts/llm/mailohls_sft.jsonl \
  --split_json mailohls_runs/mailohls_final_family_split_s123.json \
  --normalization_stats_out "${STATS_OUT}" \
  --local_scale 1.0 \
  --local_mode aligned \
  --seed 123

echo "[DONE] Stage-2 production memory: ${MULTISCALE_OUT}"
