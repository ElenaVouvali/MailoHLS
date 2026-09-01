#!/usr/bin/env bash
set -euo pipefail

# Run once, from the repository root.  Do not tune on this locked Kalman set.
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTO_ROOT="mailohls_runs/auto_clock_v8_final"
MEMORY_DIR="artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned"
TEST_FEATURES="${AUTO_ROOT}/s123/test_features.pt"
TEST_METRICS="${AUTO_ROOT}/s123/test_metrics.json"

PYTHONPATH=. "${PYTHON_BIN}" -m LLM_branch.clock_adapt.extract_features \
  --cases_jsonl "${AUTO_ROOT}/cases/test.jsonl" \
  --memory_dir "${MEMORY_DIR}" \
  --output "${TEST_FEATURES}"

PYTHONPATH=. "${PYTHON_BIN}" -m LLM_branch.clock_adapt.evaluate \
  --features "${TEST_FEATURES}" \
  --clock_adapter_dir "${AUTO_ROOT}/s123/final" \
  --output_json "${TEST_METRICS}"

jq 'del(.rows)' "${TEST_METRICS}"
