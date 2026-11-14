#!/usr/bin/env bash
# Provided for the user; not executed during the Liono-only repair.
set -euo pipefail
R=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final
P="$R/artifacts/stage2_eval_9_20261113"
H=evouvali@davinci.microlab.ntua.gr
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
test -f "$P/export_compat9_v2/CPU_INFERENCE_PREFLIGHT_PORTABLE.json"
test -f "$P/export_compat9_v2/transfer_receipts/TRANSFER_SHA256SUMS"
ssh "$H" "test -d '$D' && test ! -e '$D/eval_sources_compat9' && test ! -e '$D/eval_gmt_memory_B_compat9' && test ! -e '$D/eval_compat9_receipts' && mkdir '$D/eval_sources_compat9' '$D/eval_gmt_memory_B_compat9' '$D/eval_compat9_receipts'"
rsync -a --checksum --info=progress2 "$P/sources/" "$H:$D/eval_sources_compat9/"
rsync -a --checksum --info=progress2 "$P/export_compat9_v2/memory_B/" "$H:$D/eval_gmt_memory_B_compat9/"
rsync -a --checksum "$P/export_compat9_v2/transfer_receipts/" "$H:$D/eval_compat9_receipts/"
ssh "$H" "cd '$D' && sha256sum -c eval_compat9_receipts/TRANSFER_SHA256SUMS"
ssh "$H" "CUDA_VISIBLE_DEVICES= '$D/venv/bin/python' '$D/eval_compat9_receipts/preflight_eval9.py' --repo '$D/code' --sources '$D/eval_sources_compat9' --memory '$D/eval_gmt_memory_B_compat9'"
echo 'Copied and CPU-input-preflighted the compatibility bank; run the actual inference runner model/prompt preflight before all-kernel inference.'
