# Stage-2 reproducibility evidence

This directory contains reviewable metadata and selected evaluation outputs. It
does not contain model adapters, datasets, memory banks, or checkpoint binaries.

## Contents

- `stage2_fusion/`: the training contract, paired step-20/40 validation
  predictions, a compact metric comparison, and the shared initial-state record.
- `memory_manifests/`: provenance for the JKN rebuild, aligned, zero, and
  deranged structural-memory variants.
- `gate_parity/`: contracts and metadata-only checkpoint reports for the
  original scale-32 and baked scale-1 configurations, plus their effective-gate
  parity audit.
- `splits/`: the exact seed-123 family partition used by the runs.
- `diagnostics/`: short diagnostic extracts without model-loading progress output.

The checkpoint reports contain SHA-256 hashes and, for every tensor, its name,
shape, dtype, norm, and finite-value status. Gate entries additionally record
raw, tanh-transformed, and effective values. The reports can be regenerated with
`LLM_branch/tools/checkpoint_inspection_report.py` and compared with
`LLM_branch/tools/compare_gate_parity_reports.py`.

The paired parity evaluation command is captured in
`scripts/reproduction/run_gate_parity_screen.sh`. Its required inputs are
explicit environment variables, and it rejects missing inputs and accidental
output overwrites.
