# Curated SALT-HLS research evidence

This directory contains unmodified summaries/compact receipts copied from named
research commits. Exact provenance, original paths and SHA-256 values are in
`SOURCE_MAP.tsv`. Some reports preserve historical filenames and in-progress
status. The top-level README is authoritative about the proposed design and
its current end-to-end qualification, not an unconditional endorsement of every
historical experiment. Private model weights, raw HLS logs and datasets are NOT
present; verify them via the referenced manifests on their original hosts.

The later [local GNN completion](latency_gnn_completed_local.md) is recorded in
[LOCAL_SOURCE_MAP.json](LOCAL_SOURCE_MAP.json), separately from the 21 copies in
`SOURCE_MAP.tsv`. Its report and decision hashes match the completion receipt;
the selected local checkpoint hash was also checked. Completion did not justify
a freeze upgrade or promote a model. These local runtime receipts were not in
the pushed research commit. Original reports remain unchanged.

Copied reports may contain relative links intended for their original paths.
Open the original path at the full commit in `SOURCE_MAP.tsv` when following
such a link. `audit_final_architecture.py` records these historical links
separately and checks maintained supervisor-document paths as a release gate.
