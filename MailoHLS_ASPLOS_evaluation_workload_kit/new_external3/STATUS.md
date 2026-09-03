# Materialization status

The preparation logic, pinned commits/blob hashes, validators, case builder, and
MLIR-generation wrapper are included. The three upstream source trees are not
vendored inside this ZIP; run `python materialize_upstream.py` once (HTTPS or
pinned local checkout) to create the final MailoHLS `kernel.*`, `kernel_info.txt`,
and `kernel_placeholders.*` files. This is intentional for provenance and license
traceability, not a claim that checkpoint-dependent memory has already been built.
