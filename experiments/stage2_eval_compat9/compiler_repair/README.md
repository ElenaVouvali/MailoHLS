# Source-grounded Kalman/Rosetta compatibility repair

The scripts and small receipts here are exact source snapshots of the qualified
Liono execution under
`artifacts/stage2_eval_9_20261113/compat9_v2_work`. See
[`docs/kalman_rosetta_compat9_v2.md`](../../../docs/kalman_rosetta_compat9_v2.md)
for the actual commands, source/compiler/checkpoint identities and limitations.

`generate_graph.py` and `compatibility_hooks_v2.py` implement the isolated
compatibility routes. `build_compiler_overlay.py` builds the opt-in native compiler;
its modified/original C++ sources and pass diff are under
`compiler_action_storage_v2/`. `check_compiler.py` validates normal post-pass
allocations and source-located loops. `export_compat9_v2.py` delegates to the
unmodified original exporter, preserved under `original_scripts/`, through a
strict global-source provenance extension. `preflight_eval9.py` validates the
resulting inference inputs using the original production memory loader.

No compiler binaries, model weights, generated graphs or memory payloads are
committed here. Reproduction requires the pinned compiler build dependencies,
authenticated source staging, frozen epoch-9 GNN and selected GMT checkpoint.
The snapshots retain their original pinned execution paths and hash guards;
this publication checkout is not a replacement for the pinned execution checkout.

Earlier candidate probes and original scripts are retained for provenance.
The qualified final route is compiler `compiler_action_storage_v2` plus the v2
provenance hooks; it does not use the original immediate-mode Kalman workaround.
