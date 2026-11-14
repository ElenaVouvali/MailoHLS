# Kalman / Rosetta compiler compatibility repair — Liono

**Local graph and frozen-memory qualification: PASS, 9/9.** The actual nine-kernel
export and production memory-loader preflight completed. This is an explicitly
versioned compatibility representation, not identical historical preprocessing.
No Davinci connection, decoder inference, `--kernel all`, HLS run or training was
performed. No frozen weights, authenticated source/header/metadata bytes, original
compiler, original exporter, seven control GEXFs or preview7 files were changed.
The worktree remains clean at `6ee1390611c9047a4b8e2b07200d223b93ee8d16`.

Paths:

```bash
R=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final
P="$R/artifacts/stage2_eval_9_20261113"
W="$P/compat9_v2_work"
# Qualified graph directory: $P/graphs_compat9_v2
# Qualified bank:            $P/export_compat9_v2/memory_B
```

## Repairs and source grounding

**Kalman passes 13/13.** The original driver marks action arrays after its first
mem2reg/canonicalization passes. The new opt-in flag
`--mailohls-preserve-action-storage` moves exact-manifest marking before those
passes and prevents promotion only for allocations carrying both the exact action
ID and opt-in storage-preservation attribute. All ordinary loop restructuring,
affine raising and unmarked promotion remain. The flag requires an action manifest.
No `-immediate` is used in the qualified route.

The distinct native build is:

`$W/compiler_action_storage_v2/bin/cgeist`

It recompiles only `driver.cc` and `PolygeistMem2Reg.cpp`, replaces the latter's
member in a private copy of the transforms archive, and links against hash-recorded
original dependencies. It never builds into or edits `/home/elvouvali/tools/Polygeist`.
The existing local mem2reg source contained a later, unbuilt type-punning patch.
To avoid incorporating that unrelated change, the old source was reconstructed
from local Git HEAD and recompiled: its object matches the original build object
**byte-for-byte** (`9374836a97df3432a4dcd70092d7461123c578262ef4ac04656f645ebae5baea`).
Only the narrow storage-preservation guard was added to that authenticated base.

`compiler_checks_final/PASS.json` proves physical `memref<64xf32>` allocations for
L5–L8 at the exact declarations and all seven action loops L3,L4,L9–L13 in normal
post-pass IR. The for-op location points inside its actual source header (Polygeist
uses the induction-variable column, the manifest uses the `for` keyword). All 13
graph actions are `source_location`; no action fallbacks were enabled.
Flag-off IR matches the original except ordering of identical DLTI layout entries.
That ordering is normalized **only for this comparison**; emitted MLIR/GEXF and
their recorded hashes are never rewritten. The missing-manifest negative test passes.

**Rosetta passes 12/12.** The existing hook was run independently first. The final
hook additionally verifies the exact `memref.global` definition and `get_global`
symbol for source L1 `zculling::z_buffer`, real shape `256x256x1xi64`, unique root,
and native compiler-traced reads/writes. Source uses and traced accesses intersect
at **322, 334 and 340**; six native access records are retained in
`graphs_compat9_v2/ROSETTA_NATIVE_BINDING.json`. Static lifetime remains unchanged.

Rosetta has **11 `source_location` bindings plus one `global_symbol_and_access`**.
The latter is never renamed `source_location`. The untouched historical tensorizer
accepts the graph and its true pinned-generator hash. Two downstream GMT scope
checks originally rejected the new binding name; a narrow runtime provenance
extension accepts only the receipt-allowlisted Rosetta graph/L1 with verified
source, metadata, compiler, MLIR, global-symbol and access proof. All other scope,
evidence, routing and encoder computations are unchanged. Generated versions of
those two check functions and original/derived source hashes are retained.
Negative tests reject a missing allowlist, wrong global, wrong source identity and
missing access evidence. No scope attribute or generator hash is rewritten.

Two pinned-builder warnings remain explicit: Rosetta L8 `fragment` and L9 `pixels`
have real allocation/call evidence but no resolved direct array-access scope edges.
Their memory/compute groups contain genuine records; each control group is absent
and remains masked. No direct access edges, missing evidence or summaries were
invented. This does not assert complete interprocedural aggregate-memory analysis.

## Complete validation and export

| Kernel | Actions | Compiler evidence records | Memory B rows |
|---|---:|---:|---:|
| GAN | 11 | 3,280 | 97 |
| Kalman | 13 | 2,012 | 103 |
| chstone-aes | 6 | 1,151 | 82 |
| chstone-jpeg | 21 | 11,408 | 127 |
| covariance | 7 | 736 | 85 |
| jacobi-2d | 5 | 796 | 79 |
| rosetta-3d-rendering | 12 | 15,387 | 98 |
| syr2k | 4 | 477 | 76 |
| trmm-opt | 3 | 264 | 73 |

Both repaired graphs were independently run through frozen epoch-9 GNN and the
600-update GMT checkpoint, then all nine were run again by the **unmodified**
`/home/elvouvali/export_stage2_eval_gmt.py` through `export_compat9_v2.py`.
Every repaired graph has finite 64×128 anchors and GMT summary shapes respectively
`(13,3,128)` and `(12,3,128)`, with at least one genuine group per action. Rosetta's
15,387 records are below the unchanged strict 16,384 limit; none were truncated.
The full export has exactly nine hash-matched `.memory.pt` files and its manifest.
Every B memory preserves its corresponding 64 A action anchors byte-for-byte.

The unchanged production structural-memory loader passes all nine on CPU with
128D memory, 64 absolute action slots, correct active IDs, complete nonempty
summary routing, finite tensors, inactive masking and source/manifest hashes.
This is **inference-input qualification**, not decoder/GPU execution. The target
inference program's own model/checkpoint/prompt preflight remains required after
copying inputs. No such program was present locally and Davinci was not accessed.

All 78 protected pre-existing files pass the final before/after check. All five
committed frozen checkpoint identities (V4, residual LoRA, FiLM, GNN and GMT) match.
The original preview7 bank remains byte-identical. In the newly recomputed nine-bank
copy, five control memory files also match preview7 exactly; AES and JPEG tensors
differ by at most `4.7683716e-7` and `8.3446503e-7`, respectively. This measured
re-inference difference is recorded, not hidden by rewriting the new manifest or
replacing old preview files. All seven source GEXFs remain exactly identical.

## Identities and receipts

| Artifact | SHA-256 |
|---|---|
| Unchanged pinned generator | `5e4f93c89ce4c275c04ff6ed4dd81a17858647a55b0abec0846b8da240da3b05` |
| Original compiler | `6f22da6242249a6ca1c2a9aea88029c1ce075852fdde75340901baef49755b5d` |
| Qualified opt-in compiler | `820d4d3a19d162c74169af8d10fe05703afe9b48df2b0c70dcc07a3f7ccec58b` |
| Kalman GEXF | `97198263c2f14050e619d7afce766703217c1db7f4c05cfe69fed3bc274c1720` |
| Rosetta GEXF | `b99c3bb75080077b7cda231662abb05166b0c0c098eaf0d4711d59163a2847f0` |
| Frozen GNN | `34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623` |
| Frozen GMT | `1d4cdf3071375349b94809062d05d30ad3e5c981bb6e2885b6f99c9e94398c12` |
| Nine-memory manifest | `17e493c37735706d2a0e8c3f6571d13f6e7022538db22f10bac4199482d1b8a2` |
| Unchanged source manifest | `54fefb269c2002dfd4ca65b5db9dea7dd609e4a6e22cfac20832ed6ba2ed0409` |

The graph receipt contains per-kernel source/metadata/compiler/MLIR/graph hashes,
actual resolution modes, compiler-build receipt hash and runtime/runner hashes.
The memory contract is `export_compat9_v2/COMPAT9_V2_EXPORT_PASS.json`, also copied
to `memory_B/EVAL_COMPATIBILITY.json`; the original exporter manifest is unchanged.
Compiler source copies, patch, object reconstruction, exact compile/link argv,
331 original link-input identities, build and qualification receipts are under
`$W/compiler_action_storage_v2` and `original_mem2reg_reconstruction`.

All code changes are new files under `$W`: compiler overlay builder/checker,
`compatibility_hooks_v2.py`, isolated graph probes, reusable `generate_graph.py`,
graph assembler, provenance-verified export wrapper, negative tests and portable
CPU input preflight. No tracked worktree file was modified. Failed early checks
and earlier candidate outputs remain in their own directories/logs for inspection.

## Executed commands and reusable entry points

After `source /tmp/salt_stage2_graph_env.sh`, the successful final commands were:

```bash
python3 "$W/build_compiler_overlay.py" --tag compiler_action_storage_v2 --reconstructed-mem2reg "$W/original_mem2reg_reconstruction/PolygeistMem2Reg.cpp"
"$GRAPH_PY" "$W/run_kalman_probe.py"
"$GRAPH_PY" "$W/check_compiler.py" --output-tag compiler_checks_final
"$GRAPH_PY" "$W/run_rosetta_verified.py"
CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 "$P/gnn_export_venv/bin/python" "$W/validate_frozen_one.py" --kernel Kalman --graph "$W/kalman_probe_v2/Kalman.gexf" --out "$W/kalman_probe_v2/FROZEN_PASS.json"
CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 "$P/gnn_export_venv/bin/python" "$W/validate_frozen_one.py" --kernel rosetta-3d-rendering --graph "$W/rosetta_verified_v3/rosetta-3d-rendering.gexf" --global-binding "$W/rosetta_verified_v3/NATIVE_BINDING.json" --out "$W/rosetta_verified_v3/FROZEN_PASS.json"
"$GRAPH_PY" "$W/assemble_graphs.py"
CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 "$P/gnn_export_venv/bin/python" "$W/export_compat9_v2.py"
CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 "$P/gnn_export_venv/bin/python" "$W/test_global_acceptance.py"
CUDA_VISIBLE_DEVICES= "$P/gnn_export_venv/bin/python" "$W/preflight_eval9.py" --repo "$R" --sources "$P/sources" --memory "$P/export_compat9_v2/memory_B" --out "$P/export_compat9_v2/CPU_INFERENCE_PREFLIGHT_PORTABLE.json"
```

These are completed, fresh-output commands; they intentionally refuse overwrite.
They also ran with `PYTHONDONTWRITEBYTECODE=1`; frozen tests/export used
`MPLCONFIGDIR=/tmp/mailohls_compat9_mpl`. The unmodified original Rosetta hook was
tested first by `run_rosetta_probe.py`; its rejection by the original GMT scope
mapper is retained in `rosetta_frozen_probe.log`. The final runtime extension
addresses that precise rejection rather than suppressing exceptions.

To generate either repaired graph again into a **new** directory, use the tested CLI:

```bash
source /tmp/salt_stage2_graph_env.sh
"$GRAPH_PY" "$W/generate_graph.py" --kernel Kalman --out "$P/kalman_new_attempt"
"$GRAPH_PY" "$W/generate_graph.py" --kernel rosetta-3d-rendering --out "$P/rosetta_new_attempt"
```

To repeat the portable input preflight without overwriting a receipt:

```bash
CUDA_VISIBLE_DEVICES= "$P/gnn_export_venv/bin/python" "$W/preflight_eval9.py" --repo "$R" --sources "$P/sources" --memory "$P/export_compat9_v2/memory_B"
```

## Safe transfer commands — provided, not executed

Run on Liono. These use fresh names, do not replace preview7, and do not delete
anything. The first command refuses if any destination already exists.

```bash
set -euo pipefail
R=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final
P="$R/artifacts/stage2_eval_9_20261113"
H=evouvali@davinci.microlab.ntua.gr
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
ssh "$H" "test -d '$D' && test ! -e '$D/eval_sources_compat9' && test ! -e '$D/eval_gmt_memory_B_compat9' && test ! -e '$D/eval_compat9_receipts' && mkdir '$D/eval_sources_compat9' '$D/eval_gmt_memory_B_compat9' '$D/eval_compat9_receipts'"
rsync -a --checksum "$P/sources/" "$H:$D/eval_sources_compat9/"
rsync -a --checksum "$P/export_compat9_v2/memory_B/" "$H:$D/eval_gmt_memory_B_compat9/"
rsync -a --checksum "$P/export_compat9_v2/transfer_receipts/" "$H:$D/eval_compat9_receipts/"
ssh "$H" "cd '$D' && sha256sum -c eval_compat9_receipts/TRANSFER_SHA256SUMS"
ssh "$H" "CUDA_VISIBLE_DEVICES= '$D/venv/bin/python' '$D/eval_compat9_receipts/preflight_eval9.py' --repo '$D/code' --sources '$D/eval_sources_compat9' --memory '$D/eval_gmt_memory_B_compat9'"
```

Keep the compatibility receipts with the bank. Only after those target-side checks
and the actual inference runner's own model/prompt preflight pass should its
all-kernel mode be used. No `--kernel all` launch command is issued here.

**Remaining blockers:** none for the requested local compiler/GNN/GMT/memory input
qualification. Rosetta's two explicit aggregate-access limitations remain. Target
decoder preflight and actual inference are pending; they were not executed on
Davinci under this Liono-only task.
