# Hotspot V2.2 MLIR and FPGA-hardware graph export

This development export materializes the current local `rodinia-hotspot-2-pipeline` and `rodinia-hotspot-3-unroll` representations. It does not replace the frozen 63-snapshot RC3 archive. Independent human semantic adjudication of RC3 remains pending.

## Result and scope

- Two freshly generated MLIR modules and native canonical/compact compiler graphs.
- Two authenticated RC3 canonical/compact program-graph pairs, independently checked against the fresh compiler facts.
- **48 serialized MLIR+hardware graphs:** two kernels × three profiles × four clock periods × canonical/compact views.
- Profiles: **ZCU104, Alveo U200, Alveo U50**.
- Clock periods: **3.33, 5, 6.67, 10 ns**. Each deployment uses full physical resource availability, with no additional platform reservation or user budget.
- **176 existing graph/overlay tests passed**. All generated overlays pass structural/evidence validation, program restoration, canonical/compact semantic equivalence and deterministic repeat serialization. Exported overlays are read back and validated before packaging.

| Kernel | Canonical nodes / edges | Compact nodes / edges | Enriched canonical nodes / edges | Registered / matched actions |
|---|---:|---:|---:|---:|
| Hotspot pipeline | 695 / 1,505 | 561 / 1,371 | 700 / 1,669 | 19 / 14 |
| Hotspot unroll | 695 / 1,505 | 561 / 1,371 | 700 / 1,669 | 19 / 14 |

Each enrichment adds five hardware nodes and 164 edges. Compiler-side witnesses and documented hardware properties do not establish actual synthesis mappings: 151 compatibility edges are possible mappings/relevance; 13 belonging/clock relationships are not implementation mappings. No HLS synthesis, QoR evaluation, neural training or tensor encoding runs here. No directive CSV label or measured QoR value enters a graph feature.

## Important result about the repaired labels

The current local C/C++ sources, `hotspot.h` files and authoritative `kernel_info.txt` files are **byte-identical to the frozen RC3 inputs**. Fresh generation reproduces every compiler node and edge. Thus this is a new verified materialization, not a fabricated new set of compiler facts.

All 19 source actions are registered, but **L1, L2, L3, L4 and L6 remain compiler-unresolved in both kernels**. These correspond to `temp_top`, `temp_left`, `temp_right`, `temp_bottom` and `power_center`. The compiler analysis does not establish an independent storage target for those source arrays. The absence of a target does not prove elimination; overlapping source-use lines do not establish array identity. No targets, memory roots or hardware links were guessed for these actions. Sparse absolute slot identities remain intact.

Fixing missing directive labels in a CSV does not, by itself, change a pragma-free compiler graph or establish those storage identities. If the corrected C/C++ or action metadata lives outside the checked-out paths recorded in `inputs.json`, it must be supplied as a different source snapshot and regenerated separately.

## Which files to use

- `canonical/` and `compact/`: the authenticated program graphs used by the hardware overlays. These retain the original release identity and provenance because current compiler facts match that release.
- `target_aware/<profile>/<period>ns_full/<view>/`: materialized target graphs. For example, `target_aware/zcu104/10ns_full/canonical/rodinia-hotspot-2-pipeline-c448680ca9be.json`.
- `release_raw_mlir/`: archived raw MLIR whose hashes match the primary authenticated compiler graphs.
- `raw_mlir/`: fresh MLIR generated from the current source files.
- `native_canonical/` and `native_compact/`: independently regenerated compiler graphs corresponding to the fresh MLIR.
- `sources/`: sealed source, header and action-metadata copies.
- `reference/`: exact FPGA profile registry and immutable release-manifest copies.
- `manifest.json`: per-file identities, provenance, compiler/target statistics and validation receipts.
- `SHA256SUMS`: integrity hashes for the export. The sibling `.tar.gz.sha256` authenticates the downloadable bundle.

Fresh and archived compilation contexts have different raw-MLIR hashes and graph identities; both provenance chains are retained. Cross-build query-fact comparison excludes only `graph_identity` and `source_provenance.mlir_sha256`. It does not modify stored query results or graphs. Within each canonical/compact pair, the existing strict reconstruction/query-equivalence checks run unchanged. All non-context metadata, compiler nodes/edges and pinned compiler/header hashes must match before release graphs may be reused.

Program schema: `salt-hls-mlir-structural-v2.2`, semantic revision `rc3`.

Enriched schema: `salt-hls-target-enriched-v22-v1`. These graphs are inputs for a future encoder; they are not V1 production-GNN tensors or structural-memory embeddings.

## Reproduce

Run from the MailoHLS repository root with the existing pinned Polygeist build and Python environments. No replacement compiler is downloaded. Use a new output directory:

```bash
PY=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
NATIVE=/home/elvouvali/.mlir-python311/bin/python
CGEIST=/home/elvouvali/tools/Polygeist/build-mailohls-assertions/bin/cgeist
EXPORT=GNN_branch/MLIR_graphs/hotspot_v22_20261028

PYTHONPATH=. "$PY" -B "$EXPORT/reproduce.py" \
  --output /tmp/hotspot-v22-rebuild --native-python "$NATIVE" --cgeist "$CGEIST"

POLYGEIST_BUILD=/home/elvouvali/tools/Polygeist/build-mailohls-assertions \
MLIR_PYTHON="$NATIVE" PYTHONPATH=. "$PY" -B -m pytest -q -p no:cacheprovider \
  tests/test_mlir_graph_v2.py tests/test_mlir_graph_v22.py tests/test_target_graph_v22.py \
  > /tmp/hotspot-v22-rebuild/logs/tests.log

PYTHONPATH=. "$PY" -B /tmp/hotspot-v22-rebuild/package.py \
  --directory /tmp/hotspot-v22-rebuild --archive /tmp/hotspot-v22-rebuild.tar.gz
```

Inspect or validate a selected hardware graph through the existing API/CLI:

```bash
"$PY" -B -m GNN_branch.target_graph_v22 inspect \
  --graph "$EXPORT/target_aware/zcu104/10ns_full/canonical/rodinia-hotspot-2-pipeline-c448680ca9be.json"
"$PY" -B -m GNN_branch.target_graph_v22 validate \
  --graph "$EXPORT/target_aware/u50/6p67ns_full/compact/rodinia-hotspot-3-unroll-7c11a20afd22.json"
```

The materializer intentionally refuses to claim authenticated RC3 reuse if current source/compiler facts differ from the release. Such a change requires a separately versioned, reviewed snapshot. Existing release archives/manifests and model artifacts stay unchanged.
