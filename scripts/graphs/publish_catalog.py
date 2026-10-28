"""Build, authenticate, and unpack the single public V2.2 graph catalog.

CPU only. Uses the existing overlay/validators without changing their schema.
Full graph JSON is gzip-compressed with fixed headers; review pages stay plain.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TARGETS = ('zcu104', 'u200', 'u50')
CLOCKS = (3.33, 5.0, 6.67, 10.0)
CATALOG = ROOT / 'graphs/latest'
PUBLISHED = CATALOG
ARCHIVE = ROOT / 'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
KALMAN = 'serrano-kalman-filter-96b4d608969f'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def clock_name(clock):
    return f'{clock:g}'.replace('.', 'p') + 'ns_full'


def initialize_worker(output):
    # Explicit initialization also works with Python's spawn/forkserver modes.
    global CATALOG
    CATALOG = Path(output)


def copy_review_evidence(destination):
    """Keep the copied guide's links valid and retain prior, separately scoped receipts."""
    guide = (ROOT / 'docs/target-aware-mlir-v22/README.md').read_text()
    links = {'../V22_FINAL_RELEASE_REVIEW.md': '../../docs/V22_FINAL_RELEASE_REVIEW.md',
             '../structural-graph-v22.md': '../../docs/structural-graph-v22.md',
             '../graph-release-v22-rc3.json': 'release-manifest.json',
             '../../mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1/training_contract.json':
                 'validation/baseline-verification.json'}
    for name in ('baseline-verification.json', 'corpus-validation.json',
                 'final-verification.json', 'reproduction-verification.json'):
        links[name] = 'validation/' + name
        source = ROOT / 'docs/target-aware-mlir-v22' / name
        target = destination / 'validation' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for old, new in links.items():
        guide = guide.replace('](' + old + ')', '](' + new + ')')
    (destination / 'GUIDE.md').write_text(guide)
    for name in ('hotspot-native-comparison.json', 'kalman.json'):
        source = PUBLISHED / 'validation' / name
        target = destination / 'validation' / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    for source in (PUBLISHED / 'kernels').glob('*/native-verification'):
        target = destination / 'kernels' / source.parent.name / source.name
        if source.resolve() != target.resolve():
            shutil.copytree(source, target)


def compressed(path, data):
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0, compresslevel=6) as out:
        out.write(data)
    payload = stream.getvalue()
    if gzip.decompress(payload) != data:
        raise ValueError('Gzip round trip failed')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {'path': str(path.relative_to(CATALOG)), 'sha256': digest(payload),
            'json_sha256': digest(data), 'bytes': len(payload), 'json_bytes': len(data)}


@lru_cache(maxsize=1)
def load_pair(root, filename):
    from GNN_branch import mlir_graph_v2 as p, target_graph_v22 as h
    pair = [h.authenticated_input(Path(root) / view / filename) for view in ('canonical', 'compact')]
    if p.graph_payload(p.expand_compact(pair[1][0])) != p.graph_payload(pair[0][0]):
        raise ValueError(f'Lossless reconstruction failed: {filename}')
    return pair


def materialize(job):
    from GNN_branch import mlir_graph_v2 as p, target_graph_v22 as h
    root, filename, target, clock = job
    started = time.perf_counter()
    pair = load_pair(root, filename)
    outputs, signatures = [], []
    summary = None
    for view, (program, auth) in zip(('canonical', 'compact'), pair):
        graph = h.enrich(program, target, clock, authentication=auth)
        summary_view = h.validate(graph)  # Restores original and independently rederives every overlay edge.
        signatures.append(h.overlay_signature(graph))
        data = h.serialized(graph).encode()
        if data != h.serialized(graph).encode():
            raise ValueError('Nondeterministic serialization')
        path = CATALOG / 'kernels' / filename[:-5] / 'hardware' / target / clock_name(clock) / (view + '.json.gz')
        item = compressed(path, data)
        item.update(view=view, target=target, clock_period_ns=clock)
        outputs.append(item)
        if view == 'canonical':
            summary = summary_view
            if target == 'zcu104' and clock == 10:
                payload = p.graph_payload(graph)
                save(path.parents[3] / 'hardware-review.json', {
                    'notice': 'Exact overlay excerpt, not a complete graph; actual synthesis bindings unknown.',
                    'context': {'target': target, 'clock_period_ns': clock, 'full_physical_capacity': True},
                    'nodes': [n for n in payload['nodes'] if n.get('_target_overlay')],
                    'edges': [e for e in payload['edges'] if e.get('_target_overlay')],
                    'action_slots': graph.graph['action_slots'],
                    'action_annotations': graph.graph['action_annotations'],
                    'excluded_relationships': graph.graph['excluded_relationships']})
    if signatures[0] != signatures[1]:
        raise ValueError(f'Canonical/compact overlay query mismatch: {filename}')
    return {'snapshot': filename[:-5], 'target': target, 'clock_period_ns': clock,
            'summary': summary, 'files': outputs, 'runtime_seconds': time.perf_counter() - started,
            'validation': 'pass; original restored, overlay rederived, canonical/compact semantics equal, deterministic serialization, gzip round trip'}


def neighborhood(directory):
    """A small literal graph excerpt; never invent diagram edges."""
    program = json.loads(gzip.decompress((directory / 'canonical.json.gz').read_bytes()))
    overlay = json.loads((directory / 'hardware-review.json').read_text())
    edges = []
    for relation in ('target:arithmetic_compatible', 'target:storage_compatible',
                     'target:action_resource_relevance', 'target:requested_clock'):
        match = next((e for e in overlay['edges'] if e['relation_type'] == relation), None)
        if match is not None:
            edges.append(match)
    actions = {e['source'] for e in edges if e['source'].startswith('action:')}
    edges += [e for e in program['edges'] if e['source'] in actions and e['relation_type'] == 'targets'][:2]
    resources = {e['target'] for e in edges if e['target'].startswith('hardware:resource:')}
    edges += [e for e in overlay['edges'] if e['source'] in resources and e['relation_type'] == 'hardware:belongs_to']
    used = {e[side] for e in edges for side in ('source', 'target')}
    nodes = [n for n in program['nodes'] + overlay['nodes'] if n['id'] in used]
    save(directory / 'graph-neighborhood.json', {'notice': 'Literal subset of canonical program plus ZCU104/10ns/full overlay; no inferred diagram edges.',
                                               'nodes': nodes, 'edges': edges})
    ids = {name: f'n{i}' for i, name in enumerate(sorted(used))}
    diagram = ['```mermaid', 'flowchart LR']
    for name in sorted(used):
        label = html.escape(name.replace('"', "'"))
        diagram.append(f'  {ids[name]}["{label}"]')
    for edge in edges:
        status = edge.get('compatibility', {}).get('status', edge.get('evidence_status', 'unknown'))
        diagram.append(f'  {ids[edge["source"]]} -->|"{edge["relation_type"]} ({status})"| {ids[edge["target"]]}')
    diagram.append('```')
    return diagram


def pages(manifest):
    rows = []
    for row in manifest['snapshots']:
        name = row['snapshot']
        directory = CATALOG / 'kernels' / name
        contexts = [c for c in manifest['contexts'] if c['snapshot'] == name]
        sources = [d for d in (CATALOG / 'sources').iterdir() if d.name.startswith(name.rsplit('-', 1)[-1])]
        if len(sources) != 1:
            raise ValueError(f'No unique source snapshot for {name}')
        baseline = next(c for c in contexts if c['target'] == 'zcu104' and c['clock_period_ns'] == 10)
        s = baseline['summary']
        status = ('Recommended development Kalman (13 registrations)' if name == KALMAN else
                  'Historical development Kalman (11 registrations)' if name == 'serrano-kalman-filter-41c8975cabb3' else
                  'External Kalman snapshot' if name == 'serrano-kalman-filter-b90908f83f76' else 'RC3 snapshot')
        lines = [f'# {name}', '', '[Back to all graphs](../../README.md)', '', status + '.', '',
                 '| Representation | Nodes | Edges | File |', '| --- | ---: | ---: | --- |']
        for view in ('canonical', 'compact'):
            v = row[view]
            lines.append(f'| MLIR {view} | {v["nodes"]:,} | {v["edges"]:,} | [{view} JSON.gz]({view}.json.gz) |')
            lines.append(f'| MLIR + hardware {view} | {v["nodes"]+5:,} | {v["edges"]+s["added_edges"]:,} | Select a target below |')
        lines += ['', f'Actions: **{s["matched_actions"]}/{s["registered_actions"]} compiler-matched/registered**. '
                  + 'Unresolved: ' + (', '.join(s['unresolved_actions']) or 'none') + '.', '',
                  f'Review [sealed source, headers and metadata](../../sources/{sources[0].name}/), '
                  '[raw MLIR](module.mlir), [exact action registrations](action-points.json), '
                  '[hardware nodes and every added edge, including evidence](hardware-review.json). '
                  'The hardware excerpt uses ZCU104, 10 ns, full capacity. Actual mappings remain unknown.', '',
                  'A small, exact neighborhood (all node IDs and edges are copied from the stored graphs):', '']
        lines += neighborhood(directory)
        lines += ['', '[Exact nodes, edges and evidence behind this diagram](graph-neighborhood.json). '
                  '`proven` describes a compiler relationship; `possible` describes hardware compatibility. '
                  'The clock edge is a requested constraint, not a mapping or feasibility result.', '',
                  '| Device | Clock (ns) | Full canonical graph | Full compact graph |', '| --- | ---: | --- | --- |']
        for c in contexts:
            base = f'hardware/{c["target"]}/{clock_name(c["clock_period_ns"])}'
            lines.append(f'| {c["target"]} | {c["clock_period_ns"]:g} | [JSON.gz]({base}/canonical.json.gz) | [JSON.gz]({base}/compact.json.gz) |')
        lines += ['', 'All contexts use full physical resources; platform reservations are unknown. '
                  'Resource compatibility is possible, never a proven synthesis binding. '
                  'Compiler facts, action IDs and uncertainty are preserved. '
                  'Independent human semantic adjudication remains pending.', '',
                  'Decompress with `gzip -dk canonical.json.gz`, or use the catalog unpack command. '
                  'See the [catalog guide](../../GUIDE.md) for schemas and the evidence contract.', '']
        if (directory / 'native-verification').is_dir():
            lines += ['The latest independent Hotspot regeneration is also retained here: '
                      '[fresh raw MLIR](native-verification/module.mlir), '
                      '[fresh canonical graph](native-verification/canonical.json.gz), '
                      '[fresh compact graph](native-verification/compact.json.gz). '
                      'Its compiler nodes and edges match the primary authenticated graph; '
                      'its compilation-context provenance and graph identity are intentionally distinct. '
                      'Hardware graphs above use the authenticated primary context. '
                      'See the [native comparison receipt](../../validation/hotspot-native-comparison.json).', '']
        (directory / 'README.md').write_text('\n'.join(lines))
        rows.append(f'| [{name}](kernels/{name}/) | {row["canonical"]["nodes"]:,} / {row["canonical"]["edges"]:,} | '
                    f'{s["added_nodes"]} / {s["added_edges"]} | {s["matched_actions"]}/{s["registered_actions"]} | {status} |')
    text = '''# Latest MLIR and MLIR + FPGA hardware graphs

**One place to review and download the current graph collection.** Each kernel folder contains raw MLIR, both compiler-graph views, action registrations, a readable hardware-evidence excerpt, and every full hardware-enriched graph.

- **63 authenticated V2.2 RC3 snapshots**, 126 original canonical/compact graphs.
- **1,512 full MLIR + hardware graphs**: 63 snapshots × ZCU104/U200/U50 × 3.33/5/6.67/10 ns × canonical/compact.
- Full physical resource availability in the published contexts. Arbitrary clocks and restricted budgets remain supported by the existing composition CLI.
- Corrected development **[Kalman](kernels/serrano-kalman-filter-96b4d608969f/)** and latest **[Hotspot pipeline](kernels/rodinia-hotspot-2-pipeline-c448680ca9be/)** / **[Hotspot unroll](kernels/rodinia-hotspot-3-unroll-7c11a20afd22/)** are easy to find here. Historical Kalman alternatives are explicitly marked below.
- The fresh Hotspot native MLIR and four independently regenerated compiler graphs are retained inside the corresponding kernel folders, alongside the primary authenticated compilation context.

RC3 is technically frozen; **independent human semantic adjudication remains pending**. Hardware edges express documented capabilities and possible compatibility, not measured resource bindings or QoR. Production GNN checkpoints still consume V1 tensors. No synthesis, training, or QoR inference was performed for this catalog.

## Review and download

Open a kernel below, inspect its raw MLIR/action registrations/hardware evidence in GitHub, then download a complete graph for the desired target and clock. Complete graph JSON is gzip-compressed to keep duplicate compiler facts from inflating the repository; no custom graph format or Git LFS is needed. The underlying schemas remain `salt-hls-mlir-structural-v2.2` and `salt-hls-target-enriched-v22-v1`.

[Architecture and evidence guide](GUIDE.md) · [Target specifications](profiles.json) · [Manifest and graph identities](manifest.json) · [Checksums](SHA256SUMS) · [Validation report](validation/catalog.json) · [Sealed source/header/metadata snapshots](sources/)

From a checkout of this branch, with Python 3.10+:

```bash
# Check every published file against SHA256SUMS (standard library only).
python scripts/graphs/publish_catalog.py verify
# Extract all graphs and raw MLIR for one kernel into a fresh directory.
python scripts/graphs/publish_catalog.py unpack \\
  --kernel serrano-kalman-filter-96b4d608969f --output /tmp/kalman-graphs
# Validate a complete enriched graph using the existing independent validator.
python -m GNN_branch.target_graph_v22 validate \\
  --graph /tmp/kalman-graphs/hardware/zcu104/10ns_full/canonical.json
```

Unpack/verify use only the standard library. Graph validation/build require the repository's existing NetworkX dependency. Reading these graphs does not require Polygeist, MLIR Python bindings, Torch, a GPU, or Vitis.

To reproduce the complete catalog from the immutable release, use a fresh output directory and `python scripts/graphs/publish_catalog.py build --output /tmp/rebuilt-graph-catalog --workers 8`. Full-corpus validation independently rederives every hardware overlay, restores every program graph, compares canonical/compact supported semantics, checks repeated serialization, and verifies gzip round trips. Runtime receipts can vary between builds; graph bytes and gzip headers are deterministic for the same implementation, inputs and runtime.

## Kernel index

Counts below are original canonical nodes/edges, added hardware nodes/edges per context, and compiler-matched/registered actions. All 63 archived snapshots are retained; this count is not a claim of 63 distinct development kernels. The release comprises 55 development snapshots and eight external snapshots.

| Kernel / source snapshot | Compiler nodes / edges | Added nodes / edges | Actions | Selection note |
| --- | ---: | ---: | ---: | --- |
'''
    text += '\n'.join(rows) + '''

## Provenance and limitations

The compiler graph files decompress to the exact authenticated RC3 bytes. The archive, its historical manifest and checksums remain unchanged. The manifest records their hashes, profile identity and overlay source hashes. All compiler queries use the current repository implementation, including the incomplete-memory-root-set correction; archived helper code is not substituted.

The recent Hotspot native regeneration established unchanged compiler facts. All 19 actions are registered, but L1/L2/L3/L4/L6 still have no established compiler target. Corrected development Kalman registers 13 actions, with L7/L8 unresolved. These remain unresolved here; slots are not renumbered and targets are not guessed. See [Hotspot regeneration evidence](validation/hotspot-native-comparison.json) and [Kalman receipt](validation/kalman.json).

Physical capacities and requested deployment constraints remain separate. Published full-resource contexts assume no platform reservation; actual shell/platform availability is unknown. Hardware mapping, achievable II, scheduling, bank conflicts and timing closure are not established. A new V2.2 heterogeneous GNN encoder, training and Stage-2 integration remain future work.
'''
    (CATALOG / 'README.md').write_text(text)


def build(args):
    from GNN_branch import mlir_graph_v2 as p, target_graph_v22 as h
    global CATALOG
    CATALOG = Path(args.output).resolve()
    if CATALOG.exists():
        raise FileExistsError('Build into a fresh directory')
    CATALOG.mkdir(parents=True)
    began = time.perf_counter()
    original_archive_hash = h.file_hash(ARCHIVE)
    manifest = {'schema': 'salt-hls-graph-catalog-v1', 'program_schema': p.SCHEMA_V22,
                'overlay_schema': h.SCHEMA, 'semantic_revision': 'rc3',
                'independent_adjudication': 'pending', 'targets': list(TARGETS),
                'clock_periods_ns': list(CLOCKS), 'resource_budget': 'full physical capacity; no known platform reservation',
                'implementation': h.implementation_identity(), 'catalog_builder_sha256': h.file_hash(Path(__file__)),
                'build_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'recommended_development_kalman': KALMAN, 'snapshots': [], 'contexts': [], 'failures': []}
    with h.authenticated_release(ARCHIVE) as (root, release):
        manifest['release_authentication'] = {'archive_sha256': original_archive_hash,
            'manifest_sha256': h.file_hash(h.RELEASE), 'authenticated_members': len(release['files_sha256'])}
        shutil.copytree(root / 'source_snapshots', CATALOG / 'sources')
        shutil.copy2(h.PROFILES, CATALOG / 'profiles.json')
        copy_review_evidence(CATALOG)
        shutil.copy2(h.RELEASE, CATALOG / 'release-manifest.json')
        paths = sorted((root / 'canonical').glob('*.json'))
        for path in paths:
            row = {'snapshot': path.stem}
            directory = CATALOG / 'kernels' / path.stem
            for view, (graph, auth) in zip(('canonical', 'compact'), load_pair(str(root), path.name)):
                raw = (root / view / path.name).read_bytes()
                item = compressed(directory / (view + '.json.gz'), raw)
                item.update(nodes=len(graph), edges=graph.number_of_edges(), graph_identity=graph.graph['graph_identity'])
                row[view] = item
                if view == 'canonical':
                    save(directory / 'action-points.json', graph.graph['action_point_coverage'])
                    row['provenance'] = graph.graph['provenance']
            shutil.copy2(root / 'raw_mlir' / (path.stem + '.mlir'), directory / 'module.mlir')
            if h.file_hash(directory / 'module.mlir') != row['provenance']['mlir_sha256']:
                raise ValueError('Raw MLIR differs from compiler provenance')
            manifest['snapshots'].append(row)
        jobs = [(str(root), path.name, target, clock) for path in paths for target in TARGETS for clock in CLOCKS]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                 initargs=(str(CATALOG),)) as pool:
            futures = {pool.submit(materialize, job): job for job in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    manifest['contexts'].append(future.result())
                except Exception as exc:
                    manifest['failures'].append({'job': list(futures[future][1:]), 'reason': str(exc)})
                if i % 12 == 0 or i == len(jobs):
                    print(f'{i}/{len(jobs)} contexts completed; failures={len(manifest["failures"])}', flush=True)
    if manifest['failures']:
        save(CATALOG / 'validation/failed-build.json', manifest)
        raise ValueError(f'{len(manifest["failures"])} contexts failed; nothing is silently skipped')
    if h.file_hash(ARCHIVE) != original_archive_hash:
        raise ValueError('Historical archive changed')
    manifest['contexts'].sort(key=lambda c: (c['snapshot'], c['target'], c['clock_period_ns']))
    save(CATALOG / 'manifest.json', manifest)
    save(CATALOG / 'validation/catalog.json', {'snapshots': len(paths), 'contexts_passed': len(jobs),
        'program_graphs': len(paths)*2, 'enriched_graphs': len(jobs)*2, 'failed': 0, 'blocked': 0,
        'runtime_seconds': time.perf_counter() - began, 'implementation': h.implementation_identity(),
        'archive_unchanged': True, 'independent_adjudication': 'pending', 'synthesis_or_training': False,
        'checks': ['archive and all 538 release members authenticated', 'canonical/compact lossless reconstruction',
                   'original program restored and fingerprint validated in every enrichment',
                   'all overlays independently rederived with existing validator',
                   'canonical/compact supported overlay semantics equal in every context',
                   'repeat serialization identical', 'all gzip payloads round trip', 'raw MLIR provenance verified']})
    pages(manifest)
    checksums(CATALOG)


def checksums(root):
    rows = [f'{digest(p.read_bytes())}  {p.relative_to(root).as_posix()}\n'
            for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS']
    (root / 'SHA256SUMS').write_text(''.join(rows))


def verify(root):
    listed = set()
    for row in (root / 'SHA256SUMS').read_text().splitlines():
        expected, relative = row.split('  ', 1)
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or relative in listed:
            raise ValueError('Unsafe or duplicate checksum path')
        if digest(path.read_bytes()) != expected:
            raise ValueError(f'Checksum mismatch: {relative}')
        listed.add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    if listed != actual:
        raise ValueError('Missing or unlisted catalog files')
    print(f'PASS: {len(listed)} files authenticated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('build')
    command.add_argument('--output', default=str(CATALOG))
    command.add_argument('--workers', type=int, default=8)
    for mode in ('verify', 'unpack'):
        command = sub.add_parser(mode)
        command.add_argument('--catalog', type=Path, default=CATALOG)
        if mode == 'unpack':
            command.add_argument('--kernel', required=True)
            command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args)
    elif args.command == 'verify':
        verify(args.catalog)
    else:
        verify(args.catalog)
        source = args.catalog / 'kernels' / args.kernel
        if not source.is_dir() or source.parent.resolve() != (args.catalog / 'kernels').resolve():
            raise ValueError('Select an exact snapshot name from the index')
        if args.output.exists():
            raise FileExistsError('Unpack into a fresh output directory')
        shutil.copytree(source, args.output)
        for path in args.output.rglob('*.gz'):
            destination = path.with_suffix('')
            destination.write_bytes(gzip.decompress(path.read_bytes()))
            path.unlink()
        print(f'Unpacked {args.kernel} into {args.output}')


if __name__ == '__main__':
    main()
