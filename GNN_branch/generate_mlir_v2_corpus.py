"""Resumable corpus orchestration for the existing, explicitly selected V2.1 builder.

This module does not compile or interpret MLIR itself. Its inventory has no QoR
inputs; all compiler and relationship checks delegate to the existing pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from GNN_branch import mlir_graph_v2 as v2
from GNN_branch.generate_mlir_dataset import load_all_kernel_names, load_application_rows
from GNN_branch.action_point_relations import action_point_views, get_loop_semantics
from experiments.refinement import configurations as c
from experiments.refinement.relations import compiler_tools

DEFAULT_OUTPUT = c.ROOT / 'GNN_branch/MLIR_graphs/v21_corpus'
DEFAULT_REUSE = c.ROOT / 'experiments/structural_audit/results/graph_v21/validated/graphs'
BUILD = str(Path.home() / 'tools/Polygeist/build-mailohls-assertions')
SPLITS = ('artifacts/reproducibility/splits/stage1_stage2_family_split_s123.json',
          'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/experiment_split.json',
          'mailohls_runs/mailohls_final_family_split_s123.json')


def headers(source):
    return {p.name: c.sha256(p) for p in sorted(Path(source).parent.glob('*.h'))}


def inventory(root=c.ROOT):
    """Read the authoritative 55-kernel list through AST, never config imports.

    Metadata copies with equal content collapse; different metadata stays a
    different snapshot even when the C/C++ source is byte-identical.
    """
    root = Path(root)
    names = load_all_kernel_names(root / 'GNN_branch/config.py')
    applications = load_application_rows(root / 'Data/ApplicationInformation.csv')
    memberships, split_hashes = {}, {}
    for filename in SPLITS:
        path = root / filename
        if not path.is_file():
            continue
        split_hashes[filename] = c.sha256(path)
        data = json.loads(path.read_text())
        for split in ('train', 'val', 'test'):
            for name in data.get(split + '_kernels', []):
                memberships.setdefault(name, {})[filename] = split
    discovered = []
    for name in names:
        row = applications[name]
        directory = root / 'Data/ApplicationDataset' / name
        discovered.append((name, directory / row.file_name, directory / 'kernel_info.txt',
                           row.top_level_function, 'development', memberships.get(name, {})))
    for directory in sorted((root / 'evaluation_set').iterdir()):
        if not directory.is_dir():
            continue
        metadata = sorted(directory.rglob('kernel_info.txt'))
        # Generated placeholder variants are not original source snapshots.
        sources = sorted(p for p in directory.iterdir() if p.suffix in ('.c', '.cpp', '.cc')
                         and 'placeholder' not in p.name)
        if not metadata:
            continue
        for meta in metadata:
            top = meta.read_text().splitlines()[0].strip()
            source = sources[0] if len(sources) == 1 else None
            discovered.append((directory.name, source, meta, top, 'external',
                               memberships.get(directory.name, {})))
    snapshots = {}
    for name, source, meta, top, origin, split in discovered:
        error = None
        try:
            if source is None:
                raise ValueError('Source selection is ambiguous; explicit source mapping required')
            if meta.read_text().splitlines()[0].strip() != top:
                raise ValueError('Inventory top and action metadata disagree')
            source_hash, meta_hash, local_headers = c.sha256(source), c.sha256(meta), headers(source)
            # Validate the existing source/action binding contract without native imports.
            from GNN_branch.mlir_graph_gen import load_kernel_info_actions
            actual_top, _ = load_kernel_info_actions(source.resolve(), meta.resolve())
            if actual_top != top:
                raise ValueError('Source/action top mismatch')
            identity = c.object_hash([source_hash, meta_hash, local_headers, top])
        except (OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
            source_hash = meta_hash = None; local_headers = {}
            identity = c.object_hash([str(source), str(meta), top, error])
        alias = {'kernel': name, 'family':c.mc.family_id_from_kernel_name(name),
                 'origin': origin, 'source': str(source) if source else None,
                 'metadata': str(meta), 'split_membership': split or {'status': 'unidentified'}}
        if identity in snapshots:
            snapshots[identity]['aliases'].append(alias)
            continue
        snapshots[identity] = {'snapshot_id': identity, 'kernel': name, 'top': top,
            'source': str(source) if source else None, 'metadata': str(meta),
            'source_sha256': source_hash, 'metadata_sha256': meta_hash,
            'local_header_sha256': local_headers, 'aliases': [alias], 'inventory_error': error}
    return {'development_kernel_count': len(names), 'split_sha256': split_hashes,
            'split_note':'The stage1_stage2_family_split_s123 artifact has JSONL row indices only; kernel membership is recorded from separately saved kernel-name split manifests, not guessed from those indices.',
            'snapshots': list(snapshots.values())}


def authenticate(path, snapshot, tools):
    graph = v2.read_graph(path)
    if graph.graph['schema'] != v2.SCHEMA_V21 or graph.graph.get('view') != 'canonical':
        raise ValueError('A canonical V2.1 artifact is required')
    v2.validate(graph)
    provenance = graph.graph['provenance']
    expected = {key: snapshot[key] for key in ('source_sha256', 'metadata_sha256', 'local_header_sha256')}
    expected.update(analysis_extension_sha256=tools['analysis_extension_sha256'],
                    v1_generator_sha256=c.sha256(c.ROOT / 'GNN_branch/mlir_graph_gen.py'),
                    v2_builder_sha256=c.sha256(c.ROOT / 'GNN_branch/mlir_graph_v2.py'),
                    semantic_annotation_sha256=c.sha256(c.ROOT / 'GNN_branch/mlir_semantics.py'))
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError('Graph provenance mismatch: ' + key)
    frontend = provenance.get('v1_provenance', provenance)
    if frontend.get('cgeist_sha256') != tools['cgeist_sha256']:
        raise ValueError('Graph frontend build mismatch')
    if frontend.get('kernel', snapshot['top']) != snapshot['top']:
        raise ValueError('Graph top mismatch')
    return graph


def checked_views(graph, compact=None):
    """Check the existing validators and all public semantic queries, not counts."""
    from experiments.structural_audit.compare_v2 import _relations
    v2.validate(graph)
    compact = v2.compact_view(graph) if compact is None else compact
    v2.validate(compact)
    if v2.graph_payload(v2.expand_compact(compact)) != v2.graph_payload(graph):
        raise ValueError('Canonical reconstruction differs')
    relations = _relations(graph)
    if relations != _relations(compact) or action_point_views(graph) != action_point_views(compact):
        raise ValueError('Canonical/compact action-point queries differ')
    for label in relations:
        if graph.nodes['action:' + label]['action_kind'] == 'loop':
            if get_loop_semantics(graph, label) != get_loop_semantics(compact, label):
                raise ValueError('Canonical/compact loop semantics differ: ' + label)
    return compact, relations


def coverage(graph, relations):
    from experiments.structural_audit.compare_v2 import semantic_counts
    counts = semantic_counts(graph)
    records = graph.graph['action_point_coverage']['records']
    # Source labels missing from metadata are not failed compiler mappings.
    if isinstance(records, dict):
        records = list(records.values())
    active = sorted(relations, key=lambda s: int(s[1:]))
    edges = list(graph.edges(data=True))
    counts.update(registered=sum(bool(r['metadata_registered']) for r in records),
        compiler_active=len(active), source_only=sum(r['source_labeled'] and not r['metadata_registered'] for r in records),
        compiler_unresolved=sum(r['metadata_registered'] and not r['compiler_matched'] for r in records),
        maximum_action_index=max((int(label[1:]) for label in active), default=0),
        active_action_points=active,
        # Current production memory contract is absolute Lk slots, not active count.
        exceeds_production_64_slots=any(int(label[1:]) > 64 for label in active),
        concrete_access_roles=dict(Counter(e.get('access') for _, _, e in edges if e['relation_type']=='memory_access')),
        views=sum(e['relation_type']=='references' for _, _, e in edges),
        aliases=sum(e['relation_type']=='alias' for _, _, e in edges),
        query_relations=dict(Counter(r['relation_type'] + ':' + r['evidence_status'] for rows in relations.values() for r in rows)),
        no_verified_neighbors=[label for label, rows in relations.items() if not any(
            r['evidence_status']=='proven' and r.get('neighbor') != label for r in rows)])
    return counts


def process(snapshot, output, tools, reuse_paths=(), previous=None, timeout=600):
    started = time.monotonic()
    record = dict(snapshot)
    key = snapshot['kernel'] + '-' + snapshot['snapshot_id'][:12]
    canonical = output / 'canonical' / (key + '.json')
    compact_path = output / 'compact' / (key + '.json')
    try:
        if snapshot['inventory_error']:
            raise ValueError(snapshot['inventory_error'])
        candidates = [canonical] if canonical.is_file() else []
        candidates += [Path(p) for p in reuse_paths]
        graph = None; rejected = []
        for path in candidates:
            try:
                if previous and path == canonical and c.sha256(path) != previous['canonical_sha256']:
                    raise ValueError('Completed graph changed after manifest reservation')
                candidate = authenticate(path, snapshot, tools)
                graph = candidate; record['reused_from'] = str(path)
                record['status'] = 'authenticated_reuse'
                break
            except (ValueError, KeyError, OSError) as exc:
                # Most cache files belong to other snapshots: do not log those as defects.
                if path == canonical:
                    raise ValueError('invalid_artifact: ' + str(exc)) from exc
                rejected.append(str(exc))
        if graph is None:
            raw = output / 'raw_mlir' / (key + '.mlir')
            command = [tools['python'], '-B', '-m', 'GNN_branch.mlir_graph_v2', '--revision', '2.1',
                '--source', snapshot['source'], '--metadata', snapshot['metadata'],
                '--cgeist', tools['cgeist'], '--mlir-output', str(raw), '--output', str(canonical)]
            if snapshot['kernel'] in c.SPECS and c.SPECS[snapshot['kernel']].get('compiler_symbol'):
                command += ['--compiler-symbol', c.SPECS[snapshot['kernel']]['compiler_symbol']]
            record['command'] = command
            env = {**os.environ, 'PYTHONPATH': tools['pythonpath'], 'PYTHONHASHSEED': '0'}
            result = subprocess.run(command, cwd=c.ROOT, env=env, capture_output=True, text=True, timeout=timeout)
            log = output / 'logs' / (key + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(result.stdout + '\n' + result.stderr)
            record['log'] = str(log)
            if result.returncode:
                raise RuntimeError('Native builder failed: ' + result.stderr[-2500:])
            graph = authenticate(canonical, snapshot, tools)
            record.update(status='new_generation', raw_mlir=str(raw), raw_mlir_sha256=c.sha256(raw))
        compact, relations = checked_views(graph)
        v2.write_graph(graph, canonical); v2.write_graph(compact, compact_path)
        record.update(canonical=str(canonical), compact=str(compact_path), schema=graph.graph['schema'],
            canonical_sha256=c.sha256(canonical), compact_sha256=c.sha256(compact_path),
            canonical_nodes=len(graph), canonical_edges=graph.number_of_edges(),
            compact_nodes=len(compact), compact_edges=compact.number_of_edges(),
            serialized_bytes=canonical.stat().st_size, compact_bytes=compact_path.stat().st_size,
            validation='pass', reconstruction='exact', semantic_queries='identical',
            coverage=coverage(graph, relations), numerical_fidelity=graph.graph.get('numerical_fidelity'),
            header_limit=graph.graph['provenance'].get('header_limit'))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        record.update(status='invalid_artifact' if str(exc).startswith('invalid_artifact:') else 'blocked',
                      error=str(exc), validation='not_completed')
    record['elapsed_seconds'] = time.monotonic() - started
    return record


def csv_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def run(args):
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        found = inventory()
        tools = compiler_tools(args.build, args.native_python)
        code = {c.display_path(p): c.sha256(p) for p in [Path(__file__),
            c.ROOT/'GNN_branch/mlir_graph_v2.py', c.ROOT/'GNN_branch/mlir_graph_gen.py',
            c.ROOT/'GNN_branch/mlir_semantics.py', c.ROOT/'GNN_branch/action_point_relations.py']}
        contract = {'inventory': found, 'toolchain': tools, 'code_sha256': code, 'schema': v2.SCHEMA_V21}
        manifest_path = output / 'corpus_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {**contract, 'records': {}}
        if any(manifest[k] != value for k, value in contract.items()):
            raise ValueError('Frozen corpus inputs/code/toolchain changed; select a fresh output directory')
        reused = sorted(Path(args.reuse_dir).glob('*.json'))
        selected = set(args.only or [])
        for snapshot in found['snapshots']:
            if selected and not selected.intersection(a['kernel'] for a in snapshot['aliases']):
                continue
            key = snapshot['snapshot_id']; previous = manifest['records'].get(key)
            if previous and previous['status'] in ('blocked', 'invalid_artifact') and not args.retry_blocked:
                continue
            record = process(snapshot, output, tools, reused, previous, args.timeout)
            if previous and previous.get('validation') == 'pass' and record.get('validation') == 'pass':
                record = {**previous, 'resume_validation': 'pass'}
            manifest['records'][key] = record
            c.write_json(manifest_path, manifest)
            print(snapshot['kernel'], record['status'], record.get('error', '')[:200], flush=True)
        rows = list(manifest['records'].values())
        csv_rows(output/'generation_summary.csv', [{k:v for k,v in r.items() if k not in ('coverage','aliases')} for r in rows])
        csv_rows(output/'graph_coverage.csv', [{'kernel':r['kernel'], 'snapshot_id':r['snapshot_id'],
            'status':r['status'], **{k:r.get(k) for k in ('canonical_nodes','canonical_edges','compact_nodes','compact_edges','elapsed_seconds','semantic_queries')},
            **{k:v for k,v in r.get('coverage',{}).items() if k in ('registered','compiler_active','source_only','compiler_unresolved','maximum_action_index','exceeds_production_64_slots')}} for r in rows])
        csv_rows(output/'relation_coverage.csv', [{'kernel':r['kernel'], 'snapshot_id':r['snapshot_id'],
            **r.get('coverage',{})} for r in rows])
        families = defaultdict(Counter)
        for row in rows:
            for family in {a['family'] for a in row['aliases']}:
                families[family].update(snapshots=1, validated=int(row.get('validation')=='pass'),
                    active_action_points=row.get('coverage',{}).get('compiler_active',0))
        manifest['family_summary'] = {k:dict(v) for k,v in sorted(families.items())}
        c.write_json(manifest_path,manifest)
        print(json.dumps(dict(Counter(r['status'] for r in rows)), sort_keys=True))
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--reuse-dir', default=str(DEFAULT_REUSE))
    parser.add_argument('--build', default=BUILD)
    parser.add_argument('--native-python', default=str(Path.home()/'.mlir-python311/bin/python'))
    parser.add_argument('--only', nargs='+'); parser.add_argument('--retry-blocked', action='store_true')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args(argv)
    if args.timeout <= 0: parser.error('--timeout must be positive')
    run(args)


if __name__ == '__main__':
    main()
