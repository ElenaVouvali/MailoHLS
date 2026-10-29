"""Audit affine tensor and native-dependence coverage in authenticated RC3 graphs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from GNN_branch import semantic_tensor_features_v22 as semantic
from GNN_branch import target_graph_v22 as overlay

ROOT = Path(__file__).resolve().parents[2]
REPO_MANIFEST = ROOT / 'docs/graph-release-v22-rc3.json'


def _raw_affine_shape(text):
    if not isinstance(text, str):
        return None, None
    match = re.fullmatch(r'\s*affine_map<\((.*?)\)(?:\[(.*?)\])?\s*->\s*\((.*)\)>\s*', text)
    if not match:
        return None, None
    inputs = len([x for x in match.group(1).split(',') if x.strip()])
    inputs += len([x for x in (match.group(2) or '').split(',') if x.strip()])
    results = semantic._split_top_level(match.group(3))
    return inputs, len(results) if results is not None else None


def audit(release_dir):
    release_dir = Path(release_dir)
    manifest = json.loads(REPO_MANIFEST.read_text())
    archive = ROOT / manifest['archive']
    if overlay.file_hash(archive) != manifest['archive_sha256']:
        raise ValueError('RC3 archive SHA256 differs from graph release manifest')
    extracted = json.loads((release_dir / 'release.json').read_text())
    if extracted.get('schema') != 'salt-hls-mlir-structural-v2.2' or extracted.get('graph_count') != 63:
        raise ValueError('Expected authenticated 63-snapshot V2.2 release')
    if extracted.get('files_sha256') != manifest.get('files_sha256'):
        raise ValueError('Extracted archive member map differs from repository release manifest')
    for member, expected in manifest['files_sha256'].items():
        path = release_dir / member
        if not path.is_file() or overlay.file_hash(path) != expected:
            raise ValueError('Release member failed SHA256 verification: ' + member)

    snapshots = []
    for path in sorted((release_dir / 'canonical').glob('*.json')):
        graph, authentication = overlay.authenticated_input(path)
        rows = defaultdict(list)
        edge_rows = list(graph.edges(keys=True, data=True))
        for source, target, key, attrs in edge_rows:
            if attrs.get('relation_type') == 'memory_access':
                access_id = (source, attrs.get('operand_index'))
                rows[access_id].append((target, str(key), attrs))

        action_records = {
            row['action_id']: row for row in graph.graph['action_point_coverage']['records']
            if row.get('metadata_registered') and row.get('compiler_matched')
        }
        action_targets = {
            graph.nodes[source]['action_id']: target
            for source, target, _, attrs in edge_rows
            if attrs.get('relation_type') == 'targets'
            and graph.nodes[source].get('category') == 'action_point'
        }
        loop_actions, array_actions = defaultdict(set), defaultdict(set)
        for label, record in action_records.items():
            target = action_targets.get(label)
            if target is None:
                continue
            kind = (record.get('source_binding') or {}).get('kind')
            if kind == 'loop':
                for operation, attrs in graph.nodes(data=True):
                    if attrs.get('category') == 'operation' and (
                        operation == target or target in attrs.get('loop_context', ())
                    ):
                        loop_actions[operation].add(label)
            elif kind == 'array':
                for source, target_node, _, attrs in edge_rows:
                    if attrs.get('relation_type') == 'memory_access' and target in (source, target_node):
                        array_actions[source].add(label)

        bindings, maps = semantic._index_context(graph)
        affine = Counter()
        maximum_inputs = maximum_results = 0
        per_action = defaultdict(Counter)
        for (operation, operand_index), witnesses in rows.items():
            affine['unique_memory_access_operations'] += 1
            # A valid access op may have multiple memory operands. Count that as
            # one access only when operand identity is identical across roots.
            infos = [row[2].get('access_indices') or {} for row in witnesses]
            signatures = {json.dumps(info, sort_keys=True) for info in infos}
            info = infos[0]
            if len(signatures) > 1:
                affine['conflicting_or_incomplete_root_edge_metadata'] += 1
                info = {}
            text = info.get('affine_map')
            is_recorded = info.get('status') == 'compiler_recorded' and isinstance(text, str) and bool(text)
            if not is_recorded:
                affine['missing_compiler_affine_evidence'] += 1
                status = 'missing_compiler_evidence'
                n_inputs = n_results = None
                parsed = None
            else:
                affine['affine_map_recorded'] += 1
                n_inputs, n_results = _raw_affine_shape(text)
                maximum_inputs = max(maximum_inputs, n_inputs or 0)
                maximum_results = max(maximum_results, n_results or 0)
                parsed = semantic.parse_linear_affine_map(text)
                # The historical V3 tensorizer cap was four; count this
                # independently after V4 lifts it without widening edge rows.
                if n_inputs is not None and n_inputs > 4:
                    affine['maps_over_four_inputs'] += 1
                if n_results is not None and n_results > semantic.MAX_AFFINE_RESULTS:
                    affine['maps_over_three_results'] += 1
                quasi = set(re.findall(r'\b(mod|floordiv|ceildiv)\b', text))
                for operator in quasi:
                    affine['maps_using_' + operator] += 1
                if parsed is None:
                    affine['parser_unsupported'] += 1
                    affine['quasi_affine_unsupported' if quasi else 'other_expression_unsupported'] += 1
                    status = 'unsupported_quasi_affine' if quasi else 'unsupported_expression'
                else:
                    affine['additive_maps_parsed'] += 1
                    verified = semantic._validated_map_inputs(graph, operation, info)
                    if verified is None:
                        affine['verified_ssa_operand_bindings_failed'] += 1
                        status = 'binding_incomplete_or_conflicting'
                    else:
                        affine['verified_ssa_operand_bindings'] += 1
                        map_row = maps.get(operation)
                        if map_row and map_row.get('conflict'):
                            affine['conflicting_map_witnesses'] += 1
                            status = 'map_witness_conflict'
                        elif map_row and map_row.get('supported'):
                            affine['supported_by_current_tensorizer'] += 1
                            status = 'current_tensor_supported'
                        else:
                            affine['currently_overflow_limited'] += 1
                            status = 'current_capacity_limit'

            related = loop_actions.get(operation, set()) | array_actions.get(operation, set())
            for label in related:
                per_action[label]['unique_accesses'] += 1
                per_action[label][status] += 1
                per_action[label]['input_count'] += n_inputs or 0
                per_action[label]['result_count'] += n_results or 0

        views = []
        for _, _, _, attrs in edge_rows:
            mapping = attrs.get('view_mapping')
            if isinstance(mapping, dict):
                arrays = [mapping.get(name) for name in ('offsets', 'sizes', 'strides')]
                encoded_rank = max([len(array) for array in arrays if isinstance(array, list)] + [0])
                source_shape, result_shape = mapping.get('source_shape'), mapping.get('result_shape')
                source_rank = mapping.get('source_rank')
                result_rank = mapping.get('result_rank')
                if source_rank is None and isinstance(source_shape, list):
                    source_rank = len(source_shape)
                if result_rank is None and isinstance(result_shape, list):
                    result_rank = len(result_shape)
                shape_rank = max([r for r in (source_rank, result_rank) if type(r) is int] + [0])
                views.append((mapping, encoded_rank, shape_rank))

        native = graph.graph.get('compiler_analysis', {})
        dependences = native.get('dependences', [])
        dep_counts = Counter(row.get('result', 'missing') for row in dependences)
        dep_reasons = Counter(row.get('reason', 'missing') for row in dependences)
        dep_edges = [(u, v, key, attrs) for u, v, key, attrs in edge_rows
                     if attrs.get('relation_type') == 'depends_on']
        depth_histogram = Counter()
        multi_depth = []
        for _, _, _, attrs in dep_edges:
            depth_count = len(attrs.get('common_affine_loops') or [])
            depth_histogram[depth_count] += 1
            if depth_count > 1:
                multi_depth.append(attrs)
        unchecked = sum(
            max(0, len(attrs.get('common_affine_loops') or []) + 1 - attrs.get('dependence_depth', 0))
            for attrs in multi_depth
            if attrs.get('evidence_status') == 'proven' and attrs.get('dependence_depth') is not None
        )
        dep_action_ids = defaultdict(set)
        for source, target, key, _ in dep_edges:
            for label, record in action_records.items():
                if (record.get('source_binding') or {}).get('kind') != 'loop':
                    continue
                action_target = action_targets.get(label)
                if action_target is not None and any(
                    operation == action_target or action_target in graph.nodes[operation].get('loop_context', ())
                    for operation in (source, target)
                ):
                    dep_action_ids[label].add(str(key))

        stem = path.stem
        kernel = stem.rsplit('-', 1)[0]
        snapshots.append({
            'snapshot':path.name, 'kernel':kernel, 'graph_identity':graph.graph['graph_identity'],
            'sha256':authentication['file_sha256'], 'nodes':graph.number_of_nodes(),
            'edges':graph.number_of_edges(), 'affine_accesses':dict(affine),
            'maximum_affine_inputs':maximum_inputs, 'maximum_affine_results':maximum_results,
            'action_accesses':{label:dict(counts) for label, counts in sorted(per_action.items())},
            'registered_matched_actions':sorted(action_records),
            'view_count':len(views),
            'views_with_any_static_mapping_vector':sum(
                any(isinstance(mapping.get(name), list) for name in ('offsets', 'sizes', 'strides'))
                for mapping, _, _ in views),
            'view_encoded_dimension_overflow_count':sum(rank > semantic.MAX_VIEW_DIMS for _, rank, _ in views),
            'maximum_encoded_view_dimensions':max([rank for _, rank, _ in views] + [0]),
            'maximum_view_shape_rank':max([rank for _, _, rank in views] + [0]),
            'view_shape_rank_over_four_count':sum(rank > 4 for _, _, rank in views),
            'dependence_result_counts':dict(dep_counts), 'dependence_reason_counts':dict(dep_reasons),
            'dependence_edge_count':len(dep_edges), 'common_loop_depth_histogram':dict(depth_histogram),
            'multi_common_depth_edge_count':len(multi_depth),
            'proven_multi_depth_unchecked_query_estimate':unchecked,
            'dependence_action_edge_counts':{label:len(keys) for label, keys in sorted(dep_action_ids.items())},
        })
    return {
        'schema':'salt-hls-v22-semantic-coverage-audit-v1',
        'graph_archive_sha256':manifest['archive_sha256'],
        'authenticated_snapshots':len(snapshots), 'snapshots':snapshots,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.release_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'snapshots':result['authenticated_snapshots'], 'output':str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
