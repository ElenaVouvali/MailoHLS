"""Audit source bindings, legacy projection and saved model interfaces on CPU."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import networkx as nx
import torch

from experiments.refinement import configurations as c, relations as legacy
from LLM_branch.common import directive_domains as domains
from LLM_branch.common.structural_memory import resolve_relation_mask

HERE = Path(__file__).resolve().parent
DEFAULT_SHARED = c.ROOT / 'experiments/refinement/results/search/pilot/shared/manifest.json'
ORIGINALS = ('spcl_example_00', 'spcl_example_01', 'machsuite-gemm-ncubed',
             'machsuite-spmv-ellpack', 'rodinia_lud_1_tiling_0')
VERIFIER_POLICY = 'strict_callsite_and_access_v1'
BITS = {'loop_array_access': 8, 'nearest_child_loop': 4, 'dependence_forward': 16}


def strict_verify_pair(graph, array, loop, original=None, artifact_hash=None):
    """Require an explicit call ID AND observed operand/callee/position agreement.

    The frozen verifier's positional fallback remains available only as a
    diagnostic. Removing unsupported binding edges in a graph copy isolates this
    stricter policy without editing legacy code, graphs, or model inputs.
    """
    filtered = graph.copy()
    rejected, access_adaptations = [], []
    array_anchor = legacy._anchor(graph, array, 'array')
    relevant_nodes = {v for _, v, e in graph.out_edges(array_anchor, data=True) if int(e.get('flow', -1)) == 7}
    # Native writes intentionally use operation -> root. The old verifier only
    # recognizes root -> operation with the forward role. Adapt this orientation
    # only when both native directed edges agree. Calls are summaries, not direct
    # accesses; context-insensitive roots from another function are also not proof.
    for u, v, key, edge in list(graph.edges(keys=True, data=True)):
        if int(edge.get('flow', -1)) != 10:
            continue
        root, operation = (v, u) if edge.get('role') in ('write', 'read_reverse', 'readwrite_reverse') else (u, v)
        op = graph.nodes[operation]
        summary = op.get('text') in ('func.call', 'func.call_indirect', 'llvm.call')
        cross_function = op.get('function') != graph.nodes[root].get('function')
        if summary or cross_function:
            filtered.remove_edge(u, v, key)
            if root in relevant_nodes or operation in relevant_nodes:
                access_adaptations.append({'source': u, 'target': v, 'edge': dict(edge),
                    'decision': 'exclude_call_summary_or_cross_function_root_as_direct_proof'})
        elif edge.get('role') == 'write_reverse':
            forward = [e for _, target, e in graph.out_edges(operation, data=True)
                       if target == root and int(e.get('flow', -1)) == 10 and e.get('role') == 'write'
                       and e.get('operand_index') == edge.get('operand_index')]
            if forward:
                filtered.add_edge(root, operation, **{**edge, 'role': 'write', 'audit_normalized_write_orientation': True})
                if root in relevant_nodes or operation in relevant_nodes:
                    access_adaptations.append({'source': u, 'target': v, 'edge': dict(edge), 'forward_edges': forward,
                                               'decision': 'recognize_native_directed_write'})
    for actual, formal, key, edge in graph.edges(keys=True, data=True):
        if edge.get('role') != 'actual_to_formal' or int(edge.get('flow', -1)) != 2:
            continue
        call = str(edge.get('callsite', ''))
        operands = [] if call not in graph else [
            e for _, target, e in graph.out_edges(actual, data=True)
            if target == call and int(e.get('flow', -1)) == 1 and e.get('role') == 'operand'
        ]
        callee_functions = set() if call not in graph else {
            graph.nodes[target].get('function') for _, target, e in graph.out_edges(call, data=True)
            if int(e.get('flow', -1)) == 2 and e.get('role') == 'calls'
        }
        positions = {int(e['position']) for e in operands if 'position' in e}
        valid = (call in graph and graph.nodes[call].get('text') == 'func.call'
                 and graph.nodes[formal].get('function') in callee_functions
                 and len(positions) == 1 and int(edge.get('position', -1)) == 1 + next(iter(positions)))
        if not valid:
            rejected.append({'actual': str(actual), 'formal': str(formal), 'edge': dict(edge),
                             'reason': 'Missing or inconsistent explicit originating callsite/operand/callee binding'})
            filtered.remove_edge(actual, formal, key)
    result = legacy.verify_pair(filtered, array, loop, original, artifact_hash)
    previous = legacy.verify_pair(graph, array, loop, original, artifact_hash)
    result.update(verification_policy=VERIFIER_POLICY,
                  legacy_verifier_class=previous['verified_relation_class'],
                  legacy_supporting_operations=previous['supporting_mlir_operations'],
                  rejected_binding_count=len(rejected), rejected_binding_scope='entire_graph',
                  access_evidence_adaptations=access_adaptations)
    if result['verified_relation_class'] != previous['verified_relation_class']:
        result['reason'] += ' Audit uses explicit callsite proof and native directed access semantics; see adaptation evidence.'
    return result


def semantic_anchors(graph):
    found = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        if attrs.get('action_id') and (int(attrs.get('is_loop', 0)) == 1 or int(attrs.get('type', -1)) == 104):
            found[attrs['action_id']].append(str(node))
    return dict(found)


def project(graph, actions, slots=None):
    anchors = semantic_anchors(graph)
    slots = slots or max(int(label[1:]) for label in actions)
    ids, active = [-1] * slots, [False] * slots
    for label, nodes in anchors.items():
        if label not in actions or len(nodes) != 1:
            raise ValueError(f'Unknown or duplicated semantic anchor: {label}')
        index = int(label[1:]) - 1
        if index >= slots:
            raise ValueError('Active action would be truncated')
        ids[index], active[index] = int(nodes[0]), True
    mask, bits, stats = legacy.existing_projector()(graph, node_ids=ids, active_mask=active, max_slots=slots)
    return ids, active, mask, bits, stats


def alignment(graph, actions, source=None):
    anchors = semantic_anchors(graph)
    roots, bound, root_total = 0, 0, 0
    for label, nodes in anchors.items():
        if len(nodes) == 1:
            node = nodes[0]
            bound += graph.nodes[node].get('action_resolution') == 'source_location'
            if actions[label].kind == 'array':
                root_total += 1
                targets = [v for _, v, e in graph.out_edges(node, data=True)
                           if e.get('role') == 'array_root' and int(e.get('flow', -1)) == 7]
                roots += len(targets) == 1 and int(graph.nodes[targets[0]].get('is_memory_root', 0)) == 1
    bindings = [e for _, _, e in graph.edges(data=True) if e.get('role') == 'actual_to_formal']
    source_bindings = {}
    if source:
        lines = source.read_text().splitlines()
        for label, nodes in anchors.items():
            label_lines = [i + 1 for i, line in enumerate(lines) if re.search(r'\b' + label + r':', line)]
            line = graph.nodes[nodes[0]].get('action_source_line') if len(nodes) == 1 else None
            source_bindings[label] = {'label_lines': label_lines, 'compiler_action_source_line': line,
                'source_context': [{'line': i + 1, 'text': lines[i]} for i in range(max(0, int(line or 1) - 3), min(len(lines), int(line or 1) + 1))],
                'status': 'requires_review' if len(label_lines) != 1 or line is None or not 0 <= int(line) - label_lines[0] <= 2 else 'near_exact_source_label'}
    return {'source_actions': len(actions), 'semantic_anchors': sum(map(len, anchors.values())),
            'missing_actions': sorted(set(actions) - set(anchors)),
            'duplicate_actions': {k: v for k, v in anchors.items() if len(v) != 1},
            'exact_source_bindings': bound, 'array_anchors': root_total, 'resolved_memory_roots': roots,
            'actual_formal_bindings': len(bindings),
            'source_binding_checks': source_bindings,
            'bindings_with_explicit_callsite': sum('callsite' in e for e in bindings),
            'action_anchor_coverage': bound / len(actions) if actions else None,
            'memory_root_resolution_coverage': roots / root_total if root_total else None,
            'callsite_binding_coverage': sum('callsite' in e for e in bindings) / len(bindings) if bindings else None}


def encoder_helpers():
    """Execute only the production's pure feature functions, without its CLI/config.

    As with the existing projector adapter, this runs the actual function bodies;
    it does not replace their feature transformations with an approximation.
    """
    path = c.ROOT / 'GNN_branch/mlir_data.py'
    names = {'_as_int', '_as_float', '_as_bool01', '_signed_log1p', '_log1p_nonnegative', '_json_list',
             'node_kind', 'pragma_kind', 'canonical_op_family', 'canonical_type', '_memory_volume_log',
             'node_numeric_features', 'edge_role', 'edge_certainty', 'edge_numeric_features',
             'node_categorical_row', 'edge_categorical_row', '_node_id_order', '_iter_edges'}
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError('Production encoder changed; review pure-function adapter')
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    env = dict(constants, math=math, json=json, re=re, nx=nx, Any=Any, Mapping=Mapping)
    exec(compile(ast.Module(functions, type_ignores=[]), str(path), 'exec'), env)
    return env


def encoded_features(graph, schema):
    helpers = encoder_helpers()
    result, oov = {}, {}
    for family in ('node', 'edge'):
        attrs = [graph.nodes[n] for n in helpers['_node_id_order'](graph)[0]] if family == 'node' else [
            a for _, _, _, a in helpers['_iter_edges'](graph)]
        fields = schema[family + '_categorical_fields']
        categories = schema[family + '_categories']
        unknown, absent, rows = Counter(), Counter(), []
        unknown_examples = defaultdict(Counter)
        for a in attrs:
            row = []
            values = helpers[family + '_categorical_row'](a)
            for field, value in zip(fields, values):
                known = categories[field]
                if value not in known:
                    unknown[field] += 1
                    unknown_examples[field][value] += 1
                    value = '<unknown>'
                row.extend(float(v == value) for v in known)
            row.extend(helpers[family + '_numeric_features'](a))
            for field in fields:
                if field not in a:
                    absent[field] += 1
            rows.append(row)
        tensor = torch.tensor(rows, dtype=torch.float32)
        result[family] = tensor if family == 'node' else tensor.half()
        oov[family] = {'rows': len(rows), 'unknown_by_field': dict(unknown),
                       'unknown_value_examples': {field: counts.most_common(6) for field, counts in unknown_examples.items()},
                       'missing_raw_attributes': dict(absent),
                       'note': 'Node categorical fields are derived; missing raw names are not necessarily missing features'}
    edges = [[int(u), int(v)] for u, v, _, _ in helpers['_iter_edges'](graph)]
    result['edge_index'] = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return result, oov


def check_memory(pack, graph, actions, schema_hash, contract_hash):
    emb, active = pack['node_embs'], pack['node_embs_mask'].bool()
    if emb.ndim != 2 or tuple(active.shape) != (emb.size(0),) or emb.size(1) != int(pack['gnn_dim']):
        raise ValueError('Invalid structural-memory dimensions')
    if not torch.isfinite(emb).all() or pack['feature_schema_sha256'] != schema_hash or pack['gnn_contract_sha256'] != contract_hash:
        raise ValueError('Nonfinite memory or incompatible saved feature/contract hash')
    slots = emb.size(0)
    if int(pack['max_slots']) != slots or not torch.equal(pack['slot_ids'], torch.arange(1, slots + 1)):
        raise ValueError('Incorrect absolute slot IDs')
    anchors = semantic_anchors(graph)
    mapped = {}
    for i in range(slots):
        if active[i]:
            label = f'L{i + 1}'
            if int(pack['labels'][i]) != i + 1 or anchors.get(label) != [str(pack['node_ids'][i])]:
                raise ValueError('Sparse absolute action/anchor alignment mismatch')
            expected_kind = 1 if actions[label].kind == 'loop' else 2
            if int(pack['slot_cats'][i]) != expected_kind:
                raise ValueError('Action type/slot category mismatch')
            mapped[label] = int(pack['node_ids'][i])
        elif int(pack['labels'][i]) != -1 or int(pack['node_ids'][i]) != -1 or emb[i].count_nonzero():
            raise ValueError('Inactive slot has an identity or nonzero embedding')
    if set(mapped) != set(anchors):
        raise ValueError('Structural memory silently drops active anchors')
    mask, bits, _ = legacy.existing_projector()(graph, node_ids=pack['node_ids'], active_mask=active, max_slots=slots)
    saved_bits, saved_mask = pack['action_relation_bits'], pack['action_relation_mask']
    if not torch.equal(bits, saved_bits) or not torch.equal(mask, saved_mask):
        raise ValueError('Saved relation tensor does not reproduce from graph')
    if not torch.equal(saved_bits.ne(0), saved_mask):
        raise ValueError('Typed bits and Boolean projection disagree')
    if (saved_mask & ~(active[:, None] & active[None, :])).any():
        raise ValueError('Inactive memory slots are visible')
    if not torch.all(saved_bits.diagonal()[active].bitwise_and(1).bool()):
        raise ValueError('Active self visibility lost')
    routed = resolve_relation_mask({'relation_mask': saved_mask, 'relation_bits': saved_bits}, 'compiler_relational')
    if not torch.equal(routed, saved_bits.ne(0)):
        raise ValueError('Production routing differs from Boolean projection')
    return {'status': 'validated', 'shape': list(emb.shape), 'active_slots': int(active.sum()),
            'mapping': mapped, 'pragma_free': pack['disable_pragma_injection'],
            'typed_values': dict(Counter(map(int, saved_bits.flatten().tolist()))),
            'routing_dtype': str(routed.dtype), 'inactive_slots_excluded': True,
            'embedding_fact_recoverability': 'not_tested; dimensional/alignment checks do not prove learned semantic retention'}


def load_kernel(name, shared):
    if name in shared.get('kernels', {}):
        kernel = shared['kernels'][name]
        prov, record = kernel['source_provenance'], kernel['relations']
        if record['status'] != 'available':
            raise FileNotFoundError(record.get('reason', 'Missing graph'))
        source, metadata, path = map(c.repo_path, (prov['snapshot_source'], prov['snapshot_metadata'], record['artifact']))
        if c.sha256(source) != prov['source_sha256'] or c.sha256(metadata) != prov['metadata_sha256'] or c.sha256(path) != record['artifact_hash']:
            raise ValueError('Archived source/metadata/graph hash mismatch')
        for entry in prov.get('files', []):
            if c.sha256(c.repo_path(entry['snapshot'])) != entry['sha256']:
                raise ValueError('Archived source dependency hash mismatch')
        actions = domains.load_source_actions(name, metadata.parent)
        verified = legacy.graph_relations(path, c.sha256(source), c.sha256(metadata), actions,
            c.repo_path(record['compiler_active_metadata_path']) if record.get('compiler_active_metadata_path') else None)
        if verified['action_relation_bits'] != record['action_relation_bits']:
            raise ValueError('Archived projection differs from unchanged projector')
    else:
        path = c.ROOT / 'GNN_branch/MLIR_graphs' / (name + '.gexf')
        graph = nx.read_gexf(path)
        meta = json.loads(graph.graph['name'].split(':', 1)[1])
        directory = c.ROOT / 'Data/ApplicationDataset' / name
        metadata = directory / 'kernel_info.txt'
        matches = [p for p in directory.iterdir() if p.suffix in ('.cpp', '.c', '.cc') and c.sha256(p) == meta['source_sha256']]
        if len(matches) != 1 or c.sha256(metadata) != meta['action_sha256']:
            raise ValueError('Original kernel source/metadata cannot be authenticated')
        source = matches[0]
        actions = domains.load_source_actions(name, metadata.parent)
        verified = legacy.graph_relations(path, c.sha256(source), c.sha256(metadata), actions)
    graph = nx.read_gexf(path)
    return source, metadata, path, graph, actions, verified


def compare_annotations(annotations):
    """Unknowns are excluded, draft labels never become reviewed precision/recall."""
    groups = defaultdict(Counter)
    for row in annotations:
        key = (row['kernel'], row['relationship_type'])
        tally = groups[key]
        if row.get('expected') is None or row.get('observed') is None:
            tally['unresolved'] += 1
            continue
        reviewed = row.get('review_status') == 'independently_reviewed' and bool(row.get('reviewer'))
        outcome = ('TP' if row['observed'] else 'FN') if row['expected'] else ('FP' if row['observed'] else 'TN')
        tally[('' if reviewed else 'draft_') + outcome] += 1
    result = []
    for (kernel, relation), tally in sorted(groups.items()):
        counts = {k: tally[k] for k in ('TP', 'FP', 'FN', 'TN', 'draft_TP', 'draft_FP', 'draft_FN', 'draft_TN', 'unresolved')}
        result.append({'kernel': kernel, 'relationship_type': relation, **counts,
            'precision': tally['TP'] / (tally['TP'] + tally['FP']) if tally['TP'] + tally['FP'] else None,
            'recall': tally['TP'] / (tally['TP'] + tally['FN']) if tally['TP'] + tally['FN'] else None,
            'reference_status': 'reviewed_counts_only; draft counts are diagnostic, not validated accuracy'})
    return result


def annotate(spec, source, graph, actions, projected, artifact_hash):
    """Attach observations AFTER reading independent, source-written expectations."""
    row = dict(spec)
    row.setdefault('review_status', 'draft')
    row.setdefault('annotator', 'machine_assisted_source_reading')
    if row['review_status'] == 'independently_reviewed' and not row.get('reviewer'):
        raise ValueError('Reviewed reference requires an explicit reviewer')
    if spec.get('source_sha256') and spec['source_sha256'] != c.sha256(source):
        raise ValueError('Annotation source hash mismatch')
    lines = source.read_text().splitlines()
    evidence = []
    for quote in spec['source_quotes']:
        matches = [{'line': i + 1, 'text': line} for i, line in enumerate(lines) if quote in line]
        if not matches:
            raise ValueError(f'Annotation source quote absent: {quote}')
        evidence.append({'quote': quote, 'matches': matches})
    row.update(source=c.display_path(source), source_sha256=c.sha256(source), source_evidence=evidence,
               graph_sha256=artifact_hash)
    a, b = row['action_i'], row['action_j']
    if a not in actions or b not in actions:
        raise ValueError('Annotation uses unknown action identity')
    active = set(semantic_anchors(graph))
    if a not in active or b not in active:
        row.update(observed=None, status='inactive_or_missing_anchor')
        return row
    bits = projected['action_relation_bits'][int(a[1:]) - 1][int(b[1:]) - 1]
    row.update(observed=bool(bits & BITS[row['relationship_type']]), projected_bits=bits,
               status='compared', action_anchors={k: legacy._operation(graph, semantic_anchors(graph)[k][0]) for k in (a, b)})
    if {actions[a].kind, actions[b].kind} == {'array', 'loop'}:
        array, loop = (a, b) if actions[a].kind == 'array' else (b, a)
        original = next((p for p in projected['pairs'] if {p['action_i'], p['action_j']} == {a, b}), None)
        row['strict_verification'] = strict_verify_pair(graph, array, loop, original, artifact_hash)
    return row


def raw_artifacts(name, path, graph):
    meta = json.loads(graph.graph['name'].split(':', 1)[1])
    archive = c.ROOT / 'experiments/refinement/results/r0_local_headroom/graphs'
    possibilities = [path.with_suffix('.mlir'), archive / (name + '.mlir'),
                     c.ROOT / 'GNN_branch/MLIR_graphs/audit_mlir' / (name + '.mlir')]
    raw = next((p for p in possibilities if p.is_file() and c.sha256(p) == meta.get('mlir_sha256')), None)
    if raw is None:
        # Resolve display-name aliases by content, never by a guessed kernel name.
        candidates = sorted((c.ROOT / 'GNN_branch/MLIR_graphs/audit_mlir').glob('*.mlir'))
        raw = next((p for p in candidates if c.sha256(p) == meta.get('mlir_sha256')), None)
    analyses = [path.with_suffix('.analysis.json'), archive / (name + '.analysis.json')]
    analysis = next((p for p in analyses if p.is_file()), None)
    # Analysis records are additional observations, not authenticated just by name.
    analysis_verified = False
    if analysis:
        data = json.loads(analysis.read_text())
        analysis_verified = meta.get('mlir_sha256') in json.dumps(data)
    return {'raw_mlir': c.display_path(raw) if raw else None,
            'raw_mlir_status': 'hash_verified' if raw else 'unavailable; graph operation text is retained, not a substitute for raw IR',
            'analysis_record': c.display_path(analysis) if analysis else None,
            'analysis_binding_status': 'contains_matching_mlir_hash' if analysis_verified else 'unverified_or_unavailable'}


def projection_witness(graph, array_label, loop_label):
    """Expose the actual legacy call-reachability path, not a semantic proof."""
    array, loop = legacy._anchor(graph, array_label, 'array'), legacy._anchor(graph, loop_label, 'loop')
    allowed = nx.DiGraph()
    for u, v, edge in graph.edges(data=True):
        flow, role = int(edge.get('flow', -1)), edge.get('role')
        if (flow == 8 and role == 'region_entry' or flow == 0 and role in ('next_in_block', 'block_successor')
                or flow == 2 and role == 'calls'):
            allowed.add_edge(u, v, attributes=dict(edge))
    witnesses = []
    for _, access, edge in graph.out_edges(array, data=True):
        if int(edge.get('flow', -1)) != 7 or edge.get('role') not in ('array_read', 'array_write', 'array_readwrite'):
            continue
        if loop in allowed and access in allowed and nx.has_path(allowed, loop, access):
            nodes = nx.shortest_path(allowed, loop, access)
            witnesses.append({'array_access_edge': {'source': array, 'target': access, **dict(edge)},
                'region_path': [{'source': u, 'target': v, **allowed[u][v]['attributes']} for u, v in zip(nodes, nodes[1:])],
                'operations': [legacy._operation(graph, n) for n in nodes if graph.nodes[n].get('text') in ('func.call', 'func.func') or n == access]})
    return witnesses


def run_audit(args):
    output = c.repo_path(args.output_dir)
    shared_path = c.repo_path(args.shared_manifest)
    shared = json.loads(shared_path.read_text()) if shared_path.is_file() else {}
    protected = {c.display_path(p): c.sha256(p) for pattern in
                 ('**/manifest.json', '**/protocol.json', '**/*MEASUREMENTS.jsonl', '**/synthesis_results.jsonl', '**/search_trace.jsonl')
                 for p in (c.ROOT / 'experiments/refinement/results').glob(pattern)}
    specs = json.loads(c.repo_path(args.annotations).read_text())
    finals = json.loads((c.ROOT / 'FINAL_ARTIFACTS.json').read_text())
    checkpoint = c.repo_path(finals['gnn']['checkpoint'])
    contract_path = checkpoint.parent / 'gnn_checkpoint_contract.json'
    schema_path = checkpoint.parent / 'provenance/feature_schema.json'
    schema = json.loads(schema_path.read_text()) if schema_path.is_file() else {}
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    if schema and contract and contract['feature_schema']['sha256'] != c.sha256(schema_path):
        raise ValueError('Checkpoint schema provenance mismatch')
    memory_dir = c.repo_path(finals['gnn']['memory'])
    # External display names need not equal the original kernel identity:
    # Kalman is stored as serrano-kalman-filter. Match the exact graph hash,
    # rather than treating a similar name as proof of artifact compatibility.
    memory_by_graph = defaultdict(list)
    for memory_path in sorted(memory_dir.glob('*.memory.pt')):
        metadata_pack = torch.load(memory_path, map_location='cpu', weights_only=False)
        memory_by_graph[metadata_pack.get('source_gexf_sha256')].append(memory_path)
    stage2_contract = c.repo_path(finals['stage2']['PARETO_ADP']['path']) / 'training_contract.json'
    trace = {'saved_feature_schema': schema, 'checkpoint_configuration': contract,
             'saved_configuration_status': 'available' if schema and contract else 'blocked; checkpoint feature schema or contract unavailable; semantic audit continues',
             'stage2_training_contract': json.loads(stage2_contract.read_text()) if stage2_contract.is_file() else None,
             'kernels': {}, 'interface': interface_contract()}
    stage2 = trace['stage2_training_contract'] or {}
    trace['actual_consumer_settings'] = {
        'gnn': {k: contract.get('model_init_flags', {}).get(k) for k in ('gnn_type', 'encode_edge', 'encode_edge_position', 'D', 'num_layers')},
        'stage2_structural': stage2.get('structural'), 'gnn_model_init': contract.get('model_init')}
    files = [Path(__file__), HERE / '__init__.py', HERE / 'fixtures/semantics.json', c.repo_path(args.annotations), schema_path, contract_path,
             c.ROOT / 'FINAL_ARTIFACTS.json', *[c.ROOT / p for p in (
                 'GNN_branch/mlir_graph_gen.py', 'GNN_branch/mlir_data.py', 'GNN_branch/model.py',
                 'GNN_branch/build_structural_memory.py', 'LLM_branch/common/structural_memory.py',
                 'LLM_branch/common/structural_xattn.py', 'LLM_branch/train/train_SFT_xattn_new.py',
                 'experiments/refinement/relations.py', 'experiments/refinement_search/policies.py',
                 'experiments/refinement_search/controller.py', 'experiments/refinement_search/benchmark.py')]]
    files = [p for p in files if p.is_file()]
    files += [p for p in (shared_path, stage2_contract, checkpoint) if p.is_file()]
    fixture_results = output / 'fixtures/results.json'
    if fixture_results.is_file():
        fixture_manifest = output / 'fixtures/manifest.json'
        if not fixture_manifest.is_file() or json.loads(fixture_manifest.read_text()).get('audit_sha256') != c.sha256(Path(__file__)):
            raise ValueError('Fixture receipts belong to another audit implementation; preserve them and use a new output directory')
        trace['fixture_results'] = json.loads(fixture_results.read_text())
        files += [p for p in (fixture_results, output / 'fixtures/manifest.json') if p.is_file()]
    annotations, errors = [], []
    for name in dict.fromkeys(args.kernels + list(ORIGINALS if args.include_originals else ())):
        try:
            source, metadata, path, graph, actions, projected = load_kernel(name, shared)
        except (OSError, ValueError) as exc:
            trace['kernels'][name] = {'status': 'blocked', 'reason': str(exc)}
            errors.append({'kernel': name, 'kind': 'missing_or_invalid_artifact', 'reason': str(exc)})
            continue
        files.extend([source, metadata, path])
        headers = sorted(source.parent.glob('*.h')) + sorted(source.parent.glob('*.hpp'))
        files.extend(headers)
        header_context = [{'path': c.display_path(p), 'sha256': c.sha256(p),
                           'analysis_stub_marker': 'AP_FIXED_STUB' in p.read_text(errors='replace') or 'AP_INT_STUB' in p.read_text(errors='replace')}
                          for p in headers]
        active = {k: a for k, a in actions.items() if k not in projected['inactive_actions']}
        audited = [strict_verify_pair(graph, a, b, next((p for p in projected['pairs'] if {p['action_i'], p['action_j']} == {a, b}), None), c.sha256(path))
                   for a in active if active[a].kind == 'array' for b in active if active[b].kind == 'loop']
        audited += legacy.audit_loop_pairs(graph, active, projected['pairs'], c.sha256(path))
        encoded, oov = encoded_features(graph, schema) if schema else ({}, {'status': 'blocked; no saved schema'})
        record = {'status': 'audited', 'source': c.display_path(source), 'metadata': c.display_path(metadata),
                  'graph': c.display_path(path), 'graph_sha256': c.sha256(path),
                  'local_header_context': header_context,
                  'header_provenance_limit': 'Current/local snapshot hashes recorded; graph metadata does not seal the complete transitive include closure. Analysis AP stubs are not vendor arithmetic.',
                  'graph_metadata': projected['graph_metadata'], **raw_artifacts(name, path, graph),
                  'alignment': alignment(graph, actions, source), 'compiler_inactive_actions': projected['inactive_actions'],
                  'projected_nonself_pairs': len(projected['pairs']), 'verified_pair_classifications': dict(Counter(r['verified_relation_class'] for r in audited)),
                  'verified_relations': audited, 'feature_oov': oov,
                  'relation_bit_counts_directed': {label: int(projected_bits_count(projected, bit)) for bit, label in legacy.RELATIONS.items()},
                  'graph_analysis_coverage': {
                      'memory_access_roles': dict(Counter(e.get('role') for _, _, e in graph.edges(data=True) if int(e.get('flow', -1)) == 10)),
                      'dependence_certainties': dict(Counter(e.get('certainty', '<missing>') for _, _, e in graph.edges(data=True) if int(e.get('flow', -1)) == 12)),
                      'memory_view_edges': sum(int(e.get('flow', -1)) == 9 for _, _, e in graph.edges(data=True)),
                      'call_nodes': sum(v.get('text') == 'func.call' for _, v in graph.nodes(data=True))},
                  'encoded_dimensions': {k: list(encoded[k].shape) for k in ('node', 'edge') if k in encoded},
                  'saved_tensor': {'status': 'unavailable'}, 'memory': {'status': 'unavailable; no GNN encoding performed'}}
        for family in ('node', 'edge'):
            if encoded and encoded[family].shape[1] != schema[family + '_feature_dim']:
                raise ValueError('Encoded feature dimension differs from checkpoint schema')
        tensor_path = c.ROOT / 'GNN_branch/MLIR_dataset/all_kernels/graphs' / (name + '.pt')
        if not tensor_path.is_file():
            for memory_path in memory_by_graph.get(c.sha256(path), []):
                memory_metadata = torch.load(memory_path, map_location='cpu', weights_only=False)
                candidate_path = c.repo_path(memory_metadata['pt_path']) if memory_metadata.get('pt_path') else None
                if candidate_path and candidate_path.is_file():
                    tensor_path = candidate_path
                    break
        if tensor_path.is_file() and encoded:
            saved = torch.load(tensor_path, map_location='cpu', weights_only=False)
            matches = {key: torch.equal(saved[key], encoded[family]) for key, family in [('x', 'node'), ('edge_attr', 'edge'), ('edge_index', 'edge_index')]}
            record['saved_tensor'] = {'status': 'validated' if all(matches.values()) else 'mismatch', 'exact_tensor_matches': matches,
                                      'graph_provenance': saved.get('graph_provenance'), 'recorded_oov': saved.get('categorical_oov')}
            files.append(tensor_path)
            if not all(matches.values()):
                errors.append({'kernel': name, 'kind': 'encoded_tensor_mismatch', 'matches': matches})
        matching_packs = memory_by_graph.get(c.sha256(path), [])
        if len(matching_packs) > 1:
            raise ValueError('Ambiguous memory snapshots for the same graph; select one explicitly')
        pack_path = matching_packs[0] if matching_packs else memory_dir / (name + '.memory.pt')
        if pack_path.is_file() and schema and contract:
            pack = torch.load(pack_path, map_location='cpu', weights_only=False)
            files.append(pack_path)
            if pack['source_gexf_sha256'] != c.sha256(path):
                record['memory'] = {'status': 'different_graph_snapshot', 'saved_graph_sha256': pack['source_gexf_sha256'],
                                    'note': 'Not aligned against a different graph; historical pack preserved'}
            else:
                record['memory'] = check_memory(pack, graph, actions, c.sha256(schema_path), c.sha256(contract_path))
                record['memory']['artifact'] = c.display_path(pack_path)
                record['memory']['checkpoint_bytes_status'] = 'hash_verified' if checkpoint.is_file() else 'unavailable'
                if checkpoint.is_file() and pack['gnn_checkpoint_sha256'] != c.sha256(checkpoint):
                    raise ValueError('Memory checkpoint hash mismatch')
        legacy_memory = []
        if name in shared.get('kernels', {}):
            original_source = c.repo_path(shared['kernels'][name]['source_provenance']['source'])
            for old_path in sorted((original_source.parent / 'code_to_memory_outputs/memory').glob('*.memory.pt')):
                old_pack = torch.load(old_path, map_location='cpu', weights_only=False)
                required = ('source_gexf_sha256', 'gnn_checkpoint_sha256', 'feature_schema_sha256', 'gnn_contract_sha256')
                missing = [key for key in required if not old_pack.get(key)]
                legacy_memory.append({'artifact': c.display_path(old_path), 'shape': list(old_pack['node_embs'].shape),
                    'status': 'historical_unverified' if missing else 'requires_separate_snapshot_authentication',
                    'missing_provenance': missing,
                    'note': 'Not substituted for an authenticated current-checkpoint pack'})
                files.append(old_path)
        record['external_legacy_memory_inventory'] = legacy_memory
        for spec in specs:
            if spec['kernel'] == name:
                row = annotate(spec, source, graph, actions, projected, c.sha256(path))
                if 'strict_verification' in row:
                    row['legacy_projection_witnesses'] = projection_witness(graph,
                        row['strict_verification']['action_i'], row['strict_verification']['action_j'])
                raw_path = c.repo_path(record['raw_mlir']) if record['raw_mlir'] else None
                row['raw_mlir_evidence'] = []
                if raw_path:
                    raw_lines = raw_path.read_text().splitlines()
                    operations = list(row.get('action_anchors', {}).values()) + row.get('strict_verification', {}).get('supporting_mlir_operations', [])
                    for op in operations:
                        text = op.get('full_text', '').splitlines()[0].strip() if op.get('full_text') else ''
                        matches = [{'line': i + 1, 'text': s} for i, s in enumerate(raw_lines) if text and text in s]
                        row['raw_mlir_evidence'].append({'node': op['graph_node'], 'op_uid': op.get('op_uid'),
                            'raw_mlir': record['raw_mlir'], 'matches': matches,
                            'status': 'text_witness_not_unique_operation_identity' if matches else 'no_exact_text_match'})
                annotations.append(row)
                if row.get('observed') is not None and row['expected'] is not None and row['observed'] != row['expected']:
                    errors.append({'kind': 'reference_disagreement', 'kernel': name, 'annotation': row,
                                   'status': 'draft_finding_requires_independent_review' if row['review_status'] == 'draft' else 'reviewed_reference_disagreement'})
        trace['kernels'][name] = record
        files += [c.repo_path(record[k]) for k in ('raw_mlir', 'analysis_record') if record.get(k)]
    input_hashes = {c.display_path(p): c.sha256(p) for p in files}
    manifest = {'experiment': 'structural-representation-audit', 'verifier_policy': VERIFIER_POLICY,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip(),
                'worktree_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=c.ROOT, text=True),
                'python': sys.version, 'torch_version': torch.__version__, 'networkx_version': nx.__version__,
                'inputs_sha256': input_hashes,
                'protected_historical_sha256': protected,
                'reference_policy': 'Real annotations are drafts unless explicitly independently reviewed; no draft accuracy claims',
                'no_hls_no_training_no_model_inference': True}
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists() and json.loads((output / 'manifest.json').read_text())['inputs_sha256'] != input_hashes:
        raise ValueError('Audit inputs changed; use a new output directory to preserve the previous audit')
    if not (output / 'manifest.json').exists():
        c.write_json(output / 'manifest.json', manifest)
    c.write_jsonl(output / 'relationship_annotations.jsonl', annotations)
    comparisons = compare_annotations(annotations)
    with (output / 'relationship_comparison.csv').open('w') as stream:
        fields = list(comparisons[0]) if comparisons else ['kernel', 'relationship_type']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(comparisons)
    c.write_jsonl(output / 'relation_errors.jsonl', errors)
    c.write_json(output / 'representation_trace.json', trace)
    write_report(output, trace, annotations, comparisons, errors)
    if any(c.sha256(c.repo_path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError('An audit input changed during execution')
    if any(c.sha256(c.repo_path(path)) != digest for path, digest in protected.items()):
        raise ValueError('A protected historical experiment changed during execution')
    print(json.dumps({'output': c.display_path(output), 'annotations': len(annotations),
                      'kernels': {k: v['status'] for k, v in trace['kernels'].items()}, 'findings': len(errors)}, indent=2))
    return trace, annotations


def projected_bits_count(projected, bit):
    return sum(bool(value & bit) for row in projected['action_relation_bits'] for value in row)


def interface_contract():
    return {
        'graph_to_gnn': 'Directed multigraph edges; categorical one-hot encoding with explicit unknown fallback plus numeric attributes; TransformerConv consumes edge_attr when saved encode_edge=True',
        'gnn_to_memory': 'Semantic loop/array anchor at absolute Lk-1 slot; pragma-free multiscale concat of JKN and centered conv1, 128 dimensions in production; no proof every fact is recoverable',
        'stage2_routing': 'Saved ADP contract compiler_relational; trainer stacks relation_mask, attention casts to bool. Typed bits are exported and checked, not supplied as typed attention bias',
        'routing_retains': ['connectivity', 'absolute identity', 'self visibility', 'inactive masking'],
        'routing_discards_explicitly': ['relation type', 'dependence direction when connectivity is symmetric', 'certainty labels', 'multiple relation multiplicity'],
        'legacy_false_visibility': 'A legacy false-positive mask edge remains visible to Stage-2; the separate refinement verifier does not remove it',
        'r1_refinement': 'Verified relationship eligibility and provenance, direct/indirect priority, loop-array preference, current complete directives and source-derived domains; no GNN embeddings or learned utility ranking',
        'ranker_feature_contract': {
            'loop_depth_hierarchy_trip_count': 'available; respect compiler static/unknown flags and nearest versus transitive hierarchy',
            'array_access_roots': 'available for exact roots; helper contexts require strict verification',
            'memory_dependence': 'available; use direction, certainty, bounds and unresolved status explicitly',
            'arithmetic': 'available as operation families and local operations; requires action-region aggregation',
            'related_current_directives': 'available from current search assignment; NOT injected in frozen pragma-free memory',
            'partition_dimension_factor_and_access_pattern': 'current values/source dimensions available; candidate-specific bank conflict/parallelism effect not established',
            'frozen_embeddings': 'available on original kernels; semantic recoverability/probe not measured by this audit',
            'target_resource_headroom': 'available from request and measured incumbent; absent as dynamic features in pragma-free static action memory'},
        'future_schema_requirements': ['actual_to_formal.callsite: unique originating call node/op_uid',
            'callee symbol/function identity', 'observed actual operand index and formal argument index',
            'call-scoped memory-root/alias provenance; never shared-helper reachability alone']}


def write_report(output, trace, annotations, comparisons, errors):
    lines = ['# Structural representation audit', '',
        'No synthesis, GNN/LLM inference, retraining or production artifact modification was performed.', '',
        '## Reference status', '',
        'Real-source annotations were written from source semantics, independently of projected bits, but are **machine-assisted drafts**. '
        'They require independent human adjudication. Reviewed precision/recall remain unavailable when there are no reviewed labels. '
        'Draft disagreements identify inspectable failure cases, not an independently validated dataset accuracy estimate.', '',
        '| Kernel | Source actions | Anchors | Projected pairs | Verified / uncertain / no evidence | Saved memory |',
        '|---|---:|---:|---:|---|---|']
    for name, rec in trace['kernels'].items():
        if rec['status'] != 'audited':
            lines.append(f'| {name} | blocked | | | {rec["reason"]} | |'); continue
        a = rec['alignment']; counts = rec['verified_pair_classifications']
        verified = counts.get('direct_verified', 0) + counts.get('indirect_verified', 0)
        lines.append(f'| {name} | {a["source_actions"]} | {a["semantic_anchors"]} | {rec["projected_nonself_pairs"]} | '
                     f'{verified} / {counts.get("conservative_unknown", 0)} / {counts.get("no_evidenced_relation", 0)} | {rec["memory"]["status"]} |')
    lines += ['', '## Source-reference comparisons', '',
              '| Kernel/type | Draft TP | Draft FP | Draft FN | Draft TN | Unresolved | Reviewed precision / recall |',
              '|---|---:|---:|---:|---:|---:|---|']
    for row in comparisons:
        lines.append(f'| {row["kernel"]}/{row["relationship_type"]} | {row["draft_TP"]} | {row["draft_FP"]} | '
                     f'{row["draft_FN"]} | {row["draft_TN"]} | {row["unresolved"]} | {row["precision"]} / {row["recall"]} |')
    lines += ['', '## Available artifacts and category coverage', '']
    for name, rec in trace['kernels'].items():
        if rec['status'] != 'audited':
            continue
        lines.append(f'- {name}: node OOV {rec["feature_oov"].get("node", {}).get("unknown_by_field")}; '
                     f'edge OOV {rec["feature_oov"].get("edge", {}).get("unknown_by_field")}; '
                     f'raw IR {rec["raw_mlir_status"]}; legacy external memory {rec["external_legacy_memory_inventory"]}.')
    lines += ['', 'Unseen memory-rank/shape tokens are mapped to `<unknown>`; exact examples are recorded in the trace. '
              'This is a measured vocabulary limitation, not evidence that all structural information is lost: numeric rank/volume, '
              'topology and known edge features still reach the GNN. Legacy external 64-dimensional memory files without '
              'graph/checkpoint/schema provenance cannot be authenticated as the current 128-dimensional production memory.', '']
    lines += ['', '## GAN trace', '']
    for row in annotations:
        if row['kernel'] == 'GAN' and {row['action_i'], row['action_j']} in ({'L2', 'L11'}, {'L3', 'L11'}):
            v = row.get('strict_verification', {})
            lines += [f'- {row["action_i"]}–{row["action_j"]}: draft source access expected={row["expected"]}; '
                      f'legacy bits {row.get("projected_bits")}; legacy verifier {v.get("legacy_verifier_class")}; '
                      f'strict verifier {v.get("verified_relation_class")}; anchors {row.get("action_anchors")}; '
                      f'root nodes {[r["graph_node"] for r in v.get("relevant_memory_root", [])]}; '
                      f'shared helper nodes {v.get("shared_helper_nodes", [])}; '
                      f'inspected nodes (not proof when uncertain) {v.get("supporting_graph_nodes", [])}; '
                      f'legacy witness operations {v.get("legacy_supporting_operations", [])}.']
    schema = trace['saved_feature_schema']
    lines += ['', '## Call-site accuracy', '',
        '`_add_call_edges()` writes actual-to-formal position but no originating call ID. '
        'The unchanged refinement verifier can accept position+callee as proof. '
        'This audit removes unscoped/inconsistent binding edges in a graph copy under `strict_callsite_and_access_v1` before verification. '
        'It also recognizes native operation-to-root writes from paired directed edges and excludes call-effect summaries as direct-access proof. '
        'The stricter policy can reduce recall; uncertain is not negative. Production graphs and past results remain unchanged. '
        'The two-call regression establishes rejection of unscoped evidence; it does not claim every old positional match was factually wrong. '
        'Synthetic regression outcomes are in `fixtures/results.json` when fixture validation is run.', '',
        '## Learned representation and routing', '',
        f'Saved configuration: {trace["saved_configuration_status"]}. '
        f'Checkpoint schema `{schema.get("feature_schema_version")}`, graph `{schema.get("graph_schema_version")}`; '
        f'node dimension {schema.get("node_feature_dim")}, edge dimension {schema.get("edge_feature_dim")}.', '',
        f'Edge categories: {schema.get("edge_categorical_fields")}. Numeric fields: {schema.get("edge_numeric_fields")}.', '',
        'The recorded checkpoint enables edge-aware TransformerConv. Exact CPU re-encoding is compared with available saved tensors. '
        'OOV counts and raw missing attributes are reported separately; missing derived node-field names are not evidence of lost features. '
        'Known categories are one-hot encoded, unseen values use `<unknown>`; edge numerics are stored in float16. '
        'Direction and parallel-edge ordering are tested against saved `edge_index`.', '',
        'Existing original-kernel memory packs are checked against source graph, checkpoint and feature-schema hashes, '
        'absolute slots, action types, finite embedding dimensions and exact reprojected bits. '
        'External-kernel refinement graphs are not evidence that matching frozen embeddings exist. '
        'A graph-snapshot mismatch blocks an end-to-end memory claim rather than aligning unrelated artifacts.', '',
        'Local AP analysis stubs and header hashes are recorded separately from vendor HLS arithmetic. '
        'The graph metadata seals the source translation unit but not the complete transitive include closure. '
        'These control/dataflow graphs must not be treated as bit-accurate models of vendor fixed-point arithmetic. '
        'A future schema migration should seal headers as well as call-site provenance.', '',
        'Stage-2 uses Boolean relation visibility, retaining self and active-slot masks. Explicit relation type, direction and certainty '
        'are not supplied as typed attention features. This does not imply those properties are absent from learned GNN embeddings. '
        'The deterministic R1 selector does not consume those embeddings at all.', '',
        '## Priorities', '',
        '1. Independently review source-reference drafts, especially negative and dependence cases; expand reviewed precision/recall coverage.',
        '2. Use strict call-site verification for future refinement evidence; retain legacy checkpoints and graph tensors. '
        'A future graph migration needs unique call IDs and call-scoped root/formal bindings, followed by a controlled compatibility evaluation.',
        '3. Once audited evidence coverage is adequate, develop a lightweight site-and-value selector using current neighboring directives '
        'and trusted relation context. Compare feature-only, frozen-embedding and explicit-relation inputs under the same budget.',
        '4. Treat typed-routing ablation as optional architecture research, not a proven fix. Do not retrain solely because deterministic pair search was weak.', '',
        'This audit supports a targeted evidence-layer correction before relying on indirect relationships for learning. '
        'It does not justify replacing the GNN or claiming that all graph relationships are inaccurate. '
        'Embedding fact recoverability and downstream predictive value remain unmeasured.', '',
        'Semantic references: [MLIR Func](https://mlir.llvm.org/docs/Dialects/Func/), '
        '[MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/), [Affine](https://mlir.llvm.org/docs/Dialects/Affine/), '
        '[language reference](https://mlir.llvm.org/docs/LangRef/).', '']
    if trace.get('fixture_results'):
        lines += ['## Executed fixtures', '', '| Fixture | Unit check | Real compiler check |', '|---|---|---|']
        for row in trace['fixture_results']:
            lines.append(f'| {row["name"]} | {row["unit_status"]} | {row["compiler_status"]} |')
    (output / 'structural_audit_report.md').write_text('\n'.join(lines))


def fixture_graph(case):
    """Hand-authored unit graph, NEVER represented as compiler-generated output."""
    graph = nx.MultiDiGraph()
    for label, node, root in [('L1', '1', '2'), ('L3', '3', '4')]:
        graph.add_node(node, type=104, action_id=label, array_var=label,
                       action_resolution='source_location', function=0)
        graph.add_node(root, type=1, is_memory_root=1, function=0)
        graph.add_edge(node, root, flow=7, role='array_root')
    for label, node, block, operation in [('L9', '9', '10', '11'), ('L19', '19', '20', '21')]:
        graph.add_node(node, type=0, is_loop=1, action_id=label,
                       action_resolution='source_location', function=0)
        graph.add_node(block, type=4, function=0)
        graph.add_node(operation, type=0, text='affine.load', function=0)
        graph.add_edge(node, block, flow=8, role='region_entry')
        graph.add_edge(block, operation, flow=0, role='next_in_block')
        pragma = str(int(node) + 100)
        graph.add_node(pragma, type=100, text='UNROLL', action_id=label)
        graph.add_edge(pragma, block, flow=4, role='pragma_scope')
    graph.add_edge('1', '11', flow=7, role='array_read')
    graph.add_edge('2', '11', flow=10, role='read')
    graph.add_edge('3', '21', flow=7, role='array_write')
    graph.add_edge('4', '21', flow=10, role='write')
    graph.nodes['21']['text'] = 'affine.store'
    if case == 'direct_write':
        for u, v, key, edge in list(graph.edges(keys=True, data=True)):
            if v == '11' and edge.get('role') in ('array_read', 'read'):
                edge['role'] = 'array_write' if edge['role'] == 'array_read' else 'write'
        graph.nodes['11']['text'] = 'affine.store'
        for key in list(graph['2']['11']):
            graph['2']['11'][key]['role'] = 'write_reverse'
            graph['2']['11'][key]['operand_index'] = 1
        graph.add_edge('11', '2', flow=10, role='write', operand_index=1)
    if case == 'nested_loops':
        graph.add_edge('10', '19', flow=0, role='next_in_block')
    if case in ('shared_helper', 'bound_helper'):
        for u, v, key, edge in list(graph.edges(keys=True, data=True)):
            if v in ('11', '21') and edge.get('role') in ('array_read', 'array_write', 'read', 'write'):
                graph.remove_edge(u, v, key)
        graph.add_node('30', type=0, text='func.func', function=1)
        graph.add_node('31', type=4, function=1)
        graph.add_node('32', type=1, function=1, is_memory_root=1, is_block_argument=1)
        graph.add_node('33', type=0, text='affine.load', function=1)
        graph.add_edge('30', '31', flow=8, role='region_entry')
        graph.add_edge('31', '33', flow=0, role='next_in_block')
        graph.add_edge('32', '33', flow=10, role='read')
        for call, root, scope in [('11', '2', '1'), ('21', '4', '3')]:
            graph.nodes[call]['text'] = 'func.call'
            graph.add_edge(call, '30', flow=2, role='calls')
            graph.add_edge(root, call, flow=1, role='operand', position=0)
            edge = {'flow': 2, 'role': 'actual_to_formal', 'position': 1}
            if case == 'bound_helper':
                edge['callsite'] = call
            graph.add_edge(root, '32', **edge)
            graph.add_edge(scope, '33', flow=7, role='array_read')
    if case == 'resolved_view':
        graph.add_node('40', type=1, function=0, text='memref.subview')
        graph.add_edge('2', '40', flow=9, role='compiler_derived_view', certainty='must_alias')
        graph.add_edge('40', '11', flow=1, role='operand', position=0)
        # Concrete root-access edge above is an explicit input assumption.
    if case == 'dependencies':
        graph.nodes['11']['text'] = 'affine.store'
        graph.add_edge('11', '21', flow=12, role='RAW', certainty='proven')
        graph.add_edge('21', '11', flow=12, role='WAR', certainty='may')
    return graph


def fixture_checks(name):
    graph = fixture_graph(name)
    if name == 'nested_loops':
        return {'nested_region': '19' in legacy.local_region(graph, '9')}
    if name == 'dependencies':
        from LLM_branch.common.directive_domains import SourceAction
        actions = {k: SourceAction(kind='loop', trip_count=8) for k in ('L9', 'L19')}
        actions.update({k: SourceAction(kind='array', dimensions=((1, 8),)) for k in ('L1', 'L3')})
        _, _, _, bits, _ = project(graph, actions, 19)
        return {'proven_forward': bool(bits[8, 18] & 16), 'uncertain_reverse': bool(bits[18, 8] & 16)}
    result = {'own_array': strict_verify_pair(graph, 'L1', 'L9')['verified_relation_class']}
    if name in ('different_arrays', 'shared_helper', 'bound_helper'):
        result['other_array'] = strict_verify_pair(graph, 'L3', 'L9')['verified_relation_class']
    return result


def validate_fixtures(args):
    output = c.repo_path(args.output_dir) / 'fixtures'
    output.mkdir(parents=True, exist_ok=True)
    specs = json.loads((HERE / 'fixtures/semantics.json').read_text())
    try:
        tools = legacy.compiler_tools(build=args.polygeist_build, python=args.mlir_python) if args.compile_fixtures else None
        blocker = 'Not requested; use --compile-fixtures for actual compiler validation'
    except FileNotFoundError as exc:
        tools, blocker = None, str(exc)
    manifest = {'policy': VERIFIER_POLICY, 'audit_sha256': c.sha256(Path(__file__)),
                'specs_sha256': c.sha256(HERE / 'fixtures/semantics.json'), 'compiler_tools': tools,
                'compile_requested': args.compile_fixtures}
    manifest_path = output / 'manifest.json'
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Fixture inputs changed; choose a new output directory')
    if not manifest_path.is_file():
        c.write_json(manifest_path, manifest)
    results_path = output / 'results.json'
    if results_path.is_file():
        previous = json.loads(results_path.read_text())
        for record in previous:
            for path, digest in record.get('artifact_sha256', {}).items():
                if c.sha256(c.repo_path(path)) != digest:
                    raise ValueError('Prepared fixture artifact changed; preserve it and select a new directory')
        print(json.dumps({'fixture_resume': 'authenticated_existing_results', 'count': len(previous)}))
        return previous
    records = []
    for spec in specs:
        observed = fixture_checks(spec['name'])
        if observed != spec['unit_expected']:
            raise ValueError(f'Fixture {spec["name"]}: {observed} != independently specified {spec["unit_expected"]}')
        record = {'name': spec['name'], 'unit_graph_origin': 'hand_authored_test_input_not_compiler_output',
                  'unit_expected': spec['unit_expected'], 'unit_observed': observed, 'unit_status': 'pass',
                  'compiler_status': 'blocked' if args.compile_fixtures and not tools else 'not_run', 'reason': blocker,
                  'source_semantics': spec['semantics'], 'source_sha256': None}
        if tools:
            directory = output / spec['name']; directory.mkdir(exist_ok=True)
            source = directory / 'kernel.cpp'; metadata = directory / 'kernel_info.txt'
            source.write_text(spec['program']); metadata.write_text(spec['metadata'])
            record['source_sha256'] = c.sha256(source)
            try:
                generated = directory / 'kernel.gexf'
                receipt = legacy.regenerate_graph(source, generated, tools, timeout=120)
                graph = nx.read_gexf(generated)
                actions = domains.load_source_actions(spec['name'], directory)
                projected = legacy.graph_relations(generated, c.sha256(source), c.sha256(metadata), actions)
                observations = []
                for pair in spec.get('compiler_access_checks', []):
                    a, b = pair['array'], pair['loop']
                    if a not in semantic_anchors(graph) or b not in semantic_anchors(graph):
                        observations.append({**pair, 'status': 'blocked_anchor_not_retained'})
                        continue
                    actual = strict_verify_pair(graph, a, b, artifact_hash=c.sha256(generated))
                    observations.append({**pair, 'status': 'compared', 'observed': actual['verified_relation_class'],
                                         'matches_reference': actual['verified_relation_class'] in pair['allowed_classes'], 'evidence': actual})
                record.update(compiler_status='generated', compiler_receipt=receipt, compiler_tools=tools,
                              compiler_observations=observations, graph_sha256=c.sha256(generated),
                              projection_stats=projected['stats'], compiler_alignment=alignment(graph, actions),
                              compiler_memory_view_edges=sum(int(e.get('flow', -1)) == 9 for _, _, e in graph.edges(data=True)),
                              compiler_dependence_certainties=dict(Counter(e.get('certainty') for _, _, e in graph.edges(data=True) if int(e.get('flow', -1)) == 12)))
                checks = []
                for check in spec.get('compiler_relation_checks', []):
                    observed = bool(projected['action_relation_bits'][int(check['action_i'][1:]) - 1][int(check['action_j'][1:]) - 1] & check['bit'])
                    checks.append({**check, 'observed': observed, 'matches_reference': observed == check['expected']})
                record['compiler_relation_checks'] = checks
                if any(r.get('matches_reference') is False for r in observations + checks):
                    record['compiler_status'] = 'reference_disagreement'
                record['artifact_sha256'] = {c.display_path(p): c.sha256(p) for p in directory.iterdir() if p.is_file()}
            except (OSError, ValueError, RuntimeError) as exc:
                record.update(compiler_status='blocked', reason=str(exc))
        records.append(record)
    c.write_json(output / 'results.json', records)
    print(json.dumps([{'name': r['name'], 'unit': r['unit_status'], 'compiler': r['compiler_status']} for r in records], indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('audit', 'fixtures'), default='audit')
    parser.add_argument('--kernels', nargs='+', default=['GAN', 'covariance', 'Kalman'])
    parser.add_argument('--output-dir', default=str(HERE / 'results'))
    parser.add_argument('--shared-manifest', default=str(DEFAULT_SHARED))
    parser.add_argument('--annotations', default=str(HERE / 'fixtures/annotations.json'))
    parser.add_argument('--include-originals', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile-fixtures', action='store_true')
    parser.add_argument('--polygeist-build')
    parser.add_argument('--mlir-python')
    args = parser.parse_args()
    if args.mode == 'audit':
        run_audit(args)
    else:
        validate_fixtures(args)


if __name__ == '__main__':
    main()
