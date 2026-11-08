"""Versioned action-level routing derived from compiler-native RC3 witnesses.

This is a small projection over GNN_branch.action_point_relations, not a
traversal of the historical GEXF/numeric-node representation.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from GNN_branch import action_point_relations as rc3, mlir_graph_v2 as compiler

SCHEMA = 'salt-hls-rc3-action-relations-v1'
ROUTING_POLICY = 'proven_loop_scope_array_access_and_loop_hierarchy_v1'
MAX_SLOTS = 64


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


RELATION_SCHEMA_ID = _digest({
    'schema': SCHEMA, 'routing_policy': ROUTING_POLICY,
    'absolute_slot_rule': 'Lk maps to k-1; never compact',
    'evidence_filter': 'proven for route eligibility; retain non-proven states in ledger',
    'relations': ['self', 'loop_child', 'loop_parent', 'array_accessed_by_loop',
                  'loop_accesses_array', 'loop_dependence'],
    'absence_semantics': 'not_established is not independence',
})


def _slots(graph, max_slots):
    coverage = graph.graph.get('action_point_coverage', {}).get('records', [])
    by_id = {row['action_id']: row for row in coverage}
    points = {attrs['action_id']: node for node, attrs in graph.nodes(data=True)
              if attrs.get('category') == 'action_point'}
    labels = sorted(set(by_id) | set(points), key=lambda x: int(x[1:]))
    if any(int(label[1:]) > max_slots for label in labels):
        raise ValueError('Absolute action slot exceeds configured capacity; truncation is forbidden')
    registered = [False] * max_slots
    matched = [False] * max_slots
    unresolved = [False] * max_slots
    source_only = [False] * max_slots
    targets = {}
    for label in labels:
        slot = int(label[1:]) - 1
        row = by_id.get(label, {})
        registered[slot] = bool(row.get('metadata_registered'))
        source_only[slot] = bool(row.get('source_labeled')) and not registered[slot]
        point = points.get(label)
        if point is None:
            # Coverage ledgers may preserve an inactive absolute slot without
            # a corresponding action_point node. Keep the slot unresolved;
            # never synthesize a target or compact later actions.
            unresolved[slot] = registered[slot]
            continue
        target_edges = [(v, e) for _, v, _, e in graph.out_edges(point, keys=True, data=True)
                        if e.get('relation_type') == 'targets' and not e.get('is_reverse')]
        if len(target_edges) > 1:
            raise ValueError(f'{label}: multiple compiler targets')
        is_proven = (len(target_edges) == 1 and target_edges[0][1].get('evidence_status') == 'proven'
                     and row.get('compiler_matched', True))
        matched[slot] = bool(is_proven)
        unresolved[slot] = registered[slot] and not matched[slot]
        if is_proven:
            targets[label] = {'node': target_edges[0][0], 'target_edge': str(next(
                key for _, v, key, edge in graph.out_edges(points[label], keys=True, data=True)
                if v == target_edges[0][0] and edge.get('relation_type') == 'targets')),
                'source_binding': row.get('source_binding'),
                'target_binding_rule': graph.nodes[points[label]].get('target_binding_rule')}
    return labels, points, targets, {
        'slot_rule': 'Lk maps to slot k-1; no compacting',
        'registered_mask': registered, 'matched_mask': matched,
        'valid_mask': [a and b for a, b in zip(registered, matched)],
        'unresolved_mask': unresolved, 'source_only_mask': source_only,
    }


def _is_array_target(graph, targets, label):
    target = targets.get(label)
    return bool(target and graph.nodes[target['node']].get('category') == 'memory_object')


def build_action_relations(graph, *, max_slots=MAX_SLOTS, expected_graph_identity=None):
    """Return absolute-slot masks, typed proof ledger and proven routing mask.

    The mask is intentionally directed. A loop query can see arrays accessed
    in its scope; an array query sees only the outermost labeled loop proven
    to access it. Nested loop hierarchy remains separately directed. The
    nested accesses remain in the witness ledger and are never discarded as
    compiler facts.
    """
    if graph.graph.get('schema') != compiler.SCHEMA_V22:
        raise ValueError('RC3 V2.2 compiler graph required')
    if graph.graph.get('semantic_revision') != 'rc3':
        raise ValueError('RC3 semantic revision required')
    identity = graph.graph.get('graph_identity')
    if not identity or (expected_graph_identity is not None and identity != expected_graph_identity):
        raise ValueError('Compiler graph identity mismatch')
    labels, points, targets, masks = _slots(graph, max_slots)
    access_cache = {}
    raw = {label: rc3.get_action_point_relations(graph, label, evidence='all', _access_cache=access_cache)
           for label in sorted(points, key=lambda x: int(x[1:]))}
    hierarchy = defaultdict(set)
    for rows in raw.values():
        for row in rows:
            if row['relation_type'] == 'loop_hierarchy' and row['evidence_status'] == 'proven':
                if row.get('hierarchy') == 'child':
                    hierarchy[row['action_point']].add(row['neighbor'])
    # Candidate loops with proven access are gathered from compiler queries.
    # An array-side query routes to the outermost candidate loop, avoiding a
    # redundant nested key; loop-side queries retain every directly scoped
    # action access. Each witness remains serialized below.
    array_loops = defaultdict(set)
    for query, rows in raw.items():
        if not _is_array_target(graph, targets, query):
            continue
        for row in rows:
            if row['relation_type'] == 'memory_access' and row['evidence_status'] == 'proven':
                array_loops[query].add(row['neighbor'])
    outermost = {}
    for array, candidates in array_loops.items():
        nested = {child for parent in candidates for child in hierarchy[parent] if child in candidates}
        outermost[array] = candidates - nested

    # Proven target edge is the source of a self route.
    ledger = []
    routes = defaultdict(list)
    non_routing = []
    for label in sorted(points, key=lambda x: int(x[1:])):
        slot = int(label[1:]) - 1
        target = targets.get(label)
        if target:
            self_row = {
                'query_action': label, 'key_action': label, 'query_slot': slot, 'key_slot': slot,
                'relation_type': 'self', 'direction': 'self', 'evidence_status': 'proven',
                'routing_eligible': True, 'source_node': points[label], 'target_node': target['node'],
                'supporting_edge': target['target_edge'], 'graph_identity': identity,
                'source_binding': target['source_binding'],
            }
            ledger.append(self_row); routes[(label, label)].append(self_row)
        for row in raw[label]:
            status = row['evidence_status']
            kind = row['relation_type']
            if kind == 'memory_alias':
                # Alias is authenticated evidence but is outside this routing
                # schema. Keep it in the provenance ledger, never as a key.
                record = {'query_action': label, 'key_action': row['neighbor'],
                          'relation_type': 'memory_alias_excluded', 'direction': 'compiler_edge_direction',
                          'evidence_status': status, 'routing_eligible': False, **row}
                non_routing.append(record)
                continue
            selected = False
            rel_type = kind
            direction = 'query_to_related_action'
            if kind == 'loop_hierarchy':
                selected = status == 'proven'
                rel_type = 'loop_child' if row.get('hierarchy') == 'child' else 'loop_parent'
                direction = row.get('hierarchy')
            elif kind == 'memory_access':
                query_is_array = _is_array_target(graph, targets, label)
                neighbor = row['neighbor']
                if status == 'proven' and query_is_array:
                    selected = neighbor in outermost.get(label, set())
                    rel_type, direction = 'array_accessed_by_loop', 'array_to_accessing_loop'
                elif status == 'proven':
                    selected = True
                    rel_type, direction = 'loop_accesses_array', 'loop_to_accessed_array'
            elif kind == 'memory_dependence':
                # Action-local recurrence and memory dependences are evidence,
                # not cross-action routing. Only genuinely inter-action rows
                # with a proven compiler witness can become routed links.
                selected = status == 'proven' and row['neighbor'] != label
                rel_type = 'loop_dependence'
                direction = row.get('dependence_direction', 'compiler_direction')
            record = {
                'query_action': label, 'key_action': row.get('neighbor'),
                'query_slot': slot, 'key_slot': int(row['neighbor'][1:]) - 1,
                'relation_type': rel_type, 'source_relation_type': kind,
                'direction': direction, 'evidence_status': status,
                'routing_eligible': bool(selected), 'graph_identity': identity,
                **row,
            }
            ledger.append(record)
            if selected:
                routes[(label, row['neighbor'])].append(record)
            elif kind == 'memory_dependence' or status != 'proven':
                non_routing.append(record)

    eligible = [[False] * max_slots for _ in range(max_slots)]
    relation_types = [[[] for _ in range(max_slots)] for _ in range(max_slots)]
    for (query, key), rows in routes.items():
        i, j = int(query[1:]) - 1, int(key[1:]) - 1
        if not (masks['valid_mask'][i] and masks['valid_mask'][j]):
            raise ValueError('Routing references an invalid/unmatched action slot')
        eligible[i][j] = True
        relation_types[i][j] = sorted({r['relation_type'] for r in rows})
    relation_payload = {
        'schema': SCHEMA, 'routing_policy': ROUTING_POLICY,
        'evidence_filters': ['proven'], 'absolute_slots': masks,
        'eligibility': eligible, 'relation_types': relation_types,
        'provenance': {
            'graph_identity': identity,
            'source_sha256': graph.graph.get('provenance', {}).get('source_sha256'),
            'metadata_sha256': graph.graph.get('provenance', {}).get('metadata_sha256'),
            'mlir_sha256': graph.graph.get('provenance', {}).get('mlir_sha256'),
            'analysis_extension_sha256': graph.graph.get('provenance', {}).get('analysis_extension_sha256'),
            'v2_builder_sha256': graph.graph.get('provenance', {}).get('v2_builder_sha256'),
        },
        'ledger': sorted(ledger, key=lambda r: (r['query_slot'], r.get('key_slot', -1), r['relation_type'],
                                                r.get('supporting_edge', ''))),
        'non_routing_evidence': sorted(non_routing, key=lambda r: (r['query_action'], r.get('key_action') or '',
                                                                  r['relation_type'], r.get('supporting_edge', ''))),
        'absence_semantics': 'No emitted dependence relation means not_established, not independence.',
    }
    relation_payload['relation_schema_id'] = RELATION_SCHEMA_ID
    relation_payload['payload_sha256'] = _digest(relation_payload)
    relation_payload['compiler_target_nodes'] = targets
    return relation_payload
