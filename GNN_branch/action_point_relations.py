"""Typed, evidence-bearing views of the single authoritative V2 compiler graph."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json


def _edges(graph, node, relation, incoming=False):
    edges = graph.in_edges(node, keys=True, data=True) if incoming else graph.out_edges(node, keys=True, data=True)
    return [(u, v, key, e) for u, v, key, e in edges if e['relation_type'] == relation and not e.get('is_reverse')]


def _targets(graph):
    return {attrs['action_id']: _edges(graph, node, 'targets')[0][1]
            for node, attrs in graph.nodes(data=True) if attrs['category'] == 'action_point'}


def _roots(graph, value):
    return [(v, e['evidence_status']) for _, v, _, e in _edges(graph, value, 'resolves_to')]


def _weakest(*statuses):
    return max(statuses, key={'proven': 0, 'possible': 1, 'not_established': 2}.__getitem__)


def _caller_roots(graph, root, context, chain=(), seen=()):
    """Substitute exact formals at one call; never merge all callers' roots."""
    if root in seen:
        return [(root, 'possible', chain)]
    if root in context:
        target, status, bindings = context[root]
        return [(r, _weakest(status, s), c)
                for r, s, c in _caller_roots(graph, target, context, chain + bindings, seen + (root,))]
    attrs = graph.nodes[root]
    if attrs.get('root_kind') != 'formal_parameter':
        return [(root, 'not_established' if attrs.get('root_kind') == 'unresolved' else 'proven', chain)]
    formal = attrs['root_value']
    candidates = []
    for actual, _, _, binding in _edges(graph, formal, 'actual_to_formal', incoming=True):
        call = graph.nodes[binding['call_operation']]
        if call.get('recursive_call'):
            continue
        receipt = {**binding, 'actual': actual, 'formal': formal}
        for actual_root, status in _roots(graph, actual):
            for resolved, proof, path in _caller_roots(graph, actual_root, context, chain + (receipt,), seen + (root,)):
                candidates.append((resolved, _weakest(status, proof, binding['evidence_status'], call['call_resolution']), path))
    return candidates or [(root, 'proven', chain)]


def _accesses(graph, loop):
    function = graph.nodes[loop]['function']
    owned = [node for node, attrs in graph.nodes(data=True)
             if attrs['category'] == 'operation' and attrs.get('function') == function and loop in attrs.get('loop_context', [])]
    bindings_by_call, bodies = defaultdict(list), defaultdict(list)
    for actual, formal, edge in graph.edges(data=True):
        if edge['relation_type'] == 'actual_to_formal':
            bindings_by_call[edge['call_operation']].append((actual, formal, edge))
    for node, attrs in graph.nodes(data=True):
        if attrs['category'] == 'operation' and ':op' in attrs.get('op_uid', ''):
            bodies[attrs['op_uid'].rsplit(':op', 1)[0]].append(node)

    def visit(operations, context, chain, functions):
        found = []
        for operation in sorted(operations):
            for _, root, key, edge in _edges(graph, operation, 'memory_access'):
                for resolved, status, path in _caller_roots(graph, root, context):
                    found.append({'memory_root': resolved, 'source_operation': operation,
                        'access': edge['access'], 'evidence_status': _weakest(edge['evidence_status'], status),
                        'access_path': 'through_call' if chain or path else 'local',
                        'call_chain': list(dict.fromkeys(chain + tuple(p['call_operation'] for p in path))),
                        'callsite_bindings': list(path), 'supporting_edge': str(key),
                        'memory_operand': edge['memory_operand'], 'operand_index': edge.get('operand_index')})
            node = graph.nodes[operation]
            if not node.get('call_identity'):
                continue
            callee = node.get('callee_function')
            bindings = bindings_by_call[operation]
            if node.get('call_resolution') != 'proven' or callee in functions:
                actuals = {u for u, _, _, e in _edges(graph, operation, 'operand', incoming=True)}
                for actual in sorted(actuals):
                    for root, _ in _roots(graph, actual):
                        for resolved, _, path in _caller_roots(graph, root, context):
                            found.append({'memory_root': resolved, 'source_operation': operation,
                                'access': 'readwrite', 'evidence_status': 'possible', 'access_path': 'through_call',
                                'call_chain': list(dict.fromkeys(chain + (operation,) + tuple(p['call_operation'] for p in path))),
                                'callsite_bindings': list(path), 'reason': 'Recursive/unresolved call; no concrete interprocedural proof'})
                continue
            next_context = dict(context)
            for actual, formal, binding in bindings:
                actual_roots, formal_roots = _roots(graph, actual), _roots(graph, formal)
                receipt = {**binding, 'actual': actual, 'formal': formal}
                # An unresolved binding must not fall back to roots from some
                # other call to the same helper.
                for formal_root, _ in formal_roots:
                    next_context[formal_root] = (formal_root, 'possible', (receipt,))
                if len(actual_roots) != 1 or len(formal_roots) != 1:
                    continue
                proof = _weakest(actual_roots[0][1], formal_roots[0][1], binding['evidence_status'])
                next_context[formal_roots[0][0]] = (actual_roots[0][0], proof, (receipt,))
            # Each access follows only bindings selected by its formal root.
            # Do not attach every argument as evidence for every memory access.
            found.extend(visit(bodies[callee], next_context, chain + (operation,), functions + (callee,)))
        return found

    symbol = graph.nodes[loop]['op_uid'].rsplit(':op', 1)[0]
    return visit(owned, {}, (), (symbol,))


def _base(graph, point, neighbor, relation_type, **attrs):
    return {'action_point': point, 'neighbor': neighbor, 'relation_type': relation_type,
            'evidence_status': 'proven', 'access_path': 'not_applicable',
            'graph_identity': graph.graph['graph_identity'],
            'source_provenance': {k: graph.graph['provenance'].get(k) for k in
                ('source_sha256', 'metadata_sha256', 'mlir_sha256', 'analysis_extension_sha256', 'v2_builder_sha256')}, **attrs}


def get_action_point_relations(graph, action_point, evidence='proven', *, _access_cache=None):
    """Derive facts on demand; no secondary action-point graph or QoR oracle."""
    if evidence not in ('proven', 'possible', 'not_established', 'all'):
        raise ValueError('Unknown evidence filter')
    targets = _targets(graph)
    if action_point not in targets:
        raise ValueError(f'Unknown/inactive action point {action_point}')
    loops = {label: target for label, target in targets.items() if graph.nodes[target]['category'] == 'operation'}
    arrays = defaultdict(list)
    for label, target in targets.items():
        if graph.nodes[target]['category'] == 'memory_object':
            arrays[target].append(label)
    rows = []
    access_cache = {} if _access_cache is None else _access_cache
    relevant_loops = loops if action_point not in loops else {action_point: loops[action_point]}
    for label, loop in sorted(relevant_loops.items()):
        if loop not in access_cache:
            access_cache[loop] = _accesses(graph, loop)
        for access in access_cache[loop]:
            for array in arrays.get(access['memory_root'], []):
                if action_point not in (label, array):
                    continue
                neighbor = array if action_point == label else label
                rows.append(_base(graph, action_point, neighbor, 'memory_access', **access))
    if action_point in loops:
        own = loops[action_point]
        by_loop = {node: label for label, node in loops.items()}
        # Nearest *labeled* ancestors/descendants, preserving unlabeled loops.
        for incoming, direction in ((False, 'child'), (True, 'parent')):
            pending = [(own, [])]
            visited = set()
            while pending:
                node, path = pending.pop()
                if node in visited: continue
                visited.add(node)
                for u, v, key, edge in _edges(graph, node, 'loop_child', incoming):
                    target = u if incoming else v
                    new_path = path + [str(key)]
                    if target in by_loop:
                        rows.append(_base(graph, action_point, by_loop[target], 'loop_hierarchy', hierarchy=direction,
                                          supporting_edges=new_path))
                    else:
                        pending.append((target, new_path))
        def owner(operation):
            context = graph.nodes[operation].get('loop_context', [])
            return next((by_loop[n] for n in reversed(context) if n in by_loop), None)
        for consumer, producer, key, edge in graph.edges(keys=True, data=True):
            if edge['relation_type'] != 'depends_on': continue
            left, right = owner(producer), owner(consumer)
            if not left or not right or left == right or action_point not in (left, right): continue
            rows.append(_base(graph, action_point, right if action_point == left else left, 'memory_dependence',
                evidence_status=edge['evidence_status'], dependence_direction='precedes' if action_point == left else 'depends_on',
                dependence_kind=edge['dependence_kind'], source_operation=producer, target_operation=consumer,
                supporting_edge=str(key), compiler_evidence=edge['source_analysis']))
    # Array-alias evidence is distinct from an access or optimization coupling.
    if targets[action_point] in arrays:
        root = targets[action_point]
        values = {u for u, _, _, _ in _edges(graph, root, 'resolves_to', incoming=True)}
        for u, v, key, edge in graph.edges(keys=True, data=True):
            if edge['relation_type'] != 'alias' or not ({u, v} & values): continue
            other = v if u in values else u
            for other_root, status in _roots(graph, other):
                for label in arrays.get(other_root, []):
                    if label == action_point: continue
                    rows.append(_base(graph, action_point, label, 'memory_alias',
                        evidence_status=edge['evidence_status'], alias_kind=edge['alias_kind'],
                        supporting_edge=str(key), compiler_evidence=edge['source_analysis']))
    unique = {json.dumps(row, sort_keys=True): row for row in rows
              if evidence == 'all' or row['evidence_status'] == evidence}
    return [unique[key] for key in sorted(unique)]


def action_point_views(graph, max_slots=None):
    targets = _targets(graph)
    count = max_slots or max((int(k[1:]) for k in targets), default=0)
    if any(int(label[1:]) > count for label in targets):
        raise ValueError('Action point would be truncated')
    active = [f'L{i + 1}' in targets for i in range(count)]
    bits, connectivity = [[0] * count for _ in range(count)], [[False] * count for _ in range(count)]
    records, unrepresented, access_cache = [], [], {}
    for point in sorted(targets, key=lambda s: int(s[1:])):
        i = int(point[1:]) - 1; bits[i][i] = 1; connectivity[i][i] = True
        for row in get_action_point_relations(graph, point, evidence='proven', _access_cache=access_cache):
            records.append(row)
            j = int(row['neighbor'][1:]) - 1
            kind, bit = row['relation_type'], 0
            if kind == 'memory_access': bit = 8
            elif kind == 'loop_hierarchy': bit = 4 if row['hierarchy'] == 'child' else 2
            elif kind == 'memory_dependence': bit = 16 if row['dependence_direction'] == 'precedes' else 32
            if not (kind == 'memory_alias' and row.get('alias_kind') == 'no_alias'):
                connectivity[i][j] = True
            bits[i][j] |= bit
            if not bit: unrepresented.append(row)
    return {'labels': [i + 1 if active[i] else -1 for i in range(count)], 'active_mask': active,
            'typed_relations': records, 'connectivity': connectivity,
            'legacy_relation_bits': bits, 'legacy_relation_mask': [[bool(v) for v in row] for row in bits],
            'unrepresented_in_legacy_bits': unrepresented,
            'note': 'Derived V2 views only; never substituted into production Stage-2 memory'}


def historical_relation(record):
    """Factorize old labels without presenting unverified historical proof as V2."""
    old = record['verified_relation_class']
    status, path = {'direct_verified': ('proven', 'local'), 'indirect_verified': ('proven', 'through_call'),
                    'conservative_unknown': ('possible', 'not_applicable'),
                    'no_evidenced_relation': ('not_established', 'not_applicable')}[old]
    return {'relation_type': record.get('verified_relation_types', record.get('original_relation_types', ['unspecified'])),
            'evidence_status': 'possible' if status == 'proven' else status,
            'access_path': path, 'historical_evidence_status': status,
            'mapping_ambiguity': 'Historical confidence has not been checked against V2 call/root evidence; relation types may combine access and non-access facts',
            'legacy_classification': old, 'legacy_record': deepcopy(record)}
