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


def _return_evidence(graph, value, seen=()):
    if value in seen: return []
    own = graph.nodes[value].get('return_root_evidence')
    if own: return [own]
    parents = [v for _,v,e in graph.out_edges(value,data=True) if e['relation_type']=='references']
    return [record for parent in parents for record in _return_evidence(graph,parent,seen+(value,))]


def _memory_views(graph, value, seen=()):
    if value in seen: return []
    rows=[]
    for _,base,_,edge in _edges(graph,value,'references'):
        if edge.get('view_mapping'):
            rows.append({'value':value,'base':base,**edge['view_mapping']})
        rows.extend(_memory_views(graph,base,seen+(value,)))
    return rows


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
    if graph.graph.get('schema', '').endswith('v2.1'):
        # Unbound formals remain symbolic. A query inside a helper is not a
        # licence to substitute buffers from every caller into that helper.
        return [(root, 'proven', chain)]
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


def _bind_call(graph, operation, context, bindings):
    next_context=dict(context)
    for actual,formal,binding in bindings:
        actual_roots,formal_roots=_roots(graph,actual),_roots(graph,formal)
        receipt={**binding,'actual':actual,'formal':formal}
        for formal_root,_ in formal_roots:
            next_context[formal_root]=(formal_root,'possible',(receipt,))
        if len(actual_roots)==len(formal_roots)==1:
            proof=_weakest(actual_roots[0][1],formal_roots[0][1],binding['evidence_status'])
            next_context[formal_roots[0][0]]=(actual_roots[0][0],proof,(receipt,))
    return next_context


def _call_contexts(graph, caller, callee):
    """Enumerate exact verified invocations, not a union of formal roots."""
    calls=defaultdict(list)
    for call,node in graph.nodes(data=True):
        if not node.get('call_identity') or node.get('call_resolution')!='proven':continue
        for _,target,_,edge in _edges(graph,call,'calls'):
            if edge['evidence_status']=='proven':calls[node['function']].append((call,graph.nodes[target]['function']))
    def walk(function,path,seen):
        if function==callee:return [path]
        return [result for call,target in sorted(calls[function]) if target not in seen
                for result in walk(target,path+(call,),seen+(target,))]
    return walk(caller,(),(caller,))


def _accesses(graph, loop, call_context=()):
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
                    row = {'memory_root': resolved, 'source_operation': operation,
                        'access': edge['access'], 'evidence_status': _weakest(edge['evidence_status'], status),
                        'access_path': 'through_call' if chain or path else 'local',
                        'call_chain': list(dict.fromkeys(chain + tuple(p['call_operation'] for p in path))),
                        'callsite_bindings': list(path), 'supporting_edge': str(key),
                        'memory_operand': edge['memory_operand'], 'operand_index': edge.get('operand_index')}
                    if graph.graph.get('schema', '').endswith('v2.1'):
                        from GNN_branch.mlir_semantics import index_dependencies
                        local_calls=chain[len(call_context):]
                        origin = local_calls[0] if local_calls else operation
                        scope = graph.nodes[origin].get('loop_context', [])
                        extra = _return_evidence(graph,edge['memory_operand'])
                        return_calls = [b['call_operation'] for record in extra for b in record['bindings']]
                        values=[edge['memory_operand']]+[b['actual'] for b in path]+[b['actual'] for record in extra for b in record['bindings'] if 'actual' in b]
                        views={json.dumps(v,sort_keys=True):v for value in values for v in _memory_views(graph,value)}
                        view_rows=[views[k] for k in sorted(views)]
                        row.update(access_scope='through_call' if chain else 'direct_body' if scope and scope[-1] == loop else 'nested_descendant',
                            origin_loop=edge.get('origin_loop'), call_origin_scope='not_applicable' if call_context and not local_calls else 'direct_body' if scope and scope[-1] == loop else 'nested_descendant',
                            access_indices=edge.get('access_indices'), return_alias_evidence=extra,
                            memory_object=resolved, supporting_operation=operation, concrete=True,
                            memory_views=view_rows, caller_view_induction_loops=index_dependencies(graph,
                                [arg['value'] for view in view_rows for arg in view['operands'] if arg['position']>0 and not graph.nodes[arg['value']].get('is_memory')]))
                        if return_calls:
                            row['call_chain']=list(dict.fromkeys(row['call_chain']+return_calls))
                            row['access_path']='through_call'
                    found.append(row)
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
                            row = {'memory_root': resolved, 'source_operation': operation,
                                'access': 'readwrite', 'evidence_status': 'possible', 'access_path': 'through_call',
                                'call_chain': list(dict.fromkeys(chain + (operation,) + tuple(p['call_operation'] for p in path))),
                                'callsite_bindings': list(path), 'reason': 'Recursive/unresolved call; no concrete interprocedural proof'}
                            if graph.graph.get('schema','').endswith('v2.1'):
                                row.update(access_scope='through_call', memory_object=resolved,supporting_operation=operation,
                                           access_indices={'status':'not_established'},concrete=False, origin_loop=(node.get('loop_context',[]) or [None])[-1])
                            found.append(row)
                continue
            next_context = _bind_call(graph,operation,context,bindings)
            # Each access follows only bindings selected by its formal root.
            # Do not attach every argument as evidence for every memory access.
            found.extend(visit(bodies[callee], next_context, chain + (operation,), functions + (callee,)))
        return found

    symbol = graph.nodes[loop]['op_uid'].rsplit(':op', 1)[0]
    context,functions={},[]
    for call in call_context:
        if call not in graph or graph.nodes[call].get('call_resolution')!='proven':
            raise ValueError('Invocation context requires a verified originating call')
        node=graph.nodes[call]
        if functions and node['caller_function']!=functions[-1]:raise ValueError('Invocation context is not a connected call chain')
        if not functions:functions.append(node['caller_function'])
        if node['callee_function'] in functions:raise ValueError('Recursive invocation context is not proven')
        functions.append(node['callee_function'])
        context=_bind_call(graph,call,context,bindings_by_call[call])
    if functions and functions[-1]!=symbol:raise ValueError('Invocation context does not end at the loop function')
    rows=visit(owned,context,tuple(call_context),tuple(functions) if functions else (symbol,))
    if graph.graph.get('schema','').endswith('v2.1'):
        for row in rows:row['invocation_context']=list(call_context)
    return rows


def _base(graph, point, neighbor, relation_type, **attrs):
    return {'action_point': point, 'neighbor': neighbor, 'relation_type': relation_type,
            'evidence_status': 'proven', 'access_path': 'not_applicable',
            'graph_identity': graph.graph['graph_identity'],
            'source_provenance': {k: graph.graph['provenance'].get(k) for k in
                ('source_sha256', 'metadata_sha256', 'mlir_sha256', 'analysis_extension_sha256', 'v2_builder_sha256')}, **attrs}


def get_action_point_relations(graph, action_point, evidence='proven', *, call_context=None, _access_cache=None):
    """Derive facts on demand; no secondary action-point graph or QoR oracle."""
    if evidence not in ('proven', 'possible', 'not_established', 'all'):
        raise ValueError('Unknown evidence filter')
    targets = _targets(graph)
    if action_point not in targets:
        raise ValueError(f'Unknown/inactive action point {action_point}')
    if call_context:
        for call in call_context:
            if call not in graph or graph.nodes[call].get('call_resolution')!='proven':
                raise ValueError('Invocation context requires a verified originating call')
        if graph.nodes[targets[action_point]]['category']=='operation' and graph.nodes[targets[action_point]]['op_uid'].rsplit(':op',1)[0]!=graph.nodes[call_context[-1]]['callee_function']:
            raise ValueError('Invocation context does not end at the action-point function')
    loops = {label: target for label, target in targets.items() if graph.nodes[target]['category'] == 'operation'}
    arrays = defaultdict(list)
    for label, target in targets.items():
        if graph.nodes[target]['category'] == 'memory_object':
            arrays[target].append(label)
    rows = []
    access_cache = {} if _access_cache is None else _access_cache
    relevant_loops = loops if action_point not in loops else {action_point: loops[action_point]}
    for label, loop in sorted(relevant_loops.items()):
        contexts={tuple(call_context or ())}
        if call_context is not None and call_context:
            callee=graph.nodes[call_context[-1]].get('callee_function') if call_context[-1] in graph else None
            if graph.nodes[loop]['op_uid'].rsplit(':op',1)[0]!=callee:continue
        elif call_context is None and graph.graph.get('schema','').endswith('v2.1'):
            # Join action points in different functions only through complete,
            # individually verified invocation contexts. Roots are never unioned.
            for root,points in arrays.items():
                callers={graph.nodes['action:'+point].get('function',graph.nodes[root].get('function')) for point in points}
                for caller in callers:
                    callee=graph.nodes[loop]['function']
                    if caller is None or caller<0 or caller==callee:continue
                    key=('call_contexts',caller,callee)
                    if key not in access_cache:access_cache[key]=_call_contexts(graph,caller,callee)
                    contexts.update(access_cache[key])
        accesses=[]
        for context in sorted(contexts):
            key=(loop,context)
            if key not in access_cache:access_cache[key]=_accesses(graph,loop,context)
            accesses.extend(access_cache[key])
        for access in accesses:
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
            if graph.graph.get('schema', '').endswith('v2.1') and edge.get('carrying_loop') == own:
                rows.append(_base(graph, action_point, action_point, 'memory_dependence',
                    evidence_status=edge['evidence_status'], dependence_direction='carried_by',
                    carrying_loop=own, dependence_kind=edge['dependence_kind'],
                    source_operation=producer, target_operation=consumer, supporting_edge=str(key),
                    distance_bounds=edge.get('distance_bounds'), distance_exact=edge.get('distance_exact'),
                    compiler_evidence=edge['source_analysis']))
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
    return [deepcopy(unique[key]) for key in sorted(unique)]


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
            elif kind == 'memory_dependence' and row['dependence_direction'] in ('precedes','depends_on'):
                bit = 16 if row['dependence_direction'] == 'precedes' else 32
            if not (kind == 'memory_alias' and row.get('alias_kind') == 'no_alias'):
                connectivity[i][j] = True
            bits[i][j] |= bit
            if not bit: unrepresented.append(row)
    return {'labels': [i + 1 if active[i] else -1 for i in range(count)], 'active_mask': active,
            'typed_relations': records, 'connectivity': connectivity,
            'legacy_relation_bits': bits, 'legacy_relation_mask': [[bool(v) for v in row] for row in bits],
            'unrepresented_in_legacy_bits': unrepresented,
            'note': 'Derived V2 views only; never substituted into production Stage-2 memory'}


def get_loop_semantics(graph, action_point, *, call_context=()):
    """Read loop state/accesses, including memory without a directive point.

    Self/within-loop dependences are not discarded just because both accesses
    have the same nearest labelled loop. Counts refer to compiler operations,
    not new physical accesses for every enclosing loop.
    """
    target = _targets(graph).get(action_point)
    if target is None or not graph.nodes[target].get('is_loop'):
        raise ValueError(f'Not an active loop action point: {action_point}')
    dependencies, state = [], []
    for u, v, key, edge in graph.edges(keys=True, data=True):
        if edge['relation_type'] == 'depends_on' and (target in graph.nodes[u].get('loop_context', []) or target in graph.nodes[v].get('loop_context', [])):
            dependencies.append({'consumer': u, 'producer': v, 'supporting_edge': str(key), **deepcopy(edge)})
        if edge['relation_type'] == 'loop_carried' and edge.get('carrying_loop') == target:
            state.append({'source_value': u, 'target_value': v, 'supporting_edge': str(key), **deepcopy(edge)})
    node = next(a for _, a in graph.nodes(data=True) if a.get('category') == 'action_point' and a['action_id'] == action_point)
    return deepcopy({'action_point': action_point, 'compiler_loop': target,
        'metadata_value': node.get('metadata_value', node.get('source_loop_metadata_value')),
        'compiler_proven_trip_count': node.get('compiler_proven_trip_count',graph.nodes[target].get('trip_count') if graph.nodes[target].get('trip_count_static') else None),
        'tcad_trip_count': node.get('tcad_trip_count'),
        'trip_count_provenance': node.get('trip_count_provenance',graph.nodes[target].get('trip_count_source')),
        'accesses': _accesses(graph, target,call_context),
        'ssa_state': sorted(state, key=lambda r: r['supporting_edge']),
        'memory_dependences': sorted(dependencies, key=lambda r: r['supporting_edge'])})


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
