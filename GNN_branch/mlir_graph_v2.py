"""Explicitly selected MLIR structural graph; independent of V1 feature tensors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time

import networkx as nx

from GNN_branch import mlir_graph_gen as v1

SCHEMA = 'salt-hls-mlir-structural-v2'
SCHEMA_V21 = 'salt-hls-mlir-structural-v2.1'
NODE_KINDS = {'operation', 'ssa_value', 'constant_or_feature', 'control_region', 'memory_object', 'action_point'}
EDGE_KINDS = {
    'owns_region', 'contains_block', 'contains_operation', 'declares_argument',
    'next_operation', 'control_flow', 'defines', 'operand', 'literal', 'feature',
    'loop_carried', 'loop_child', 'calls', 'actual_to_formal', 'returns_to',
    'references', 'alias', 'resolves_to', 'memory_access', 'depends_on',
    'targets', 'analysis_summary',
}
EVIDENCE = {'proven', 'possible', 'not_established'}
COMPACT_FEATURES = {'memory_shape', 'affine_access', 'loop_trip_count'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _symbol(value):
    """Normalize a printed MLIR symbol without damaging quoted static names."""
    text = str(value or '').strip()
    if text.startswith('@'):
        text = text[1:]
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] == '"' else text


def _decode(value):
    if isinstance(value, str) and value[:1] in ('[', '{'):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def add_relation(graph, source, target, relation_type, evidence_status='proven', **attrs):
    """Deduplicate identical facts; independent relation kinds remain parallel edges."""
    if source not in graph or target not in graph:
        raise ValueError(f'Missing relation endpoint: {source}, {target}')
    data = {'relation_type': relation_type, 'evidence_status': evidence_status,
            'access_path': 'not_applicable', 'is_reverse': False, **attrs}
    key = hashlib.sha256(canonical_json([source, target, data]).encode()).hexdigest()[:24]
    graph.add_edge(source, target, key=key, **data)


def graph_payload(graph):
    return {'schema': graph.graph['schema'], 'metadata': deepcopy(graph.graph),
            'nodes': [{'id': node, **deepcopy(attrs)} for node, attrs in sorted(graph.nodes(data=True))],
            'edges': [{'source': u, 'target': v, 'key': key, **deepcopy(attrs)}
                      for u, v, key, attrs in sorted(graph.edges(keys=True, data=True), key=lambda e: (e[0], e[1], str(e[2])))]}


def write_graph(graph, path):
    validate(graph)
    path = Path(path)
    payload = canonical_json(graph_payload(graph)) + '\n'
    if path.exists() and path.read_text() != payload:
        raise ValueError(f'Refusing to overwrite a different graph: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def read_graph(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') not in (SCHEMA, SCHEMA_V21):
        raise ValueError('Not a V2 graph; V1 tensors are not accepted')
    if data['metadata'].get('schema') != data['schema']:
        raise ValueError('Graph and metadata schema disagree')
    graph = nx.MultiDiGraph()
    graph.graph.update(data['metadata'])
    for node in data['nodes']:
        graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    for edge in data['edges']:
        graph.add_edge(edge['source'], edge['target'], key=edge['key'],
                       **{k: v for k, v in edge.items() if k not in ('source', 'target', 'key')})
    validate(graph)
    return graph


def _family(name):
    if name in v1.LOOP_OPS: return 'loop'
    if name in v1.CALL_OPS or name == 'func.call_indirect': return 'call'
    if name in v1.READ_OPS: return 'memory_read'
    if name in v1.WRITE_OPS: return 'memory_write'
    if name in v1.VIEW_OPS: return 'memory_view'
    if name.startswith(('arith.', 'math.')): return 'arithmetic'
    if name in ('func.func', 'llvm.func'): return 'function'
    return 'other'


def _build_nodes(result):
    graph = nx.MultiDiGraph(schema=SCHEMA, view='canonical', tensor_compatibility='not_compatible_with_v1_checkpoint')
    graph.add_node('module', category='control_region', control_kind='module')
    mapping = {}
    for node, attrs in sorted(result.graph.nodes(data=True), key=lambda item: int(item[0])):
        if attrs.get('op_uid'):
            ident, category = 'op:' + attrs['op_uid'], 'operation'
        elif attrs.get('ssa_id'):
            ident, category = 'value:' + attrs['ssa_id'], 'ssa_value'
        elif int(attrs.get('type', -1)) == v1.NODE_TYPE_IMMEDIATE:
            # The original deterministic node ID disambiguates equal auxiliary
            # literals; compiler-owned operation/value IDs remain authoritative.
            ident, category = f'feature:{node}', 'constant_or_feature'
        else:
            continue  # Pragmas and array_scope are consolidated below.
        mapping[str(node)] = ident
        native = {k: _decode(v) for k, v in attrs.items()}
        graph.add_node(ident, **native, category=category, legacy_node_id=str(node))
        if category == 'operation':
            graph.nodes[ident]['operation_family'] = _family(attrs['text'])
        if category == 'constant_or_feature':
            graph.nodes[ident]['feature_role'] = 'semantic_constant' if attrs.get('feature_kind') == 'constant' else 'auxiliary'
    # Convert existing *SSA* edges, not legacy context-insensitive memory roots.
    for u, v, attrs in result.graph.edges(data=True):
        if str(u) not in mapping or str(v) not in mapping:
            continue
        flow, role = int(attrs.get('flow', -1)), attrs.get('role', '')
        source, target = mapping[str(u)], mapping[str(v)]
        if flow == 1:
            relation = {'result': 'defines', 'operand': 'operand', 'immediate': 'literal'}.get(role, 'feature')
        elif flow == 11:
            relation = 'loop_carried'
        elif flow == 0:
            relation = 'next_operation' if role == 'next_in_block' else 'control_flow'
        elif flow == 13:
            relation = 'analysis_summary'
        else:
            continue
        add_relation(graph, source, target, relation,
                     'possible' if flow == 13 else 'proven', legacy_evidence=dict(attrs))
    return graph, mapping


def _control_and_calls(graph, result, mapping):
    by_key = {r.key: 'op:' + f'{r.function_name}:op{r.function_ordinal}' for r in result.operation_records}
    function_nodes = {fid: 'op:' + result.graph.nodes[node]['op_uid'] for fid, node in result.function_nodes.items()}
    for fid, node in function_nodes.items():
        graph.nodes[node]['function_symbol'] = result.functions[fid]
        graph.nodes[node]['has_body'] = any(b.function_id == fid for b in result.blocks.values())
        add_relation(graph, 'module', node, 'contains_operation')
    for block in result.blocks.values():
        owner = v1.unwrap_operation(block.block.owner)
        owner_key = v1.mlir_wrapper_key(owner)
        if owner_key not in by_key and v1.operation_name(owner) not in ('func.func','llvm.func'):
            raise ValueError('Real block owner is not an indexed compiler operation')
        owner_id = by_key.get(owner_key, function_nodes[block.function_id])
        region = f'region:{owner_id}:{block.region_index}'
        block_id = f'block:{result.functions[block.function_id]}:{block.block_id}'
        graph.add_node(region, category='control_region', control_kind='region', owner=owner_id,
                       region_index=block.region_index, function=block.function_id)
        graph.add_node(block_id, category='control_region', control_kind='block', function=block.function_id,
                       block=block.block_id, parent_region=region)
        add_relation(graph, owner_id, region, 'owns_region')
        add_relation(graph, region, block_id, 'contains_block')
        for record in block.operations:
            operation = 'op:' + f'{record.function_name}:op{record.function_ordinal}'
            graph.nodes[operation].update(block_order=record.block_order, parent_operation=owner_id,
                loop_context=['op:' + result.loops[i].op_record.function_name + ':op' + str(result.loops[i].op_record.function_ordinal)
                              for i in record.loop_stack])
            add_relation(graph, block_id, operation, 'contains_operation')
        for node, attrs in list(graph.nodes(data=True)):
            if attrs['category'] == 'ssa_value' and int(attrs.get('function', -1)) == block.function_id and int(attrs.get('block', -1)) == block.block_id and int(attrs.get('is_block_argument', 0)):
                add_relation(graph, block_id, node, 'declares_argument')
    for loop in result.loops:
        if loop.parent_index is not None:
            parent = result.loops[loop.parent_index].op_record
            child = loop.op_record
            add_relation(graph, f'op:{parent.function_name}:op{parent.function_ordinal}',
                         f'op:{child.function_name}:op{child.function_ordinal}', 'loop_child')
    ssa_operands = defaultdict(dict)
    for u, v, edge in graph.edges(data=True):
        if edge['relation_type'] == 'operand':
            ssa_operands[v][int(edge['legacy_evidence']['position'])] = u
    call_graph = nx.DiGraph()
    for record in result.operation_records:
        if record.op_name not in v1.CALL_OPS | {'func.call_indirect'}:
            continue
        call = f'op:{record.function_name}:op{record.function_ordinal}'
        callee = v1.strip_mlir_string(record.attributes.get('callee', '')).lstrip('@')
        callee_id = result.function_name_to_id.get(callee)
        graph.nodes[call].update(caller_function=record.function_name, callee_function=callee,
                                 call_identity=graph.nodes[call]['op_uid'], call_resolution='not_established')
        if callee_id is None:
            continue
        formals = result.function_arguments.get(callee_id, [])
        has_body = any(b.function_id == callee_id for b in result.blocks.values())
        matched = len(formals) == len(record.operands) and has_body
        graph.nodes[call]['call_resolution'] = 'proven' if matched else 'possible'
        add_relation(graph, call, function_nodes[callee_id], 'calls', 'proven' if matched else 'possible',
                     call_operation=call, call_uid=graph.nodes[call]['op_uid'],
                     caller_function=record.function_name, callee_function=callee)
        call_graph.add_edge(record.function_name, callee)
        for position, formal in enumerate(formals):
            actual = ssa_operands[call].get(position)
            formal_id = mapping[str(formal)]
            if actual is None:
                continue
            type_agrees = graph.nodes[actual].get('value_type') == graph.nodes[formal_id].get('value_type')
            add_relation(graph, actual, formal_id, 'actual_to_formal', 'proven' if matched and type_agrees else 'possible',
                         call_operation=call, call_uid=graph.nodes[call]['op_uid'],
                         caller_function=record.function_name, callee_function=callee, argument_index=position)
    recursive = {f for component in nx.strongly_connected_components(call_graph)
                 if len(component) > 1 or any(call_graph.has_edge(n, n) for n in component) for f in component}
    for node, attrs in graph.nodes(data=True):
        if attrs.get('call_identity') and attrs.get('callee_function') in recursive:
            attrs['call_resolution'] = 'possible'
            attrs['recursive_call'] = True
    # Return forwarding remains scoped to this exact call; it is not a global
    # alias union. Queries currently leave returned-memory roots unresolved.
    for record in result.operation_records:
        if record.op_name not in v1.CALL_OPS:
            continue
        call = f'op:{record.function_name}:op{record.function_ordinal}'
        callee = graph.nodes[call].get('callee_function')
        for returned in result.operation_records:
            if returned.function_name != callee or returned.op_name not in v1.RETURN_OPS:
                continue
            for index, value in ssa_operands[f'op:{returned.function_name}:op{returned.function_ordinal}'].items():
                results = [v for _, v, e in graph.out_edges(call, data=True) if e['relation_type'] == 'defines' and e['legacy_evidence'].get('position') == index]
                for target in results:
                    add_relation(graph, value, target, 'returns_to', 'possible', call_operation=call,
                                 call_uid=graph.nodes[call]['op_uid'], result_index=index,
                                 reason='Call-scoped return; branch/alias feasibility not established')


def _memory(graph, result, revision='2.0'):
    analysis = result.compiler_analysis
    values = {attrs['ssa_id']: node for node, attrs in graph.nodes(data=True) if attrs['category'] == 'ssa_value'}
    memory_values = {node for node in values.values() if int(graph.nodes[node].get('is_memory', 0))}
    constraints = defaultdict(set)
    for evidence in analysis.get('views', []):
        base, view = values[evidence['source']], values[evidence['target']]
        if base in memory_values and view in memory_values:
            if graph.nodes[base].get('function') != graph.nodes[view].get('function'):
                raise ValueError('A local SSA view cannot cross isolated function scopes')
            if revision == '2.1':
                defining = [u for u, _, e in graph.in_edges(view, data=True) if e['relation_type']=='defines']
                operation = defining[0] if len(defining)==1 else None
                supported = operation and graph.nodes[operation].get('text') in (v1.VIEW_OPS - {'unrealized_conversion_cast'}) | {'polygeist.subindex'}
                direct = supported and any(u==base and e['relation_type']=='operand' and e['legacy_evidence'].get('position')==0 for u,_,e in graph.in_edges(operation,data=True))
                if not direct:
                    # BufferViewFlowAnalysis returns transitive reachability,
                    # including branch/call forwarding; it is not must-alias.
                    add_relation(graph,view,base,'analysis_summary','possible',
                        summary_kind='buffer_view_reachability',source_analysis=evidence)
                    continue
            constraints[view].add(base)
            add_relation(graph, view, base, 'references', source_analysis=evidence)
    for evidence in analysis.get('aliases', []):
        left, right = values[evidence['lhs']], values[evidence['rhs']]
        result_kind = evidence['result']
        status = 'proven' if result_kind in ('must_alias', 'no_alias') else 'possible'
        add_relation(graph, min(left, right), max(left, right), 'alias', status,
                     alias_kind=result_kind, symmetric=True, source_analysis=evidence)
        if result_kind == 'must_alias' and graph.nodes[left].get('function') == graph.nodes[right].get('function'):
            constraints[left].add(right); constraints[right].add(left)
    for u, v, edge in graph.edges(data=True):
        if edge['relation_type'] == 'loop_carried' and u in memory_values and v in memory_values:
            constraints[v].add(u)
    roots = {node: set() for node in memory_values}
    for node in sorted(memory_values):
        attrs = graph.nodes[node]
        defining = [u for u, _, e in graph.in_edges(node, data=True) if e['relation_type'] == 'defines']
        allocation = any(graph.nodes[u].get('text') in ('memref.alloc', 'memref.alloca', 'memref.get_global', 'llvm.alloca') for u in defining)
        formal = int(attrs.get('function_argument_index', -1)) >= 0
        if allocation or formal:
            globals_ = [u for u in defining if graph.nodes[u].get('text') == 'memref.get_global']
            reference = graph.nodes[globals_[0]].get('mlir_attrs', {}).get('name', '') if globals_ else ''
            symbol = _symbol(reference) if revision == '2.1' else v1.strip_mlir_string(reference).lstrip('@')
            if globals_ and not symbol: raise ValueError('Global memory reference has no compiler symbol')
            obj = 'memory:global:' + symbol if symbol else 'memory:' + attrs['ssa_id']
            if obj not in graph:
                graph.add_node(obj, category='memory_object', function=-1 if symbol else attrs['function'],
                               root_value=node, root_values=[], root_kind='global' if symbol else 'allocation' if allocation else 'formal_parameter',
                               global_symbol=symbol, value_type=attrs.get('value_type'), memory_shape=attrs.get('memory_shape'),
                               memory_rank=attrs.get('memory_rank'))
            if graph.nodes[obj]['value_type'] != attrs.get('value_type'):
                raise ValueError('One global storage symbol has inconsistent memory types')
            graph.nodes[obj]['root_values'].append(node)
            roots[node].add(obj)
    for _ in range(len(memory_values) + 1):
        changed = False
        for node in sorted(memory_values):
            inherited = set().union(*(roots.get(source, set()) for source in constraints[node])) if constraints[node] else set()
            before = len(roots[node]); roots[node].update(inherited); changed |= before != len(roots[node])
        if not changed: break
    if revision == '2.1':
        from GNN_branch.mlir_semantics import resolve_returned_memory
        resolve_returned_memory(graph, roots, constraints)
    # Keep unresolved storage explicit instead of dropping concrete accesses.
    # One placeholder per source SCC preserves view/iter-argument forwarding
    # without claiming an allocation, a unique caller, or a proven alias.
    unresolved = {node for node in memory_values if not roots[node]}
    forwarding = nx.DiGraph()
    forwarding.add_nodes_from(unresolved)
    forwarding.add_edges_from((source, target) for target in unresolved for source in constraints[target] if source in unresolved)
    for component in nx.strongly_connected_components(forwarding):
        if any(source not in component for node in component for source in forwarding.predecessors(node)):
            continue
        seed = min(component); attrs = graph.nodes[seed]
        obj = 'memory:unresolved:' + attrs['ssa_id']
        graph.add_node(obj, category='memory_object', function=attrs['function'], root_value=seed,
                       root_kind='unresolved', reason='No proven allocation/formal origin; placeholder only')
        roots[seed].add(obj)
    for _ in range(len(memory_values) + 1):
        changed = False
        for node in sorted(memory_values):
            inherited = set().union(*(roots.get(source, set()) for source in constraints[node])) if constraints[node] else set()
            before = len(roots[node]); roots[node].update(inherited); changed |= before != len(roots[node])
        if not changed: break
    for node in sorted(memory_values):
        unique_unknown = len(roots[node]) == 1 and graph.nodes[next(iter(roots[node]))]['root_kind'] == 'unresolved'
        graph.nodes[node]['root_resolution'] = 'not_established' if unique_unknown or not roots[node] else 'proven' if len(roots[node]) == 1 and not graph.nodes[node].get('possible_return_alias') else 'possible'
        graph.nodes[node]['memory_root_count'] = len(roots[node])
        graph.nodes[node]['is_memory_root'] = int(any(graph.nodes[r]['root_value'] == node for r in roots[node]))
        for root in sorted(roots[node]):
            add_relation(graph, node, root, 'resolves_to', graph.nodes[node]['root_resolution'],
                         reason='Function-local allocation/formal and compiler view/must-alias/region forwarding; no cross-call root union')
    # Compiler effects refer to exact SSA operands. Calls remain summaries.
    for effect in analysis.get('effects', []):
        operation = f'op:{effect["function"]}:op{effect["ordinal"]}'
        op = graph.nodes[operation]
        modes = defaultdict(set)
        for entry in effect.get('effects', []):
            if entry['kind'] in ('read', 'write') and entry.get('value') in values:
                modes[(values[entry['value']], entry.get('operand_index'))].add(entry['kind'])
        for (value, position), kinds in sorted(modes.items(), key=str):
            mode = 'readwrite' if len(kinds) == 2 else next(iter(kinds))
            summary = op.get('operation_family') == 'call'
            for root in sorted(roots.get(value, set())):
                add_relation(graph, operation, root, 'analysis_summary' if summary else 'memory_access',
                             'possible' if summary else graph.nodes[value]['root_resolution'],
                             access=mode, operand_index=position, memory_operand=value,
                             access_path='through_call' if summary else 'local',
                             concrete=not summary, source_analysis=effect)
        if effect['status'] == 'unknown':
            op['memory_effect_status'] = 'not_established'
    for dep in analysis.get('dependences', []):
        unresolved = revision=='2.1' and dep['result']=='not_applicable' and dep.get('fallback_required')
        if dep['result'] not in ('dependence', 'unknown') and not unresolved:
            continue
        source = f'op:{dep["source"]["function"]}:op{dep["source"]["ordinal"]}'
        target = f'op:{dep["target"]["function"]}:op{dep["target"]["ordinal"]}'
        status = 'proven' if dep['result'] == 'dependence' else 'possible'
        if graph.nodes[source].get('operation_family') == 'call' or graph.nodes[target].get('operation_family') == 'call':
            status = 'possible'
        add_relation(graph, target, source, 'depends_on', status,
                     dependence_kind=dep['kind'], compiler_source=source, compiler_target=target,
                     loop_carried=dep.get('loop_carried', dep.get('possible_loop_carried')),
                     distance_components=dep.get('components'), dependence_depth=dep.get('dependence_depth'),
                     source_analysis=dep)
    return roots


def build_from_result(result, provenance=None, paired_v1=None, global_definitions=None, revision='2.0', source=None, metadata=None):
    """Use live compiler objects while their owning module/context are alive."""
    if result.compiler_analysis is None:
        raise ValueError('V2 requires native compiler analysis; no approximate graph fallback')
    graph, mapping = _build_nodes(result)
    if revision not in ('2.0', '2.1'):
        raise ValueError('Unknown structural graph revision')
    graph.graph['schema'] = SCHEMA if revision == '2.0' else SCHEMA_V21
    _control_and_calls(graph, result, mapping)
    roots = _memory(graph, result, revision)
    for _, attrs in graph.nodes(data=True):
        if attrs.get('root_kind') == 'global':
            attrs['global_definition'] = (global_definitions or {}).get(attrs['global_symbol'],
                {'status': 'not_established', 'reason': 'External or unavailable global declaration'})
    for spec in result.actions:
        if not spec.matched:
            continue
        label = spec.action_id
        anchors = [(n, a) for n, a in result.graph.nodes(data=True) if a.get('action_id') == label and
                   (int(a.get('is_loop', 0)) == 1 or int(a.get('type', -1)) == v1.NODE_TYPE_ARRAY_SCOPE)]
        if len(anchors) != 1 or anchors[0][1].get('action_resolution') != 'source_location':
            raise ValueError(f'{label}: missing/ambiguous exact compiler target')
        old_node, attrs = anchors[0]
        point = 'action:' + label
        graph.add_node(point, category='action_point', action_id=label, action_kind=spec.kind,
                       absolute_index=int(label[1:]), supported_directives=list(spec.directives),
                       function=attrs['function'], source_file=spec.source_file,
                       source_line=spec.source_line, source_column=spec.source_column)
        if spec.kind == 'loop':
            target, status = mapping[str(old_node)], 'proven'
            graph.nodes[point]['target_binding_rule'] = 'exact_compiler_loop_source_binding'
            graph.nodes[point]['source_loop_metadata_value'] = spec.kernel_info_loop_value
        else:
            legacy_targets = [mapping[str(v)] for _, v, e in result.graph.out_edges(old_node, data=True)
                       if int(e.get('flow', -1)) == 7 and e.get('role') == 'array_root']
            tagged = [n for n,a in graph.nodes(data=True) if a['category']=='operation' and
                      v1.strip_mlir_string(a.get('mlir_attrs',{}).get('mailohls.action_id','')) == label]
            native_targets = [v for op in tagged for _,v,e in graph.out_edges(op,data=True)
                              if e['relation_type']=='defines' and int(graph.nodes[v].get('is_memory',0))]
            if tagged and len(native_targets) != 1:
                raise ValueError(f'{label}: native action attribute has ambiguous memory results')
            targets = native_targets or legacy_targets
            options = roots.get(targets[0], set()) if len(targets) == 1 else set()
            if len(options) != 1:
                raise ValueError(f'{label}: exact array target has unresolved/multiple local memory roots')
            target, status = next(iter(options)), graph.nodes[targets[0]]['root_resolution']
            graph.nodes[point]['array_variable'] = spec.variable
            graph.nodes[point]['source_array_dimensions'] = list(spec.array_dimensions)
            graph.nodes[point]['target_binding_rule'] = 'native_action_attribute' if native_targets else 'exact_compiler_array_source_binding'
            graph.nodes[point]['legacy_array_root_values'] = legacy_targets
            graph.nodes[point]['legacy_target_disagreement'] = bool(legacy_targets and target not in set().union(*(roots.get(v,set()) for v in legacy_targets)))
            binding = graph.nodes[targets[0]].get('global_action_binding')
            if binding:
                graph.nodes[point].update(global_action_binding=deepcopy(binding),
                    target_binding_rule='exact_compiler_global_source_binding')
        add_relation(graph, point, target, 'targets', status, source_binding=dict(attrs))
    compatibility = {}
    old_graph = paired_v1 if paired_v1 is not None else result.graph
    by_op = {a.get('op_uid'): n for n, a in graph.nodes(data=True) if a.get('op_uid')}
    by_ssa = {a.get('ssa_id'): n for n, a in graph.nodes(data=True) if a.get('ssa_id')}
    def feature_signature(old, node, attrs):
        anchors = []
        for u, v, e in list(old.in_edges(node, data=True)) + list(old.out_edges(node, data=True)):
            other = v if u == node else u
            peer = old.nodes[other]
            if peer.get('op_uid') or peer.get('ssa_id'):
                anchors.append((peer.get('op_uid', peer.get('ssa_id')), u == node,
                                {k: _decode(v) for k,v in e.items() if k != 'id'}))
        semantic = {k: _decode(v) for k,v in attrs.items() if k not in ('label','id')}
        return canonical_json([semantic, sorted(anchors, key=str)])
    features = defaultdict(list)
    for node, attrs in result.graph.nodes(data=True):
        if int(attrs.get('type', -1)) == v1.NODE_TYPE_IMMEDIATE:
            features[feature_signature(result.graph, node, attrs)].append(mapping[str(node)])
    for node, attrs in old_graph.nodes(data=True):
        category = int(attrs.get('type', -1))
        if category in (v1.NODE_TYPE_PRAGMA, v1.NODE_TYPE_ARRAY_SCOPE):
            target = 'action:' + str(attrs.get('action_id'))
            reason = 'consolidated_action_point'
        elif category == v1.NODE_TYPE_PSEUDO_BLOCK:
            fid, block = int(attrs['function']), int(attrs['block'])
            target = f'block:{result.functions[fid]}:{block}'
            reason = 'real_block_ownership_replaces_synthetic_membership'
            function = f'op:{result.functions[fid]}:function'
            if block == -1 and not graph.nodes[function]['has_body']:
                target = function
                reason = 'declaration_has_no_real_block; synthetic_scope_contracts_to_function'
        elif category == v1.NODE_TYPE_IMMEDIATE:
            candidates = features.get(feature_signature(old_graph, node, attrs), [])
            target = candidates[0] if len(candidates) == 1 else None
            reason = 'preserved_feature' if target else 'unresolved_feature_mapping_retained_for_review'
        else:
            target = by_op.get(attrs.get('op_uid')) or by_ssa.get(attrs.get('ssa_id'))
            reason = 'preserved_compiler_entity'
        compatibility[str(node)] = {'target': target if target in graph else None, 'reason': reason,
                                     'legacy_attributes': dict(attrs) if category in (v1.NODE_TYPE_PRAGMA, v1.NODE_TYPE_ARRAY_SCOPE, v1.NODE_TYPE_PSEUDO_BLOCK) else
                                     {k: attrs[k] for k in ('type', 'action_id', 'op_uid', 'ssa_id') if k in attrs}}
    graph.graph.update(provenance=provenance or {}, compiler_analysis=deepcopy(result.compiler_analysis),
                       compatibility_mapping=compatibility,
                       feature_encoding={'kind': 'typed_attributes', 'tensor_feature_dimension': None,
                                         'note': 'No V2 learned tensor encoder or checkpoint is implemented'})
    if revision == '2.1':
        from GNN_branch.mlir_semantics import annotate_semantics
        annotate_semantics(graph, result, source, metadata)
    graph.graph['graph_identity'] = hashlib.sha256(canonical_json({
        'schema': graph.graph['schema'], 'provenance': provenance or {}, 'builder_sha256': digest(__file__)}).encode()).hexdigest()
    validate(graph)
    return graph


def compact_view(graph):
    """Losslessly fold leaf auxiliary tokens; retain SSA, constants and evidence."""
    if graph.graph.get('view') != 'canonical':
        raise ValueError('Compact view must derive from canonical V2')
    compact = deepcopy(graph)
    removed = {}
    for node, attrs in sorted(graph.nodes(data=True)):
        if attrs['category'] != 'constant_or_feature' or attrs.get('feature_role') != 'auxiliary' or attrs.get('feature_kind') not in COMPACT_FEATURES:
            continue
        incident = [(u, v, k, dict(e)) for u, v, k, e in graph.edges(keys=True, data=True) if u == node or v == node]
        neighbors = {v if u == node else u for u, v, _, _ in incident}
        if len(neighbors) != 1 or any(e['relation_type'] != 'feature' for _, _, _, e in incident):
            continue
        target = next(iter(neighbors))
        record = {'attributes': deepcopy(attrs), 'edges': incident, 'transferred_to': target}
        removed[node] = record
        compact.nodes[target].setdefault('folded_auxiliary_features', []).append({
            'node': node, 'feature_kind': attrs['feature_kind'], 'token': attrs.get('text')})
        compact.remove_node(node)
    compact.graph.update(view='compact', compaction_mapping=removed,
                         compaction_policy='leaf auxiliary memory-shape/affine/trip tokens only; exact reconstruction retained')
    validate(compact)
    return compact


def expand_compact(graph):
    expanded = deepcopy(graph)
    for node, record in expanded.graph.pop('compaction_mapping', {}).items():
        expanded.add_node(node, **record['attributes'])
        for u, v, key, attrs in record['edges']:
            expanded.add_edge(u, v, key=key, **attrs)
    for _, attrs in expanded.nodes(data=True):
        attrs.pop('folded_auxiliary_features', None)
    expanded.graph.pop('compaction_policy', None)
    expanded.graph['view'] = 'canonical'
    return expanded


def validate(graph):
    if graph.graph.get('schema') not in (SCHEMA, SCHEMA_V21):
        raise ValueError('V2 schema is required')
    points = set()
    for node, attrs in graph.nodes(data=True):
        if attrs.get('category') not in NODE_KINDS:
            raise ValueError(f'Unknown node category: {node}')
        if attrs['category'] == 'action_point':
            label = attrs['action_id']
            if node != 'action:' + label or label in points or int(label[1:]) != attrs['absolute_index']:
                raise ValueError('Action point identity/alignment mismatch')
            points.add(label)
            targets = [(v, e) for _, v, e in graph.out_edges(node, data=True) if e['relation_type'] == 'targets']
            if len(targets) != 1:
                raise ValueError('Action point must have one canonical target')
            target = targets[0][0]
            kind = attrs.get('action_kind')
            expected = 'operation' if kind == 'loop' else 'memory_object' if kind == 'array' else None
            if graph.nodes[target]['category'] != expected or (kind == 'loop' and not int(graph.nodes[target].get('is_loop', 0))):
                raise ValueError('Action point target kind mismatch')
            binding = attrs.get('global_action_binding')
            if binding:
                operation, value = binding['operation'], binding['value']
                if (graph.nodes[target].get('root_kind') != 'global' or operation not in graph or value not in graph
                        or graph.nodes[operation].get('text') != 'memref.get_global'
                        or _symbol(graph.nodes[operation].get('mlir_attrs', {}).get('name')) != graph.nodes[target].get('global_symbol')
                        or binding['symbol'] != graph.nodes[target].get('global_symbol')
                        or graph.nodes[operation].get('source_line') != attrs['source_line']
                        or graph.nodes[operation].get('function') != graph.nodes[value].get('function')
                        or not binding['source_use_lines'] or value not in graph.nodes[target].get('root_values', [])
                        or not any(v == value and e['relation_type'] == 'defines' for _,v,e in graph.out_edges(operation,data=True))):
                    raise ValueError('Static/global action point has an invalid compiler binding')
                witnesses = binding.get('access_operations', [])
                if not witnesses or any(access not in graph
                        or graph.nodes[access].get('function') != graph.nodes[operation].get('function')
                        or graph.nodes[access].get('source_line') not in binding['source_use_lines']
                        or not any(v == target and e['relation_type'] == 'memory_access' and e.get('concrete')
                                   for _,v,e in graph.out_edges(access,data=True)) for access in witnesses):
                    raise ValueError('Static/global action point has an invalid compiler binding: access witness')
    for u, v, edge in graph.edges(data=True):
        if edge.get('relation_type') not in EDGE_KINDS or edge.get('evidence_status') not in EVIDENCE:
            raise ValueError('Unknown edge relation/evidence category')
        if edge['relation_type'] == 'memory_access':
            if graph.nodes[u]['category'] != 'operation' or graph.nodes[v]['category'] != 'memory_object' or graph.nodes[u].get('operation_family') == 'call' or not edge.get('concrete'):
                raise ValueError('A call summary is not concrete memory access')
        if edge['relation_type'] == 'actual_to_formal':
            call = edge.get('call_operation')
            if call not in graph or graph.nodes[call].get('op_uid') != edge.get('call_uid'):
                raise ValueError('Binding lost its originating call identity')
            position = edge.get('argument_index')
            actual = any(w == call and e['relation_type'] == 'operand' and e['legacy_evidence'].get('position') == position
                         for _, w, e in graph.out_edges(u, data=True))
            formal = graph.nodes[v].get('function_argument_index') == position
            callee = any(e['relation_type'] == 'calls' and graph.nodes[w].get('function_symbol') == graph.nodes[call].get('callee_function')
                         and graph.nodes[w].get('function') == graph.nodes[v].get('function')
                         for _, w, e in graph.out_edges(call, data=True))
            if not actual or not formal or not callee:
                raise ValueError('Binding operand/formal/callee mismatch')
    if graph.graph.get('schema') == SCHEMA_V21:
        from GNN_branch.mlir_semantics import validate_semantics
        validate_semantics(graph)
    if 'module' in graph and not nx.is_weakly_connected(graph):
        raise ValueError('Canonical semantic levels contain disconnected entities')
    return {'nodes': len(graph), 'edges': graph.number_of_edges(), 'action_points': len(points),
            'node_categories': dict(sorted(Counter(a['category'] for _, a in graph.nodes(data=True)).items())),
            'edge_categories': dict(sorted(Counter(a['relation_type'] for _, _, a in graph.edges(data=True)).items()))}


class _CoverageBuilder(v1.MLIRGraphBuilder):
    """V2.1 adapter: absent targets are coverage evidence, never shape repairs."""
    def _array_argument_node(self, spec):
        try:
            return super()._array_argument_node(spec)
        except RuntimeError:
            # A static local is a global object, not an allocation. Require its
            # compiler declaration location AND rooted source-use witnesses.
            # Shape only filters incompatible candidates (including C++ init
            # guards); it never supplies the identity proof.
            function, _ = self._resolve_action_function_id(spec)
            records = {record.node: record for record in self.operation_records}
            matches = {}
            for record in self.operation_records:
                if record.op_name != 'memref.get_global' or record.function_id != function or self._source_distance(record, spec) is None:
                    continue
                for key in record.result_keys:
                    node = self.value_nodes[key]
                    shape = self._node_memory_shape(node)
                    accesses = [records[access.op_node] for access in self.memory_accesses
                        if access.root_node == node and access.op_node in records
                        and records[access.op_node].function_id == function
                        and records[access.op_node].op_name in v1.READ_OPS | v1.WRITE_OPS]
                    uses = {point.line for access in accesses for point in access.source_points
                        if self._same_source_file(point.filename, spec.source_file)}
                    overlap = set(spec.source_use_lines) & uses
                    if tuple(shape[:len(spec.array_dimensions)]) != spec.array_dimensions or not overlap:
                        continue
                    symbol = _symbol(record.attributes.get('name'))
                    if symbol:
                        witnesses = sorted({'op:' + access.function_name + ':op' + str(access.function_ordinal)
                            for access in accesses if any(point.line in overlap for point in access.source_points)})
                        matches.setdefault(symbol, (record, node, sorted(overlap), witnesses))
            if len(matches) != 1:
                raise
            symbol, (record, node, uses, witnesses) = next(iter(matches.items()))
            self.graph.nodes[node]['global_action_binding'] = {
                'symbol': symbol, 'operation': 'op:' + record.function_name + ':op' + str(record.function_ordinal),
                'value': 'value:' + self.graph.nodes[node]['ssa_id'], 'source_use_lines': uses,
                'access_operations': witnesses,
                'evidence': 'compiler_get_global_declaration_location_and_rooted_source_accesses'}
            return record.function_id, node, record.block_id, 'source_location'

    def _add_memory_shape_features(self):
        super()._add_memory_shape_features()
        # V1 resolves loop labels during trip-count annotation, before action
        # attachment. Filter unsupported matches before *both* phases. The
        # complete registration list is restored on the public result below.
        self.unresolved_action_evidence = {}
        eligible = []
        for spec in self.actions:
            if spec.kind != 'loop':
                eligible.append(spec); continue
            try:
                _, resolution = self._resolve_loop_action(spec)
                if resolution != 'source_location':
                    raise ValueError('No exact compiler source-location binding; ordinal fallback is not accepted')
                eligible.append(spec)
            except (RuntimeError, ValueError) as exc:
                self.unresolved_action_evidence[spec.action_id] = str(exc)
        self.actions = eligible

    def _attach_array_actions(self):
        registered = self.actions
        try:
            for spec in registered:
                if spec.kind != 'array': continue
                self.actions = [spec]
                try:
                    super()._attach_array_actions()
                except RuntimeError as exc:
                    if not str(exc).startswith(f'Array action {spec.action_id}/'): raise
                    self.unresolved_action_evidence[spec.action_id] = str(exc)
        finally:
            self.actions = registered


def generate(source, metadata, mlir, paired_v1=None, provenance=None, revision='2.0'):
    """Keep the owning MLIR module alive until every live-object query finishes."""
    from mlir._mlir_libs import _mailohls_analysis
    fixture = (provenance or {}).get('native_fixture', {})
    valid_fixture = revision == '2.1' and fixture.get('origin') == 'handwritten_MLIR_native_verified' and fixture.get('mlir_sha256') == digest(mlir)
    if paired_v1 is None and not ((provenance and provenance.get('cgeist_sha256')) or valid_fixture):
        raise ValueError('External raw MLIR requires an authenticated paired V1 graph')
    _, actions = v1.load_kernel_info_actions(Path(source).resolve(), Path(metadata).resolve())
    if paired_v1 is not None:
        name = paired_v1.graph.get('name', '')
        if not name.startswith('mailohls-meta-v1:'):
            raise ValueError('Paired V1 graph lacks native provenance')
        recorded = json.loads(name.split(':', 1)[1])
        for key, actual in [('source_sha256', digest(source)), ('action_sha256', digest(metadata)), ('mlir_sha256', digest(mlir)),
                            ('analysis_extension_sha256', digest(_mailohls_analysis.__file__))]:
            if recorded.get(key) != actual:
                raise ValueError(f'V1/V2 comparison context differs: {key}')
        provenance = {**(provenance or {}), 'v1_provenance': recorded}
    context, module, text = v1.parse_mlir_module(Path(mlir).resolve(), True)
    builder = result = None
    started = time.perf_counter()
    try:
        if revision == '2.1' and not module.operation.verify():
            raise ValueError('Native MLIR verification failed')
        # Globals do not define function-local SSA results, but their constant
        # initializers/visibility are still semantic facts. Keep them once on
        # the corresponding memory object, not as auxiliary token nodes.
        definitions = {}
        def collect_globals(operation):
            name = v1.operation_name(operation)
            if name in ('memref.global','llvm.mlir.global'):
                attrs = v1.attribute_items(operation)
                symbol = _symbol(attrs.get('sym_name','')) if revision == '2.1' else v1.strip_mlir_string(attrs.get('sym_name',''))
                if symbol:
                    definitions[symbol] = {'operation': name, 'attributes': attrs, 'status': 'compiler_recorded'}
            elif name not in ('func.func','llvm.func'):
                for region in v1.operation_regions(operation):
                    for block in v1.region_blocks(region):
                        for child in v1.block_operations(block): collect_globals(child)
        collect_globals(module.operation)
        analysis = _mailohls_analysis.analyze_module(module.operation)
        builder_class = v1.MLIRGraphBuilder if revision=='2.0' else _CoverageBuilder
        builder = builder_class(module, text, actions, compiler_analysis=analysis,
            analysis_extension_sha256=digest(_mailohls_analysis.__file__), require_actions=revision == '2.0')
        result = builder.build()
        if revision=='2.1': result.actions = actions
        shared_seconds = time.perf_counter() - started
        started = time.perf_counter()
        graph = build_from_result(result, {**(provenance or {}), 'source_sha256': digest(source),
            'metadata_sha256': digest(metadata), 'mlir_sha256': digest(mlir),
            'analysis_extension_sha256': digest(_mailohls_analysis.__file__),
            'v1_generator_sha256': digest(v1.__file__), 'v2_builder_sha256': digest(__file__),
            'local_header_sha256': {p.name: digest(p) for p in sorted(Path(source).parent.glob('*.h'))},
            'header_limit': 'Local headers sealed; transitive system include closure is not captured. AP analysis stubs are not vendor arithmetic.',
            **({'unresolved_registration_evidence':builder.unresolved_action_evidence} if revision=='2.1' else {})}, paired_v1, definitions, revision, source, metadata)
        timing = {'shared_native_analysis_and_v1_builder_seconds': shared_seconds,
                  'additional_v2_construction_seconds': time.perf_counter() - started,
                  'historical_v1_construction_seconds': None}
        return graph, timing
    finally:
        result = builder = module = None
        gc.collect()
        context.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('generate', 'compact', 'validate'), default='generate')
    parser.add_argument('--revision', choices=('2.0', '2.1'), default='2.0')
    parser.add_argument('--source'); parser.add_argument('--metadata'); parser.add_argument('--mlir')
    parser.add_argument('--mlir-output', help='Preserve frontend IR for a fresh source compilation')
    parser.add_argument('--input'); parser.add_argument('--output')
    parser.add_argument('--v1-graph'); parser.add_argument('--cgeist', default='cgeist')
    parser.add_argument('--compiler-symbol'); parser.add_argument('--cflag', action='append', default=[])
    args = parser.parse_args()
    if args.mode != 'generate':
        if not args.input or (args.mode == 'compact' and not args.output):
            parser.error('--input is required, and compact also requires --output')
        graph = read_graph(args.input)
        if args.mode == 'compact':
            graph = compact_view(graph)
            write_graph(graph, args.output)
        print(canonical_json(validate(graph)))
        return
    if not args.source or not args.output:
        parser.error('generate requires --source and --output')
    if args.mlir and not args.v1_graph:
        parser.error('--mlir requires --v1-graph to authenticate the source/compiler context')
    source = Path(args.source).resolve()
    metadata = Path(args.metadata).resolve() if args.metadata else source.with_name('kernel_info.txt')
    paired = nx.read_gexf(args.v1_graph) if args.v1_graph else None
    if args.mlir:
        graph, timing = generate(source, metadata, args.mlir, paired, revision=args.revision)
    else:
        with tempfile.TemporaryDirectory(prefix='mailohls_v2_') as directory:
            path = Path(directory)
            kernel, actions = v1.load_kernel_info_actions(source, metadata)
            manifest = path / 'actions.tsv'
            manifest.write_text(''.join(f'{a.action_id}\t{a.kind}\t{a.source_file}\t{a.source_line}\t{a.source_column}\n' for a in actions))
            flags = list(args.cflag) + [f'--force-attribute={name}:noinline' for name in sorted({a.function for a in actions if a.function != kernel})]
            command, _ = v1.compile_source_to_mlir(source=source, kernel=kernel, mlir_output=path/'input.mlir',
                action_manifest=manifest, cgeist=args.cgeist, cflags=flags, compiler_symbol=args.compiler_symbol)
            if args.mlir_output:
                saved = Path(args.mlir_output)
                if saved.exists() and saved.read_bytes() != (path/'input.mlir').read_bytes():
                    raise ValueError('Refusing to overwrite a different frontend module')
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes((path/'input.mlir').read_bytes())
            graph, timing = generate(source, metadata, path/'input.mlir', paired,
                {'cgeist_sha256': digest(command[0]), 'frontend_arguments': [a for a in command if str(path) not in a]}, args.revision)
    write_graph(graph, args.output)
    print(canonical_json({**validate(graph), **timing}))


if __name__ == '__main__':
    main()
