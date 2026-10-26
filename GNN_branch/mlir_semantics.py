"""V2.1 evidence annotations over the existing graph, not another builder.

Native MLIR attributes and SSA identities are authoritative. Source snippets are
labelled lexical evidence and never establish aliasing, dependence or bit widths.
"""
from collections import defaultdict
from pathlib import Path
import re

from GNN_branch import mlir_graph_gen as frontend


def operands(graph, operation):
    return {int(e['legacy_evidence']['position']): u for u, _, e in graph.in_edges(operation, data=True)
            if e['relation_type'] == 'operand'}


def definitions(graph, value):
    return [(u, int(e['legacy_evidence']['position'])) for u, _, e in graph.in_edges(value, data=True)
            if e['relation_type'] == 'defines']


def resolve_returned_memory(graph, roots, constraints):
    """Resolve finite return/view/select paths with exact call-scoped bindings.

    All possible return operands must be understood for a unique proof. Multiple
    alternatives remain possible. No argument-position-only fallback is used.
    """
    returns, bindings = defaultdict(list), defaultdict(list)
    for u, v, edge in graph.edges(data=True):
        if edge['relation_type'] == 'returns_to': returns[v].append((u, edge))
        if edge['relation_type'] == 'actual_to_formal': bindings[edge['call_operation']].append((u, v, edge))

    def resolve(value, context, calls=(), seen=()):
        if value in seen: return {}, False, []
        if value in context: return context[value]
        defining = definitions(graph, value)
        if len(defining) == 1:
            operation, index = defining[0]
            node = graph.nodes[operation]
            if node.get('call_identity'):
                if operation in calls or node.get('call_resolution') != 'proven': return {}, False, []
                call_context, receipts = {}, []
                for actual, formal, binding in bindings[operation]:
                    if not graph.nodes[formal].get('is_memory'): continue
                    found, complete, path = resolve(actual, context, calls, seen + (value,))
                    receipt = {**binding, 'actual': actual, 'formal': formal}
                    path = path + [receipt]
                    if binding['evidence_status'] != 'proven':
                        found = {root: 'possible' for root in found}
                    call_context[formal] = (found, complete, path)
                options = [(u, e) for u, e in returns[value] if e['call_operation'] == operation and e['result_index'] == index]
                if not options: return {}, False, []
                merged, complete = {}, True
                for returned, edge in options:
                    found, known, path = resolve(returned, call_context, calls + (operation,), ())
                    complete &= known
                    for root, status in found.items():
                        if root in merged and merged[root] != status: status = 'possible'
                        merged[root] = status
                    receipts.extend(path + [{'call_operation': operation, 'call_uid': node['op_uid'],
                        'result_index': index, 'return_operand': returned, 'result_value': value}])
                if not complete or len(merged) != 1: merged = {root: 'possible' for root in merged}
                return merged, complete, receipts
            if node.get('text') in ('affine.load','memref.load') and graph.nodes[value].get('is_memory'):
                # This frontend sometimes lowers conditional pointer returns
                # through a local rank-zero memref<memref<...>> temporary.
                # Preserve store candidates; without reaching-definition and
                # branch coverage proof, even one candidate is not must-alias.
                slot = operands(graph, operation).get(0)
                if slot and graph.nodes[slot].get('memory_rank') == 0:
                    candidates = [args[0] for n,a in graph.nodes(data=True)
                        if a.get('text') in ('affine.store','memref.store')
                        for args in [operands(graph,n)] if args.get(1)==slot and graph.nodes[args[0]].get('is_memory')]
                    found, receipts = {}, []
                    for candidate in candidates:
                        next_roots, _, path = resolve(candidate,context,calls,seen+(value,))
                        found.update({root:'possible' for root in next_roots});receipts.extend(path)
                    return found, False, receipts
            # Selects and region results keep alternatives, never type-based alias proof.
            candidates = []
            if node.get('text') == 'arith.select':
                candidates = [v for i, v in operands(graph, operation).items() if i in (1, 2)]
            elif node.get('text') == 'scf.if':
                candidates = [operands(graph, n)[index] for n, a in graph.nodes(data=True)
                    if a.get('parent_operation') == operation and a.get('text') == 'scf.yield'
                    and index in operands(graph, n)]
            else:
                candidates = [v for _, v, e in graph.out_edges(value, data=True) if e['relation_type'] == 'references']
            if candidates:
                found, complete, path = {}, True, []
                for candidate in candidates:
                    next_roots, known, next_path = resolve(candidate, context, calls, seen + (value,))
                    complete &= known
                    for root,status in next_roots.items():
                        found[root]='possible' if found.get(root)=='possible' or status=='possible' else 'proven'
                    path.extend(next_path)
                if not complete or len(found) != 1: found = {root: 'possible' for root in found}
                return found, complete, path
        known = roots.get(value, set())
        # A formal is a symbolic object inside its own function. It is never
        # substituted with every actual argument of every caller.
        return {root: 'proven' if len(known) == 1 and not graph.nodes[value].get('possible_return_alias') else 'possible' for root in known}, bool(known), []

    for value in sorted(returns):
        found, complete, receipts = resolve(value, {})
        if not found: continue
        roots[value].update(found)
        attrs = graph.nodes[value]
        attrs['return_root_evidence'] = {'roots': found, 'all_returns_resolved': complete, 'bindings': receipts}
        if not complete or len(found) != 1 or any(status != 'proven' for status in found.values()):
            attrs['possible_return_alias'] = True
    # Forward call-result evidence through local compiler-proven views/iter args.
    for _ in range(len(roots) + 1):
        changed = False
        for value in sorted(roots):
            before = (len(roots[value]), bool(graph.nodes[value].get('possible_return_alias')))
            for parent in constraints[value]:
                roots[value].update(roots.get(parent, set()))
                if graph.nodes[parent].get('possible_return_alias'): graph.nodes[value]['possible_return_alias'] = True
            changed |= before != (len(roots[value]), bool(graph.nodes[value].get('possible_return_alias')))
        if not changed: break


def _dense_attribute(operation, name):
    """Decode compiler arrays if supported; always retain their original text."""
    raw = frontend.get_raw_attribute(operation, name)
    if raw is None: return None
    from mlir import ir
    for cls in ('DenseI64ArrayAttr', 'DenseIntElementsAttr'):
        try:
            return [None if int(v) == -(1 << 63) else int(v) for v in getattr(ir, cls)(raw)]
        except (TypeError, ValueError, AttributeError): pass
    return {'unresolved_encoding': str(raw)}


def index_dependencies(graph, indices):
    """SSA dependence on induction variables; does not claim affine independence."""
    pending, seen, variables = list(indices), set(), set()
    while pending:
        value = pending.pop()
        if value in seen: continue
        seen.add(value)
        if graph.nodes[value].get('induction_for'):
            variables.add(graph.nodes[value]['induction_for']); continue
        for operation, _ in definitions(graph, value):
            pending.extend(operands(graph, operation).values())
    return sorted(variables)


def _loop_annotations(graph):
    blocks = {n: a for n, a in graph.nodes(data=True) if a.get('control_kind') == 'block'}
    for block, attrs in blocks.items():
        region = graph.nodes[attrs['parent_region']]
        owner = region['owner']
        if graph.nodes[owner].get('text') not in ('scf.for', 'affine.for'): continue
        args = [v for _, v, e in graph.out_edges(block, data=True) if e['relation_type'] == 'declares_argument']
        for value in args:
            # All single-IV for bodies use block argument zero for the IV.
            if graph.nodes[value].get('block_argument_index') == 0 or graph.nodes[value]['ssa_id'].endswith(':a0'):
                graph.nodes[value]['induction_for'] = owner
            else:
                graph.nodes[value]['iteration_argument_for'] = owner
    for u, v, edge in graph.edges(data=True):
        if edge['relation_type'] == 'loop_carried':
            role = edge['legacy_evidence']['role']
            owner = graph.nodes[v].get('iteration_argument_for')
            if not owner and role == 'loop_result':
                owner = next((op for op, _ in definitions(graph, v) if graph.nodes[op].get('is_loop')), None)
            if owner:
                edge.update(carrying_loop=owner, state_kind='memory_reference' if graph.nodes[v].get('is_memory') else 'scalar',
                            forwarding_role=role, evidence_origin='verified_MLIR_region_SSA')
        elif edge['relation_type'] == 'depends_on':
            dep = edge['source_analysis']
            edge['loop_carried'] = dep.get('loop_carried')
            edge['possible_loop_carried'] = dep.get('possible_loop_carried')
            left = [n for n in graph.nodes[v].get('loop_context', []) if graph.nodes[n].get('text') == 'affine.for']
            right = [n for n in graph.nodes[u].get('loop_context', []) if graph.nodes[n].get('text') == 'affine.for']
            common = []
            for a, b in zip(left, right):
                if a != b: break
                common.append(a)
            depth = dep.get('dependence_depth', 0)
            carried = edge['evidence_status'] == 'proven' and dep.get('loop_carried') and 0 < depth <= len(common)
            edge['carrying_loop'] = common[depth - 1] if carried else None
            edge['carrying_loop_status'] = 'proven' if carried else 'not_established'
            edge['common_affine_loops'] = common
            edge['distance_bounds'] = [{'loop': loop, **component} for loop, component in zip(common, dep.get('components', []))]
            edge['distance_exact'] = [d['lower'] if d['lower'] is not None and d['lower'] == d['upper'] else None for d in edge['distance_bounds']]
            edge['analysis_limit'] = 'Native extension records the first successful depth; absence of other carrying loops is not independence'


def _coverage(graph, result, source, metadata):
    labels = frontend._source_actions(Path(source))
    registered = {a.action_id: a for a in result.actions}
    active = {a['action_id']: n for n, a in graph.nodes(data=True) if a['category'] == 'action_point'}
    records = []
    for label in sorted(set(labels) | set(registered), key=lambda s: int(s[1:])):
        spec, point = registered.get(label), active.get(label)
        item = {'action_id': label, 'source_labeled': label in labels, 'metadata_registered': spec is not None,
                'compiler_matched': point is not None, 'compiler_eliminated': False if point else None,
                'compiler_status': 'compiler_matched' if point else 'compiler_unresolved' if spec else 'not_registered',
                'source_binding': labels.get(label), 'action_point_node': point,
                'elimination_note': 'Absence of a match does not establish compiler elimination'}
        if spec and not point:
            item['resolution_evidence']=graph.graph['provenance'].get('unresolved_registration_evidence',{}).get(label)
        if spec: item['metadata_value'] = spec.kernel_info_loop_value if spec.kind == 'loop' else list(spec.array_dimensions)
        if point:
            target = next(v for _, v, e in graph.out_edges(point, data=True) if e['relation_type'] == 'targets')
            item['compiler_target'] = target
            attrs = graph.nodes[target]
            if spec.kind == 'loop':
                count = attrs.get('trip_count') if attrs.get('trip_count_static') else None
                graph.nodes[point].update(metadata_value=spec.kernel_info_loop_value,
                    compiler_proven_trip_count=count, tcad_trip_count=count,
                    trip_count_provenance=attrs.get('trip_count_source', 'unknown'))
                item.update(compiler_proven_trip_count=count, trip_count_provenance=attrs.get('trip_count_source', 'unknown'))
            else:
                item['compiler_memory_shape'] = attrs.get('memory_shape')
                line=Path(source).read_text().splitlines()[spec.source_line-1]
                match=frontend.SOURCE_LABEL_RE.match(line)
                body=match.group('body') if match else line
                prefix=body.split(spec.variable,1)[0].strip() if spec.variable in body else None
                graph.nodes[point]['source_declared_type']={'spelling':prefix,'evidence':'source_text_not_compiler_verified','bit_width':None,'signedness':None}
        records.append(item)
    graph.graph['action_point_coverage'] = {'source': Path(source).name, 'registration_metadata': Path(metadata).name, 'records': records}


def _numeric_fidelity(graph, source):
    from mlir import ir
    files = [Path(source)] + sorted(Path(source).parent.glob('*.h'))
    declarations, headers = [], []
    for path in files:
        text = path.read_text(errors='replace')
        if path.name in ('ap_fixed.h', 'ap_int.h'):
            known_stub = bool(re.search(r'AP_(?:FIXED|INT)_STUB|(?:fixed.point|control.flow).*analysis|using\s+ap_fixed\s*=\s*float', text, re.I))
            headers.append({'file': path.name, 'analysis_stub_marker': known_stub,
                            'hardware_exact': False if known_stub else None})
        for match in re.finditer(r'\btypedef\s+[^;]*\b(?:ap_fixed|ap_uint|ap_int)\s*<[^;]+;', text):
            declarations.append({'file': path.name, 'line': text.count('\n', 0, match.start()) + 1,
                'declaration': match.group(), 'evidence': 'source_text_not_compiler_verified',
                'source_bit_width': None, 'source_signedness': None})
    graph.graph['numerical_fidelity'] = {'source_declarations': declarations, 'arithmetic_headers': headers,
        'hardware_exact': False if any(h['analysis_stub_marker'] for h in headers) else None,
        'header_hash_scope':'current local snapshot; historical raw-IR include closure is not authenticated' if graph.graph['provenance'].get('v1_provenance') else 'local frontend snapshot only; transitive include closure is not captured',
        'reason': 'Lowered widths describe this compiler module only; source template parameters are not hardware-verified. Transitive headers are not fully sealed.'}
    for _, attrs in graph.nodes(data=True):
        if attrs['category'] != 'ssa_value': continue
        spelling = attrs.get('value_type', '')
        match = re.fullmatch(r'(s?i|ui|f)(\d+)', spelling)
        attrs['lowered_numerical_type'] = {'mlir_type': spelling,
            'bit_width': int(match[2]) if match else None,
            'signedness': {'i':'signless', 'si':'signed', 'ui':'unsigned', 'f':'not_applicable'}.get(match[1]) if match else None,
            'source_type_correspondence': 'not_established'}
        try:
            parsed=ir.Type.parse(spelling)
            if ir.MemRefType.isinstance(parsed):
                element=ir.MemRefType(parsed).element_type
                lowered=attrs['lowered_numerical_type']
                lowered['element_type']=str(element)
                match=re.fullmatch(r'(s?i|ui|f)(\d+)',str(element))
                lowered['element_bit_width']=int(match[2]) if match else None
        except (TypeError,ValueError,RuntimeError): pass


def annotate_semantics(graph, result, source, metadata):
    """Attach native evidence before the owning MLIR context is released."""
    from GNN_branch.mlir_graph_v2 import digest
    graph.graph['provenance']['semantic_annotation_sha256'] = digest(__file__)
    _loop_annotations(graph)
    records = {f'op:{r.function_name}:op{r.function_ordinal}': r for r in result.operation_records}
    for u, v, edge in graph.edges(data=True):
        if edge['relation_type'] == 'references':
            defining = definitions(graph, u)
            if not defining: continue
            operation = defining[0][0]; record = records.get(operation)
            if not record: continue
            name = record.op_name
            semantics = {'memref.subview':'relative_to_source_view', 'memref.reinterpret_cast':'relative_to_underlying_storage',
                         'memref.cast':'shape_layout_compatible_identity'}.get(name, 'mapping_retained_not_normalized')
            edge['view_mapping'] = {'operation': operation, 'operation_kind': name, 'offset_semantics': semantics,
                'offsets': _dense_attribute(record.operation, 'static_offsets'),
                'sizes': _dense_attribute(record.operation, 'static_sizes'),
                'strides': _dense_attribute(record.operation, 'static_strides'),
                'operands': [{'position': i, 'value': v} for i, v in sorted(operands(graph, operation).items())], 'attributes': dict(record.attributes),
                'source_type': graph.nodes[v].get('value_type'), 'result_type': graph.nodes[u].get('value_type'),
                'source_shape': graph.nodes[v].get('memory_shape'), 'result_shape': graph.nodes[u].get('memory_shape'),
                'result_rank': graph.nodes[u].get('memory_rank'), 'composed_offset': None,
                'storage_identity':'same_underlying_object', 'element_overlap':'not_established'}
        if edge['relation_type'] == 'memory_access':
            op = graph.nodes[u]; args = operands(graph, u); index_positions = []
            if op.get('text') in ('memref.load', 'affine.load'): index_positions = list(range(1, len(args)))
            if op.get('text') in ('memref.store', 'affine.store'): index_positions = list(range(2, len(args)))
            indices = [args[i] for i in index_positions]
            edge['access_indices'] = {'ssa_operands': indices, 'operand_positions': index_positions,
                'affine_map': op.get('mlir_attrs', {}).get('map'), 'induction_loops': index_dependencies(graph, indices),
                'status': 'compiler_recorded' if index_positions or op.get('text') in ('memref.load','memref.store','affine.load','affine.store') else 'not_established'}
            edge['origin_loop'] = op.get('loop_context', [])[-1] if op.get('loop_context') else None
    _coverage(graph, result, source, metadata)
    _numeric_fidelity(graph, source)


def validate_semantics(graph):
    """Check evidence chains, not merely graph connectivity or plausible counts."""
    def root_path(value, root, seen=(), context=None, calls=()):
        context=context or {}
        if value in seen: return False
        if value in context:
            actual,caller_context=context[value]
            return root_path(actual,root,seen+(value,),caller_context,calls)
        obj=graph.nodes[root]
        if obj.get('root_kind')=='unresolved':return False
        if value in obj.get('root_values',[obj.get('root_value')]):
            return bool(definitions(graph,value)) if obj.get('root_kind')!='formal_parameter' else graph.nodes[value].get('function_argument_index',-1)>=0
        defining=definitions(graph,value)
        if len(defining)==1:
            operation,index=defining[0];node=graph.nodes[operation]
            if node.get('call_identity'):
                if operation in calls or node.get('call_resolution')!='proven':return False
                bound={v:(u,context) for u,v,e in graph.edges(data=True) if e['relation_type']=='actual_to_formal' and e['call_operation']==operation and e['evidence_status']=='proven'}
                returned=[u for u,_,e in graph.in_edges(value,data=True) if e['relation_type']=='returns_to' and e['call_operation']==operation and e['result_index']==index]
                return bool(returned) and all(root_path(v,root,(),bound,calls+(operation,)) for v in returned)
            candidates=[]
            if node.get('text')=='arith.select':candidates=[v for i,v in operands(graph,operation).items() if i in (1,2)]
            if node.get('text')=='scf.if':
                candidates=[operands(graph,n)[index] for n,a in graph.nodes(data=True) if a.get('parent_operation')==operation and a.get('text')=='scf.yield' and index in operands(graph,n)]
            if candidates:return all(root_path(v,root,seen+(value,),context,calls) for v in candidates)
        parents=[v for _,v,e in graph.out_edges(value,data=True) if e['relation_type']=='references']
        parents += [u for u,_,e in graph.in_edges(value,data=True) if e['relation_type']=='loop_carried']
        parents += [v if u==value else u for u,v,e in list(graph.in_edges(value,data=True))+list(graph.out_edges(value,data=True))
                    if e['relation_type']=='alias' and e.get('alias_kind')=='must_alias' and graph.nodes[u].get('function')==graph.nodes[v].get('function')]
        return any(root_path(parent,root,seen+(value,),context,calls) for parent in parents)
    for value,attrs in graph.nodes(data=True):
        evidence=attrs.get('return_root_evidence')
        if not evidence:continue
        for receipt in evidence['bindings']:
            call=receipt['call_operation']
            valid=call in graph and graph.nodes[call].get('op_uid')==receipt['call_uid']
            if 'formal' in receipt:
                valid &= any(v==receipt['formal'] and e['relation_type']=='actual_to_formal' and e['call_operation']==call and e['argument_index']==receipt['argument_index'] for _,v,e in graph.out_edges(receipt['actual'],data=True))
            else:
                valid &= any(v==receipt['result_value'] and e['relation_type']=='returns_to' and e['call_operation']==call and e['result_index']==receipt['result_index'] for _,v,e in graph.out_edges(receipt['return_operand'],data=True))
            if not valid:raise ValueError('Return memory evidence chain has an invalid binding receipt')
    for u, v, edge in graph.edges(data=True):
        kind = edge['relation_type']
        if kind == 'resolves_to' and edge['evidence_status'] == 'proven' and not root_path(u, v):
            raise ValueError('Proven memory root lacks a valid evidence chain')
        if kind == 'references' and 'view_mapping' not in edge:
            raise ValueError('Memory view lost its compiler mapping')
        if kind == 'references' and 'view_mapping' in edge:
            mapping = edge['view_mapping']; operation = mapping['operation']
            if operation not in graph or operation not in {op for op, _ in definitions(graph, u)} or v not in operands(graph, operation).values():
                raise ValueError(f'View mapping has no matching compiler operands/results: {u} -> {v}, operation {operation}, operands {operands(graph, operation)}')
            if mapping['operation_kind']!=graph.nodes[operation]['text'] or mapping['attributes']!=graph.nodes[operation].get('mlir_attrs',{}):
                raise ValueError('View mapping disagrees with compiler attributes')
        if kind == 'actual_to_formal' and edge['evidence_status'] == 'proven':
            call = graph.nodes[edge['call_operation']]
            if graph.nodes[u].get('function') != call.get('function') or graph.nodes[u].get('value_type') != graph.nodes[v].get('value_type') or edge.get('caller_function')!=call.get('caller_function') or edge.get('callee_function')!=call.get('callee_function'):
                raise ValueError('Proven binding crosses the wrong caller/type context')
        if kind == 'depends_on':
            source, target = edge['source_analysis']['source'], edge['source_analysis']['target']
            if v != f'op:{source["function"]}:op{source["ordinal"]}' or u != f'op:{target["function"]}:op{target["ordinal"]}':
                raise ValueError('Dependence direction disagrees with compiler record')
            carrying = edge.get('carrying_loop')
            if carrying and (edge['evidence_status'] != 'proven' or carrying not in graph.nodes[u].get('loop_context', []) or carrying not in graph.nodes[v].get('loop_context', [])):
                raise ValueError('Dependence carrying loop is outside access scopes')
        if kind == 'memory_access':
            if edge.get('origin_loop') != (graph.nodes[u].get('loop_context', []) or [None])[-1]:
                raise ValueError('Access origin disagrees with compiler loop context')
            if not set(edge['access_indices']['ssa_operands']) <= set(operands(graph, u).values()):
                raise ValueError('Access index is not a compiler operand')
            info=edge['access_indices']
            if info['affine_map']!=graph.nodes[u].get('mlir_attrs',{}).get('map') or info['induction_loops']!=index_dependencies(graph,info['ssa_operands']):
                raise ValueError('Access map/index dependencies disagree with compiler operands')
    for node,attrs in graph.nodes(data=True):
        if attrs.get('category')!='operation' or 'loop_context' not in attrs:continue
        parent=attrs.get('parent_operation');loops=[];seen=set()
        while parent is not None:
            if parent in seen:raise ValueError('Cyclic control-region ownership')
            seen.add(parent)
            if graph.nodes[parent].get('text') in frontend.LOOP_OPS:loops.append(parent)
            parent=graph.nodes[parent].get('parent_operation')
        if list(reversed(loops))!=attrs['loop_context']:raise ValueError('Loop context disagrees with real region ownership')
    coverage = graph.graph.get('action_point_coverage', {}).get('records', [])
    for record in coverage:
        if record['compiler_matched'] != ('action:' + record['action_id'] in graph):
            raise ValueError('Action-point coverage disagrees with graph')
        if record['compiler_matched'] and not record['metadata_registered']:
            raise ValueError('An unregistered source label became an active action point')
        if record['compiler_matched']:
            point=graph.nodes[record['action_point_node']]
            target=next(v for _,v,e in graph.out_edges(record['action_point_node'],data=True) if e['relation_type']=='targets')
            if target!=record['compiler_target']:raise ValueError('Action-point coverage has a different compiler target')
            if point['action_kind']=='loop':
                count=graph.nodes[target].get('trip_count') if graph.nodes[target].get('trip_count_static') else None
                if point['metadata_value']!=record['metadata_value'] or point['metadata_value']!=point['source_loop_metadata_value'] or point['compiler_proven_trip_count']!=count or record['compiler_proven_trip_count']!=count or point['tcad_trip_count']!=count:
                    raise ValueError('Source/compiler trip-count distinction was corrupted')
