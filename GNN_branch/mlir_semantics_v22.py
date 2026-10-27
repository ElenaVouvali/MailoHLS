"""Opt-in V2.2 evidence checks; historical graph semantics stay in their modules.

Storage identity is weaker than element aliasing or dependence. Pointer views
below preserve an allocation root, never establish equal addresses or overlap.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import ctypes
from functools import lru_cache

from GNN_branch.mlir_semantics import operands, definitions

POINTER_VIEWS = {
    'llvm.getelementptr': 'base_pointer_with_element_indices',
    'llvm.bitcast': 'pointer_representation_cast',
    'llvm.addrspacecast': 'pointer_address_space_cast',
    'polygeist.memref2pointer': 'aligned_pointer_of_source_memref',
    'polygeist.pointer2memref': 'view_of_source_pointer',
}
LOOPS = {'scf.for','scf.forall','scf.parallel','scf.while','affine.for','affine.parallel'}


class _Handle(ctypes.Structure):
    _fields_=[('ptr',ctypes.c_void_p)]


def native_pointer(value):
    capsule=value._CAPIPtr
    name=ctypes.pythonapi.PyCapsule_GetName
    name.argtypes=[ctypes.py_object];name.restype=ctypes.c_char_p
    pointer=ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes=[ctypes.py_object,ctypes.c_char_p];pointer.restype=ctypes.c_void_p
    return pointer(capsule,name(capsule))


@lru_cache(maxsize=1)
def _capi():
    from mlir._mlir_libs import _mlir
    library=ctypes.CDLL(str(Path(_mlir.__file__).parent/'libMLIRPythonCAPI.so'))
    library.mlirOperationGetNumSuccessors.argtypes=[_Handle]
    library.mlirOperationGetNumSuccessors.restype=ctypes.c_ssize_t
    library.mlirOperationGetSuccessor.argtypes=[_Handle,ctypes.c_ssize_t]
    library.mlirOperationGetSuccessor.restype=_Handle
    return library


def native_successors(operation):
    """Pinned Python lacks Operation.successors; use its public, read-only C API."""
    library=_capi();handle=_Handle(native_pointer(operation))
    return [library.mlirOperationGetSuccessor(handle,i).ptr
            for i in range(library.mlirOperationGetNumSuccessors(handle))]


def attach_cfg(graph,result):
    from GNN_branch.mlir_graph_v2 import add_relation
    blocks={native_pointer(r.block):r for r in result.blocks.values()}
    for record in result.operation_records:
        source=f'op:{record.function_name}:op{record.function_ordinal}'
        for index,pointer in enumerate(native_successors(record.operation)):
            block=blocks[pointer]
            if not block.operations:raise ValueError('CFG successor has no terminator operation')
            first=block.operations[0];target=f'op:{first.function_name}:op{first.function_ordinal}'
            if any(v==target and e['relation_type']=='control_flow' and e.get('legacy_evidence',{}).get('position')==index+1
                   for _,v,e in graph.out_edges(source,data=True)):continue
            add_relation(graph,source,target,'control_flow',successor_index=index,
                native_evidence='mlirOperationGetSuccessor',
                legacy_evidence={'flow':0,'role':'block_successor','position':index+1})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def canonical_analysis(analysis):
    """Native view roots iterate a Block* map; order is not a semantic fact.

    Sort identities, retaining every fact and its multiplicity. The pinned
    binary is unchanged; aliases/dependences and their outcomes are untouched.
    """
    normalized=deepcopy(analysis)
    normalized['views']=sorted(normalized['views'],key=lambda r:json.dumps(r,sort_keys=True))
    return normalized


def pointer_constraints(graph, constraints):
    """Forward only native, function-local, storage-preserving pointer operands."""
    from GNN_branch.mlir_graph_v2 import add_relation
    for operation,node in list(graph.nodes(data=True)):
        if node.get('text') not in POINTER_VIEWS:
            continue
        base = operands(graph,operation).get(0)
        results = [v for _,v,e in graph.out_edges(operation,data=True) if e['relation_type']=='defines']
        if base is None or not graph.nodes[base].get('is_memory'):
            continue
        for value in results:
            if not graph.nodes[value].get('is_memory') or graph.nodes[value]['function']!=graph.nodes[base]['function']:
                continue
            constraints[value].add(base)
            add_relation(graph,value,base,'references',source_analysis={
                'operation':operation,'operand_index':0,'reason':'native_storage_preserving_pointer_operation'},
                storage_relation=POINTER_VIEWS[node['text']], element_overlap='not_established')


def region_roots(graph, roots, constraints):
    """Preserve select/conditional alternatives without treating may as must."""
    alternatives = {}
    for value in roots:
        defs = definitions(graph,value)
        if len(defs)!=1: continue
        operation,index=defs[0];kind=graph.nodes[operation].get('text')
        if kind=='arith.select':
            args=operands(graph,operation);alternatives[value]=[args[i] for i in (1,2) if i in args]
        elif kind in ('scf.if','affine.if'):
            alternatives[value]=[operands(graph,n)[index] for n,a in graph.nodes(data=True)
                if a.get('parent_operation')==operation and a.get('text') in ('scf.yield','affine.yield')
                and index in operands(graph,n)]
    for _ in range(len(roots)+1):
        changed=False
        for value,choices in alternatives.items():
            found=set().union(*(roots.get(v,set()) for v in choices)) if choices else set()
            complete=len(choices)==2 and all(roots.get(v) for v in choices)
            possible=not complete or len(found)!=1 or any(graph.nodes[v].get('possible_return_alias') for v in choices)
            before=(len(roots[value]),bool(graph.nodes[value].get('possible_return_alias')))
            roots[value].update(found)
            if possible:graph.nodes[value]['possible_return_alias']=True
            graph.nodes[value]['region_root_evidence']={'alternatives':choices,'all_alternatives_resolved':complete,
                'storage_uniqueness':'possible' if possible else 'proven','operation':definitions(graph,value)[0][0]}
            changed |= before!=(len(roots[value]),bool(graph.nodes[value].get('possible_return_alias')))
        for value in roots:
            for parent in constraints[value]:
                before=(len(roots[value]),bool(graph.nodes[value].get('possible_return_alias')))
                roots[value].update(roots.get(parent,set()))
                if graph.nodes[parent].get('possible_return_alias'):graph.nodes[value]['possible_return_alias']=True
                changed |= before!=(len(roots[value]),bool(graph.nodes[value].get('possible_return_alias')))
        if not changed:break


def annotate(graph, source):
    from GNN_branch.mlir_graph_v2 import digest
    graph.graph['provenance']['semantic_extension_v22_sha256']=digest(__file__)
    graph.graph['evidence_policy']={
        'positive_statuses':['proven','possible','not_established'],
        'explicit_negative_results':'compiler_analysis aliases:no_alias and dependence no_dependence/proven_no_alias; not positive connectivity',
        'missing_relation':'no independence conclusion',
        'native_dependence_depth_coverage':'first successful depth only; other depths unverified',
        'pointer_root':'same underlying object, not equal address or proven element overlap',
    }
    for value,attrs in graph.nodes(data=True):
        if attrs.get('category')!='ssa_value' or not attrs.get('is_memory'):continue
        defs=definitions(graph,value)
        attrs['root_resolution_diagnostic']={
            'status':attrs.get('root_resolution'),
            'defining_operations':[{'operation':op,'kind':graph.nodes[op]['text']} for op,_ in defs],
            'reason':'unique_compiler_storage_chain' if attrs.get('root_resolution')=='proven' else
                     'multiple_or_incomplete_memory_alternatives' if attrs.get('root_resolution')=='possible' else
                     'loaded_pointer_requires_reaching_store_and_invocation_analysis' if any(graph.nodes[op]['text'] in ('affine.load','memref.load','llvm.load') for op,_ in defs) else
                     'no_supported_unique_allocation_formal_view_or_return_chain'}
    for u,v,edge in graph.edges(data=True):
        if edge['relation_type']=='loop_carried' and edge['legacy_evidence']['role']=='loop_zero_trip_result':
            owner=definitions(graph,v)[0][0]
            edge.update(carrying_loop=owner,state_kind='memory_reference',forwarding_role='loop_zero_trip_result',
                evidence_origin='native_for_initial_result_on_zero_iteration_path')
        if edge['relation_type']=='references' and graph.nodes[edge['view_mapping']['operation']].get('text') in POINTER_VIEWS:
            edge['view_mapping']['offset_semantics']=POINTER_VIEWS[graph.nodes[edge['view_mapping']['operation']]['text']]
        if edge['relation_type']=='memory_access':
            edge['access_class']='concrete_resolved_storage' if edge['evidence_status']=='proven' else 'concrete_possible_storage' if edge['evidence_status']=='possible' else 'concrete_unresolved_storage'
            if graph.nodes[u]['text'] in ('llvm.load','llvm.store'):
                pointer=edge['memory_operand'];chain=[];seen=set()
                while pointer not in seen:
                    seen.add(pointer);found=definitions(graph,pointer)
                    if len(found)!=1:break
                    op,_=found[0]
                    if graph.nodes[op]['text'] not in POINTER_VIEWS:break
                    args=operands(graph,op)
                    chain.append({'operation':op,'operation_kind':graph.nodes[op]['text'],
                        'base':args.get(0),'index_operands':[v for i,v in sorted(args.items()) if i>0],
                        'attributes':graph.nodes[op].get('mlir_attrs',{})})
                    pointer=args.get(0)
                    if pointer is None:break
                edge['access_indices']['pointer_expression_chain']=chain
                from GNN_branch.mlir_semantics import index_dependencies
                edge['access_indices']['pointer_induction_loops']=index_dependencies(graph,
                    [value for step in chain for value in step['index_operands']])
                edge['access_indices']['status']='compiler_recorded_pointer_SSA; affine_form_not_established'
    for record in graph.graph['action_point_coverage']['records']:
        if not record['metadata_registered'] or record['compiler_matched']:continue
        binding=record.get('source_binding') or {};lines=set(binding.get('source_use_lines',[]))
        record['resolution_diagnostic']={
            'disposition':'compiler_unresolved; no independent storage target established',
            'elimination_proven':False,
            'source_declaration':{'file':binding.get('source_file'),'line':binding.get('source_line'),'array':binding.get('array_name')},
            'source_line_context_not_object_identity':[n for n,a in graph.nodes(data=True)
                if a.get('category')=='operation' and a.get('source_line') in lines and a.get('source_file')==binding.get('source_file')],
            'explanation':'Scalar computation may survive without a separate array. Shared use-line overlap cannot identify a declaration; no transformation receipt proves elimination.'}
    fidelity=graph.graph['numerical_fidelity']
    fidelity['type_evidence_contract']={'source':'lexical declarations, not AST-verified template substitution',
        'lowered':'native MLIR types checked occurrence by occurrence','correspondence':'not_established',
        'quantization_rounding_overflow':'only explicit source spellings; defaults and hardware behavior unverified'}
    templates=[]
    for path in [Path(source)]+sorted(Path(source).parent.glob('*.h')):
        for match in re.finditer(r'\b(ap_fixed|ap_ufixed|ap_int|ap_uint)\s*<([^<>;\n]+)>',path.read_text(errors='replace')):
            templates.append({'file':path.name,'line':path.read_text(errors='replace').count('\n',0,match.start())+1,
                'template':match[1],'arguments_lexical':[x.strip() for x in match[2].split(',')],
                'evidence':'source_text_not_compiler_AST_verified','resolved_width':None,
                'resolved_integer_bits':None,'hardware_semantics_verified':False})
    fidelity['source_template_spellings']=templates


def validate_evidence(graph):
    """Additional portable consistency checks; native inspection is separate."""
    native=graph.graph['compiler_analysis']
    dep_records={fingerprint(r) for r in native['dependences']}
    aliases={fingerprint(r) for r in native['aliases']}
    for u,v,edge in graph.edges(data=True):
        kind=edge['relation_type']
        if edge.get('is_reverse'):raise ValueError('Neural reverse edges are not compiler facts')
        if kind=='depends_on':
            dep=edge['source_analysis']
            if fingerprint(dep) not in dep_records:raise ValueError('Dependence has no native analysis record')
            expected='proven' if dep['result']=='dependence' else 'possible'
            if edge['evidence_status']!=expected:raise ValueError('Dependence certainty disagrees with native outcome')
            if edge['dependence_kind']!=dep['kind']:raise ValueError('Dependence kind mismatch')
            common=edge['common_affine_loops'];depth=dep.get('dependence_depth',0)
            carrying=common[depth-1] if expected=='proven' and dep.get('loop_carried') and 0<depth<=len(common) else None
            if edge.get('carrying_loop')!=carrying:raise ValueError('Dependence carrying-loop depth mismatch')
            if edge.get('distance_components')!=dep.get('components'):raise ValueError('Dependence distance evidence mismatch')
        elif kind=='alias':
            evidence=edge['source_analysis']
            if fingerprint(evidence) not in aliases or edge['alias_kind']!=evidence['result']:
                raise ValueError('Alias disagrees with native classification')
            if {u,v}!={'value:'+evidence['lhs'],'value:'+evidence['rhs']}:raise ValueError('Alias operand identity mismatch')
            expected='proven' if evidence['result'] in ('must_alias','no_alias') else 'possible'
            if edge['evidence_status']!=expected:raise ValueError('Alias certainty mismatch')
        elif kind=='references':
            mapping=edge['view_mapping'];operation=mapping['operation'];op=graph.nodes[operation]
            if op['text'] in POINTER_VIEWS:
                if operands(graph,operation).get(0)!=v or graph.nodes[u]['function']!=graph.nodes[v]['function']:
                    raise ValueError('Pointer root has wrong base or function context')
                if mapping['offset_semantics']!=POINTER_VIEWS[op['text']]:raise ValueError('Pointer view semantics mismatch')
        elif kind=='memory_access':
            operand=edge['memory_operand']
            roots=[e for _,target,e in graph.out_edges(operand,data=True) if target==v and e['relation_type']=='resolves_to']
            if not roots or any(e['evidence_status']!=edge['evidence_status'] for e in roots):
                raise ValueError('Concrete access storage/certainty does not match its operand root')
            if operands(graph,u).get(edge['operand_index'])!=operand:
                raise ValueError('Concrete access has a different native operand position')
            effect=edge['source_analysis']
            if u!=f'op:{effect["function"]}:op{effect["ordinal"]}':raise ValueError('Concrete access native effect identity mismatch')
            modes={e['kind'] for e in effect['effects'] if e.get('value')==graph.nodes[operand]['ssa_id'] and e.get('operand_index')==edge['operand_index']}
            expected={'read','write'} if edge['access']=='readwrite' else {edge['access']}
            if not expected<=modes:raise ValueError('Concrete access mode disagrees with native memory effect')
            if 'pointer_expression_chain' in edge['access_indices']:
                from GNN_branch.mlir_semantics import index_dependencies
                chain=edge['access_indices']['pointer_expression_chain']
                expected=index_dependencies(graph,[value for step in chain for value in step['index_operands']])
                if edge['access_indices']['pointer_induction_loops']!=expected:raise ValueError('Pointer index dependency mismatch')
        elif kind=='targets' and graph.nodes[u]['action_kind']=='array':
            if graph.nodes[u]['target_binding_rule']=='native_action_attribute':
                label=graph.nodes[u]['action_id']
                if not any(graph.nodes[op].get('mlir_attrs',{}).get('mailohls.action_id','').strip('"')==label
                    for value in graph.nodes[v].get('root_values',[graph.nodes[v].get('root_value')])
                    for op,_ in definitions(graph,value)):
                    raise ValueError('Array action target contradicts native action identity')
    receipt=graph.graph.get('native_correspondence')
    if receipt and receipt.get('graph_witness_sha256')!=fingerprint(graph_witness(graph)):
        raise ValueError('Native occurrence witness mismatch: operation/SSA/CFG/call or loop evidence was changed')


def graph_witness(graph):
    return {'operations':sorted((n,a.get('text'),a.get('operand_count'),a.get('result_count'),
                                a.get('parent_operation'),a.get('loop_context'),a.get('mlir_attrs'))
                               for n,a in graph.nodes(data=True) if a['category']=='operation'),
        'ssa':sorted((n,a['value_type']) for n,a in graph.nodes(data=True) if a['category']=='ssa_value'),
        'edges':sorted((u,v,e['relation_type'],e.get('legacy_evidence',{}).get('position',-1),
                        e.get('legacy_evidence',{}).get('role',''),e.get('call_operation',''),e.get('argument_index',-1),e.get('result_index',-1))
                       for u,v,e in graph.edges(data=True) if e['relation_type'] in
                       ('operand','defines','control_flow','next_operation','loop_child','loop_carried','actual_to_formal','returns_to','owns_region','contains_block','declares_argument','contains_operation'))}


def native_correspondence(module, graph):
    """Independent occurrence-level MLIR walk: no builder indexes or graph IDs used as an oracle."""
    from mlir import ir
    operations={};values={};blocks={};records=[];block_number=0;expected_state=[];block_owners={}
    functions=[];globals_={}
    def discover(op):
        op=op.operation
        if op.name in ('func.func','llvm.func'):
            functions.append(op);return
        if op.name in ('memref.global','llvm.mlir.global'):
            symbol=ir.StringAttr(op.attributes['sym_name']).value
            globals_[symbol]={'operation':op.name,'attributes':{a.name:str(a.attr) for a in op.attributes},'status':'compiler_recorded'}
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:discover(child)
    discover(module.operation)
    if globals_!=graph.graph.get('global_definitions',{}):raise ValueError('Native global-definition coverage differs')
    functions.sort(key=lambda op:ir.StringAttr(op.attributes['sym_name']).value)
    for function in functions:
        symbol=ir.StringAttr(function.attributes['sym_name']).value
        top='op:'+symbol+':function';operations[top]=function
        ordinal=0
        def walk(owner,parent,ancestors):
            nonlocal ordinal,block_number
            for region_index,region in enumerate(owner.regions):
                for block in region.blocks:
                    number=block_number;block_number+=1
                    block_id=f'block:{symbol}:{number}';blocks[block]=block_id
                    block_owners[block_id]=(parent,f'region:{parent}:{region_index}')
                    for index,value in enumerate(block.arguments):values[value]=f'value:{symbol}:b{number}:a{index}'
                    for order,child in enumerate(block.operations):
                        op=child.operation;node=f'op:{symbol}:op{ordinal}';ordinal+=1
                        operations[node]=op
                        for index,value in enumerate(op.results):values[value]=f'value:{symbol}:op{ordinal-1}:r{index}'
                        records.append((node,op,parent,tuple(ancestors),block_id,order,region_index))
                        walk(op,node,ancestors+([node] if op.name in LOOPS else []))
        walk(function,top,[])
    actual_ops={n for n,a in graph.nodes(data=True) if a['category']=='operation'}
    if actual_ops!=set(operations):raise ValueError('Native operation identity coverage differs')
    actual_values={n for n,a in graph.nodes(data=True) if a['category']=='ssa_value'}
    if actual_values!=set(values.values()):
        raise ValueError('Native SSA identity coverage differs: missing='+str(sorted(set(values.values())-actual_values))+' extra='+str(sorted(actual_values-set(values.values()))))
    expected_operands=Counter();expected_defines=Counter();expected_order=set();expected_successors=set();previous={}
    for node,op,parent,ancestors,block_id,order,region in records:
        attrs=graph.nodes[node]
        if attrs['text']!=op.name or attrs['operand_count']!=len(op.operands) or attrs['result_count']!=len(op.results):
            raise ValueError('Native operation kind/arity mismatch: '+node)
        if attrs.get('mlir_attrs',{})!={a.name:str(a.attr) for a in op.attributes}:
            raise ValueError('Native operation attributes mismatch: '+node)
        if tuple(attrs['loop_context'])!=ancestors or attrs['parent_operation']!=parent or attrs['block_order']!=order:
            raise ValueError('Native control/loop ownership mismatch: '+node)
        incoming=[u for u,_,e in graph.in_edges(node,data=True) if e['relation_type']=='contains_operation']
        if incoming!=[block_id]:raise ValueError('Native block ownership mismatch: '+node)
        for index,value in enumerate(op.operands):expected_operands[values[value],node,index]+=1
        for index,value in enumerate(op.results):expected_defines[node,values[value],index]+=1
        if block_id in previous:expected_order.add((previous[block_id],node))
        previous[block_id]=node
        for position,pointer in enumerate(native_successors(op)):
            successor=next(block for block in blocks if native_pointer(block)==pointer)
            children=list(successor.operations)
            if children:
                target=next(n for n,o in operations.items() if o==children[0].operation)
                expected_successors.add((node,target,position+1))
        # For-loop and while forwarding checked from native region operands.
        if op.name in ('scf.for','affine.for'):
            body=op.regions[0].blocks[0];iterations=list(body.arguments)[1:]
            initial=list(op.operands)[3:] if op.name=='scf.for' else list(op.operands)[-len(iterations):] if iterations else []
            yielded=list(body.operations)[-1].operation
            for i,(a,b) in enumerate(zip(initial,iterations)):expected_state.append((values[a],values[b],'iter_init',i))
            for i,(a,result) in enumerate(zip(initial,op.results)):
                if str(result.type).startswith(('memref<','!llvm.ptr')):
                    expected_state.append((values[a],values[result],'loop_zero_trip_result',400+i))
            for i,value in enumerate(yielded.operands):
                if i<len(iterations):expected_state.append((values[value],values[iterations[i]],'loop_backedge',100+i))
                if i<len(op.results):expected_state.append((values[value],values[op.results[i]],'loop_result',200+i))
        elif op.name=='scf.while':
            before,after=op.regions[0].blocks[0],op.regions[1].blocks[0]
            condition=list(before.operations)[-1].operation;yielded=list(after.operations)[-1].operation
            for i,(a,b) in enumerate(zip(op.operands,before.arguments)):expected_state.append((values[a],values[b],'while_init',i))
            for i,value in enumerate(list(condition.operands)[1:]):
                if i<len(after.arguments):expected_state.append((values[value],values[after.arguments[i]],'while_condition_to_after',100+i))
                if i<len(op.results):expected_state.append((values[value],values[op.results[i]],'while_result',200+i))
            for i,(a,b) in enumerate(zip(yielded.operands,before.arguments)):expected_state.append((values[a],values[b],'while_backedge',300+i))
    for value,node in values.items():
        if graph.nodes[node]['value_type']!=str(value.type):raise ValueError('Native SSA type mismatch: '+node)
        try:argument=ir.BlockArgument(value)
        except (TypeError,ValueError):argument=None
        if argument is not None:
            owner=blocks[argument.owner]
            if not any(u==owner and e['relation_type']=='declares_argument' for u,_,e in graph.in_edges(node,data=True)):
                raise ValueError('Native block argument ownership mismatch: '+node)
    for block,(owner,region) in block_owners.items():
        if graph.nodes[block]['parent_region']!=region or graph.nodes[region]['owner']!=owner:
            raise ValueError('Native region/block owner mismatch')
    expected_children={(ancestors[-1],node) for node,op,_,ancestors,_,_,_ in records if op.name in LOOPS and ancestors}
    if {(u,v) for u,v,e in graph.edges(data=True) if e['relation_type']=='loop_child'}!=expected_children:
        raise ValueError('Native immediate loop hierarchy mismatch')
    functions_by_symbol={ir.StringAttr(op.attributes['sym_name']).value:op for op in functions}
    for u,v,e in graph.edges(data=True):
        if e['relation_type']=='actual_to_formal':
            call=operations[e['call_operation']]
            callee=ir.FlatSymbolRefAttr(call.attributes['callee']).value
            function=functions_by_symbol[callee];formal=function.regions[0].blocks[0].arguments[e['argument_index']]
            if u!=values[call.operands[e['argument_index']]] or v!=values[formal] or e['call_uid']!=e['call_operation'][3:]:
                raise ValueError('Native actual/formal call occurrence mismatch')
        elif e['relation_type']=='returns_to':
            call=operations[e['call_operation']]
            if v!=values[call.results[e['result_index']]]:
                raise ValueError('Native call result occurrence mismatch')
            callee=ir.FlatSymbolRefAttr(call.attributes['callee']).value
            candidates=[values[o.operands[e['result_index']]] for n,o in operations.items()
                if n.startswith('op:'+callee+':op') and o.name in ('func.return','llvm.return') and e['result_index']<len(o.operands)]
            if u not in candidates:raise ValueError('Native returned operand occurrence mismatch')
    for kind,expected in [('operand',expected_operands),('defines',expected_defines)]:
        actual=Counter((u,v,int(e['legacy_evidence']['position'])) for u,v,e in graph.edges(data=True) if e['relation_type']==kind)
        if actual!=expected:raise ValueError('Native occurrence-level '+kind+' correspondence differs')
    if {(u,v) for u,v,e in graph.edges(data=True) if e['relation_type']=='next_operation'}!=expected_order:
        raise ValueError('Native sequential operation order mismatch')
    actual_successors={(u,v,int(e['legacy_evidence']['position'])) for u,v,e in graph.edges(data=True)
        if e['relation_type']=='control_flow' and e['legacy_evidence'].get('role')=='block_successor'}
    if actual_successors!=expected_successors:raise ValueError('Native CFG successors mismatch')
    actual_state=Counter((u,v,e['legacy_evidence']['role'],int(e['legacy_evidence']['position'])) for u,v,e in graph.edges(data=True) if e['relation_type']=='loop_carried')
    if actual_state!=Counter(expected_state):raise ValueError('Native loop-carried SSA forwarding mismatch')
    # Read/write effects are checked against native analysis, not root guesses.
    expected_accesses=set()
    for effect in graph.graph['compiler_analysis']['effects']:
        op=f'op:{effect["function"]}:op{effect["ordinal"]}'
        if graph.nodes[op]['operation_family']=='call':continue
        for e in effect['effects']:
            value='value:'+str(e.get('value'))
            if e['kind'] in ('read','write') and value in graph and graph.nodes[value].get('is_memory'):
                expected_accesses.add((op,value,e.get('operand_index'),e['kind']))
    actual_accesses={(u,e['memory_operand'],e.get('operand_index'),mode) for u,_,e in graph.edges(data=True)
        if e['relation_type']=='memory_access' for mode in (('read','write') if e['access']=='readwrite' else (e['access'],))}
    if actual_accesses!=expected_accesses:raise ValueError('Supported native concrete memory effects lost or fabricated')
    return {'status':'pass','oracle':'independent native MLIR Python traversal, plus native MemoryEffectOpInterface records',
        'operations':len(operations),'ssa_values':len(values),'operand_occurrences':sum(expected_operands.values()),
        'result_occurrences':sum(expected_defines.values()),'blocks':len(blocks),
        'cfg_successors':len(expected_successors),'loop_state_occurrences':len(expected_state),
        'concrete_effect_occurrences':len(expected_accesses),'graph_witness_sha256':fingerprint(graph_witness(graph)),
        'global_definitions':len(globals_),
        'compiler_module_attributes':{a.name:str(a.attr) for a in module.operation.attributes},
        'scope_limit':'Function operations/SSA/regions checked occurrence by occurrence; global definitions checked as symbol-keyed attributes. Native dependence truth beyond recorded outcomes is not exhaustively independently proven.'}
