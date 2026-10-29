"""Compiler-grounded, non-synthesis action workload and candidate features.

Static descriptors are target/pragma/QoR independent. Candidate descriptors
require a complete source-valid action assignment and never predict a schedule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path

from GNN_branch import action_point_relations as queries
from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import target_graph_v22 as overlay

STATIC_SCHEMA = 'salt-hls-action-workload-v22-v1'
CANDIDATE_SCHEMA = 'salt-hls-candidate-demand-v22-v1'
IMPLEMENTATION_VERSION = '1.0.0'
ARITHMETIC = ('integer_add_sub', 'floating_add_sub', 'integer_multiply',
              'floating_multiply', 'division', 'other_arithmetic', 'unsupported_arithmetic')
NUMERIC_FEATURES = (
    'trip_count', 'loop_depth', 'direct_arithmetic', 'descendant_arithmetic',
    'integer_add_sub', 'floating_add_sub', 'integer_multiply', 'floating_multiply',
    'division', 'other_arithmetic', 'unsupported_arithmetic',
    'static_reads', 'static_writes', 'static_readwrites', 'invocation_access_witnesses',
    'resolved_roots', 'proven_root_relations', 'possible_root_relations', 'unestablished_root_relations',
    'incomplete_root_accesses', 'direct_accesses', 'nested_accesses', 'interprocedural_accesses',
    'proven_RAW', 'possible_RAW', 'proven_WAR', 'possible_WAR', 'proven_WAW', 'possible_WAW',
    'loop_carried_proven', 'loop_carried_possible', 'ssa_loop_carried_state',
    'known_affine_accesses', 'unknown_index_accesses', 'access_dimensions',
    'current_loop_index_accesses', 'other_loop_index_accesses', 'unknown_loop_index_accesses',
    'analysis_stub_header_count', 'source_type_correspondence_not_established',
    'hardware_exactness_not_established',
)
CANDIDATE_FEATURES = (
    'pipeline_requested', 'unroll_factor', 'nominal_lanes',
    'nominal_arithmetic_occurrences', 'nominal_read_occurrences',
    'nominal_write_occurrences', 'nominal_readwrite_occurrences',
    'nominal_memory_access_occurrences', 'known_dependence_count', 'array_partitioned',
    'partition_factor', 'partition_dimension', 'nominal_bank_count',
    'distinct_banks_touched', 'bank_index_collisions', 'bank_analysis_access_count',
)


def _sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def implementation_identity():
    return {'version': IMPLEMENTATION_VERSION,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _program(graph):
    if graph.graph.get('schema') == overlay.SCHEMA:
        graph=overlay.strip_overlay(graph)
    if graph.graph.get('schema') != compiler.SCHEMA_V22 or graph.graph.get('semantic_revision') != 'rc3':
        raise ValueError('V2.2 RC3 program graph or existing target overlay required')
    compiler.validate(graph)
    if graph.graph.get('view') == 'compact':
        graph=compiler.expand_compact(graph)
    return graph


def _target_map(graph):
    return {a['action_id']: v for u, v, _, e in graph.edges(keys=True, data=True)
            if e['relation_type'] == 'targets' and graph.nodes[u].get('category') == 'action_point'
            for a in [graph.nodes[u]]}


def _operation_bucket(attrs):
    name = attrs.get('text', '')
    if name == 'arith.constant': return None
    types=list(attrs.get('result_types') or [])+list(attrs.get('operand_types') or [])
    integer_types=bool(types) and all(re.fullmatch(r'[su]?i\d+',str(t)) or str(t)=='index' for t in types)
    float_types=bool(types) and all(re.fullmatch(r'(?:bf|f)\d+',str(t)) for t in types)
    if name in ('arith.addi', 'arith.subi'): return 'integer_add_sub' if integer_types else 'unsupported_arithmetic'
    if name in ('arith.addf', 'arith.subf'): return 'floating_add_sub' if float_types else 'unsupported_arithmetic'
    if name == 'arith.muli': return 'integer_multiply' if integer_types else 'unsupported_arithmetic'
    if name == 'arith.mulf': return 'floating_multiply' if float_types else 'unsupported_arithmetic'
    if name.startswith('arith.divf'):
        return 'division' if float_types else 'unsupported_arithmetic'
    if name.startswith(('arith.divsi','arith.divui','arith.ceildivsi','arith.ceildivui',
                        'arith.floordivsi','arith.floordivui')):
        return 'division' if integer_types else 'unsupported_arithmetic'
    if name.startswith(('arith.div','arith.ceildiv','arith.floordiv')):
        return 'unsupported_arithmetic'
    if name.startswith(('arith.rem', 'arith.max',
                        'arith.min', 'arith.negf', 'arith.fma', 'arith.absi')):
        return 'other_arithmetic' if types else 'unsupported_arithmetic'
    # These Arith dialect operations are not arithmetic workload operations.
    if name.startswith(('arith.cmp', 'arith.select', 'arith.index_cast', 'arith.ext',
                        'arith.trunc', 'arith.bitcast', 'arith.andi', 'arith.ori',
                        'arith.xori', 'arith.shli', 'arith.shrsi', 'arith.shrui')):
        return None
    if name.startswith('arith.'):
        return 'unsupported_arithmetic'
    return None


def _known(value, status='proven', witnesses=(), interpretation=''):
    return {'value': value, 'known': value is not None, 'evidence_status': status if value is not None else 'not_established',
            'source_witnesses': sorted(set(map(str, witnesses))), 'interpretation': interpretation}


def _access_summary(accesses, loop=None):
    # One static access operation may have alternative roots; keep operation,
    # root relation and invocation witness counts as separate units.
    byop = defaultdict(list)
    for row in accesses:
        byop[row['source_operation']].append(row)
    witness_keys = set()
    access_witness_keys = defaultdict(set)
    roots = set(); possible_root_candidates=set(); proven_roots = set(); possible_roots = set(); unknown_roots=set()
    incomplete = set()
    scopes = set()
    affine = []
    access_witnesses=[]
    for op, rows in byop.items():
        for row in rows:
            context = tuple(row.get('invocation_context', row.get('call_chain', ())))
            witness_keys.add((op, context))
            if row.get('access') in ('read','write','readwrite'):
                access_witness_keys[row['access']].add((op,context))
            scopes.add((op, context, row.get('access_scope', row.get('access_path', 'unknown'))))
            root = row.get('memory_root')
            if root:
                relation = (op, root);status=row.get('evidence_status')
                if status=='proven': roots.add(root);proven_roots.add(relation)
                elif status=='possible': possible_root_candidates.add(root);possible_roots.add(relation)
                else: unknown_roots.add(relation)
            if not row.get('root_candidates_complete', False): incomplete.add((op, context))
            access_witnesses.append({'operation':op,'memory_root':root,'access':row.get('access'),
                'root_evidence_status':row.get('evidence_status','not_established'),
                'root_candidates_complete':row.get('root_candidates_complete',False),
                'scope':row.get('access_scope',row.get('access_path','unknown')),
                'call_chain':list(context),'supporting_edge':row.get('supporting_edge'),
                'access_indices':deepcopy(row.get('access_indices'))})
            indices = row.get('access_indices') or {}
            if indices.get('status') == 'compiler_recorded' and indices.get('affine_map'):
                affine.append((op, indices))
    scope_counts = Counter(scope for _, _, scope in scopes)
    counts={'static_operations':len(byop),'reads':len({op for op,rs in byop.items() if any(r.get('access')=='read' for r in rs)}),
            'writes':len({op for op,rs in byop.items() if any(r.get('access')=='write' for r in rs)}),
            'readwrites':len({op for op,rs in byop.items() if any(r.get('access')=='readwrite' for r in rs)}),
            'invocation_witnesses':len(witness_keys),'distinct_roots':len(roots),
            'possible_root_candidates':len(possible_root_candidates),
            'invocation_reads':len(access_witness_keys['read']),'invocation_writes':len(access_witness_keys['write']),
            'invocation_readwrites':len(access_witness_keys['readwrite']),
            'proven_root_relations':len(proven_roots),'possible_root_relations':len(possible_roots),
            'incomplete_root_accesses':len(incomplete),'unestablished_root_relations':len(unknown_roots)}
    op_mode={mode:{op for op,rows in byop.items() if any(row.get('access')==mode for row in rows)}
             for mode in ('read','write','readwrite')}
    edge_witnesses=[r for r in access_witnesses if r.get('supporting_edge') is not None]
    witnesses={
        'static_operations':sorted(byop),
        'reads':sorted(op_mode['read']),'writes':sorted(op_mode['write']),'readwrites':sorted(op_mode['readwrite']),
        'distinct_roots':[r['supporting_edge'] for r in edge_witnesses if r['root_evidence_status']=='proven'],
        'possible_root_candidates':[r['supporting_edge'] for r in edge_witnesses if r['root_evidence_status']=='possible'],
        'proven_root_relations':[r['supporting_edge'] for r in edge_witnesses if r['root_evidence_status']=='proven'],
        'possible_root_relations':[r['supporting_edge'] for r in edge_witnesses if r['root_evidence_status']=='possible'],
        'unestablished_root_relations':[r['supporting_edge'] for r in edge_witnesses if r['root_evidence_status']=='not_established'],
        'incomplete_root_accesses':[r['supporting_edge'] for r in edge_witnesses if not r['root_candidates_complete']],
        'invocation_witnesses':[r['supporting_edge'] for r in edge_witnesses],
        'invocation_reads':[r['supporting_edge'] for r in edge_witnesses if r['access']=='read'],
        'invocation_writes':[r['supporting_edge'] for r in edge_witnesses if r['access']=='write'],
        'invocation_readwrites':[r['supporting_edge'] for r in edge_witnesses if r['access']=='readwrite']}
    access_features={name:_known(value,witnesses=witnesses.get(name,()),
        interpretation='Observed compiler access operations/invocation witnesses; not dynamic operations or hardware ports')
        for name,value in counts.items()}
    return {**counts, 'count_features':access_features,
            'scope_witnesses':dict(sorted(scope_counts.items())),
            'affine_accesses': affine,
            'access_witnesses':sorted(access_witnesses,key=lambda r:(r['operation'],r['call_chain'],str(r['supporting_edge']))),
            'operation_ids': sorted(byop)}


def _dependence_summary(graph, loop, semantics):
    rows = {r['supporting_edge']: r for r in semantics.get('memory_dependences', [])}
    kinds = Counter((r.get('dependence_kind'), r.get('evidence_status')) for r in rows.values())
    carried = Counter(r.get('evidence_status') for r in rows.values() if r.get('loop_carried') is True)
    carried_possible = sum(r.get('loop_carried') is None and r.get('evidence_status') == 'possible'
                           for r in rows.values())
    state = semantics.get('ssa_state', [])
    carried_proven=carried['proven'];carried_uncertain=carried['possible']+carried_possible
    return {'dependence_analysis_complete': _known(None, witnesses=(), interpretation='Absence of dependence edges is not proof of independence'),
            'dependence_edge_count': len(rows),
            'counts': {f'{kind}_{status}': count for (kind, status), count in sorted(kinds.items())},
            'count_features':{f'{kind}_{status}':_known(count,witnesses=[r['supporting_edge'] for r in rows.values()
                if r.get('dependence_kind')==kind and r.get('evidence_status')==status],
                interpretation='Observed compiler dependence edges for this kind/evidence state')
                for (kind,status),count in sorted(kinds.items())},
            'loop_carried_proven': carried_proven, 'loop_carried_possible': carried_uncertain,
            'scalar_features':{'loop_carried_proven':_known(carried_proven,witnesses=[r['supporting_edge'] for r in rows.values() if r.get('loop_carried') is True and r.get('evidence_status')=='proven'],interpretation='Observed dependence records with proven carrying status'),
                'loop_carried_possible':_known(carried_uncertain,witnesses=[r['supporting_edge'] for r in rows.values() if r.get('loop_carried') is True or r.get('loop_carried') is None],interpretation='Observed records with possible dependence/carrying evidence'),
                'ssa_loop_carried_state':_known(len(state),witnesses=[r['supporting_edge'] for r in state],interpretation='Compiler-recorded loop-carried SSA state edges')},
            'dependences': [dict(kind=r.get('dependence_kind'), evidence_status=r.get('evidence_status'),
                                 loop_carried=r.get('loop_carried'), carrying_loop=r.get('carrying_loop'),
                                 distance_exact=r.get('distance_exact'), distance_bounds=r.get('distance_bounds'),
                                 witness=r['supporting_edge']) for r in sorted(rows.values(), key=lambda x:x['supporting_edge'])],
            'ssa_loop_carried_state': len(state),
            'ssa_state_witnesses': sorted(r['supporting_edge'] for r in state)}


def _index_summary(accesses, loop):
    records = {}
    for row in accesses:
        index = row.get('access_indices') or {}
        op = row['source_operation']
        records.setdefault(op, index)
    known = [r for r in records.values() if r.get('status') == 'compiler_recorded' and r.get('affine_map')]
    unknown = [r for r in records.values() if r not in known]
    current = [r for r in known if loop in r.get('induction_loops', [])]
    other = [r for r in known if loop not in r.get('induction_loops', [])]
    dimensions=[]
    for record in known:
        parsed=_split_affine_map(record['affine_map'])
        dimensions.append(len(parsed['results']) if parsed is not None else None)
    unknown_loop=len(unknown)
    return {'known_affine_accesses': len(known), 'unknown_accesses': len(unknown),
            'count_features':{'known_affine_accesses':_known(len(known),witnesses=[op for op,r in records.items() if r in known],
                    interpretation='Unique access operations with compiler-recorded affine maps'),
                'unknown_accesses':_known(len(unknown),witnesses=[op for op,r in records.items() if r in unknown],
                    interpretation='Unique access operations without a usable affine map')},
            'access_dimensions': dimensions,
            'access_dimension_features':[_known(size,witnesses=[op],interpretation='Result rank of a compiler-recorded memory access affine map')
                for (op,record),size in zip(((op,r) for op,r in records.items() if r in known),dimensions)],
            'current_loop_accesses': len(current), 'other_loop_accesses': len(other),
            'unknown_loop_dependency_accesses': unknown_loop,
            'scalar_features':{'current_loop_accesses':_known(len(current),witnesses=[r.get('affine_map') for r in current],interpretation='Compiler-recorded accesses whose induction list contains this loop'),
                'other_loop_accesses':_known(len(other),witnesses=[r.get('affine_map') for r in other],interpretation='Compiler-recorded accesses not containing this loop'),
                'unknown_loop_dependency_accesses':_known(unknown_loop,witnesses=[op for op,r in records.items() if r in unknown],interpretation='Accesses with no usable affine description; loop-index relation is unresolved')},
            'witnesses': [{'operation': op, 'status': attrs.get('status', 'not_established'),
                           'affine_map': attrs.get('affine_map'), 'induction_loops': attrs.get('induction_loops'),
                           'ssa_operands': attrs.get('ssa_operands'), 'operand_positions': attrs.get('operand_positions')}
                          for op, attrs in sorted(records.items())]}


def _static_loop(graph, action, target, coverage):
    semantics = queries.get_loop_semantics(graph, action)
    ops = [n for n, a in graph.nodes(data=True) if a.get('category') == 'operation' and target in a.get('loop_context', [])]
    direct, descendant = [], []
    counts = Counter()
    arithmetic_witnesses = {}
    for op in ops:
        attrs = graph.nodes[op]
        bucket = _operation_bucket(attrs)
        if bucket:
            counts[bucket] += 1
            arithmetic_witnesses.setdefault(bucket, []).append(op)
            (direct if (attrs.get('loop_context') or [None])[-1] == target else descendant).append(op)
    accesses = semantics['accesses']
    access = _access_summary(accesses)
    deps = _dependence_summary(graph, target, semantics)
    idx = _index_summary(accesses, target)
    depth = len(graph.nodes[target].get('loop_context', []))
    trip = semantics.get('compiler_proven_trip_count')
    return {'action_id': action, 'kind': 'loop', 'matched': True, 'compiler_target': target,
            'program_graph_identity': graph.graph['graph_identity'],
            'numerical_fidelity': deepcopy(graph.graph.get('numerical_fidelity', {})),
            'features': {
                'compiler_proven_trip_count': _known(trip, witnesses=[target] if trip is not None else (),
                    interpretation='Static compiler trip count; not multiplied through nested loops'),
                'source_metadata_trip_count': _known(semantics.get('tcad_trip_count'),
                    status='possible' if semantics.get('tcad_trip_count') is not None else 'not_established',
                    witnesses=[action], interpretation='Source metadata only; not compiler proof'),
                'loop_depth': _known(depth, witnesses=[target], interpretation='Lexical parent-loop count'),
                'direct_arithmetic_operation_count': _known(len(direct), witnesses=direct,
                    interpretation='Unique compiler operation nodes directly in this loop body'),
                'descendant_arithmetic_operation_count': _known(len(descendant), witnesses=descendant,
                    interpretation='Unique compiler operation nodes in nested loop descendants'),
                'arithmetic_operation_counts': {name:_known(counts[name], witnesses=arithmetic_witnesses.get(name, ()),
                    interpretation='Unique compiler operation nodes; not DSP/LUT usage') for name in ARITHMETIC},
                'memory_accesses': access,
                'dependences': deps,
                'access_indices': idx,
                'nested_loop_actions': sorted(set(r['neighbor'] for r in queries.get_action_point_relations(graph, action)
                    if r['relation_type']=='loop_hierarchy' and r.get('hierarchy')=='child')),
                'unknown_dynamic_multiplicity': _known(None, interpretation='Dynamic executions, path predicates and nested-loop product not inferred'),
            },
            'provenance': {'graph_identity': graph.graph['graph_identity'], 'action_coverage': deepcopy(coverage),
                           'query_api': 'GNN_branch.action_point_relations.get_loop_semantics',
                           'implementation': implementation_identity()}}


def _array_shape(graph, target, coverage):
    attrs = graph.nodes[target]
    shape = attrs.get('memory_shape')
    status = 'proven' if isinstance(shape, (list, tuple)) and all(type(x) is int and x > 0 for x in shape) else 'not_established'
    action_attrs=graph.nodes.get('action:'+coverage.get('action_id',''),{})
    source_dims=action_attrs.get('source_array_dimensions',coverage.get('source_binding',{}).get('array_dimensions'))
    return {'compiler_shape': _known(list(shape) if status == 'proven' else None, status, [target],
                'Compiler memory-object shape; physical allocation is not asserted'),
            'source_declared_dimensions': _known(source_dims, 'possible' if source_dims else 'not_established',
                [coverage.get('action_id',''),action_attrs.get('source_file',''),action_attrs.get('source_line','')],
                'Source action metadata dimensions, separate from compiler shape'),
            'rank': _known(len(shape) if status == 'proven' else None, status, [target], 'Compiler-known rank')}


def _static_array(graph, action, target, coverage):
    relations = queries.get_action_point_relations(graph, action, evidence='all')
    accesses = [r for r in relations if r['relation_type'] == 'memory_access']
    access = _access_summary(accesses)
    roots = []
    for _, root, key, edge in graph.edges(keys=True, data=True):
        if edge['relation_type'] == 'resolves_to' and root == target:
            roots.append({'root':_, 'status':edge['evidence_status'], 'witness':str(key)})
    loops = sorted(set(r['neighbor'] for r in relations if r['relation_type']=='memory_access' and r['neighbor'].startswith('L')))
    view_rows={json.dumps(view,sort_keys=True):deepcopy(view) for row in accesses for view in row.get('memory_views',[])}
    view_edges=[view_rows[key] for key in sorted(view_rows)]
    return {'action_id':action,'kind':'array','matched':True,'compiler_target':target,
            'program_graph_identity':graph.graph['graph_identity'],
            'numerical_fidelity':deepcopy(graph.graph.get('numerical_fidelity',{})),
            'features':{'target_memory_object':_known(target,witnesses=[target],interpretation='Compiler-matched memory object'),
                **_array_shape(graph,target,coverage), 'memory_accesses':access,
                'access_indices':_index_summary(accesses,None), 'accessing_loop_actions':loops,
                'memory_roots':roots,'root_certainty':'proven' if roots and all(x['status']=='proven' for x in roots) else 'possible_or_incomplete',
                'views':view_edges, 'storage_binding':_known(None,interpretation='Actual local/distributed/external storage binding requires synthesis/platform mapping')},
            'provenance':{'graph_identity':graph.graph['graph_identity'],'action_coverage':deepcopy(coverage),
                'query_api':'GNN_branch.action_point_relations.get_action_point_relations','implementation':implementation_identity()}}


def static_action_descriptors(graph):
    """Return one deterministic compiler-only descriptor for each registered action.

    Unresolved actions are retained in the output with unknown feature masks and
    no guessed compiler target. Overlay contexts are stripped before querying.
    """
    program = _program(graph)
    coverage = {r['action_id']:r for r in program.graph.get('action_point_coverage',{}).get('records',[])}
    targets = _target_map(program)
    result = {}
    for action in sorted(coverage,key=lambda x:int(x[1:])):
        row=coverage[action]
        target=targets.get(action)
        if not row.get('metadata_registered'):
            result[action]={'action_id':action,'kind':row.get('source_binding',{}).get('kind'),
                'matched':False,'source_only':True,'compiler_target':None,
                'program_graph_identity':program.graph['graph_identity'],'features':None,
                'known':False,'evidence_status':'not_established',
                'reason':'Source-only registration is outside the metadata action contract',
                'provenance':{'graph_identity':program.graph['graph_identity'],'action_coverage':deepcopy(row),
                              'implementation':implementation_identity()}}
        elif not row.get('compiler_matched') or target is None:
            result[action]={'action_id':action,'kind':row.get('source_binding',{}).get('kind'),'matched':False,
                'compiler_target':None,'program_graph_identity':program.graph['graph_identity'],
                'features':None,'known':False,'evidence_status':'not_established',
                'reason':'Compiler target unresolved; no program demand inferred',
                'provenance':{'graph_identity':program.graph['graph_identity'],'action_coverage':deepcopy(row),
                              'implementation':implementation_identity()}}
        elif program.nodes[target]['category']=='operation' and program.nodes[target].get('is_loop'):
            result[action]=_static_loop(program,action,target,row)
        elif program.nodes[target]['category']=='memory_object':
            result[action]=_static_array(program,action,target,row)
        else:
            result[action]={'action_id':action,'kind':row.get('source_binding',{}).get('kind'),'matched':False,
                'compiler_target':target,'program_graph_identity':program.graph['graph_identity'],'features':None,
                'known':False,'evidence_status':'not_established','reason':'Unsupported compiler target category'}
    return {'schema':STATIC_SCHEMA,'program_graph_identity':program.graph['graph_identity'],
            'implementation':implementation_identity(),
            'actions':{label:{'action_number':int(label[1:]),'tensor_slot':int(label[1:])-1,**item} for label,item in result.items()}}


def _split_affine_map(text):
    match=re.fullmatch(r'\s*affine_map<\(([^)]*)\)(?:\[[^]]*\])?\s*->\s*\((.*)\)>\s*',str(text))
    if not match:return None
    dims=[x.strip() for x in match.group(1).split(',') if x.strip()]
    body=match.group(2);items=[];depth=0;start=0
    for i,ch in enumerate(body):
        if ch in '([':depth+=1
        elif ch in ')]':depth-=1
        elif ch==',' and depth==0:items.append(body[start:i].strip());start=i+1
    items.append(body[start:].strip())
    if not body.strip():items=[]
    out=[]
    for expr in items:
        coeff={d:0 for d in dims};offset=0
        tokens=re.findall(r'[+-]?\s*(?:\d+\s*\*\s*d\d+|d\d+\s*\*\s*\d+|d\d+|\d+)',expr)
        if not tokens or ''.join(tokens).replace(' ','')!=expr.replace(' ',''):return None
        for token in tokens:
            t=token.replace(' ','');sign=-1 if t.startswith('-') else 1;t=t.lstrip('+-')
            if '*' in t:
                left,right=t.split('*')
                if left.startswith('d'): d,c=left,right
                else: c,d=left,right
                if d not in coeff:return None
                coeff[d]+=sign*int(c)
            elif t.startswith('d'):
                if t not in coeff:return None
                coeff[t]+=sign
            else:offset+=sign*int(t)
        out.append({'coefficients':coeff,'offset':offset})
    return {'dimensions':dims,'results':out}


def cyclic_bank(index,factor):
    if type(index) is not int or type(factor) is not int or factor<1:raise ValueError('integer index and positive factor required')
    return index%factor


def block_bank(index,extent,factor):
    if any(type(x) is not int for x in (index,extent,factor)) or extent<1 or factor<1 or factor>extent or index<0 or index>=extent:
        raise ValueError('block bank requires an in-range index and 1 <= factor <= extent')
    return min(index//math.ceil(extent/factor),factor-1)


def _candidate_access_bank(row, loop_target, lane_count, mode, factor, dimension, shape):
    if mode == 'none':
        return [0] * lane_count
    info=row.get('access_indices') or {}
    affine=_split_affine_map(info.get('affine_map','')) if info.get('status')=='compiler_recorded' else None
    loops=info.get('induction_loops') or []
    if affine is None or loop_target not in loops:return None
    domain=loops.index(loop_target);dim=dimension-1
    if dimension==0:
        # Partition-all-dimensions needs multidimensional bank tuples; the
        # scalar feature API deliberately leaves that case unknown.
        return None
    if dim<0 or dim>=len(affine['results']) or dim>=len(shape):return None
    coeff=affine['results'][dim]['coefficients'].get('d'+str(domain))
    if coeff is None:return None
    if any(value for variable,value in affine['results'][dim]['coefficients'].items()
           if variable!='d'+str(domain)):
        return None
    # Unknown base iteration indices translate lane banks for cyclic partition
    # but leave pairwise collisions invariant. For block banks require origin.
    offset=affine['results'][dim]['offset']
    if mode=='cyclic':return [cyclic_bank(coeff*i+offset,factor) for i in range(lane_count)]
    if mode=='block':
        # Zero-based lane scenario is stated explicitly; unknown base is not inferred.
        if len(loops)>1:return None
        return [block_bank(coeff*i+offset,shape[dim],factor) for i in range(lane_count)]
    if mode=='complete':
        extent=shape[dim]
        return [coeff*i+offset if 0<=coeff*i+offset<extent else None for i in range(lane_count)]
    return None


def candidate_demand_descriptors(graph, assignments, actions):
    """Analyze a complete legal assignment; candidate data never enter static vectors.

    `assignments` uses the directive-domain AUTO keys and `actions` is the
    complete source-derived mapping accepted by directive_domains.
    """
    from LLM_branch.common import directive_domains as domains
    if graph.graph.get('schema') != overlay.SCHEMA:
        raise ValueError('Candidate demand requires an existing target overlay to identify FPGA/clock/budget context')
    program=_program(graph)
    domains.validate_assignments(assignments,actions)
    static=static_action_descriptors(program)['actions']
    targets=_target_map(program)
    config_hash=_sha(assignments)
    output={}
    matched_arrays={a for a,d in static.items() if d.get('matched') and d.get('kind')=='array'}
    for action,desc in static.items():
        values={name:_known(None,interpretation='Unsupported/unmatched candidate analysis') for name in CANDIDATE_FEATURES}
        if not desc.get('matched'):
            output[action]={'features':values,'matched':False,'reason':'Compiler action unresolved'};continue
        if desc['kind']=='loop':
            pipe=int(assignments[f'AUTO{{_PIPE_{action}}}']);raw=int(assignments[f'AUTO{{_UNROLL_{action}}}'])
            lanes=max(1,raw)
            f=desc['features'];access=f['memory_accesses'];arith=sum(f['arithmetic_operation_counts'][x]['value'] or 0 for x in ARITHMETIC)
            dep=f['dependences'];known_dep=dep['dependence_edge_count']
            values.update({'pipeline_requested':_known(pipe,witnesses=[action]),
                'unroll_factor':_known(raw,witnesses=[action]),'nominal_lanes':_known(lanes,witnesses=[action],interpretation='Requested factor; factor 0 represented as one lane'),
                'nominal_arithmetic_occurrences':_known(arith*lanes,witnesses=[action],interpretation='Repeated static-body assumption; not hardware operator instances'),
                'nominal_read_occurrences':_known(access['invocation_reads']*lanes,witnesses=access['operation_ids'],interpretation='Call-scoped observed read occurrences times lanes; nested dynamic counts excluded; not simultaneous ports'),
                'nominal_write_occurrences':_known(access['invocation_writes']*lanes,witnesses=access['operation_ids'],interpretation='Call-scoped observed write occurrences times lanes; nested dynamic counts excluded; not simultaneous ports'),
                'nominal_readwrite_occurrences':_known(access['invocation_readwrites']*lanes,witnesses=access['operation_ids'],interpretation='Call-scoped readwrite occurrences times lanes; nested dynamic counts excluded; not simultaneous ports'),
                'nominal_memory_access_occurrences':_known(access['invocation_witnesses']*lanes,witnesses=access['operation_ids'],interpretation='Call-scoped static access occurrences times lanes; nested dynamic counts excluded; not cycle count'),
                'known_dependence_count':_known(known_dep,witnesses=[d['witness'] for d in dep['dependences']],interpretation='Observed dependence edges; absence is unknown, not independence')})
            related=[r['neighbor'] for r in queries.get_action_point_relations(program,action,evidence='all') if r['relation_type']=='memory_access' and r['neighbor'] in matched_arrays]
            bank_records=[]
            for array in sorted(set(related)):
                atarget=targets[array]; aa=assignments[f'AUTO{{_ARRAY_T_{array}}}']; ff=int(assignments[f'AUTO{{_ARRAY_F_{array}}}']);dd=int(assignments[f'AUTO{{_ARRAY_D_{array}}}'])
                mode=aa.lower(); shape=program.nodes[atarget].get('memory_shape')
                if not isinstance(shape,(tuple,list)) or any(type(s) is not int or s<1 for s in shape):shape=[]
                factor=(shape[dd-1] if mode=='complete' and dd>0 and len(shape)>=dd else ff)
                rows=[r for r in queries.get_loop_semantics(program,action)['accesses'] if r.get('memory_object')==atarget or r.get('memory_root')==atarget]
                known_patterns=[];unknown=0
                for row in rows:
                    banks=_candidate_access_bank(row,targets[action],lanes,mode,factor,dd,shape) if mode=='none' or (mode in ('cyclic','block','complete') and factor) else None
                    if banks is None:unknown+=1
                    else:known_patterns.append({'operation':row['source_operation'],'call_chain':row.get('call_chain',[]),
                        'root_evidence_status':row.get('evidence_status','not_established'),
                        'root_candidates_complete':row.get('root_candidates_complete',False),
                        'banks':banks,'witness':row.get('supporting_edge')})
                allbanks=[b for rec in known_patterns for b in rec['banks'] if b is not None]
                collisions=sum(len(rec['banks'])-len(set(rec['banks'])) for rec in known_patterns)
                bank_records.append({'array_action':array,'memory_object':atarget,'partition':{'type':mode,'factor':ff,'dimension':dd},
                    'lane_scenario':{'induction_variable_values':list(range(lanes)),
                        'base_iteration_assumption':'zero for displayed bank IDs; cyclic pairwise collision pattern is invariant to a uniform base-index shift; block mapping is scenario-only'},
                    'program_access_root_evidence':{'status':'not_established' if not known_patterns else 'possible' if any(x['root_evidence_status']!='proven' for x in known_patterns) else 'proven',
                        'witnesses':[x['witness'] for x in known_patterns]},
                    'nominal_bank_count':factor or (1 if mode=='none' else None),'distinct_banks_touched':len(set(allbanks)) if known_patterns and not unknown else None,
                    'bank_index_collisions':collisions if known_patterns and not unknown else None,'known_access_patterns':known_patterns,
                    'unknown_access_count':unknown,'all_relevant_accesses_analyzed':unknown==0 and bool(rows) and all(x['root_candidates_complete'] for x in known_patterns),
                    'interpretation':'Scenario-specific bank-index calculation only; no port conflict or schedule is inferred'})
            values['bank_analysis_access_count']=_known(sum(len(x['known_access_patterns']) for x in bank_records),interpretation='Unique compiler access witnesses analyzed')
            values['distinct_banks_touched']=_known(sum(x['distinct_banks_touched'] or 0 for x in bank_records) if bank_records and all(x['distinct_banks_touched'] is not None for x in bank_records) else None,interpretation='Per related array; not summed into throughput')
            values['bank_index_collisions']=_known(sum(x['bank_index_collisions'] or 0 for x in bank_records) if bank_records and all(x['bank_index_collisions'] is not None for x in bank_records) else None,interpretation='Repeated bank indices in nominal lanes only')
            output[action]={'features':values,'banking':bank_records,'matched':True}
        else:
            typ=assignments[f'AUTO{{_ARRAY_T_{action}}}'].lower();factor=int(assignments[f'AUTO{{_ARRAY_F_{action}}}']);dimension=int(assignments[f'AUTO{{_ARRAY_D_{action}}}'])
            shape=desc['features']['compiler_shape']['value']
            banks=None
            if typ=='none':banks=1
            elif typ=='complete' and shape:
                banks=math.prod(shape) if dimension==0 else shape[dimension-1] if dimension<=len(shape) else None
            elif typ in ('cyclic','block') and factor>0:banks=factor
            output[action]={'matched':True,'features':{'array_partitioned':_known(int(typ!='none'),witnesses=[action]),
                'partition_factor':_known(factor,witnesses=[action]),'partition_dimension':_known(dimension,witnesses=[action]),
                'nominal_bank_count':_known(banks,witnesses=[targets[action]],interpretation='Analytical partition bank count; storage class/ports unknown')},
                'analysis':'Array access evidence is represented in static descriptors; banking projections are emitted from matching loop actions.'}
    deployment=deepcopy(graph.graph.get('deployment')) if graph.graph.get('schema')==overlay.SCHEMA else None
    action_contract={label:{'kind':action.kind,'trip_count':action.trip_count,
                            'dimensions':[list(row) for row in action.dimensions]}
                     for label,action in sorted(actions.items())}
    return {'schema':CANDIDATE_SCHEMA,'program_graph_identity':program.graph['graph_identity'],
            'target_context_identity':graph.graph.get('target_context_identity'),
            'target_context':{'profile_id':graph.graph.get('target_profile_id'),
                'physical_resources':deepcopy(graph.graph.get('target_profile',{}).get('resources')) if graph.graph.get('schema')==overlay.SCHEMA else None,
                'clock_period_ns':deployment.get('clock_period_ns') if deployment else None,
                'available_resources':deployment.get('available_resources') if deployment else None,
                'requested_resources':deployment.get('requested_resources') if deployment else None},
            'candidate_configuration_sha256':config_hash,
            'source_action_contract_sha256':_sha(action_contract),
            'candidate_configuration':dict(sorted(assignments.items())),'actions':output,
            'implementation':implementation_identity()}


def action_feature_vector(descriptor):
    """Deterministic normalized value/known pairs for tensor action slots."""
    features=descriptor.get('features') if descriptor.get('matched') else None
    values={}
    if features:
        fidelity=descriptor.get('numerical_fidelity',{})
        contract=fidelity.get('type_evidence_contract',{})
        values['analysis_stub_header_count']=len(fidelity['arithmetic_headers']) if 'arithmetic_headers' in fidelity else None
        correspondence=contract.get('correspondence')
        values['source_type_correspondence_not_established']=(int(correspondence=='not_established') if correspondence is not None else None)
        values['hardware_exactness_not_established']=(int(fidelity['hardware_exact'] is None) if 'hardware_exact' in fidelity else None)
        if descriptor.get('kind')=='loop':
            values.update(trip_count=features.get('compiler_proven_trip_count',{}).get('value'),
                          loop_depth=features.get('loop_depth',{}).get('value'),
                          direct_arithmetic=features.get('direct_arithmetic_operation_count',{}).get('value'),
                          descendant_arithmetic=features.get('descendant_arithmetic_operation_count',{}).get('value'))
            values.update({key:item.get('value') for key,item in features.get('arithmetic_operation_counts',{}).items()})
        access=features.get('memory_accesses',{})
        values.update(static_reads=access.get('reads'),static_writes=access.get('writes'),
            static_readwrites=access.get('readwrites'),invocation_access_witnesses=access.get('invocation_witnesses'),
            resolved_roots=access.get('distinct_roots'),proven_root_relations=access.get('proven_root_relations'),
            possible_root_relations=access.get('possible_root_relations'),unestablished_root_relations=access.get('unestablished_root_relations'),
            incomplete_root_accesses=access.get('incomplete_root_accesses'))
        scopes=access.get('scope_witnesses',{})
        for name,scope in (('direct_accesses','direct_body'),('nested_accesses','nested_descendant'),
                           ('interprocedural_accesses','through_call')):
            if 'scope_witnesses' in access: values[name]=scopes.get(scope,0)
        if 'dependences' in features:
            dep=features['dependences'];counts=dep.get('counts',{})
            for certainty in ('proven','possible'):
                for kind in ('RAW','WAR','WAW'):
                    values[certainty+'_'+kind]=counts.get(kind+'_'+certainty,0)
            values.update(loop_carried_proven=dep.get('loop_carried_proven'),
                          loop_carried_possible=dep.get('loop_carried_possible'),
                          ssa_loop_carried_state=dep.get('ssa_loop_carried_state'))
        if 'access_indices' in features:
            idx=features['access_indices']
            values.update(known_affine_accesses=idx.get('known_affine_accesses'),
                unknown_index_accesses=idx.get('unknown_accesses'),
                access_dimensions=max((x for x in idx.get('access_dimensions',[]) if x is not None),default=0)
                    if idx.get('access_dimensions') else None,
                current_loop_index_accesses=idx.get('current_loop_accesses'),
                other_loop_index_accesses=idx.get('other_loop_accesses'),
                unknown_loop_index_accesses=idx.get('unknown_loop_dependency_accesses'))
    out=[]
    for name in NUMERIC_FEATURES:
        value=values.get(name)
        known=value is not None
        normalized=math.copysign(math.log2(1+abs(value)),value)/24 if known else 0.
        out.extend((normalized,1. if known else 0.))
    return out


def candidate_feature_vector(record):
    """Value/known pairs in `CANDIDATE_FEATURES` order for an action slot."""
    features=record.get('features',{})
    out=[]
    for name in CANDIDATE_FEATURES:
        item=features.get(name,{})
        value=item.get('value') if isinstance(item,dict) else None
        known=value is not None
        normalized=math.copysign(math.log2(1+abs(value)),value)/24 if known else 0.
        out.extend((normalized,1. if known else 0.))
    return out
