"""Focused tests for compiler-grounded V2.2 action-demand descriptors."""
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from GNN_branch import action_demand_v22 as demand
from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import target_graph_v22 as overlay
from GNN_branch.action_point_relations import get_action_point_relations

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
MANIFEST = json.loads((ROOT/'docs/graph-release-v22-rc3.json').read_text())
FIXTURES = {
    'stencil': 'canonical/machsuite-stencil2d-624ee88fef89.json',
    'gemm': 'canonical/machsuite-gemm-ncubed-8692fbcbe7b2.json',
    'lud': 'canonical/rodinia_lud_1_tiling_0-e35f0c4cd5d4.json',
    'kalman': 'canonical/serrano-kalman-filter-96b4d608969f.json',
}


@pytest.fixture(scope='module')
def graphs(tmp_path_factory):
    assert overlay.file_hash(ARCHIVE) == MANIFEST['archive_sha256']
    root=tmp_path_factory.mktemp('action-demand-rc3')
    result={}
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        for name,member in FIXTURES.items():
            expected=MANIFEST['files_sha256'][member]
            raw=archive.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest()==expected
            path=root/Path(member).name
            path.write_bytes(raw)
            result[name]=compiler.read_graph(path)
            compiler.validate(result[name])
    return result


def test_static_features_are_compiler_only_and_target_invariant(graphs):
    graph=graphs['stencil']
    before=compiler.canonical_json(compiler.graph_payload(graph))
    base=demand.static_action_descriptors(graph)
    for device,clock in (('zcu104',10),('u200',10),('u50',10),('zcu104',6.67)):
        assert demand.static_action_descriptors(overlay.enrich(graph,device,clock))==base
    assert compiler.canonical_json(compiler.graph_payload(graph))==before
    assert base['schema']==demand.STATIC_SCHEMA
    loop=base['actions']['L3']
    assert loop['matched'] and loop['features']['loop_depth']['known']
    counts=loop['features']['arithmetic_operation_counts']
    assert all(item['known'] for item in counts.values())
    assert counts['integer_multiply']['value'] >= 1
    assert 'not DSP/LUT usage' in counts['integer_multiply']['interpretation']


def test_memory_workload_and_formal_roots_do_not_require_storage_binding(graphs):
    graph=graphs['stencil']
    formals={n for n,a in graph.nodes(data=True) if a.get('category')=='memory_object' and a.get('root_kind')=='formal_parameter'}
    assert formals
    accessed={r['memory_root'] for action in ('L1','L2','L3','L4')
              for r in __import__('GNN_branch.action_point_relations',fromlist=['get_loop_semantics']).get_loop_semantics(graph,action)['accesses']
              if r.get('memory_root')}
    assert formals & accessed
    desc=demand.static_action_descriptors(graph)['actions']['L1']
    assert desc['features']['memory_accesses']['static_operations'] > 0
    assert desc['features']['memory_accesses']['reads'] + desc['features']['memory_accesses']['writes'] > 0
    assert desc['features']['dependences']['dependence_analysis_complete']['known'] is False
    assert desc['features']['dependences']['dependence_analysis_complete']['interpretation'].startswith('Absence')


def test_compact_input_expands_to_same_static_descriptors(graphs,tmp_path):
    member='compact/machsuite-stencil2d-624ee88fef89.json'
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        raw=archive.extractfile(member).read()
    assert hashlib.sha256(raw).hexdigest()==MANIFEST['files_sha256'][member]
    path=tmp_path/Path(member).name;path.write_bytes(raw)
    compact=compiler.read_graph(path)
    assert demand.static_action_descriptors(compact)==demand.static_action_descriptors(graphs['stencil'])


def test_lud_array_action_collects_accesses_without_claiming_binding(graphs):
    desc=demand.static_action_descriptors(graphs['lud'])['actions']['L5']
    assert desc['kind']=='array' and desc['matched']
    features=desc['features']
    assert features['memory_accesses']['static_operations'] > 0
    assert features['compiler_shape']['known']
    assert features['source_declared_dimensions']['known']
    assert features['storage_binding']['known'] is False
    assert features['accessing_loop_actions']


def test_formal_parameter_array_access_demand_is_not_filtered_by_storage_policy(graphs):
    graph=graphs['stencil'].copy()
    formal=next(n for n,a in graph.nodes(data=True) if a.get('category')=='memory_object' and a.get('root_kind')=='formal_parameter')
    # A query-only synthetic action binds to an existing formal memory object;
    # it is not added to or serialized as a canonical benchmark graph.
    synthetic='action:L1000'
    graph.add_node(synthetic,category='action_point',action_id='L1000',absolute_index=1000,
                   action_kind='array',supported_directives=['ARRAY_PARTITION'])
    graph.add_edge(synthetic,formal,key='synthetic-target',relation_type='targets',evidence_status='proven')
    desc=demand._static_array(graph,'L1000',formal,{'action_id':'L1000','compiler_matched':True})
    assert desc['features']['memory_accesses']['static_operations']>0
    assert desc['features']['storage_binding']['known'] is False


def test_unresolved_kalman_actions_keep_absolute_slots_and_unknowns(graphs):
    records=demand.static_action_descriptors(graphs['kalman'])['actions']
    for label in ('L7','L8'):
        assert not records[label]['matched']
        assert records[label]['compiler_target'] is None
        assert records[label]['features'] is None
    vector=demand.action_feature_vector(records['L7'])
    assert len(vector)==2*len(demand.NUMERIC_FEATURES)
    assert not any(vector[i] for i in range(1,len(vector),2))


def test_affine_subset_and_banking_math():
    assert demand._split_affine_map('affine_map<(d0) -> (d0)>')['results'][0]['coefficients']['d0']==1
    assert demand._split_affine_map('affine_map<(d0) -> (2*d0 + 1)>')['results'][0]=={'coefficients':{'d0':2},'offset':1}
    assert demand._split_affine_map('affine_map<(d0,d1)->(d0+d1*64)>')['results'][0]['coefficients']=={'d0':1,'d1':64}
    assert demand._split_affine_map('affine_map<(d0)->(d0 floordiv 4)>') is None
    assert [demand.cyclic_bank(i,4) for i in range(4)]==[0,1,2,3]
    assert [demand.cyclic_bank(2*i,4) for i in range(4)]==[0,2,0,2]
    assert [demand.block_bank(i,10,4) for i in range(10)]==[0,0,0,1,1,1,2,2,2,3]


def test_candidate_bank_scenarios_are_explicit_and_unknown_when_unsupported():
    row={'access_indices':{'status':'compiler_recorded','affine_map':'affine_map<(d0)->(2*d0)>','induction_loops':['loop']}}
    assert demand._candidate_access_bank(row,'loop',4,'cyclic',4,1,[64])==[0,2,0,2]
    row['access_indices']['affine_map']='affine_map<(d0)->(d0+1)>'
    assert demand._candidate_access_bank(row,'loop',4,'cyclic',4,1,[64])==[1,2,3,0]
    row['access_indices']['affine_map']='affine_map<(d0)->(d0)>'
    assert demand._candidate_access_bank(row,'loop',4,'block',4,1,[10])==[0,0,0,1]
    assert demand._candidate_access_bank(row,'loop',4,'complete',4,1,[10])==[0,1,2,3]
    row['access_indices']['affine_map']='opaque_dynamic_expression'
    assert demand._candidate_access_bank(row,'loop',4,'cyclic',4,1,[64]) is None
    assert demand._candidate_access_bank(row,'loop',4,'block',4,1,[64]) is None
    row={'access_indices':{'status':'compiler_recorded','affine_map':'affine_map<(d0,d1)->(d0,d1)>','induction_loops':['outer','inner']}}
    assert demand._candidate_access_bank(row,'inner',4,'cyclic',4,1,[32,64]) is None
    assert demand._candidate_access_bank(row,'inner',4,'cyclic',4,2,[32,64])==[0,1,2,3]
    assert demand._candidate_access_bank(row,'inner',4,'complete',4,0,[32,64]) is None


def test_action_relations_and_static_vectors_are_repeatable(graphs):
    graph=graphs['lud']
    left=demand.static_action_descriptors(graph)
    right=demand.static_action_descriptors(graph)
    assert json.dumps(left,sort_keys=True,separators=(',',':'))==json.dumps(right,sort_keys=True,separators=(',',':'))
    assert len(demand.action_feature_vector(left['actions']['L6']))==2*len(demand.NUMERIC_FEATURES)
    rows=get_action_point_relations(graph,'L6',evidence='all')
    assert any(r['relation_type']=='memory_access' and r['neighbor']=='L5' for r in rows)
    roots=left['actions']['L46']['features']['memory_accesses']
    assert roots['possible_root_relations']>0 and roots['unestablished_root_relations']>0


def test_complete_candidate_changes_only_candidate_demand(graphs):
    from LLM_branch.common import directive_domains as domains
    from GNN_branch.action_point_relations import get_loop_semantics
    from LLM_branch.common.directive_domains import SourceAction
    graph=graphs['lud']
    joint=overlay.enrich(graph,'zcu104',10)
    targets={u[7:]:v for u,v,_,edge in graph.edges(keys=True,data=True)
             if u.startswith('action:') and edge.get('relation_type')=='targets'}
    actions={}
    for row in graph.graph['action_point_coverage']['records']:
        label=row['action_id'];kind=row['source_binding']['kind']
        if kind=='loop':
            trip=get_loop_semantics(graph,label)['compiler_proven_trip_count']
            actions[label]=SourceAction('loop',trip_count=trip)
        else:
            node=targets[label];shape=graph.nodes[node].get('memory_shape') or [256]
            actions[label]=SourceAction('array',dimensions=tuple((i+1,int(n)) for i,n in enumerate(shape) if int(n)>0))
    configs=[]
    for partition,unroll in (('none',4),('cyclic',4),('cyclic',2)):
        values={}
        for label,action in actions.items():
            if action.kind=='loop':
                values[f'AUTO{{_PIPE_{label}}}']='0'
                values[f'AUTO{{_UNROLL_{label}}}']=str(unroll) if label=='L6' else '0'
            elif label=='L5' and partition=='cyclic':
                values[f'AUTO{{_ARRAY_T_{label}}}']='cyclic';values[f'AUTO{{_ARRAY_F_{label}}}']='4';values[f'AUTO{{_ARRAY_D_{label}}}']='1'
            else:
                values[f'AUTO{{_ARRAY_T_{label}}}']='none';values[f'AUTO{{_ARRAY_F_{label}}}']='0';values[f'AUTO{{_ARRAY_D_{label}}}']='0'
        domains.validate_assignments(values,actions)
        configs.append(demand.candidate_demand_descriptors(joint,values,actions))
    static=demand.static_action_descriptors(graph)
    assert static==demand.static_action_descriptors(overlay.enrich(graph,'zcu104',10))
    assert configs[0]['actions']['L6']['features']['nominal_lanes']['value']==4
    # L6's configured nominal factor and the linked array partition are consumed
    # together; these are lane/bank descriptors, not generated HLS accesses.
    assert configs[0]['candidate_configuration_sha256']!=configs[1]['candidate_configuration_sha256']
    assert configs[0]['actions']['L6']['banking']!=configs[1]['actions']['L6']['banking']
    assert configs[1]['actions']['L6']['features']['nominal_lanes']['value']==4
    assert configs[2]['actions']['L6']['features']['nominal_lanes']['value']==2
    assert configs[1]['actions']['L6']['features']['nominal_read_occurrences']['value'] == 2 * configs[2]['actions']['L6']['features']['nominal_read_occurrences']['value']
    other=overlay.enrich(graph,'u50',6.67)
    other_result=demand.candidate_demand_descriptors(other,configs[1]['candidate_configuration'],actions)
    assert other_result['program_graph_identity']==configs[1]['program_graph_identity']
    assert other_result['target_context_identity']!=configs[1]['target_context_identity']
    assert other_result['actions']==configs[1]['actions']
