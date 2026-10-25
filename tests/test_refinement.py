"""CPU-only configurations, compiler evidence, authentication and budget regressions."""
import copy
import json
from argparse import Namespace
from pathlib import Path
import stat

import pytest

from LLM_branch.common import directive_domains as legacy
from LLM_branch.common.directive_domains import tcad
from experiments.refinement import configurations as c, synthesis as hls, relations
from experiments.refinement.experiment import snapshot_source,implementation_hashes
from experiments.refinement.configurations import read_jsonl,sha256,write_json,write_jsonl,repo_path
from experiments.refinement import experiment as cli
from experiments.refinement.configurations import select_design,policy_record,legacy_domains,analyze



@pytest.fixture
def source(tmp_path):
    directory=tmp_path/'input';directory.mkdir()
    (directory/'kernel_info.txt').write_text('top\nL1,array,a,1,30\nL3,array,b,1,50\nL9,loop,30\nL11,loop,50\n')
    path=directory/'kernel.cpp'
    path.write_text(
        'void top(int *x) {\nL1: int a[30];\nL3: int b[50];\nL9: for(int i=0;i<30;i++) {a[i]=x[i];}\nL11: for(int i=0;i<50;i++) {x[i]=b[i];}\n}\n'
    )
    kernel,actions,template=snapshot_source(
        'toy',
        {'source':str(path),'metadata':str(directory/'kernel_info.txt'),'top':'top'},
        tmp_path/'snapshot'
    )
    native=legacy_domains(actions,directory)
    base={k:'none' if 'ARRAY_T' in k else '0' for k in native}
    for label in ['L1','L3']:
        base.update(
            {
                c.field_key('ARRAY_T',label):'cyclic',
                c.field_key('ARRAY_F',label):'5',
                c.field_key('ARRAY_D',label):'1'
            }
        )
    base[c.field_key('UNROLL','L9')]='5';base[c.field_key('UNROLL','L11')]='5'
    kernel.update(initial_assignment=base,relations={'status':'blocked','reason':'synthetic test fixture'})
    relations=[{'action_i':a,'action_j':b,'verified_relation_class':kind,'evidence_artifact_hash':'fixture-only','reason':'test-only'}
        for a,b,kind in [('L1','L9','direct_verified'),('L3','L11','direct_verified'),('L1','L11','no_evidenced_relation'),('L3','L9','conservative_unknown')]]
    return kernel,actions,base,native,relations


def test_frozen_design_complete_distances_dedup_and_controls(source):
    kernel,actions,base,native,relations=source
    result=select_design('toy',base,actions,native,relations)
    assert result==select_design('toy',base,actions,native,relations)
    candidates,factorials,pool,limits=result
    assert len(factorials)==3 and not limits
    assert len({r['canonical_signature'] for r in candidates})==len(candidates)
    lookup={r['candidate_id']:r for r in candidates}
    for row in candidates:
        tcad.validate_assignments(row['assignment'],actions)
        assert c.parse_assignment(row['target'])==row['assignment']
        for membership in row['memberships']:
            if membership['group']=='independent':assert row['changed_action_count']==1
    for test in factorials:
        for slot,cid in test['cells'].items():
            assert lookup[cid]['changed_action_count']=={'d0':0,'dA':1,'dB':1,'dAB':2}[slot]
        if test['group']=='structural':assert test['relation']['verified_relation_class']=='direct_verified'
        else:assert test['control_label']=='no_verified_relation' and 'independence' in test['matching_note']
    assert pool and all('selection_policy' in r for r in pool)
    assert any(r['policy']['changed_values_new_under_tcad'] for r in candidates)


def test_no_verified_pairs_are_invented_and_inactive_ids_stay_absolute(source):
    _,actions,base,native,relations=source
    uncertain=[{**r,'verified_relation_class':'conservative_unknown'} for r in relations]
    cs,fs,_,limits=select_design('toy',base,actions,native,uncertain,inactive=['L9'])
    assert not fs and limits
    assert all('L9' not in r['changed_action_ids'] for r in cs)
    assert any('L11' in r['changed_action_ids'] for r in cs)


def test_full_config_versus_changed_value_domain(source):
    _,actions,base,native,_=source
    original=copy.deepcopy(native)
    info=policy_record({**base,c.field_key('ARRAY_F','L1'):'6'},base,actions,native)
    assert not info['legacy_admissible'] and info['tcad_admissible']
    assert info['changed_values_new_under_tcad']=={c.field_key('ARRAY_F','L1'):'6'}
    assert not info['baseline_exception_required'] and native==original
    legacy={**base,c.field_key('UNROLL','L9'):'7'}
    assert policy_record(legacy,legacy,actions,native)['baseline_exception_required']


def measurement(value,**kw):
    return {'ADP':value,'comparison_context_id':'same','authentication':'authenticated_new','synthesis_status':'success',
        'feasible':True,'latency_cycles':100,'resources':{k:1 for k in cli.CONTEXT['device_capacities']},**kw}


@pytest.mark.parametrize('joint,interaction,gain',[(70,0,.1),(60,-10,.2),(90,20,-.1)])
def test_interaction_and_joint_over_best_single(joint,interaction,gain):
    rows={k:measurement(j) for k,j in [('d0',100),('dA',80),('dB',90),('dAB',joint)]}
    result=analyze(rows)
    assert result['interaction_ADP']==interaction
    assert result['G_joint']==pytest.approx(gain)
    assert result['G_joint_pct']==pytest.approx(100*gain)


@pytest.mark.parametrize(
    'bad',
    [
        None,
        measurement(60,comparison_context_id='other'),
        measurement(60,feasible=False),
        measurement(60,synthesis_status='failed'),
        measurement(60,authentication='historical_unverified')
    ]
)
def test_incomparable_or_invalid_cells_cannot_win(bad):
    assert analyze({'d0':measurement(100),'dA':measurement(80),'dB':measurement(90),'dAB':bad})['G_joint'] is None


@pytest.fixture
def prepared(source,tmp_path,monkeypatch):
    kernel,actions,base,native,relations=source
    spec={
        'source':kernel['source_provenance']['source'],
        'metadata':kernel['source_provenance']['metadata'],
        'top':'top'
    }
    monkeypatch.setattr(cli,'SPECS',{'toy':spec})
    def prepare_kernel(name,spec,target,args):
        saved,loaded,template=cli.snapshot_source(name,spec,target)
        saved.update(initial_assignment=base,relations={'status':'blocked','reason':'synthetic fixture'})
        return saved,loaded,native,relations,[]
    monkeypatch.setattr(cli,'prepare_kernel',prepare_kernel)
    args=Namespace(output_dir=str(tmp_path/'out'),kernels=['toy'],seed=123,domain_policy='tcad',
        config=None,reuse=[],prior_evaluations=0,domain_audit=None)
    cli.prepare(args)
    # The real shared HLS runner executes a test-only local program, never Vitis.
    xml='''<profile><ReportVersion><Version>2021.1</Version></ReportVersion><UserAssignments><unit>ns</unit><Part>xczu7ev-ffvc1156-2-e</Part><TopModelName>top</TopModelName><TargetClockPeriod>10</TargetClockPeriod><FlowTarget>vivado</FlowTarget></UserAssignments><PerformanceEstimates><SummaryOfOverallLatency><unit>clock cycles</unit><Worst-caseLatency>100</Worst-caseLatency></SummaryOfOverallLatency></PerformanceEstimates><AreaEstimates><Resources><BRAM_18K>1</BRAM_18K><DSP>1</DSP><FF>1</FF><LUT>1</LUT></Resources><AvailableResources><BRAM_18K>624</BRAM_18K><DSP>1728</DSP><FF>460800</FF><LUT>230400</LUT></AvailableResources></AreaEstimates></profile>'''
    executable=tmp_path/'fake_hls'
    executable.write_text(
        '#!/usr/bin/python3\nimport pathlib\np=pathlib.Path("hls_project/solution1/syn/report/top_csynth.xml")\np.parent.mkdir(parents=True,exist_ok=True)\np.write_text('+repr(xml)+')\nprint("Finished Command csynth_design")\nprint("Exiting vitis_hls")\n'
    )
    executable.chmod(executable.stat().st_mode|stat.S_IXUSR)
    tool={'executable':str(executable),'version':'2021.1','executable_sha256':sha256(executable),
        'vendor_headers':{name:{'path':str(tmp_path/name),'sha256':'test-only'} for name in hls.VENDOR_HEADERS}}
    monkeypatch.setattr(hls,'tool_info',lambda *args:tool)
    return Path(args.output_dir),args,tool


def test_runner_authenticated_resume_and_immutable_input(prepared):
    output,args,_=prepared
    manifest,candidates=cli.load_output(output)
    oldhashes=implementation_hashes()
    cli.run(output,'fake')
    before=(output/'measurements.jsonl').read_bytes()
    count=len(read_jsonl(output/'attempts.jsonl'))
    cli.run(output,'fake')
    assert before==(output/'measurements.jsonl').read_bytes()
    assert len(read_jsonl(output/'attempts.jsonl'))==count==len(candidates)
    assert all(r['interaction_ADP']==0 for r in cli.summarize(output).values())
    assert oldhashes==implementation_hashes()
    cli.prepare(args)  # Idempotent preparation retains frozen inputs/results.
    (output/'candidates.jsonl').write_text('[]')
    with pytest.raises(ValueError,match='Immutable'):cli.load_output(output)


def test_durable_global_cap_tool_mismatch_and_explicit_execution(prepared,monkeypatch):
    output,_,tool=prepared
    attempts=[]
    for i in range(48):cli.reserve_attempt(attempts,'old-'+str(i))
    with pytest.raises(ValueError,match='budget'):cli.reserve_attempt(attempts,'extra')
    write_jsonl(output/'attempts.jsonl',attempts)
    monkeypatch.setattr(hls,'run_candidate',lambda *args:pytest.fail('Must not exceed hard cap'))
    cli.run(output,'fake')
    assert not read_jsonl(output/'measurements.jsonl')
    attempts[0]['tool_identity']={'wrong':'tool'}
    write_jsonl(output/'attempts.jsonl',attempts)
    with pytest.raises(ValueError,match='Tool/header build differs'):cli.run(output,'fake')


def test_source_hash_and_context_changes_block_measurement(prepared):
    output,_,_=prepared
    cli.run(output,'fake',limit=1)
    manifest,candidates=cli.load_output(output)
    row=read_jsonl(output/'measurements.jsonl')[0]
    candidate=next(c for c in candidates if c['candidate_id']==row['candidate_id'])
    bad=copy.deepcopy(manifest['kernels']['toy']);bad['source_provenance']['source_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):cli.verify_measurement(row,candidate,bad)
    badrow={**row,'comparison_context_id':'other'}
    with pytest.raises(ValueError,match='context'):cli.verify_measurement(badrow,candidate,manifest['kernels']['toy'])
    repo_path(row['report_path']).write_text('tampered')
    with pytest.raises(ValueError,match='integrity'):cli.summarize(output)


def test_recover_completed_receipt_even_with_retry_flag(prepared):
    output,_,_=prepared
    cli.run(output,'fake',limit=1)
    original=read_jsonl(output/'measurements.jsonl')[0]
    # Simulate a crash after the shared runner's result.json write, before the
    # aggregate journal was updated. A successful physical job is not repeated.
    write_jsonl(output/'measurements.jsonl',[])
    cli.run(output,'fake',retry_failed=True)
    attempts=read_jsonl(output/'attempts.jsonl')
    assert sum(a['candidate_id']==original['candidate_id'] for a in attempts)==1
    assert attempts[0]['status']=='success_recovered'
    assert len(list((output/'projects'/original['candidate_id']).glob('attempt-*')))==1


def test_loop_regions_and_proven_dependency_preserve_sparse_ids():
    import networkx as nx
    from experiments.refinement.relations import audit_loop_pairs
    graph=nx.MultiDiGraph()
    actions={
        'L3':legacy.SourceAction(kind='loop',trip_count=30),
        'L11':legacy.SourceAction(kind='loop',trip_count=50)
    }
    for label,node,block,operation in [('L3','3','4','5'),('L11','11','12','13')]:
        graph.add_node(node,type=0,is_loop=1,action_id=label,action_resolution='source_location',function=0)
        graph.add_node(block,type=4,function=0)
        graph.add_node(operation,type=0,text='affine.load',function=0)
        graph.add_edge(node,block,flow=8,role='region_entry')
        graph.add_edge(block,operation,flow=0,role='next_in_block')
    graph.add_edge('5','13',flow=12,certainty='unknown')
    row=audit_loop_pairs(graph,actions,[],'hash')[0]
    assert row['verified_relation_class']=='no_evidenced_relation'
    graph.add_edge('5','13',flow=12,certainty='proven')
    row=audit_loop_pairs(graph,actions,[],'hash')[0]
    assert row['verified_relation_class']=='direct_verified'
    assert (row['action_i'],row['action_j'])==('L3','L11')
    assert row['dependence_evidence'] and row['evidence_artifact_hash']=='hash'
    for node in ['11','12','13']:graph.nodes[node]['function']=1
    assert audit_loop_pairs(graph,actions,[],'hash')[0]['verified_relation_class']=='no_evidenced_relation'


def test_GAN_shared_helper_regression_through_refinement_verifier():
    import networkx as nx
    from experiments.refinement.relations import audit_all
    graph=nx.MultiDiGraph()
    for label,node,root in [('L2','2','20'),('L3','3','30')]:
        graph.add_node(
            node,
            type=104,
            action_id=label,
            array_var=label,
            action_resolution='source_location',
            function=0
        )
        graph.add_node(root,type=1,is_memory_root=1,function=0)
        graph.add_edge(node,root,flow=7,role='array_root')
    for node,attrs in [('11',{'type':0,'is_loop':1,'action_id':'L11','action_resolution':'source_location','function':0}),
        ('12',{'type':4,'function':0}),('13',{'type':0,'text':'func.call','function':0}),
        ('14',{'type':0,'text':'func.func','function':1}),('15',{'type':4,'function':1}),('16',{'type':0,'text':'affine.load','function':1})]:graph.add_node(node,**attrs)
    for a,b,flow,role in [
        ('11','12',8,'region_entry'),
        ('12','13',0,'next_in_block'),
        ('13','14',2,'calls'),
        ('14','15',8,'region_entry'),
        ('15','16',0,'next_in_block'),
        ('2','16',7,'array_read'),
        ('3','16',7,'array_read'),
        ('3','13',7,'array_read'),
        ('30','13',10,'read')
    ]:graph.add_edge(a,b,flow=flow,role=role)
    actions={
        'L2':legacy.SourceAction(kind='array',dimensions=((1,30),)),
        'L3':legacy.SourceAction(kind='array',dimensions=((1,50),)),
        'L11':legacy.SourceAction(kind='loop',trip_count=50)
    }
    records=audit_all(graph,actions,[],'fixture-hash')
    assert records[0]['verified_relation_class']=='conservative_unknown'
    assert records[1]['verified_relation_class']=='direct_verified'
    assert records[0]['shared_helper_nodes']==['16']


def test_compiler_static_count_overrides_bound_and_unknown_is_not_enabled(tmp_path):
    import networkx as nx
    from experiments.refinement.configurations import compiler_actions,recorded_actions
    graph=nx.MultiDiGraph()
    graph.add_node('1',type=0,is_loop=1,action_id='L10',action_resolution='source_location',trip_count=4095,
        trip_count_static=1,trip_count_source='mlir_affine_analysis',full_text='affine.for 1 to 4096')
    graph.add_node('2',type=0,is_loop=1,action_id='L6',action_resolution='source_location',trip_count=-1,
        trip_count_static=0,trip_count_source='unknown',full_text='affine.for i to 80')
    path=tmp_path/'test.gexf';nx.write_gexf(graph,path)
    actions={
        'L10':legacy.SourceAction(kind='loop',trip_count=4096),
        'L6':legacy.SourceAction(kind='loop',trip_count=80)
    }
    kernel={'relations':{'status':'available','artifact':str(path),'artifact_hash':sha256(path)}}
    verified=compiler_actions(kernel,actions)
    assert verified['L10'].trip_count==4095 and verified['L6'].trip_count is None
    assert 4096 not in tcad.factor_values(verified['L10'].trip_count)
    assert 3 in tcad.factor_values(verified['L10'].trip_count)
    assert tcad.action_tuples(verified['L6'])==(('0','0'),('1','0'))
    assert recorded_actions(kernel,actions)==verified and actions['L10'].trip_count==4096
    assert [x['status'] for x in kernel['loop_domain_evidence']]==['static_metadata_mismatch','unresolved_no_enabled_factor_proposals']


def test_authenticated_archive_reuse_charges_original_budget(prepared,tmp_path):
    output,args,_=prepared
    cli.run(output,'fake')
    before={str(p):sha256(p) for p in output.rglob('*') if p.is_file()}
    manifest,candidates=cli.load_output(output)
    inventory,audit,cost=cli.read_measurements(output)
    assert len(inventory)==len(candidates)==cost
    assert all(row['status']=='authenticated' for row in audit)
    newargs=copy.copy(args)
    newargs.output_dir=str(tmp_path/'reused')
    newargs.reuse=[str(output)]
    newmanifest=cli.prepare(newargs)
    assert newmanifest['initial_reuse_count']==len(candidates)
    assert newmanifest['prior_new_synthesis_cost']==len(candidates)
    assert newmanifest['max_new_syntheses']==48-len(candidates)
    assert before=={str(p):sha256(p) for p in output.rglob('*') if p.is_file()}
    cli.run(Path(newargs.output_dir),'fake')
    assert not read_jsonl(Path(newargs.output_dir)/'attempts.jsonl')


@pytest.mark.parametrize(
    'text',
    [
        '',
        '_PIPE_L1=0',
        'auto{_PIPE_L0} = 0',
        'auto{_PIPE_L1} = 1',
        'auto{_PIPE_L1} = 01\nauto{_UNROLL_L1} = 0',
        'auto{_PIPE_L1} = 0\nauto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 0'
    ]
)
def test_malformed_incomplete_duplicate_assignments(text):
    with pytest.raises(ValueError):
        c.parse_assignment(text)

def test_real_projector_excludes_uncertain_dependences():
    import networkx as nx
    graph = nx.MultiDiGraph()
    for n in range(6):
        graph.add_node(str(n), type=0, is_loop=int(n < 2), loop_depth=0)
    graph.add_edge('0', '2', flow=8, role='region_entry')
    graph.add_edge('1', '3', flow=8, role='region_entry')
    graph.nodes['2']['type'] = graph.nodes['3']['type'] = 4
    graph.add_node('6', type=100, action_id='L1', text='PIPELINE')
    graph.add_node('7', type=100, action_id='L2', text='PIPELINE')
    graph.add_edge('6', '2', flow=4)
    graph.add_edge('7', '3', flow=4)
    graph.add_edge('2', '4', flow=0, role='next_in_block')
    graph.add_edge('3', '5', flow=0, role='next_in_block')
    graph.add_edge('4', '5', flow=12, certainty='unknown')
    project = relations.existing_projector()
    (_, bits, _) = project(graph, node_ids=[0, 1], active_mask=[True, True], max_slots=2)
    assert bits[0, 1].item() == bits[1, 0].item() == 0
    graph.add_edge('4', '5', flow=12, certainty='proven')
    (_, bits, _) = project(graph, node_ids=[0, 1], active_mask=[True, True], max_slots=2)
    assert bits[0, 1].item() == 16 and bits[1, 0].item() == 32

def test_adp_device_capacity_and_separate_budget():
    resources = {k: hls.CAPACITIES[k] // 2 for k in hls.CAPACITIES}
    row = hls.evaluate_metrics(100, resources)
    assert row['area_proxy'] == 50 and row['ADP'] == 5000 and row['feasible']
    budget = {**hls.CAPACITIES, 'DSP': resources['DSP'] - 1}
    infeasible = hls.evaluate_metrics(100, resources, budget=budget)
    assert infeasible['area_proxy'] == 50 and (not infeasible['feasible'])
    assert hls.best_feasible([dict(infeasible, synthesis_status='success', candidate_id='bad')]) is None
    assert hls.improvement_pct(row, {**row, 'ADP': 4000}) == 20
    assert hls.evaluate_metrics(None, resources)['ADP'] is None
    assert hls.evaluate_metrics(100, {**resources, 'FF': None})['rankable'] is False
    with pytest.raises(ValueError):
        hls.evaluate_metrics(float('nan'), resources)

def xml(
    top='top',
    part=hls.PART,
    clock='10',
    version='2021.1',
    latency='100',
    estimated='7.5',
    clock_unit='ns',
    latency_unit='clock cycles'
):
    timing = '' if estimated is None else '<EstimatedClockPeriod>' + estimated + '</EstimatedClockPeriod>'
    return f'<profile><ReportVersion><Version>{version}</Version></ReportVersion>\n<UserAssignments><unit>{clock_unit}</unit><Part>{part}</Part><TopModelName>{top}</TopModelName><TargetClockPeriod>{clock}</TargetClockPeriod><FlowTarget>vivado</FlowTarget></UserAssignments>\n<PerformanceEstimates><SummaryOfOverallLatency><unit>{latency_unit}</unit><Worst-caseLatency>{latency}</Worst-caseLatency></SummaryOfOverallLatency><SummaryOfTimingAnalysis>{timing}</SummaryOfTimingAnalysis></PerformanceEstimates>\n<AreaEstimates><Resources><BRAM_18K>10</BRAM_18K><DSP>20</DSP><FF>300</FF><LUT>400</LUT></Resources></AreaEstimates></profile>'

def test_tcl_and_valid_unknown_timing_xml(tmp_path):
    tcl = hls.make_tcl('source file.cpp', 'top')
    for token in [
        'set_top top',
        'add_files {source file.cpp}',
        'open_solution',
        'set_part',
        'create_clock -period 10',
        'csynth_design',
        *hls.SETTINGS
    ]:
        assert token in tcl
    with pytest.raises(ValueError):
        hls.make_tcl('bad}\nexec', 'top')
    p = tmp_path / 'report.xml'
    p.write_text(xml(estimated=None))
    row = hls.parse_report(p, 'top')
    assert row['latency_cycles'] == 100 and row['estimated_clock_ns'] is None and (row['timing_status'] == 'unknown')
    assert row['ADP'] > 0 and row['rankable']

@pytest.mark.parametrize(
    'changes',
    [
        {'top': 'wrong'},
        {'part': 'wrong'},
        {'clock': '5'},
        {'version': '2024.1'},
        {'latency': '?'},
        {'latency': '0'},
        {'clock_unit': 'ps'},
        {'latency_unit': 'ns'}
    ]
)
def test_wrong_or_partial_report_provenance(tmp_path, changes):
    p = tmp_path / 'report.xml'
    p.write_text(xml(**changes))
    with pytest.raises(hls.ReportError):
        hls.parse_report(p, 'top')
    p.write_text('<profile>')
    with pytest.raises(hls.ReportError):
        hls.parse_report(p, 'top')


from experiments.refinement.relations import verify_pair
import networkx as nx

def graph_fixture(shared=True):
    graph = nx.MultiDiGraph()
    graph.add_node(
        '1',
        type=104,
        action_id='L3',
        array_var='d',
        action_resolution='source_location',
        function=0
    )
    graph.add_node('2', type=1, text='memref', is_memory_root=1, function=0)
    graph.add_node('3', type=0, is_loop=1, action_id='L11', action_resolution='source_location', function=0)
    graph.add_node('4', type=4, function=0)
    graph.add_node('5', type=0, text='func.call', function=0, op_uid='caller:call')
    graph.add_node('6', type=0, text='affine.load', function=1, op_uid='helper:load')
    graph.add_node('7', type=4, function=1)
    graph.add_node('8', type=0, text='func.func', function=1)
    graph.add_edge('1', '2', flow=7, role='array_root')
    graph.add_edge('3', '4', flow=8, role='region_entry')
    graph.add_edge('4', '5', flow=0, role='next_in_block')
    graph.add_edge('5', '8', flow=2, role='calls')
    graph.add_edge('8', '7', flow=8, role='region_entry')
    graph.add_edge('7', '6', flow=0, role='next_in_block')
    if shared:
        graph.add_edge('1', '6', flow=7, role='array_read')
    return graph

def test_shared_helper_not_direct_and_unknown_bindings_stay_uncertain():
    graph = graph_fixture()
    original = {'relation_bits': {'i_to_j': 8, 'j_to_i': 8}, 'relation_type': ['loop_array_access']}
    result = verify_pair(graph, 'L3', 'L11', original, 'artifact-hash')
    assert result['verified_relation_class'] == 'conservative_unknown'
    assert result['shared_helper_nodes'] == ['6']
    assert result['evidence_artifact_hash'] == 'artifact-hash'
    assert result['original_relation_bits'] == original['relation_bits']
    assert result['action_j'] == 'L11'

def test_verified_own_region_root_access():
    graph = graph_fixture()
    graph.add_edge('1', '5', flow=7, role='array_read')
    graph.add_edge('2', '5', flow=10, role='read', operand_index=0)
    result = verify_pair(graph, 'L3', 'L11', artifact_hash='proof')
    assert result['verified_relation_class'] == 'direct_verified'
    assert result['callsite_provenance'][0]['root_node'] == '2'

def test_indirect_requires_specific_call_and_formal_memory_binding():
    graph = graph_fixture()
    graph.add_node('9', type=1, function=1)
    graph.add_edge('2', '5', flow=1, role='operand', position=0)
    graph.add_edge('2', '9', flow=2, role='actual_to_formal', callsite='different_call')
    graph.add_edge('9', '6', flow=10, role='read')
    assert verify_pair(graph, 'L3', 'L11')['verified_relation_class'] == 'conservative_unknown'
    graph.add_edge('2', '9', flow=2, role='actual_to_formal', callsite='5')
    assert verify_pair(graph, 'L3', 'L11')['verified_relation_class'] == 'indirect_verified'

def test_indirect_binding_uses_observed_operand_position_and_callee_identity():
    graph = graph_fixture()
    graph.add_node('9', type=1, function=1)
    graph.add_edge('2', '5', flow=1, role='operand', position=0)
    graph.add_edge('2', '9', flow=2, role='actual_to_formal', position=1)
    graph.add_edge('9', '6', flow=10, role='read')
    result = verify_pair(graph, 'L3', 'L11')
    assert result['verified_relation_class'] == 'indirect_verified'
    assert result['callsite_provenance'][0]['actual_operand_position'] == 0
    graph.nodes['9']['function'] = 99
    assert verify_pair(graph, 'L3', 'L11')['verified_relation_class'] == 'conservative_unknown'

def test_unknown_dependence_is_not_access_proof_and_absence_not_independence():
    graph = graph_fixture(shared=False)
    graph.add_edge('5', '6', flow=12, certainty='unknown')
    assert verify_pair(graph, 'L3', 'L11')['verified_relation_class'] == 'no_evidenced_relation'
    graph.nodes['3']['action_resolution'] = 'fallback'
    with pytest.raises(ValueError):
        verify_pair(graph, 'L3', 'L11')


def test_prefixed_artifact_reader_selects_current_ledger_not_prior(tmp_path):
    actual=tmp_path/'EXPERIMENT_ATTEMPTS.jsonl'
    actual.write_text('{}\n')
    (tmp_path/'EXPERIMENT_PRIOR_ATTEMPTS.jsonl').write_text('{}\n{}\n')
    assert cli._artifact(tmp_path,['attempts.jsonl'])==actual


def test_tcl_project_name_cannot_inject_commands():
    with pytest.raises(ValueError):
        hls.make_tcl('safe.cpp','top',project='bad;exec')


def test_sparse_action_slot_alignment_and_inactive_bits():
    actions={'L1':legacy.SourceAction(kind='loop',trip_count=8),
        'L3':legacy.SourceAction(kind='array',dimensions=((1,8),)),
        'L9':legacy.SourceAction(kind='loop',trip_count=8)}
    active=[False]*9;labels=[-1]*9;bits=[[0]*9 for _ in range(9)]
    for index in [0,2,8]:active[index]=True;labels[index]=index+1;bits[index][index]=1
    bits[0][2]=bits[2][0]=8
    pairs=relations.pairs_from_bits(bits,active,labels,actions,{})
    assert [(r['action_i'],r['action_j']) for r in pairs]==[('L1','L3')]
    labels[8]=3
    with pytest.raises(ValueError):relations.pairs_from_bits(bits,active,labels,actions,{})
    labels[8]=9;bits[0][1]=8
    with pytest.raises(ValueError):relations.pairs_from_bits(bits,active,labels,actions,{})
