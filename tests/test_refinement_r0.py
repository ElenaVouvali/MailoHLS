"""CPU-only R0 contracts. Fake HLS runs are confined to pytest temp directories."""
import json
import os
from pathlib import Path
import stat
from unittest.mock import patch

import pytest

from LLM_branch.common import directive_domains as dd
from experiments.refinement import r0_candidates as c, r0_metrics as metrics
from experiments.refinement import r0_relations as relations, r0_synthesis as synthesis
from experiments.refinement import r0_local_headroom as cli
from experiments.refinement.r0_common import object_hash, read_jsonl, sha256, write_json, write_jsonl


@pytest.fixture
def source(tmp_path):
    info = tmp_path / "kernel_info.txt"
    info.write_text("top\nL1,loop,16\nL3,array,b,1,8,2,4\nL9,loop,8\n")
    path = tmp_path / "kernel.cpp"
    path.write_text("""void top(int *a) {
L1: for (int i=0;i<16;i++) {
  a[i] += 1;
}
L3: int b[8][4];
L9: for (int j=0;j<8;j++) {
  b[j][0] = a[j];
}
}
""")
    actions = dd.load_source_actions("toy", tmp_path)
    template = dd.prepare_source_template("toy", path, tmp_path)
    domains = dd.source_site_domains("toy", template, tmp_path)
    base = {lhs: ("none" if "ARRAY_T" in lhs else "0") for lhs in domains}
    return path, actions, template, domains, base


def evidence():
    return {"action_i": "L1", "action_j": "L3", "relation_type": ["loop_array_access"],
            "relation_bits": {"i_to_j": 8, "j_to_i": 8},
            "compiler_evidence_source": "synthetic unit fixture, not an experiment artifact",
            "graph_or_memory_artifact": "test-only", "artifact_hash": "test-only"}


def test_roundtrip_and_complete_order(source):
    _, actions, _, domains, base = source
    text = c.serialize(base, actions)
    assert c.parse_assignment(text) == base
    assert c.signature(c.parse_assignment(text.lower())) == c.signature(base)
    assert [c.FIELD.fullmatch(f)[2] for f in c.ordered_fields(actions)] == ["L1", "L1", "L3", "L3", "L3", "L9", "L9"]
    c.validate(base, actions, domains)


@pytest.mark.parametrize("text", ["", "_PIPE_L1=0", "auto{_PIPE_L0} = 0", "auto{_PIPE_L1} = 1",
    "auto{_PIPE_L1} = 01\nauto{_UNROLL_L1} = 0",
    "auto{_PIPE_L1} = 0\nauto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 0"])
def test_malformed_incomplete_duplicate_assignments(text):
    with pytest.raises(ValueError):
        c.parse_assignment(text)


@pytest.mark.parametrize("typ,factor,dim", [("none","2","1"),("complete","2","1"),("block","0","1"),("cyclic","4","0")])
def test_invalid_array_tuples(source, typ, factor, dim):
    _, actions, _, domains, base = source
    base.update({c.key("ARRAY_T","L3"):typ,c.key("ARRAY_F","L3"):factor,c.key("ARRAY_D","L3"):dim})
    with pytest.raises(ValueError):c.validate(base, actions, domains)


def test_valid_independent_pipe_unroll_and_domain_rejection(source):
    _, actions, _, domains, base = source
    base.update({c.key("PIPE","L1"):"1", c.key("UNROLL","L1"):"4"})
    c.validate(base, actions, domains)  # They are not mutually exclusive.
    base[c.key("UNROLL","L1")] = "3"
    with pytest.raises(c.DomainViolation):c.validate(base, actions, domains)
    with pytest.raises(ValueError):c.validate({k:v for k,v in base.items() if "L9" not in k}, actions, domains)


def test_legal_neighbors_and_determinism(source):
    _, actions, _, domains, base = source
    single, joint = c.generate_pools("toy",base,actions,domains,[evidence()])
    assert single and joint
    for pool, count in [(single,1),(joint,2)]:
        assert len({r["canonical_signature"] for r in pool}) == len(pool)
        for row in pool:
            c.validate(row["assignment"],actions,domains)
            assert row["changed_action_count"]==count
            assert c.parse_assignment(row["target"])==row["assignment"]
            assert row["canonical_signature"] != c.signature(base)
        assert c.select_diverse(pool,actions,123)==c.select_diverse(pool,actions,123)
    assert any(r["changed_action_count"]==1 and r["changed_field_count"]==3 for r in single)
    assert len(c.select_diverse(single,actions,123))==4
    assert not c.generate_pools("toy",base,actions,domains,[])[1]


def test_inactive_actions_never_selected_and_unknown_pairs_rejected(source):
    _, actions, _, domains, base = source
    single,joint=c.generate_pools("toy",base,actions,domains,[evidence()],inactive_actions=["L9"])
    assert all("L9" not in r["changed_action_ids"] for r in single+joint)
    with pytest.raises(ValueError):c.generate_pools("toy",base,actions,domains,[{**evidence(),"action_j":"L1"}])


def test_sparse_bits_self_exclusion_and_decoding(source):
    _,actions,_,_,_=source
    active=[False]*9;labels=[-1]*9;bits=[[0]*9 for _ in range(9)]
    for i in [0,2,8]:active[i]=True;labels[i]=i+1;bits[i][i]=1
    bits[0][2]=bits[2][0]=8
    bits[0][8]=4|16;bits[8][0]=2|32
    pairs=relations.pairs_from_bits(bits,active,labels,actions,{"compiler_evidence_source":"test"})
    assert [(p['action_i'],p['action_j']) for p in pairs]==[("L1","L3"),("L1","L9")]
    assert pairs[1]['relation_bits']=={'i_to_j':20,'j_to_i':34}
    assert "proven_dependence_forward" in pairs[1]['relation_type']
    labels[8]=3
    with pytest.raises(ValueError):relations.pairs_from_bits(bits,active,labels,actions,{})


def test_inactive_relation_bits_rejected(source):
    _,actions,_,_,_=source
    active=[True,False,True,False,False,False,False,False,False]
    bits=[[0]*9 for _ in range(9)];bits[0][0]=bits[2][2]=1
    labels=[1,-1,3,-1,-1,-1,-1,-1,-1]
    assert relations.pairs_from_bits(bits,active,labels,actions,{},["L9"])==[]
    bits[0][8]=8
    with pytest.raises(ValueError):relations.pairs_from_bits(bits,active,labels,actions,{},["L9"])


def test_real_projector_excludes_uncertain_dependences():
    import networkx as nx
    graph=nx.MultiDiGraph()
    # Two semantic loops, their regions, and one memory op in each region.
    for n in range(6):graph.add_node(str(n),type=0,is_loop=int(n<2),loop_depth=0)
    graph.add_edge("0","2",flow=8,role="region_entry")
    graph.add_edge("1","3",flow=8,role="region_entry")
    graph.nodes['2']['type']=graph.nodes['3']['type']=4
    graph.add_node('6',type=100,action_id='L1',text='PIPELINE')
    graph.add_node('7',type=100,action_id='L2',text='PIPELINE')
    graph.add_edge('6','2',flow=4)
    graph.add_edge('7','3',flow=4)
    graph.add_edge("2","4",flow=0,role="next_in_block")
    graph.add_edge("3","5",flow=0,role="next_in_block")
    graph.add_edge("4","5",flow=12,certainty="unknown")
    project=relations.existing_projector()
    _,bits,_=project(graph,node_ids=[0,1],active_mask=[True,True],max_slots=2)
    assert bits[0,1].item()==bits[1,0].item()==0
    graph.add_edge("4","5",flow=12,certainty="proven")
    _,bits,_=project(graph,node_ids=[0,1],active_mask=[True,True],max_slots=2)
    assert bits[0,1].item()==16 and bits[1,0].item()==32


def test_missing_or_non_mlir_graph_is_blocked(source,tmp_path):
    _,actions,_,_,_=source
    with pytest.raises(OSError):relations.graph_relations(tmp_path/'missing.gexf','x','y',actions)
    import networkx as nx
    p=tmp_path/'harp.gexf';nx.write_gexf(nx.MultiDiGraph(),p)
    with pytest.raises(ValueError,match='LLVM/HARP'):relations.graph_relations(p,'x','y',actions)


def test_adp_device_capacity_and_separate_budget():
    resources={k:metrics.CAPACITIES[k]//2 for k in metrics.CAPACITIES}
    row=metrics.evaluate_metrics(100,resources)
    assert row['area_proxy']==50 and row['ADP']==5000 and row['feasible']
    budget={**metrics.CAPACITIES,'DSP':resources['DSP']-1}
    infeasible=metrics.evaluate_metrics(100,resources,budget=budget)
    assert infeasible['area_proxy']==50 and not infeasible['feasible']
    assert metrics.best_feasible([dict(infeasible,synthesis_status='success',candidate_id='bad')]) is None
    assert metrics.improvement_pct(row,{**row,'ADP':4000})==20
    assert metrics.evaluate_metrics(None,resources)['ADP'] is None
    assert metrics.evaluate_metrics(100,{**resources,'FF':None})['rankable'] is False
    with pytest.raises(ValueError):metrics.evaluate_metrics(float('nan'),resources)


def test_source_materialization_disabled_and_complete(source):
    _,actions,template,_,base=source
    text=synthesis.materialize(template,c.serialize(base,actions))
    assert 'auto{' not in text and '#pragma HLS pipeline' not in text and '#pragma HLS array_partition' not in text
    base.update({c.key('ARRAY_T','L3'):'complete',c.key('ARRAY_D','L3'):'1'})
    text=synthesis.materialize(template,c.serialize(base,actions))
    assert 'type=complete' in text and 'factor=0' not in text and 'variable=b' in text
    with pytest.raises(ValueError):synthesis.materialize(template+'\nauto{_PIPE_L1}',c.serialize(base,actions))


def xml(top='top',part=metrics.PART,clock='10',version='2021.1',latency='100',estimated='7.5',clock_unit='ns',latency_unit='clock cycles'):
    timing='' if estimated is None else '<EstimatedClockPeriod>'+estimated+'</EstimatedClockPeriod>'
    return f'''<profile><ReportVersion><Version>{version}</Version></ReportVersion>
<UserAssignments><unit>{clock_unit}</unit><Part>{part}</Part><TopModelName>{top}</TopModelName><TargetClockPeriod>{clock}</TargetClockPeriod><FlowTarget>vivado</FlowTarget></UserAssignments>
<PerformanceEstimates><SummaryOfOverallLatency><unit>{latency_unit}</unit><Worst-caseLatency>{latency}</Worst-caseLatency></SummaryOfOverallLatency><SummaryOfTimingAnalysis>{timing}</SummaryOfTimingAnalysis></PerformanceEstimates>
<AreaEstimates><Resources><BRAM_18K>10</BRAM_18K><DSP>20</DSP><FF>300</FF><LUT>400</LUT></Resources></AreaEstimates></profile>'''


def test_tcl_and_valid_unknown_timing_xml(tmp_path):
    tcl=synthesis.make_tcl('source file.cpp','top')
    for token in ['set_top top','add_files {source file.cpp}','open_solution','set_part','create_clock -period 10','csynth_design',*synthesis.SETTINGS]:assert token in tcl
    with pytest.raises(ValueError):synthesis.make_tcl('bad}\nexec','top')
    p=tmp_path/'report.xml';p.write_text(xml(estimated=None))
    row=synthesis.parse_report(p,'top')
    assert row['latency_cycles']==100 and row['estimated_clock_ns'] is None and row['timing_status']=='unknown'
    assert row['ADP']>0 and row['rankable']


@pytest.mark.parametrize('changes',[{'top':'wrong'},{'part':'wrong'},{'clock':'5'},{'version':'2024.1'},{'latency':'?'},{'latency':'0'},{'clock_unit':'ps'},{'latency_unit':'ns'}])
def test_wrong_or_partial_report_provenance(tmp_path,changes):
    p=tmp_path/'report.xml';p.write_text(xml(**changes))
    with pytest.raises(synthesis.ReportError):synthesis.parse_report(p,'top')
    p.write_text('<profile>')
    with pytest.raises(synthesis.ReportError):synthesis.parse_report(p,'top')


@pytest.fixture
def queue_fixture(source,tmp_path):
    path,actions,template,domains,base=source
    template_path=tmp_path/'template.txt';template_path.write_text(template)
    kernel={'top':'top','template_path':str(template_path),'template_sha256':sha256(template_path),
      'source_provenance':{'source':str(path),'snapshot_source':str(path),'files':[{'snapshot':str(path),'sha256':sha256(path)}]}}
    single,joint=c.generate_pools('toy',base,actions,domains,[evidence()])
    rows=[c.candidate_record('toy','baseline',base,base,actions,domains),single[0],joint[0]]
    for row in rows:row['evaluation_context_hash']=object_hash(row['assignment'])
    fake=tmp_path/'fake_hls'
    fake.write_text('#!/usr/bin/python3\nimport os,time,pathlib,sys\n'
        'time.sleep(float(os.environ.get("FAKE_SLEEP","0")))\n'
        'p=pathlib.Path("r0_project/solution1/syn/report/top_csynth.xml");p.parent.mkdir(parents=True,exist_ok=True)\n'
        'p.write_text('+repr(xml())+')\n'
        'sys.exit(int(os.environ.get("FAKE_EXIT","0")))\n')
    fake.chmod(fake.stat().st_mode|stat.S_IXUSR)
    tool={'executable':str(fake),'version':'2021.1','executable_sha256':sha256(fake)}
    return {'kernels':{'toy':kernel}},rows,tmp_path/'out',tool


def test_resumable_runner_does_not_repeat_results(queue_fixture):
    manifest,rows,out,tool=queue_fixture
    with patch.object(synthesis,'tool_info',return_value=tool):
        assert synthesis.run_queue(manifest,rows,out,'fake')==3
        assert synthesis.run_queue(manifest,rows,out,'fake')==0
    results=read_jsonl(out/'synthesis_results.jsonl')
    assert len(results)==3 and all(r['synthesis_status']=='success' for r in results)
    assert all(r['attempt_count']==1 for r in results)
    Path(results[0]['report_path']).write_text('changed')
    with patch.object(synthesis,'tool_info',return_value=tool),pytest.raises(ValueError,match='report missing/modified'):
        synthesis.run_queue(manifest,rows,out,'fake')


def test_failed_exit_with_report_is_unrankable_and_retry_is_explicit(queue_fixture,monkeypatch):
    manifest,rows,out,tool=queue_fixture
    monkeypatch.setenv('FAKE_EXIT','7')
    with patch.object(synthesis,'tool_info',return_value=tool):
        assert synthesis.run_queue(manifest,rows,out,'fake')==1  # Arms blocked by failed baseline.
        assert synthesis.run_queue(manifest,rows,out,'fake')==0
        row=read_jsonl(out/'synthesis_results.jsonl')[0]
        assert row['synthesis_status']=='failed' and row['ADP'] is None and not row['rankable']
        monkeypatch.delenv('FAKE_EXIT')
        assert synthesis.run_queue(manifest,rows,out,'fake',retry_failed=True)==3
    rows=read_jsonl(out/'synthesis_results.jsonl')
    assert len(rows)==3 and rows[0]['attempt_count']==2


def test_timeout_and_missing_executable(queue_fixture,monkeypatch):
    manifest,rows,out,tool=queue_fixture
    monkeypatch.setenv('FAKE_SLEEP','2')
    with patch.object(synthesis,'tool_info',return_value=tool):
        assert synthesis.run_queue(manifest,rows,out,'fake',timeout=0.03)==1
    row=read_jsonl(out/'synthesis_results.jsonl')[0]
    assert row['synthesis_status']=='timeout' and row['ADP'] is None
    with pytest.raises(FileNotFoundError):synthesis.tool_info('/nonexistent/r0-vitis-hls')


def test_modified_source_snapshot_blocks_synthesis(queue_fixture):
    manifest,rows,out,tool=queue_fixture
    Path(manifest['kernels']['toy']['source_provenance']['source']).write_text('changed')
    with pytest.raises(ValueError,match='snapshot changed'):synthesis.run_candidate(rows[0],manifest['kernels']['toy'],out,tool)


def test_real_published_baselines_preserved_under_owner_factor_override():
    rows=cli.read_jsonl(cli.ROOT/'paper_plots/SALT_experiments.jsonl')
    for name,expected in [('covariance',False),('GAN',False),('Kalman',True)]:
        spec=cli.SPECS[name];info=cli.repo_path(spec['metadata']).parent
        actions=dd.load_source_actions(name,info)
        template=dd.prepare_source_template(name,cli.repo_path(spec['source']),info)
        domains=dd.source_site_domains(name,template,info)
        paper=next(r for r in rows if r['kernel']==name and r.get('objective')=='ADP' and r['stage']=='predicted-directives')
        base=c.from_paper(paper,actions)
        if expected:c.validate(base,actions,domains)
        else:
            with pytest.raises(c.DomainViolation):c.validate(base,actions,domains)
        original = dict(base)
        local = c.published_factor_domains(base, actions, domains)
        c.validate(base, actions, local)
        assert base == original
        assert all(local[k] == values for k, values in domains.items()
                   if not ('UNROLL' in k or 'ARRAY_F' in k))
        single, joint = c.generate_pools(name, base, actions, local, [])
        assert len(c.select_diverse(single, actions, 123)) == 4
        assert not joint
        for row in single:
            c.validate(row['assignment'], actions, local)


def test_published_factor_local_steps_not_historical_whitelist(source):
    _, actions, _, domains, base = source
    base[c.key('UNROLL','L1')] = '10'
    base.update({c.key('ARRAY_T','L3'):'cyclic',c.key('ARRAY_F','L3'):'5',c.key('ARRAY_D','L3'):'1'})
    local = c.published_factor_domains(base, actions, domains)
    assert local[c.key('UNROLL','L1')] == ['0','2','9','10','11']
    assert local[c.key('ARRAY_F','L3')] == ['0','2','4','5','6']
    assert '10' not in domains[c.key('UNROLL','L1')]
    bad = {**base,c.key('ARRAY_D','L3'):'9'}
    with pytest.raises(c.DomainViolation):c.published_factor_domains(bad, actions, domains)
    bad = {**base,c.key('ARRAY_T','L3'):'none'}
    with pytest.raises(ValueError):c.published_factor_domains(bad, actions, domains)


@pytest.mark.parametrize('off_domain',[False,True])
def test_prepare_without_compiler_or_hls_is_explicit_and_deterministic(source,tmp_path,off_domain):
    from argparse import Namespace
    path,actions,template,domains,base=source
    directives=[]
    for label,action in actions.items():
        if action.kind=='loop':directives.append({'action_id':label,'kind':'loop','pipeline_ii':0,'unroll_factor':3 if off_domain and label=='L1' else 0})
        else:directives.append({'action_id':label,'kind':'array','variable':'b','partition_type':'none','partition_factor':0,'partition_dim':0})
    paper=tmp_path/'paper.jsonl'
    write_jsonl(paper,[{'kernel':'covariance','objective':'ADP','stage':'predicted-directives','latency':100,
        'resources':{'bram':10,'dsp':20,'ff':300,'lut':400},'directives':directives}])
    args=Namespace(output_dir=str(tmp_path/'prepared'),baseline_jsonl=str(paper),config=None,
        polygeist_build=None,mlir_python=None,kernels=['covariance'],seed=123,graph_timeout=5)
    spec={'source':str(path),'metadata':str(path.parent/'kernel_info.txt'),'top':'top'}
    with patch.dict(cli.SPECS,{'covariance':spec}),patch.object(cli,'compiler_tools',side_effect=FileNotFoundError('test compiler unavailable')):
        first=cli.prepare(args)
        original=(Path(args.output_dir)/'candidates.jsonl').read_bytes()
        second=cli.prepare(args)
    assert first==second
    assert original==(Path(args.output_dir)/'candidates.jsonl').read_bytes()
    assert first['kernels']['covariance']['relations']['status']=='blocked'
    assert first['actual_prepared_configurations']==5
    initial = first['kernels']['covariance']['initial_assignment']
    assert initial[c.key('UNROLL','L1')] == ('3' if off_domain else '0')
    assert not read_jsonl(Path(args.output_dir)/'synthesis_results.jsonl')
    report=(Path(args.output_dir)/'R0_REPORT.md').read_text()
    assert 'not measured successfully' in report
    assert not list(Path(args.output_dir).glob('projects/*'))


def test_authenticated_kalman_graph_preserves_sparse_absolute_actions():
    spec=cli.SPECS['Kalman']
    actions=dd.load_source_actions('Kalman',cli.repo_path(spec['metadata']).parent)
    result=relations.graph_relations(cli.repo_path(spec['relations']),sha256(cli.repo_path(spec['source'])),
        sha256(cli.repo_path(spec['metadata'])),actions,cli.repo_path(spec['relation_metadata']))
    assert result['inactive_actions']==['L7','L8']
    assert result['labels'][6:8]==[-1,-1]
    assert result['active_mask'][8] and result['labels'][8]==9
    assert any('loop_array_access' in pair['relation_type'] for pair in result['pairs'])
    assert all(not {'L7','L8'} & {p['action_i'],p['action_j']} for p in result['pairs'])
