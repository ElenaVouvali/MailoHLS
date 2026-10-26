"""CPU-only corpus and adapter checks; synthetic oracle results are test-only."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import pytest
from test_refinement_search import source, oracle
from GNN_branch import generate_mlir_v2_corpus as corpus, mlir_graph_v2 as v2
from experiments.v21_refinement import experiment as e
from experiments.refinement import configurations as c, experiment as old, relations
from experiments.refinement_search import controller as ctl, policies as p


def augment(problem):
    problem = deepcopy(problem)
    problem['v21_pairs'] = [{**r,'ordering_bucket':r['verified_relation_class']}
                            for r in problem['relations'] if r['verified_relation_class'] in e.VERIFIED]
    problem['v21_features'] = {label:{'trip_count':a.trip_count,'kind':a.kind,
        'arithmetic_operations':0,'loop_depth':0,'proven_carrying_dependences':0} for label,a in problem['actions'].items()}
    return problem


def test_inventory_complete_content_identity_and_split_provenance():
    found = corpus.inventory()
    names = {a['kernel'] for r in found['snapshots'] for a in r['aliases'] if a['origin']=='development'}
    assert len(names)==found['development_kernel_count']==55
    assert 'GNN_branch.config' not in __import__('sys').modules
    assert len({r['snapshot_id'] for r in found['snapshots']})==len(found['snapshots'])
    kalman = [r for r in found['snapshots'] if any(a['kernel']=='serrano-kalman-filter' for a in r['aliases'])]
    assert len({r['metadata_sha256'] for r in kalman}) >= 2
    assert all('split_membership' in a for r in found['snapshots'] for a in r['aliases'])


def test_discovery_collapses_identical_copies_preserves_metadata_variants(tmp_path):
    (tmp_path/'GNN_branch').mkdir(); (tmp_path/'GNN_branch/config.py').write_text("ALL_KERNEL = ['toy']\nraise RuntimeError('must not import')\n")
    (tmp_path/'Data').mkdir()
    (tmp_path/'Data/ApplicationInformation.csv').write_text('app_name,top_level_function,file_name,file_name_extension\ntoy,top,kernel.cpp,cpp\n')
    (tmp_path/'evaluation_set').mkdir()
    for directory in (tmp_path/'Data/ApplicationDataset/toy',tmp_path/'evaluation_set/copy'):
        directory.mkdir(parents=True); (directory/'kernel.cpp').write_text('void top(){L1: for(int i=0;i<8;i++){} }\n')
        (directory/'kernel_info.txt').write_text('top\nL1,loop,8\n')
    first = corpus.inventory(tmp_path)
    assert len(first['snapshots'])==1 and len(first['snapshots'][0]['aliases'])==2
    (tmp_path/'evaluation_set/copy/kernel_info.txt').write_text('top\nL1,loop,7\n')
    assert len(corpus.inventory(tmp_path)['snapshots'])==2


@pytest.fixture(scope='module')
def real_graph():
    path = corpus.DEFAULT_REUSE/'GAN.v21.json'
    if not path.is_file(): pytest.skip('Locally authenticated GAN artifact unavailable')
    return path,v2.read_graph(path)


def test_real_authentication_rejects_mismatched_source_metadata_schema_tool(real_graph,tmp_path):
    path,graph = real_graph
    snap = next(r for r in corpus.inventory()['snapshots'] if r['kernel']=='GAN')
    tools = {'analysis_extension_sha256':graph.graph['provenance']['analysis_extension_sha256'],
             'cgeist_sha256':graph.graph['provenance']['v1_provenance']['cgeist_sha256']}
    corpus.authenticate(path,snap,tools)
    for key in ('source_sha256','metadata_sha256'):
        with pytest.raises(ValueError,match='provenance mismatch'): corpus.authenticate(path,{**snap,key:'different'},tools)
    with pytest.raises(ValueError,match='frontend build'): corpus.authenticate(path,snap,{**tools,'cgeist_sha256':'different'})
    changed=deepcopy(graph); changed.graph['schema']=v2.SCHEMA
    wrong=tmp_path/'old.json'; v2.write_graph(changed,wrong)
    with pytest.raises(ValueError,match='V2.1'): corpus.authenticate(wrong,snap,tools)


def test_real_canonical_compact_queries_deterministic_and_unmodified(real_graph,tmp_path):
    path,graph=real_graph; original=c.sha256(path)
    compact,facts=corpus.checked_views(graph)
    assert not any(r.get('neighbor')=='L2' and r['evidence_status']=='proven' for r in facts['L11'])
    assert any(r['neighbor']=='L11' and r['relation_type']=='memory_access' for r in facts['L3'])
    one,two=tmp_path/'one.json',tmp_path/'two.json';v2.write_graph(graph,one);v2.write_graph(graph,two)
    assert one.read_bytes()==two.read_bytes() and original==c.sha256(path)
    assert v2.graph_payload(v2.expand_compact(compact))==v2.graph_payload(graph)


def test_corpus_failure_isolation_authenticated_reuse_and_resume(real_graph,tmp_path,monkeypatch):
    path,graph=real_graph; snap=next(r for r in corpus.inventory()['snapshots'] if r['kernel']=='GAN')
    tools={'analysis_extension_sha256':graph.graph['provenance']['analysis_extension_sha256'],
           'cgeist_sha256':graph.graph['provenance']['v1_provenance']['cgeist_sha256']}
    monkeypatch.setattr(corpus.subprocess,'run',lambda *a,**kw:pytest.fail('Authenticated reuse must not compile'))
    assert corpus.process({**snap,'inventory_error':'missing source'},tmp_path,tools)['status']=='blocked'
    good=corpus.process(snap,tmp_path,tools,[path])
    assert good['status']=='authenticated_reuse' and good['semantic_queries']=='identical'
    assert corpus.process(snap,tmp_path,tools,[path],good)['canonical_sha256']==good['canonical_sha256']
    canonical=Path(good['canonical']);canonical.write_text(canonical.read_text()+' ')
    assert corpus.process(snap,tmp_path,tools,[path],good)['status']=='invalid_artifact'


def test_real_relation_adapter_preserves_call_contexts_and_evidence(real_graph):
    _,graph=real_graph; actions={a['action_id']:None for _,a in graph.nodes(data=True) if a['category']=='action_point'}
    pairs=e.verified_pairs(graph,[],actions,'graph-hash')
    assert not any({r['action_i'],r['action_j']}=={'L2','L11'} for r in pairs)
    known=next(r for r in pairs if {r['action_i'],r['action_j']}=={'L3','L11'})
    assert known['artifact_hash']=='graph-hash' and all(r['evidence_status']=='proven' for r in known['typed_evidence'])
    assert any(r.get('access')=='read' and r.get('callsite_bindings') for r in known['typed_evidence'])


def test_p2_equal_eligibility_preserves_p1_exact_order(source):
    _,original=source;problem=augment(original)
    problem['v21_pairs'][0]['verified_relation_class']='indirect_verified'
    a=p.structural_proposals(problem,problem['baseline'],123);b=e.pair_proposals(problem,problem['baseline'],123)
    assert [r['signature'] for r in a]==[r['signature'] for r in b]
    assert all(r['relation_evidence']['verified_relation_class']=='indirect_verified' for r in b)


def test_proposals_complete_legal_deterministic_no_qor_inputs(source):
    _,original=source;problem=augment(original);base=problem['baseline']
    for joint in (False,True):
        rows=e.prioritized(problem,base,123,joint)
        assert rows and rows==e.prioritized(problem,base,123,joint)
        assert len({r['signature'] for r in rows})==len(rows)
        for row in rows:
            assert set(row['assignment'])==set(base) and len(row['reference_changed_actions'])==(2 if joint else 1)
            c.policy_record(row['assignment'],base,problem['actions'],problem['native_domains'],'tcad')
            assert row['structural_priority']['actions']
    assert 'measurement' not in problem and 'cache' not in problem


def test_joint_proposals_do_not_require_beneficial_singles_and_p3_fixed_slots(source):
    _,original=source;problem=augment(original);base=problem['baseline']
    first=e.choose('v21_relations',problem,[],base,123,p.choose);history=[first]
    assert len(e.choose('v21_relations',problem,history,base,123,p.choose)['reference_changed_actions'])==2
    for step in range(2,9):
        row=e.choose('v21_priority',problem,history,base,123,p.choose)
        assert (row['proposal_type']=='structural')==(step in (4,8)); history.append(row)
    problem['v21_pairs']=[]
    assert e.choose('v21_relations',problem,history,base,123,p.choose) is None
    assert e.choose('v21_priority',problem,history[:3],base,123,p.choose)['proposal_type']=='coordinate'


@pytest.fixture
def prepared(source,tmp_path,monkeypatch):
    kernel,original=source;problem=augment(original)
    old_dir=tmp_path/'historical';old_dir.mkdir();c.write_json(old_dir/'manifest.json',{'original':True})
    original_manifest={'context':c.CONTEXT,'domain_policy':'tcad','kernels':{'toy':kernel},'native_domains':{'toy':problem['native_domains']}}
    monkeypatch.setattr(old,'load_output',lambda path:(original_manifest,[]))
    monkeypatch.setattr(old,'read_measurements',lambda path:({},[],0))
    monkeypatch.setattr(c,'from_paper',lambda row,actions:problem['baseline'])
    monkeypatch.setattr(c,'compiler_actions',lambda kernel,actions:actions)
    monkeypatch.setattr(relations,'audit_kernel',lambda kernel,actions:(problem['relations'],[]))
    protocol={'kernels':{'toy':{'pairs':problem['v21_pairs'],'features':problem['v21_features']}},
              'artifact_sha256':{},'physical_cache_roots':[],'protocol_path':str(tmp_path/'protocol.json')}
    c.write_json(tmp_path/'protocol.json',ctl._sealed(protocol));output=tmp_path/'search'
    with e.runtime(protocol):
        ctl.prepare(Namespace(input_dir=str(old_dir),output_dir=str(output),measurements=[],kernels=['toy'],
                    policies=list(e.POLICIES),max_evaluations=8,seed=123,execution='replay',fresh_synthesis=False))
    return output,protocol


def test_shared_engine_private_histories_budget_resume_and_cache_cost(prepared,oracle):
    output,protocol=prepared;oracle.values=[100,80,120,{'ADP':1,'feasible':False},
        {'synthesis_status':'unavailable','ADP':None,'rankable':False,'feasible':None},90,70,75]
    before=e.benchmark.core_hashes();old_choose=p.choose
    with e.runtime(protocol):
        for policy in e.POLICIES:
            path=output/'toy'/policy;ctl.search(path,limit=3);prefix=c.read_jsonl(path/'search_trace.jsonl');ctl.search(path)
            manifest,shared=ctl.load_run(path);traces,records,history,pending,terminal=ctl.read_state(path,manifest,shared)
            assert len(history)<=8 and not pending and len({r['signature'] for r in history})==len(history)
            assert c.read_jsonl(path/'search_trace.jsonl')[:3]==prefix
            assert history[0]['signature']==manifest['baseline_signature']
            best=[r['best_ADP_so_far'] for r in traces if r['event_type']=='query'];assert best==sorted(best,reverse=True)
            n=len(oracle);ctl.search(path);assert len(oracle)==n
            assert all(t['logical_evaluation_cost']==1 for t in traces if t['event_type']=='query')
            assert not ctl.summarize(path)[3]['coverage_complete']
    assert p.choose is old_choose and e.benchmark.core_hashes()==before


def test_protocol_tamper_and_nonempty_directory_rejected(tmp_path):
    payload={'code_sha256':e.code_hashes(),'artifact_sha256':{},'context':c.CONTEXT,'rules':e.RULES}
    c.write_json(tmp_path/'protocol.json',ctl._sealed(payload));assert e.load_protocol(tmp_path)['rules']==e.RULES
    payload['code_sha256']={};c.write_json(tmp_path/'protocol.json',ctl._sealed(payload))
    with pytest.raises(ValueError,match='implementation changed'):e.load_protocol(tmp_path)
    with pytest.raises(ValueError,match='Output exists'):e.prepare(Namespace(output_dir=str(tmp_path)))


def test_batch_safety_flag_required(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'load_protocol',lambda path:{'execution':'synthesis','kernels':{'GAN':{},'Kalman':{}}})
    with pytest.raises(ValueError,match='allow-batch'):
        e.execute(Namespace(output_dir=str(tmp_path),mode='run',kernels=None,policies=None,allow_batch=False))


def test_policy_provenance_does_not_change_unchanged_physical_engine_context(prepared):
    output,protocol=prepared
    original=ctl.physical_candidate
    with e.runtime(protocol):
        manifest,shared=ctl.load_run(output/'toy'/'coordinate')
        problem=ctl.problem_for(manifest,shared)
        proposal=p.choose('coordinate',problem,[],problem['baseline'])
        candidate=ctl.physical_candidate(manifest,shared,proposal)
        expected=c.object_hash({'source_context':e.existing.hls.source_context(shared['kernels']['toy']),
            'signature':proposal['signature'],'search_code':e.benchmark.core_hashes()})
        assert candidate['evaluation_context_hash']==expected
        assert shared['code_sha256']!=e.benchmark.core_hashes()
        altered=deepcopy(shared);altered['kernels']['toy']['source_provenance']['source_sha256']='different'
        assert ctl.physical_candidate(manifest,altered,proposal)['evaluation_context_hash']!=expected
    assert ctl.physical_candidate is original


def test_retrospective_diagnostic_uses_canonical_assignment_field_identity():
    # Serialized directives accept auto; internal assignments use canonical AUTO.
    values=c.parse_assignment('auto{_PIPE_L6} = 1\nauto{_UNROLL_L6} = 0')
    assert values[c.field_key('PIPE','L6')]=='1'
