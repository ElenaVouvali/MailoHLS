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
def real_graph(tmp_path_factory):
    path = corpus.DEFAULT_REUSE/'GAN.v21.json'
    if not path.is_file(): pytest.skip('Locally authenticated GAN artifact unavailable')
    graph = v2.read_graph(path)
    if graph.graph['provenance']['v2_builder_sha256'] != c.sha256(v2.__file__):
        try:
            tools = corpus.compiler_tools(corpus.BUILD)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        snapshot = next(r for r in corpus.inventory()['snapshots'] if r['kernel']=='GAN')
        record = corpus.process(snapshot,tmp_path_factory.mktemp('current_GAN_graph'),tools)
        assert record.get('validation')=='pass',record.get('error')
        path = Path(record['canonical']); graph = v2.read_graph(path)
    return path,graph


def test_real_authentication_rejects_mismatched_source_metadata_schema_tool(real_graph,tmp_path):
    path,graph = real_graph
    snap = next(r for r in corpus.inventory()['snapshots'] if r['kernel']=='GAN')
    tools = {'analysis_extension_sha256':graph.graph['provenance']['analysis_extension_sha256'],
             'cgeist_sha256':graph.graph['provenance'].get('v1_provenance',graph.graph['provenance'])['cgeist_sha256']}
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
           'cgeist_sha256':graph.graph['provenance'].get('v1_provenance',graph.graph['provenance'])['cgeist_sha256']}
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


def test_explicit_source_variant_preserves_original_identity(tmp_path):
    (tmp_path/'GNN_branch').mkdir();(tmp_path/'GNN_branch/config.py').write_text("ALL_KERNEL=['toy']\n")
    (tmp_path/'Data/ApplicationDataset/toy').mkdir(parents=True)
    (tmp_path/'Data/ApplicationInformation.csv').write_text('app_name,top_level_function,file_name,file_name_extension\ntoy,top,kernel.c,c\n')
    original=tmp_path/'Data/ApplicationDataset/toy/kernel.c';original.write_text('void top(void) {}\n')
    (original.parent/'kernel_info.txt').write_text('top\nL9,loop,8\n')
    (tmp_path/'evaluation_set').mkdir()
    selected=tmp_path/'variant.c';selected.write_text('int top(int x) {\nL9: for(int i=0;i<8;i++) x+=i;\nreturn x;\n}\n')
    before=original.read_bytes()
    found=corpus.inventory(tmp_path,{'toy':selected});row=found['snapshots'][0]
    assert row['inventory_error'] is None and row['source_sha256']==c.sha256(selected)
    assert row['aliases'][0]['original_source_sha256']==c.sha256(original)
    assert 'equivalence not assumed' in row['aliases'][0]['source_selection']
    assert original.read_bytes()==before
    with pytest.raises(ValueError,match='Unknown source-override'):
        corpus.inventory(tmp_path,{'absent':selected})


def test_portable_package_integrity_and_determinism(real_graph,tmp_path):
    import tarfile
    path,graph=real_graph
    snap=next(r for r in corpus.inventory()['snapshots'] if r['kernel']=='GAN')
    tools={'analysis_extension_sha256':graph.graph['provenance']['analysis_extension_sha256'],
           'cgeist_sha256':graph.graph['provenance'].get('v1_provenance',graph.graph['provenance'])['cgeist_sha256']}
    output=tmp_path/'corpus';output.mkdir()
    record=corpus.process(snap,output,tools,[path]);assert record['validation']=='pass'
    manifest={'schema':v2.SCHEMA_V21,'toolchain':tools,'inventory':{'snapshots':[snap]},
              'records':{snap['snapshot_id']:record},
              'code_sha256':{'GNN_branch/mlir_graph_v2.py':c.sha256(v2.__file__)}}
    c.write_json(output/'corpus_manifest.json',manifest)
    for name in ('graph_coverage.csv','relation_coverage.csv','generation_summary.csv'):
        (output/name).write_text('kernel,status\nGAN,pass\n')
    one,two=tmp_path/'one.tar.gz',tmp_path/'two.tar.gz'
    index=corpus.package(output,one);corpus.package(output,two)
    assert c.sha256(one)==c.sha256(two)
    with tarfile.open(one) as archive:
        portable=__import__('json').load(archive.extractfile('index.json'))
        assert portable['schema']==v2.SCHEMA_V21
        row=portable['snapshots'][0]
        assert not Path(row['canonical']).is_absolute()
        assert archive.extractfile(row['source']).read()==Path(snap['source']).read_bytes()
    compact=Path(record['compact']);compact.write_text(compact.read_text()+' ')
    with pytest.raises(ValueError,match='Graph changed'):
        corpus.package(output,tmp_path/'tampered.tar.gz')
    manifest['records']={};c.write_json(output/'corpus_manifest.json',manifest)
    with pytest.raises(ValueError,match='complete validated corpus'):
        corpus.package(output,tmp_path/'partial.tar.gz')


@pytest.fixture(scope='module')
def native_pipeline():
    try:
        return corpus.compiler_tools(corpus.BUILD)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('name,expected',[('chstone-jpeg',21),('rosetta-3d-rendering',12),('gemv',5)])
def test_recovered_external_sources_compile_with_complete_action_coverage(name,expected,native_pipeline,tmp_path):
    overrides={'gemv':c.ROOT/'evaluation_set/gemv/code_to_memory_outputs/meta/gemv.c'}
    snap=next(r for r in corpus.inventory(source_overrides=overrides)['snapshots'] if r['kernel']==name)
    result=corpus.process(snap,tmp_path,native_pipeline,timeout=180)
    assert result.get('validation')=='pass',result.get('error')
    assert result['coverage']['registered']==expected
    if name!='rosetta-3d-rendering':
        assert result['coverage']['compiler_active']==expected
    assert result['semantic_queries']=='identical' and result['reconstruction']=='exact'
    graph=v2.read_graph(result['canonical'])
    if name=='rosetta-3d-rendering':
        # Historical V2.1 misses this class-valued static array. Preserve that
        # result; require complete coverage from the explicitly selected RC3.
        assert result['coverage']['compiler_active']==11
        assert 'action:L1' not in graph
        import subprocess,os
        path=tmp_path/'rc3.json'
        import json
        script="""import json,sys
from GNN_branch import mlir_graph_v2 as v
g,_=v.generate(sys.argv[1],sys.argv[2],sys.argv[3],provenance=json.loads(sys.argv[5]),revision='2.2',semantic_revision='rc3')
v.write_graph(g,sys.argv[4])
"""
        command=[native_pipeline['python'],'-B','-c',script,snap['source'],snap['metadata'],
            result['raw_mlir'],str(path),json.dumps(graph.graph['provenance'])]
        run=subprocess.run(command,cwd=c.ROOT,env={**os.environ,'PYTHONPATH':native_pipeline['pythonpath']},capture_output=True,text=True,timeout=180)
        assert run.returncode==0,run.stderr
        graph=v2.read_graph(path);compact,facts=corpus.checked_views(graph)
        assert corpus.coverage(graph,facts)['compiler_active']==expected
        binding=graph.nodes['action:L1']['array_binding_evidence']
        assert binding['tier']=='exact_declaration_location' and binding['source_line']==314
        assert graph.graph['numerical_fidelity']['hardware_exact'] is False
    if name=='gemv':
        assert all(graph.nodes['action:'+label]['compiler_proven_trip_count']==32 for label in ('L4','L5'))
        assert 'original_source_sha256' in result['aliases'][0]


@pytest.fixture(scope='module')
def original_jpeg(tmp_path_factory):
    import subprocess,shutil
    if not shutil.which('gcc'):pytest.skip('Host GCC unavailable')
    result=subprocess.run(['git','show','6863ce0:evaluation_set/chstone-jpeg/kernel.c'],cwd=c.ROOT,capture_output=True)
    if result.returncode:pytest.skip('Original immutable JPEG reference commit unavailable')
    directory=tmp_path_factory.mktemp('original_jpeg');source=directory/'original.c';source.write_bytes(result.stdout)
    return source


@pytest.mark.parametrize('optimization',['-O0','-O2'])
def test_jpeg_full_selftest_stdout_matches_original(original_jpeg,optimization,tmp_path):
    import subprocess
    harness=tmp_path/'main.c';harness.write_text('int chstone_jpeg(void); int main(void){return chstone_jpeg();}\n')
    outputs=[]
    for i,source in enumerate((original_jpeg,c.ROOT/'evaluation_set/chstone-jpeg/kernel.c')):
        target=tmp_path/('jpeg'+str(i))
        build=subprocess.run(['gcc','-std=c99',optimization,str(source),str(harness),'-o',str(target)],capture_output=True,text=True)
        assert build.returncode==0,build.stderr
        run=subprocess.run([str(target)],capture_output=True,text=True,timeout=30)
        assert run.returncode==0,run.stdout[-200:]
        outputs.append((run.stdout,run.stderr))
    assert outputs[0]==outputs[1]


def test_jpeg_marker_edge_cases_and_all_read_word_values_match_original(original_jpeg,tmp_path):
    import subprocess
    harness=tmp_path/'markers.c'
    harness.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
static unsigned char data[64];
static void report(void) {
 printf("STATE %d %d %ld %d %d %d %d %d %d\n",main_result,i_get_sos,(long)(ReadBuf-data),
 p_jinfo_comps_info_dc_tbl_no[0],p_jinfo_comps_info_dc_tbl_no[1],p_jinfo_comps_info_dc_tbl_no[2],
 p_jinfo_comps_info_ac_tbl_no[0],p_jinfo_comps_info_ac_tbl_no[1],p_jinfo_comps_info_ac_tbl_no[2]);
}
int main(int argc,char **argv) {
 int mode=atoi(argv[1]);
 if(mode==9) {
  for(int a=0;a<256;a++)for(int b=0;b<256;b++) {
   data[0]=a;data[1]=b;ReadBuf=data;
   if(read_word()!=(short)((a<<8)|b) || ReadBuf!=data+2)return 1;
  }
  puts("65536 words matched");return 0;
 }
 atexit(report);
 p_jinfo_num_components=(mode==4)?0:3;
 for(int i=0;i<3;i++)p_jinfo_comps_info_id[i]=i+1;
 data[0]=0;data[1]=12;data[2]=(mode==3)?0:3;
 for(int i=0;i<3;i++) {
  data[3+2*i]=(mode==1)?3-i:i+1;
  data[4+2*i]=i*17;
 }
 if(mode==2)data[3]=99;
 ReadBuf=data;get_sos();return 0;
}
''')
    binaries=[]
    for i,source in enumerate((original_jpeg,c.ROOT/'evaluation_set/chstone-jpeg/kernel.c')):
        target=tmp_path/('markers'+str(i))
        # Include the benchmark in the wrapper's translation unit so ReadBuf
        # retains its actual file-local linkage. No benchmark code is changed.
        build=subprocess.run(['gcc','-std=c99','-O2','-include',str(source),str(harness),'-o',str(target)],capture_output=True,text=True)
        assert build.returncode==0,build.stderr;binaries.append(target)
    for case in (0,1,2,3,4,9):
        runs=[subprocess.run([str(binary),str(case)],capture_output=True,text=True,timeout=10) for binary in binaries]
        assert all(r.returncode==0 for r in runs)
        assert (runs[0].stdout,runs[0].stderr)==(runs[1].stdout,runs[1].stderr),case


def test_rosetta_analysis_header_masks_slices_and_wraps_without_hardware_claim(tmp_path):
    import subprocess,shutil
    if not shutil.which('g++'):pytest.skip('Host C++ compiler unavailable')
    source=tmp_path/'ap_test.cpp'
    source.write_text(r'''
#include "ap_int.h"
#include <cassert>
int main() {
 ap_uint<32> value=0;
 value(7,0)=0x12;value(15,8)=0x34;value(23,16)=0x56;value(31,24)=0x78;
 assert((std::uint64_t)value==0x78563412);
 assert((std::uint64_t)value(15,8)==0x34);
 ap_uint<8> byte=255;assert((std::uint64_t)byte++==255);assert((std::uint64_t)byte==0);
 byte=257;assert((std::uint64_t)byte==1);
 ap_uint<1> bit=1;++bit;assert((std::uint64_t)bit==0);
 ap_uint<64> full=0;full(63,0)=~std::uint64_t(0);assert((std::uint64_t)full(63,0)==~std::uint64_t(0));
 return 0;
}
''')
    result=subprocess.run(['g++','-std=c++17','-O2','-I',str(c.ROOT/'evaluation_set/rosetta-3d-rendering'),str(source),'-o',str(tmp_path/'ap_test')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert subprocess.run([str(tmp_path/'ap_test')]).returncode==0
    assert 'AP_INT_STUB' in (c.ROOT/'evaluation_set/rosetta-3d-rendering/ap_int.h').read_text()
