"""Independent source/native fixtures; handwritten IR is explicitly identified."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
from test_mlir_graph_v2 import native_tools
from GNN_branch import mlir_graph_v2 as v2
from GNN_branch.action_point_relations import get_action_point_relations as query, action_point_views, get_loop_semantics
from experiments.structural_audit import finalize_v22 as audit
from GNN_branch.mlir_semantics_v22 import canonical_analysis


def native_graph(tools,source,metadata,raw,output,provenance,semantic_revision='rc3'):
    script='''import json,sys
from GNN_branch import mlir_graph_v2 as v
g,_=v.generate(sys.argv[1],sys.argv[2],sys.argv[3],provenance=json.loads(sys.argv[5]),revision='2.2',semantic_revision=sys.argv[6])
v.write_graph(g,sys.argv[4])
'''
    result=subprocess.run([tools['python'],'-B','-c',script,str(source),str(metadata),str(raw),str(output),json.dumps(provenance),semantic_revision],
        cwd=audit.ROOT,env={**os.environ,'PYTHONPATH':tools['pythonpath'],'PYTHONHASHSEED':'0'},capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stderr
    return v2.read_graph(output)


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    return audit.baseline(tmp_path_factory.mktemp('v22_frozen'))


@pytest.fixture(scope='module')
def real_graphs(frozen,native_tools,tmp_path_factory):
    base,release=frozen;output=tmp_path_factory.mktemp('v22_real');graphs={}
    for row in release['graphs']:
        if not (row['kernel']=='rodinia_lud_1_tiling_0' or row['snapshot_id'].startswith('96b4d608')):continue
        old=v2.read_graph(base/row['canonical'])
        graphs[row['kernel']]=(old,native_graph(native_tools,base/row['source'],base/row['metadata'],base/row['raw_mlir'],
            output/(row['snapshot_id']+'.json'),old.graph['provenance']))
    return graphs


def test_external_kalman_overlap_binding_is_not_allocation_identity(real_graphs):
    old,new=real_graphs['serrano-kalman-filter']
    def target(g,label):return next(v for _,v,e in g.out_edges('action:'+label,data=True) if e['relation_type']=='targets')
    assert target(old,'L6')==target(old,'L8')  # Confirm the demonstrated legacy failure.
    assert 'action:L7' not in new and 'action:L8' not in new
    coverage={r['action_id']:r for r in new.graph['action_point_coverage']['records']}
    for label in ('L7','L8'):
        assert coverage[label]['metadata_registered'] and not coverage[label]['compiler_matched']
        assert coverage[label]['compiler_eliminated'] is None
        assert not coverage[label]['resolution_diagnostic']['elimination_proven']
    assert get_loop_semantics(new,'L10')['compiler_proven_trip_count']==4095
    assert len([r for r in coverage.values() if r['metadata_registered']])==13


def test_LUD_invocation_contexts_retain_positive_and_negative_evidence(real_graphs):
    _,graph=real_graphs['rodinia_lud_1_tiling_0']
    assert sum(a['category']=='action_point' for _,a in graph.nodes(data=True))==46
    rows=[r for r in query(graph,'L2') if r['neighbor']=='L5' and r['relation_type']=='memory_access']
    assert rows and all(r['invocation_context']==['op:lud_diagonal:op5'] and r['callsite_bindings'] for r in rows)
    assert not [r for r in query(graph,'L2',call_context=()) if r['neighbor']=='L5']
    with pytest.raises(ValueError,match='verified originating call'):query(graph,'L2',call_context=('missing',))


@pytest.fixture(scope='module')
def source_cases(native_tools,tmp_path_factory):
    entries=json.loads((audit.ROOT/'experiments/structural_audit/fixtures/semantics_v21.json').read_text())
    entries=[r for r in entries if 'mlir' not in r]
    entries.append({'name':'nested_calls','program':'''extern "C" __attribute__((noinline)) void inner(float *p){p[0]+=1;}
extern "C" __attribute__((noinline)) void outer(float *p){inner(p);}
extern "C" float top(float x){
L1: float a[8];
L3: float b[8];
a[0]=x;b[0]=x+1;
L9: for(int i=0;i<8;i++)outer(a);
L19: for(int j=0;j<8;j++)outer(b);
outer(a);return a[0]+b[0];}
''','metadata':'top\nL1,array,a,1,8\nL3,array,b,1,8\nL9,loop,8\nL19,loop,8\n'})
    entries.append({'name':'static_arrays','program':'''extern "C" float top(float x){
L1: static float a[8];
L3: static float b[8];
L9: for(int i=0;i<8;i++){a[i]+=x;b[i]=a[i]*2;}
return b[0];}
''','metadata':'top\nL1,array,a,1,8\nL3,array,b,1,8\nL9,loop,8\n'})
    output=tmp_path_factory.mktemp('v22_sources');graphs={}
    for entry in entries:
        directory=output/entry['name'];directory.mkdir()
        source=directory/'kernel.cpp';meta=directory/'kernel_info.txt';path=directory/'graph.json'
        source.write_text(entry['program']);meta.write_text(entry['metadata'])
        result=subprocess.run([native_tools['python'],'-B','-m','GNN_branch.mlir_graph_v2','--revision','2.2',
            '--source',str(source),'--metadata',str(meta),'--cgeist',native_tools['cgeist'],'--output',str(path)],
            cwd=audit.ROOT,env={**os.environ,'PYTHONPATH':native_tools['pythonpath'],'PYTHONHASHSEED':'0'},capture_output=True,text=True,timeout=180)
        assert result.returncode==0,(entry['name'],result.stderr)
        graphs[entry['name']]=v2.read_graph(path)
    return graphs


@pytest.mark.parametrize('name',['returned_alias','ambiguous_return','unresolved_call','recursive_call',
    'index_and_state','different_indices','registration','nested_calls','static_arrays'])
def test_native_source_cases_have_occurrence_level_validation(source_cases,name):
    g=source_cases[name];assert g.graph['schema']==v2.SCHEMA_V22
    assert g.graph['native_correspondence']['status']=='pass'
    v2.validate(g)


def test_two_nested_helper_invocations_do_not_merge_arrays(source_cases):
    g=source_cases['nested_calls']
    for point,array,other in [('L9','L1','L3'),('L19','L3','L1')]:
        rows=[r for r in query(g,point) if r['relation_type']=='memory_access']
        assert rows and any(r['neighbor']==array and len(r['call_chain'])==2 for r in rows)
        assert not [r for r in rows if r['neighbor']==other]


def test_ambiguous_returns_and_unresolved_calls_never_gain_false_proof(source_cases):
    rows=[r for r in query(source_cases['ambiguous_return'],'L9','all') if r['relation_type']=='memory_access']
    assert not [r for r in rows if r['evidence_status']=='proven']
    for name in ('recursive_call','unresolved_call'):
        rows=query(source_cases[name],'L9','all')
        assert any(r['evidence_status']=='possible' and not r['concrete'] for r in rows if r['relation_type']=='memory_access')


def test_returned_aliases_and_global_arrays_keep_distinct_roots(source_cases):
    for point,expected,other in [('L9','L1','L3'),('L19','L3','L1')]:
        rows=[r for r in query(source_cases['returned_alias'],point) if r['relation_type']=='memory_access']
        assert any(r['neighbor']==expected for r in rows) and not any(r['neighbor']==other for r in rows)
    graph=source_cases['static_arrays']
    roots=[v for u,v,e in graph.edges(data=True) if e['relation_type']=='targets' and graph.nodes[u]['action_kind']=='array']
    assert len(set(roots))==2 and all(graph.nodes[r]['root_kind']=='global' for r in roots)


@pytest.fixture(scope='module')
def native_ir_cases(native_tools,tmp_path_factory):
    output=tmp_path_factory.mktemp('v22_native_ir');graphs={}
    programs={
      'pointer':'''module { func.func @top(%x: f32) -> f32 {
        %a = memref.alloca() {mailohls.action_id = "L1"} : memref<8xf32> loc("kernel.cpp":2:11)
        %p = "polygeist.memref2pointer"(%a) : (memref<8xf32>) -> !llvm.ptr
        %i = llvm.mlir.constant(1 : i64) : i64
        %q = llvm.getelementptr %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %x, %q : f32, !llvm.ptr
        %v = llvm.load %q : !llvm.ptr -> f32
        return %v : f32
      } }''',
      'cfg':'''module { func.func @top(%x: f32, %flag: i1) -> f32 {
        %a = memref.alloca() {mailohls.action_id = "L1"} : memref<8xf32> loc("kernel.cpp":2:11)
        %c0 = arith.constant 0 : index
        cf.cond_br %flag, ^yes, ^no
      ^yes:
        memref.store %x, %a[%c0] : memref<8xf32>
        cf.br ^join
      ^no:
        memref.store %x, %a[%c0] : memref<8xf32>
        cf.br ^join
      ^join:
        %v = memref.load %a[%c0] : memref<8xf32>
        return %v : f32
      } }''',
      'memory_state':'''module { func.func @top(%x: f32, %n: index) -> f32 {
        %a = memref.alloca() {mailohls.action_id = "L1"} : memref<8xf32> loc("kernel.cpp":2:11)
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %q = scf.for %i = %c0 to %n step %c1 iter_args(%p = %a) -> memref<8xf32> {
          memref.store %x, %p[%c0] : memref<8xf32>
          scf.yield %p : memref<8xf32>
        } loc("kernel.cpp":4:5)
        %v = memref.load %q[%c0] : memref<8xf32>
        return %v : f32
      } }''',
      'select':'''module { func.func @top(%x: f32, %flag: i1) -> f32 {
        %a = memref.alloca() {mailohls.action_id = "L1"} : memref<8xf32> loc("kernel.cpp":2:11)
        %b = memref.alloca() {mailohls.action_id = "L3"} : memref<8xf32> loc("kernel.cpp":3:11)
        %p = arith.select %flag, %a, %b : memref<8xf32>
        %c0 = arith.constant 0 : index
        memref.store %x, %p[%c0] : memref<8xf32>
        %v = memref.load %p[%c0] : memref<8xf32>
        return %v : f32
      } }'''}
    views=next(r for r in json.loads((audit.ROOT/'experiments/structural_audit/fixtures/semantics_v21.json').read_text()) if r['name']=='strided_views')
    programs['strided_views']=views['mlir']
    programs['memory_zero_trip']=programs['memory_state'].replace(
        '        %c0 =', '        %b = memref.alloca() {mailohls.action_id = "L3"} : memref<8xf32> loc("kernel.cpp":3:11)\n        %c0 =').replace(
        'scf.yield %p','scf.yield %b')
    cell='memref<memref<8xf32>>'
    slot_stores={
        'slot_dominating':'memref.store %a, %slot[] : '+cell,
        'slot_overwritten':'memref.store %a, %slot[] : '+cell+'\nmemref.store %b, %slot[] : '+cell,
        'slot_conditional':'scf.if %flag { memref.store %a, %slot[] : '+cell+' } else { memref.store %b, %slot[] : '+cell+' }',
        'slot_incomplete':'scf.if %flag { memref.store %a, %slot[] : '+cell+' }',
        'slot_helper':'memref.store %a, %slot[] : '+cell+'\nfunc.call @set(%slot, %b) : ('+cell+', memref<8xf32>) -> ()',
        'slot_may_alias':'%q = arith.select %flag, %slot, %external : '+cell+'\nmemref.store %a, %q[] : '+cell,
    }
    for name,stores in slot_stores.items():
        programs[name]='''module {
func.func @set(%s: memref<memref<8xf32>>, %p: memref<8xf32>) {
 memref.store %p, %s[] : memref<memref<8xf32>>
 return
}
func.func @take(%p: memref<8xf32>, %x: f32) {
 %z = arith.constant 0 : index
 memref.store %x, %p[%z] : memref<8xf32>
 return
}
func.func @top(%x: f32, %flag: i1, %external: memref<memref<8xf32>>) -> f32 {
 %a = memref.alloca() {mailohls.action_id="L1"} : memref<8xf32> loc("kernel.cpp":2:11)
 %b = memref.alloca() {mailohls.action_id="L3"} : memref<8xf32> loc("kernel.cpp":3:11)
 %slot = memref.alloca() : memref<memref<8xf32>>
 '''+stores+'''
 %p = memref.load %slot[] : memref<memref<8xf32>>
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %c8 = arith.constant 8 : index
 scf.for %i = %c0 to %c8 step %c1 {
  func.call @take(%p, %x) : (memref<8xf32>, f32) -> ()
 } loc("kernel.cpp":4:5)
 return %x : f32
} }'''
    programs['slot_nested']=programs['slot_conditional'].replace('func.func @top(',
        'func.func @outer(%p:memref<8xf32>,%x:f32) { func.call @take(%p,%x) : (memref<8xf32>,f32)->() return }\nfunc.func @top(').replace(
        'func.call @take(%p, %x)', 'func.call @outer(%p, %x)').replace(
        ' } loc("kernel.cpp":4:5)', '  func.call @outer(%a,%x) : (memref<8xf32>,f32)->()\n } loc("kernel.cpp":4:5)')
    conditional_prefix='''module { func.func @top(%x:f32,%flag:i1,%n:index)->f32 {
 %a=memref.alloca() {mailohls.action_id="L1"}:memref<8xf32> loc("kernel.cpp":2:11)
 %b=memref.alloca() {mailohls.action_id="L3"}:memref<8xf32> loc("kernel.cpp":3:11)
 %z=arith.constant 0:index
 %one=arith.constant 1:index
'''
    expressions={
        'conditional_if':'%p=scf.if %flag -> memref<8xf32> {scf.yield %a:memref<8xf32>} else {scf.yield %b:memref<8xf32>}',
        'conditional_affine':'%p=affine.if affine_set<(d0):(d0>=0)>(%n) -> memref<8xf32> {affine.yield %a:memref<8xf32>} else {affine.yield %b:memref<8xf32>}',
        'conditional_while':'''%p=scf.while (%v=%a):(memref<8xf32>)->memref<8xf32> {
 scf.condition(%flag) %v:memref<8xf32>
 } do { ^bb0(%v:memref<8xf32>): scf.yield %b:memref<8xf32> }''',
        'conditional_for_zero':'%p=scf.for %i=%z to %z step %one iter_args(%v=%a)->memref<8xf32> {scf.yield %b:memref<8xf32>}',
        'conditional_for_one':'%p=scf.for %i=%z to %one step %one iter_args(%v=%a)->memref<8xf32> {scf.yield %b:memref<8xf32>}',
    }
    for name,expression in expressions.items():
        programs[name]=conditional_prefix+expression+'\nmemref.store %x,%p[%z]:memref<8xf32>\nreturn %x:f32 } }'
    programs['evidence_first']='''module {func.func @top(%x:f32) -> f32 {
 %wrong = memref.alloca() {mailohls.action_id="L3"} : memref<8xf32> loc("kernel.cpp":2:11)
 %correct = memref.alloca() {mailohls.action_id="L1"} : memref<8xf32> loc("kernel.cpp":20:11)
 %z = arith.constant 0 : index
 memref.store %x, %correct[%z] : memref<8xf32>
 return %x : f32
} }'''
    programs['overlapping_views']=views['mlir'].replace('    %r =',
        '    %q = memref.subview %a[18] [8] [2] : memref<64xf32> to memref<8xf32, strided<[2], offset: 18>>\n    %r =').replace(
        '      %v = memref.load','      memref.store %x, %q[%i] : memref<8xf32, strided<[2], offset: 18>>\n      %v = memref.load')
    for name,mlir in programs.items():
        directory=output/name;directory.mkdir();source=directory/'kernel.cpp';meta=directory/'kernel_info.txt';raw=directory/'kernel.mlir'
        source.write_text(views['program'] if 'views' in name else 'extern "C" float top(float x){\nL1: float a[8];\nL3: float b[8];\nL9: for(int i=0;i<8;i++)a[i]=x;\nreturn x;\n}\n')
        meta.write_text(views['metadata'] if 'views' in name else 'top\nL1,array,a,1,8\n'+('L3,array,b,1,8\nL9,loop,8\n' if name=='memory_zero_trip' or name.startswith('slot_') else 'L9,loop,8\n' if name=='memory_state' else 'L3,array,b,1,8\n' if name=='select' or name.startswith('conditional_') else ''));raw.write_text(mlir)
        provenance={'native_fixture':{'origin':'handwritten_MLIR_native_verified','mlir_sha256':v2.digest(raw)},
                    'fixture_semantic_expectations':'independently specified; not frontend-generated output'}
        graphs[name]=native_graph(native_tools,source,meta,raw,directory/'graph.json',provenance)
    return graphs


def test_LLVM_pointer_roots_preserve_concrete_accesses(native_ir_cases):
    graph=native_ir_cases['pointer']
    edges=[e for u,v,e in graph.edges(data=True) if e['relation_type']=='memory_access']
    assert len(edges)==2 and all(e['evidence_status']=='proven' for e in edges)
    assert all(e['access_indices']['pointer_expression_chain'][0]['operation_kind']=='llvm.getelementptr' for e in edges)
    assert not [e for _,_,e in graph.edges(data=True) if e['relation_type']=='depends_on']  # Same root isn't a dependence proof.


def test_CFG_successors_use_native_C_API(native_ir_cases):
    graph=native_ir_cases['cfg'];assert graph.graph['native_correspondence']['cfg_successors']==4
    assert sum(e['relation_type']=='control_flow' for _,_,e in graph.edges(data=True))==4


@pytest.mark.parametrize('name',['pointer','cfg','strided_views','overlapping_views','memory_state','memory_zero_trip','select'])
def test_compact_native_evidence_and_queries_are_exact(native_ir_cases,name):
    graph=native_ir_cases[name];compact=v2.compact_view(graph)
    audit.queries_equal(graph,compact);assert action_point_views(graph)==action_point_views(compact)


def test_memory_iteration_and_dynamic_count_do_not_use_metadata_as_proof(native_ir_cases):
    graph=native_ir_cases['memory_state'];loop=get_loop_semantics(graph,'L9')
    assert loop['compiler_proven_trip_count'] is None and loop['metadata_value']==8
    assert len(loop['ssa_state'])==4 and all(e['state_kind']=='memory_reference' for e in loop['ssa_state'])
    assert any(e['forwarding_role']=='loop_zero_trip_result' for e in loop['ssa_state'])


def test_zero_trip_memory_result_keeps_initial_and_yielded_buffers(native_ir_cases):
    g=native_ir_cases['memory_zero_trip']
    result=next(n for n,a in g.nodes(data=True) if a.get('text')=='scf.for')
    value=next(v for _,v,e in g.out_edges(result,data=True) if e['relation_type']=='defines')
    roots=[(v,e['evidence_status']) for _,v,e in g.out_edges(value,data=True) if e['relation_type']=='resolves_to']
    assert len(roots)==2 and all(status=='possible' for _,status in roots)
    assert {v for v,_ in roots}=={v for u,v,e in g.edges(data=True) if e['relation_type']=='targets' and u in ('action:L1','action:L3')}


def test_pointer_select_is_possible_and_view_overlap_is_not_guessed(native_ir_cases):
    graph=native_ir_cases['select']
    accesses=[e for _,_,e in graph.edges(data=True) if e['relation_type']=='memory_access']
    assert accesses and all(e['evidence_status']=='possible' for e in accesses)
    graph=native_ir_cases['overlapping_views']
    views=[e['view_mapping'] for _,_,e in graph.edges(data=True) if e['relation_type']=='references']
    assert {tuple(v['offsets']) for v in views if isinstance(v['offsets'],list)} >= {(16,),(18,),(0,)}
    assert all(v['element_overlap']=='not_established' for v in views)


@pytest.mark.parametrize('change',['operand','call_uid','carrying_loop','certainty','root','reverse'])
def test_corrupted_evidence_is_rejected(source_cases,native_ir_cases,change):
    g=deepcopy(source_cases['index_and_state'] if change in ('carrying_loop','certainty') else source_cases['nested_calls'] if change=='call_uid' else native_ir_cases['pointer'])
    if change=='operand':
        e=next(e for _,_,e in g.edges(data=True) if e['relation_type']=='operand');e['legacy_evidence']['position']=999
    elif change=='call_uid':
        e=next(e for _,_,e in g.edges(data=True) if e['relation_type']=='actual_to_formal');e['call_uid']='wrong'
    elif change=='carrying_loop':
        e=next(e for _,_,e in g.edges(data=True) if e['relation_type']=='depends_on' and e['evidence_status']=='proven');e['carrying_loop']='missing'
    elif change=='certainty':
        e=next(e for _,_,e in g.edges(data=True) if e['relation_type']=='depends_on');e['evidence_status']='possible' if e['evidence_status']=='proven' else 'proven'
    elif change=='root':
        e=next(e for _,_,e in g.edges(data=True) if e['relation_type']=='memory_access');e['evidence_status']='not_established'
    else:next(e for _,_,e in g.edges(data=True))['is_reverse']=True
    with pytest.raises((ValueError,KeyError)):v2.validate(g)


def test_numeric_fidelity_never_claims_stub_hardware_exactness(real_graphs):
    _,g=real_graphs['serrano-kalman-filter'];f=g.graph['numerical_fidelity']
    assert f['hardware_exact'] is False and f['source_template_spellings']
    assert all(r['resolved_width'] is None and not r['hardware_semantics_verified'] for r in f['source_template_spellings'])


def test_frozen_AES_uncertainty_is_distinct_not_duplicate(frozen):
    base,m=frozen;row=next(r for r in m['graphs'] if r['kernel']=='chstone-aes')
    graph=v2.read_graph(base/row['canonical']);stats=audit.statistics(graph)
    assert stats['duplicate_native_queries']==0
    assert stats['dependence_status']['possible']==43236
    assert stats['native_negative_conclusions']>0
    assert stats['native_dependence_reasons']['affine_analysis_failure']==20032


def test_historical_schemas_remain_readable_without_modifying_release(frozen):
    base,m=frozen
    for row in m['graphs'][:3]:
        before=v2.digest(base/row['canonical']);assert v2.read_graph(base/row['canonical']).graph['schema']==v2.SCHEMA_V21
        assert v2.digest(base/row['canonical'])==before


def test_native_view_order_canonicalization_retains_the_exact_multiset():
    original={'views':[{'source':'b','target':'c'},{'source':'a','target':'c'},{'source':'b','target':'c'}],
              'dependences':[{'result':'unknown'}],'aliases':[{'result':'may_alias'}]}
    shuffled={**original,'views':list(reversed(original['views']))}
    assert canonical_analysis(original)==canonical_analysis(shuffled)
    assert len(canonical_analysis(original)['views'])==3
    assert original['views'][0]['source']=='b'  # Input/legacy data is not mutated.
    assert canonical_analysis(original)['dependences']==original['dependences']


@pytest.mark.parametrize('name,expected,complete',[
    ('slot_dominating',{'L1'},True),('slot_overwritten',{'L3'},True),
    ('slot_conditional',{'L1','L3'},True),('slot_incomplete',{'L1'},False),
    ('slot_helper',{'L1'},False),('slot_may_alias',set(),False)])
def test_reaching_pointer_stores_are_control_proven(native_ir_cases,name,expected,complete):
    g=native_ir_cases[name]
    plans=[a['loaded_pointer_evidence'] for _,a in g.nodes(data=True) if a.get('loaded_pointer_evidence') and a['loaded_pointer_evidence']['load_operation'].startswith('op:top:')]
    assert len(plans)==1 and plans[0]['candidate_set_complete']==complete
    rows=[r for r in query(g,'L9','all') if r['relation_type']=='memory_access']
    assert {r['neighbor'] for r in rows}==expected
    if len(expected)>1 or not complete:assert not any(r['evidence_status']=='proven' for r in rows)
    if expected:assert all(r['root_candidates_complete']==complete for r in rows)
    audit.queries_equal(g,v2.compact_view(g))


def test_array_resolution_enumerates_native_evidence_before_legacy_selection(native_ir_cases):
    g=native_ir_cases['evidence_first'];point=g.nodes['action:L1'];proof=point['array_binding_evidence']
    assert proof['tier']=='native_action_attribute'
    assert g.nodes[proof['operation']]['source_line']==20  # Legacy location-first picks the wrong object at line 2.
    assert g.nodes[proof['operation']]['mlir_attrs']['mailohls.action_id']=='"L1"'


def test_possible_nested_calls_preserve_alternatives_and_separate_invocations(native_ir_cases):
    g=native_ir_cases['slot_nested']
    rows=[r for r in query(g,'L9','all') if r['relation_type']=='memory_access' and r.get('concrete')]
    groups={}
    for row in rows:groups.setdefault(row['call_chain'][0],[]).append(row)
    assert len(groups)==2
    alternatives=next(rs for rs in groups.values() if len(rs)==2)
    assert {r['neighbor'] for r in alternatives}=={'L1','L3'}
    assert all(r['evidence_status']=='possible' and r['root_candidates_complete'] for r in alternatives)
    single=next(rs for rs in groups.values() if len(rs)==1)
    assert single[0]['neighbor']=='L1' and single[0]['evidence_status']=='proven'
    assert all(len(r['call_chain'])==2 for r in rows)
    audit.queries_equal(g,v2.compact_view(g))


@pytest.mark.parametrize('name',['conditional_if','conditional_affine','conditional_while','conditional_for_zero','conditional_for_one'])
def test_conditional_memory_results_do_not_invent_unique_roots(native_ir_cases,name):
    g=native_ir_cases[name]
    accesses=[(v,e) for _,v,e in g.edges(data=True) if e['relation_type']=='memory_access']
    assert len(accesses)==2 and all(e['evidence_status']=='possible' for _,e in accesses)
    assert {v for v,_ in accesses}=={v for u,v,e in g.edges(data=True) if e['relation_type']=='targets'}
    # For static for counts this implementation conservatively retains both
    # state paths. It does not equate a sound overapproximation with exact aliasing.
    audit.queries_equal(g,v2.compact_view(g))


def test_loaded_pointer_witness_corruption_is_rejected(native_ir_cases):
    g=deepcopy(native_ir_cases['slot_dominating'])
    value=next(n for n,a in g.nodes(data=True) if a.get('loaded_pointer_evidence',{}).get('candidate_set_complete'))
    g.nodes[value]['loaded_pointer_evidence']['store_operations']=[]
    with pytest.raises(ValueError,match='Loaded-pointer'):
        v2.validate(g)


def test_rc2_native_mode_reproduces_frozen_topology(frozen,native_tools,tmp_path):
    base,cohort=frozen
    row=next(r for r in cohort['graphs'] if r['snapshot_id'].startswith('96b4d608'))
    directory,records=audit.rc2_baseline(tmp_path)
    record=records[row['snapshot_id']]
    reference=v2.read_graph(directory/'canonical'/Path(record['canonical']).name)
    reproduced=native_graph(native_tools,base/row['source'],base/row['metadata'],base/row['raw_mlir'],
        tmp_path/'historical_rc2.json',reference.graph['provenance'],'rc2')
    assert not reproduced.graph.get('semantic_revision')
    assert v2.graph_payload(reference)['nodes']==v2.graph_payload(reproduced)['nodes']
    assert v2.graph_payload(reference)['edges']==v2.graph_payload(reproduced)['edges']
    # Source/code hashes legitimately differ in a reproduction run. Semantics do not.
    for point,a in reference.nodes(data=True):
        if a.get('category')=='action_point':
            old=query(reference,a['action_id'],'all');new=query(reproduced,a['action_id'],'all')
            for rows in (old,new):
                for r in rows:r.pop('source_provenance',None);r.pop('graph_identity',None)
            assert old==new


def test_live_tuple_and_portable_list_have_identical_canonical_JSON(native_ir_cases,tmp_path):
    graph=deepcopy(native_ir_cases['pointer'])
    graph.graph['roundtrip_contract_fixture']={'source_use_lines':(2,4)}
    path=tmp_path/'portable.json';v2.write_graph(graph,path)
    read=v2.read_graph(path)
    assert v2.canonical_json(v2.graph_payload(graph))==v2.canonical_json(v2.graph_payload(read))


def test_resume_seals_initial_git_receipt_but_rejects_changed_code(tmp_path):
    path=tmp_path/'protocol.json'
    original={'git_commit':'initial','worktree_status':'initial local edits',
        'code_sha256':{'builder':'exact'},'source_sha256':'source'}
    audit.seal_protocol(path,original);before=path.read_bytes()
    audit.seal_protocol(path,{**original,'git_commit':'later','worktree_status':'later docs'})
    assert path.read_bytes()==before
    with pytest.raises(ValueError,match='Frozen contract changed'):
        audit.seal_protocol(path,{**original,'code_sha256':{'builder':'changed'}})
    with pytest.raises(ValueError,match='Frozen contract changed'):
        audit.seal_protocol(path,{**original,'source_sha256':'changed'})


@pytest.mark.parametrize('gap',['missing_roots','unresolved_root','incomplete_branch','unknown_status'])
def test_cyclic_reference_forwarding_keeps_incomplete_sets(gap):
    import networkx as nx
    from GNN_branch.action_point_relations import _candidate_set_complete
    g=nx.MultiDiGraph()
    g.add_node('root',root_kind='allocation')
    for value in ('a','b'):
        g.add_node(value)
        g.add_edge(value,'root',relation_type='resolves_to',evidence_status='possible')
    g.add_edge('a','b',relation_type='references',evidence_status='proven')
    g.add_edge('b','a',relation_type='references',evidence_status='proven')
    if gap=='missing_roots':
        for u,v,k,e in list(g.edges(keys=True,data=True)):
            if u=='b' and e['relation_type']=='resolves_to':g.remove_edge(u,v,k)
    elif gap=='unresolved_root':g.nodes['root']['root_kind']='unresolved'
    elif gap=='unknown_status':
        for u,v,e in g.edges(data=True):
            if u=='b' and e['relation_type']=='resolves_to':e['evidence_status']='not_established'
    else:g.nodes['b']['region_root_evidence']={'all_alternatives_resolved':False}
    assert not _candidate_set_complete(g,'a')
    assert not _candidate_set_complete(g,'b')


def test_native_loop_forwarding_does_not_close_incomplete_alternatives(native_tools,tmp_path):
    from GNN_branch.action_point_relations import _candidate_set_complete
    source=tmp_path/'kernel.cpp';meta=tmp_path/'kernel_info.txt';raw=tmp_path/'fixture.mlir'
    source.write_text('extern "C" void top(){\nL1: float a[8];\nL9: for(int i=0;i<2;i++){}\n}\n')
    meta.write_text('top\nL1,array,a,1,8\nL9,loop,2\n')
    raw.write_text('''module {
func.func private @unknown() -> memref<8xf32>
func.func @sink(%p:memref<8xf32>) { %z=arith.constant 0:index %x=memref.load %p[%z]:memref<8xf32> return }
func.func @top(%flag:i1,%n:index) {
 %a=memref.alloca() {mailohls.action_id="L1"}:memref<8xf32> loc("kernel.cpp":2:11)
 %u=func.call @unknown():()->memref<8xf32>
 %partial=arith.select %flag,%a,%u:memref<8xf32>
 %z=arith.constant 0:index
 %one=arith.constant 1:index
 %forwarded=scf.for %i=%z to %n step %one iter_args(%p=%partial)->memref<8xf32> {
   scf.yield %p:memref<8xf32>
 }
 %two=arith.constant 2:index
 scf.for %j=%z to %two step %one {
   func.call @sink(%forwarded):(memref<8xf32>)->()
 } loc("kernel.cpp":3:1)
 return
} }''')
    g=native_graph(native_tools,source,meta,raw,tmp_path/'graph.json',
        {'native_fixture':{'origin':'handwritten_MLIR_native_verified','mlir_sha256':v2.digest(raw)}})
    partial=next(n for n,a in g.nodes(data=True) if a.get('region_root_evidence'))
    assert not _candidate_set_complete(g,partial)
    loop=next(n for n,a in g.nodes(data=True) if a.get('text')=='scf.for')
    forwarded=next(v for _,v,e in g.out_edges(loop,data=True) if e['relation_type']=='defines')
    assert not _candidate_set_complete(g,forwarded)
    rows=[r for r in query(g,'L9','all') if r['relation_type']=='memory_access' and r.get('concrete')]
    assert rows and all(r['evidence_status']=='possible' and not r['root_candidates_complete'] for r in rows)
    audit.queries_equal(g,v2.compact_view(g))



def test_review_worksheet_preserves_disposition_and_rejects_changed_reviewed_evidence(tmp_path):
    from experiments.structural_audit.review_v22 import adjudication
    path=tmp_path/'worksheet.json'
    case={'case_id':'test_only','expected_semantic_relationship':'allocation A',
        'reviewer_disposition':'pending','reviewer_identity':None,'review_date':None,
        'supporting_evidence':[],'reviewer_notes':None}
    adjudication(path,[deepcopy(case)],[],'archive')
    record=json.loads(path.read_text());reviewed=record['cases'][0]
    reviewed.update(reviewer_disposition='agree',reviewer_identity='synthetic_test_identity_not_human_review',
        review_date='2000-01-01',supporting_evidence=['test fixture only'])
    path.write_text(json.dumps(record))
    adjudication(path,[deepcopy(case)],[],'archive')
    assert json.loads(path.read_text())['cases'][0]==reviewed
    with pytest.raises(ValueError,match='Reviewed evidence changed'):
        adjudication(path,[{**case,'expected_semantic_relationship':'different storage'}],[],'archive')
