"""Read-only RC3 release verification and bounded, unadjudicated references.

No graph construction or compiler analysis is replaced here. Native mode reuses
our occurrence checker and exports actual MLIR text for selected review cases.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile

from GNN_branch import mlir_graph_v2 as v
from GNN_branch.action_point_relations import get_action_point_relations as query, action_point_views, get_loop_semantics

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'experiments/structural_audit/results/v22/final_review'
REVIEW_FIELDS=('reviewer_disposition','reviewer_identity','review_date','supporting_evidence','reviewer_notes')


def save(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')


def adjudication(path,cases,snapshots,release):
    """Never erase an independent review or retain it across changed evidence."""
    old={c['case_id']:c for c in json.loads(Path(path).read_text())['cases']} if Path(path).exists() else {}
    for case in cases:
        prior=old.get(case['case_id'])
        if prior:
            before={k:x for k,x in prior.items() if k not in REVIEW_FIELDS}
            after={k:x for k,x in case.items() if k not in REVIEW_FIELDS}
            if before!=after and prior['reviewer_disposition']!='pending':raise ValueError('Reviewed evidence changed: '+case['case_id'])
            if before==after:
                for key in REVIEW_FIELDS:case[key]=prior[key]
    save(path,{'schema':'salt-hls-v22-independent-review-v1','release_archive_sha256':release,
        'scope':'bounded machine-prepared references, not ground truth or human review',
        'disposition_values':['pending','agree','disagree','unresolved'],
        'review_instructions':'Enter identity, ISO date and source/native evidence; a changed disposition alone is not independent validation.',
        'snapshots':[{**{k:v for k,v in r.items() if k not in ('action_registry','inherited_native_receipt')},
            'action_registry_reference':'canonical/'+r['kernel']+'-'+r['snapshot_id'][:12]+'.json#/metadata/action_point_coverage',
            'inherited_native_receipt_summary':{k:v for k,v in r['inherited_native_receipt'].items() if k not in ('compiler_module_attributes',)},
            'inherited_native_receipt_reference':'validation.json#/'+r['snapshot_id']+'/native/native_correspondence'} for r in snapshots],
        'cases':cases})


def unpack(archive,destination,expected):
    if v.digest(archive)!=expected:raise ValueError('Release archive hash mismatch')
    if not destination.exists():
        with tarfile.open(archive) as bundle:
            if any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in bundle):raise ValueError('Unsafe archive')
            bundle.extractall(destination)
    release=json.loads((destination/'release.json').read_text())
    for name,digest in release['files_sha256'].items():
        if v.digest(destination/name)!=digest:raise ValueError('Bundled artifact mismatch: '+name)
    return release


def frozen_query(base):
    spec=importlib.util.spec_from_file_location('rc3_original_queries',base/'code/GNN_branch/action_point_relations.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def native(args):
    """Independent live MLIR walk; no builder or analyze_module invocation."""
    from GNN_branch.mlir_graph_gen import parse_mlir_module
    from GNN_branch.mlir_semantics_v22 import native_correspondence
    from mlir import ir
    context,module,_=parse_mlir_module(Path(args.raw),True)
    try:
        assert module.operation.verify()
        graph=v.read_graph(args.graph);receipt=native_correspondence(module,graph)
        operations={}
        def visit_function(function):
            symbol=ir.StringAttr(function.attributes['sym_name']).value;ordinal=0
            def walk(owner):
                nonlocal ordinal
                for region in owner.regions:
                    for block in region.blocks:
                        for child in block.operations:
                            op=child.operation;key=f'op:{symbol}:op{ordinal}';ordinal+=1
                            operations[key]={'kind':op.name,'raw_operation':str(op).splitlines()[0],
                                'native_location':str(op.location),'operands':[{'position':i,'type':str(value.type),
                                'native_spelling':str(value).splitlines()[0]} for i,value in enumerate(op.operands)],
                                'results':[str(result.type) for result in op.results]}
                            walk(op)
            walk(function)
        for child in module.body.operations:
            op=child.operation
            if op.name in ('func.func','llvm.func'):visit_function(op)
        save(args.native_output,{'raw_mlir_sha256':v.digest(args.raw),'canonical_sha256':v.digest(args.graph),
            'oracle':'live native MLIR object walk; original alias/dependence outcomes inherited, not rerun',
            'native_correspondence':receipt,'operations':operations})
    finally:
        visit_function=op=child=module=None
        gc.collect()
        context.__exit__(None,None,None)


def review(args):
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    public=json.loads((ROOT/'docs/graph-release-v22-rc3.json').read_text())
    base=output/'rc3_archive';release=unpack(ROOT/public['archive'],base,public['archive_sha256'])
    v21=json.loads((ROOT/'docs/graph-release-v21.json').read_text())
    assert v.digest(ROOT/v21['archive'])==v21['archive_sha256']
    with tarfile.open(ROOT/v21['archive']) as bundle:cohort=json.load(bundle.extractfile('release.json'))
    records=json.loads((base/'validation.json').read_text());assert set(records)=={r['snapshot_id'] for r in cohort['graphs']} and len(records)==63
    frozen=frozen_query(base);graphs={};snapshots=[];cases=[];native_tables={};failures=[]
    selected={'GAN','rodinia_lud_1_tiling_0','rodinia-hotspot-2-pipeline','rodinia-hotspot-3-unroll','serrano-kalman-filter','covariance','chstone-aes'}
    protocol=json.loads((base/'protocol.json').read_text());tools=protocol['toolchain']
    rows={r['snapshot_id']:r for r in cohort['graphs']}
    def load(ident):
        if ident not in graphs:graphs[ident]=v.read_graph(base/'canonical'/Path(records[ident]['canonical']).name)
        return graphs[ident]
    for row in cohort['graphs']:
        ident=row['snapshot_id'];record=records[ident]
        for name in (row['source'],row['metadata'],row['raw_mlir'],*row['headers']):
            assert v.digest(base/name)==cohort['files_sha256'][name],name
        try:
            g=load(ident);c=v.read_graph(base/'compact'/Path(record['compact']).name)
            assert g.graph['schema']==v.SCHEMA_V22 and g.graph['semantic_revision']=='rc3'
            v.validate(g);v.validate(c);assert v.graph_payload(g)==v.graph_payload(v.expand_compact(c))
            original=v.graph_payload(g);before_cache={};after_cache={};compact_cache={}
            for _,a in g.nodes(data=True):
                if a.get('category')!='action_point':continue
                point=a['action_id'];current=query(g,point,'all',_access_cache=after_cache)
                assert current==query(c,point,'all',_access_cache=compact_cache),point
                assert current==frozen.get_action_point_relations(g,point,'all',_access_cache=before_cache),point
                if a['action_kind']=='loop':assert get_loop_semantics(g,point)==get_loop_semantics(c,point)
            assert action_point_views(g)==action_point_views(c)
            assert original==v.graph_payload(g),'Queries mutated canonical facts'
            coverage=g.graph['action_point_coverage']['records']
            snapshots.append({'snapshot_id':ident,'kernel':row['kernel'],'source':row['source'],'source_sha256':row['source_sha256'],
                'metadata':row['metadata'],'metadata_sha256':row['metadata_sha256'],'raw_mlir':row['raw_mlir'],
                'raw_mlir_sha256':v.digest(base/row['raw_mlir']),'compiler':release['toolchain'],
                'registered':sum(a['metadata_registered'] for a in coverage),'matched':sum(a['compiler_matched'] for a in coverage),
                'action_registry':coverage,'verification_this_run':'portable validity, exact inputs, reconstruction and current/archived public queries agree',
                'inherited_native_receipt':record['native']['native_correspondence'],
                'inherited_two_interpreter_result':record['native']['repeat_native']})
            if row['kernel'] in selected and args.native_python:
                path=output/'native'/f'{ident}.json';path.parent.mkdir(exist_ok=True)
                command=[args.native_python,'-B','-m','experiments.structural_audit.review_v22','--mode','native',
                    '--raw',str(base/row['raw_mlir']),'--graph',str(base/'canonical'/Path(record['canonical']).name),'--native-output',str(path)]
                run=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':tools['pythonpath']},capture_output=True,text=True,timeout=180)
                if run.returncode:raise ValueError('Native occurrence review: '+run.stderr[-2500:])
                native_tables[ident]=json.loads(path.read_text())
            print(row['kernel'],'PASS',flush=True)
        except (AssertionError,ValueError,KeyError) as exc:failures.append({'snapshot_id':ident,'kernel':row['kernel'],'error':str(exc)})
    assert not failures,failures
    save(output/'snapshot_verification.json',snapshots)

    def case(ident,name,expected,justification,actual,operations=(),quotes=(),comparison='unresolved'):
        row=rows[ident];g=load(ident);source=(base/row['source']).read_text().splitlines()
        source_evidence=[{'file':row['source'],'line':i+1,'source_sha256':row['source_sha256'],'expression':line.strip()}
            for quote in quotes for i,line in enumerate(source) if quote in line]
        witnesses=[]
        for operation in sorted(set(operations))[:8]:
            if operation not in g:continue
            a=g.nodes[operation];line=a.get('source_line');filename=a.get('source_file')
            candidate=base/row['source'] if filename==Path(row['source']).name else (base/row['source']).parent/str(filename)
            if line and candidate.is_file():
                lines=candidate.read_text(errors='replace').splitlines()
                if line<=len(lines):source_evidence.append({'file':str(candidate.relative_to(base)),'line':line,'source_sha256':v.digest(candidate),'expression':lines[line-1].strip()})
            witnesses.append({'operation':operation,'native':native_tables.get(ident,{}).get('operations',{}).get(operation),
                'compiler_recorded_text':a.get('full_text'),'source_location':a.get('source_location'),
                'operand_positions':[{'value':u,'position':e['legacy_evidence']['position']} for u,_,e in g.in_edges(operation,data=True) if e['relation_type']=='operand']})
        cases.append({'case_id':ident[:12]+':'+name,'snapshot_id':ident,'kernel':row['kernel'],
            'source_evidence':source_evidence,'snapshot_source_sha256':row['source_sha256'],
            'raw_mlir':row['raw_mlir'],'raw_mlir_sha256':v.digest(base/row['raw_mlir']),
            'compiler_identity':release['toolchain'],'native_operations':witnesses,
            'expected_semantic_relationship':expected,'independent_justification':justification,
            'actual_v22':actual,'machine_comparison':comparison,
            'evidence_level':'machine-prepared source/native references and handwritten expectations; no human adjudication',
            'evidence_needed_to_close':'Reviewer must inspect cited source/IR, operands and context; verify or reject the expected conclusion and record supporting evidence. Native dependence truth is not established by its recorded status alone.',
            'reviewer_disposition':'pending','reviewer_identity':None,'review_date':None,'supporting_evidence':[],'reviewer_notes':None})
    annotations=json.loads((ROOT/'experiments/structural_audit/fixtures/annotations.json').read_text())
    for annotation in annotations:
        if annotation['kernel'] not in ('GAN','Kalman','covariance','rodinia_lud_1_tiling_0'):continue
        if annotation['kernel']=='Kalman' and annotation['action_i']=='L7':continue
        row=next(r for r in cohort['graphs'] if r['kernel']==('serrano-kalman-filter' if annotation['kernel']=='Kalman' else annotation['kernel'])
            and (annotation['kernel']!='Kalman' or any(a['origin']=='development' for a in r['aliases'])))
        ident=row['snapshot_id'];g=load(ident);typ={'loop_array_access':'memory_access','nearest_child_loop':'loop_hierarchy','dependence_forward':'memory_dependence'}[annotation['relationship_type']]
        facts=[r for r in query(g,annotation['action_i'],'all') if r['neighbor']==annotation['action_j'] and r['relation_type']==typ
            and (typ!='loop_hierarchy' or r['hierarchy']=='child')]
        proven=any(r['evidence_status']=='proven' for r in facts);expected=annotation['expected']
        operations=[r.get('source_operation') for r in facts if r.get('source_operation')]
        for label in (annotation['action_i'],annotation['action_j']):
            if 'action:'+label in g:operations += [t for _,t,e in g.out_edges('action:'+label,data=True) if e['relation_type']=='targets' and g.nodes[t]['category']=='operation']
        case(ident,annotation['relationship_type']+':'+annotation['action_i']+':'+annotation['action_j'],
            {'relationship_present':expected,'scope':'only the specified direct access/nearest hierarchy/dependence, not program independence'},
            {'origin':'pre-existing machine-assisted draft source expectation, not ground truth','reason':annotation['reason']},
            {'matching_count':len(facts),'examples':facts[:3]},operations,annotation['source_quotes'],
            'unresolved' if expected is None else 'diagnostic_agreement' if proven==expected else 'diagnostic_disagreement_requires_review')
        if annotation['kernel']=='rodinia_lud_1_tiling_0' and annotation['action_i']=='L5' and annotation['action_j']=='L2':
            empty=[r for r in query(g,'L2',call_context=()) if r['neighbor']=='L5']
            case(ident,'helper_empty_context_negative','No caller-specific L5 access without an invocation context',
                'The helper formal is symbolic. Source buffer binding requires the exact lud_diagonal call, not all helper callers.',empty,operations,annotation['source_quotes'],'diagnostic_agreement' if not empty else 'diagnostic_disagreement_requires_review')
    for b in json.loads((base/'details/action_binding_audit.json').read_text()):
        case(b['snapshot_id'],'registration:'+b['action_id'],'Registered action remains unresolved; compiler elimination is not proven',
            'Exact annotation/declaration and a unique storage witness are required; shape and shared-use lines cannot identify storage.',b,
            [x['operation'] for x in b['compiler_objects']],(b['source_declaration'].strip(),),'unresolved')
    for p in json.loads((base/'details/pointer_root_audit.json').read_text()):
        case(p['snapshot_id'],'loaded_pointer:'+p['operation'],'Concrete access retained, loaded storage root not established',
            'A formal pointer-containing cell does not establish contents; complete invocation-aware reaching stores are absent.',p,
            [p['operation']]+[d['operation'] for d in p['definition_chain']],comparison='unresolved')
    aes=next(r for r in cohort['graphs'] if r['kernel']=='chstone-aes');g=load(aes['snapshot_id'])
    analysis=g.graph['compiler_analysis']
    for category in ('proven_affine_dependence','aliased_storage_not_canonicalized','affine_analysis_failure','non_affine_access'):
        candidates=[d for d in analysis['dependences'] if d['reason']==category]
        d=min(candidates,key=lambda d:json.dumps(d,sort_keys=True))
        operations=['op:'+d[k]['function']+':op'+str(d[k]['ordinal']) for k in ('source','target')]
        case(aes['snapshot_id'],'dependence:'+category,
            'Preserve source-to-target RAW/WAR/WAW direction; unknown/failure is not no-dependence',
            'Source operations and native analysis scope must be checked independently. First-successful-depth evidence is not exhaustive over depths.',d,operations,
            comparison='unresolved' if d['result']!='dependence' else 'native_record_agreement_not_independent_dependence_proof')
    for kind in ('must_alias','may_alias','no_alias'):
        d=min((d for d in analysis['aliases'] if d['result']==kind),key=lambda d:json.dumps(d,sort_keys=True))
        operations=[op for value in (d['lhs'],d['rhs']) for op,_,e in g.in_edges('value:'+value,data=True) if e['relation_type']=='defines']
        case(aes['snapshot_id'],'alias:'+kind,kind+' retained separately from storage-root identity; no_alias is negative evidence',
            'Inspect defining memrefs/views and compiler alias witnesses; matching root/type alone is insufficient.',d,operations,comparison='unresolved')
    cfg=next((u,t,e) for u,t,e in g.edges(data=True) if e['relation_type']=='control_flow')
    case(aes['snapshot_id'],'cfg_successor','Branch successor and direction match native CFG',
        'Read the branch label in raw MLIR; the native C API successor walk is separate from graph construction.',cfg,[cfg[0],cfg[1]],comparison='native_occurrence_agreement')
    gan=next(r for r in cohort['graphs'] if r['kernel']=='GAN');graph=load(gan['snapshot_id'])
    operation=next(n for n,a in sorted(graph.nodes(data=True)) if a.get('text')=='arith.mulf')
    case(gan['snapshot_id'],'ssa_def_use','Binary floating multiplication has two positioned uses and one defined result; no hardware-exact arithmetic claim',
        'Inspect the actual native MLIR binary operation syntax and its operands; native occurrence traversal checks individual uses/results rather than counts only.',
        {'operand_edges':[{'value':u,'position':e['legacy_evidence']['position']} for u,_,e in graph.in_edges(operation,data=True) if e['relation_type']=='operand'],
         'result_edges':[{'value':t,'position':e['legacy_evidence']['position']} for _,t,e in graph.out_edges(operation,data=True) if e['relation_type']=='defines']},[operation],comparison='native_occurrence_agreement')
    cov=next(r for r in cohort['graphs'] if r['kernel']=='covariance');g=load(cov['snapshot_id'])
    loop=get_loop_semantics(g,'L6');case(cov['snapshot_id'],'dynamic_trip_count','Compiler exact count remains unknown; metadata is not proof',
        'The triangular inner bound depends on an outer induction value.',
        {**{k:value for k,value in loop.items() if k not in ('accesses','ssa_state','memory_dependences')},
         'access_count':len(loop['accesses']),'ssa_state_count':len(loop['ssa_state']),
         'memory_dependence_count':len(loop['memory_dependences']),'memory_dependence_examples':loop['memory_dependences'][:2]},[t for _,t,e in g.out_edges('action:L6',data=True) if e['relation_type']=='targets'],('L6:',),'diagnostic_agreement' if loop['compiler_proven_trip_count'] is None else 'diagnostic_disagreement_requires_review')
    if args.fixture_root:
        fixture_root=Path(args.fixture_root).resolve()
        expected_fixtures={
            'strided_views':'Subview offsets are relative to the source view; reinterpret_cast offsets use underlying storage. Neither proves element overlap.',
            'overlapping_views':'Same allocation and overlapping-looking offsets do not invent an element-level dependence proof.',
            'conditional_if':'Both branch return roots are possible; complete alternatives do not make either uniquely proven.',
            'slot_incomplete':'A store in only one branch does not close the loaded-pointer candidate set.',
            'memory_state':'Loop init/backedge/result forwarding is SSA state, distinct from a proven memory dependence.',
        }
        paths=[(name,fixture_root/name,expected) for name,expected in expected_fixtures.items()]
        cycles=sorted(fixture_root.parent.glob('test_native_loop_forwarding_do[0-9]*'))
        if not cycles:raise ValueError('Fresh native loop-forwarding regression artifact is required')
        paths.append(('incomplete_loop_forwarding',cycles[0],
            'An incomplete select input stays incomplete across the loop SSA cycle and at the exact helper invocation.'))
        for name,directory,expected in paths:
            raw=directory/('fixture.mlir' if name=='incomplete_loop_forwarding' else 'kernel.mlir')
            source=directory/'kernel.cpp';meta=directory/'kernel_info.txt';graph=v.read_graph(directory/'graph.json')
            assert graph.graph['provenance']['mlir_sha256']==v.digest(raw)
            assert graph.graph['provenance']['source_sha256']==v.digest(source)
            assert graph.graph['native_correspondence']['status']=='pass'
            actual={
                'views':[{'value':u,'base':t,'mapping':e['view_mapping']} for u,t,e in graph.edges(data=True) if e['relation_type']=='references'],
                'conditional_or_pointer_evidence':[{k:a[k] for k in ('region_root_evidence','loaded_pointer_evidence') if k in a}
                    for _,a in graph.nodes(data=True) if any(k in a for k in ('region_root_evidence','loaded_pointer_evidence'))],
                'native_occurrence_receipt':graph.graph['native_correspondence'],
                'root_edges':[{'value':u,'memory_object':t,'root_kind':graph.nodes[t]['root_kind'],'certainty':e['evidence_status']} for u,t,e in graph.edges(data=True) if e['relation_type']=='resolves_to'],
                'loop_state':[{'source':u,'destination':t,**e} for u,t,e in graph.edges(data=True) if e['relation_type']=='loop_carried'],
                'current_queries':[r for _,a in graph.nodes(data=True) if a['category']=='action_point'
                    for r in query(graph,a['action_id'],'all')][:6]}
            if name=='incomplete_loop_forwarding':
                actual['archived_queries']=[r for _,a in graph.nodes(data=True) if a['category']=='action_point'
                    for r in frozen.get_action_point_relations(graph,a['action_id'],'all')][:6]
            cases.append({'case_id':'native_fixture:'+name,'kernel':'adversarial_native_fixture','snapshot_id':None,
                'source_evidence':[{'file':'kernel.cpp','source_sha256':v.digest(source),'text':source.read_text(),
                    'role':'C++ annotation stub only; not claimed as frontend translation of this handwritten MLIR'}],
                'raw_mlir':{'sha256':v.digest(raw),'text':raw.read_text()},'metadata':{'sha256':v.digest(meta),'text':meta.read_text()},
                'compiler_identity':release['toolchain'],'expected_semantic_relationship':expected,
                'independent_justification':'Independently specified handwritten MLIR expectations, checked by native verification and regression assertions; not human review.',
                'actual_v22':actual,'machine_comparison':'native_regression_agreement; archived completeness differs in the demonstrated loop case',
                'evidence_level':'handwritten MLIR verified through native object model in this run',
                'evidence_needed_to_close':'Reviewer must inspect full embedded IR, initial/yield/branch paths, view semantics and query certainty.',
                'reviewer_disposition':'pending','reviewer_identity':None,'review_date':None,'supporting_evidence':[],'reviewer_notes':None})
    recovery=[]
    bundle=json.loads((base/'V22_RECOVERY_INPUTS.json').read_text())
    for name,data in bundle['files'].items():
        if name.startswith('historical_'):continue
        current=ROOT/name
        assert current.is_file() and v.digest(current)==data['sha256'],name
        original=subprocess.run(['git','show','4244ef7:'+name],cwd=ROOT,capture_output=True)
        recovery.append({'path':name,'analysis_variant_sha256':data['sha256'],
            'original_reference':'4244ef7:'+name,'original_sha256':hashlib.sha256(original.stdout).hexdigest() if original.returncode==0 else None,
            'original_status':'recoverable immutable Git blob' if original.returncode==0 else 'not present before RC3 (analysis-only added header)',
            'in_frozen_cohort':False,'vendor_HLS_equivalence':'not_established',
            'functional_equivalence':'JPEG inherited differential tests plus current regression only; otherwise not established'})
    save(output/'recovery_provenance.json',{'variants':recovery,
        'selection_guard':'GEMV source override is explicit, original SHA retained, metadata/source labels validated; full source/header/meta hashes authenticate graph reuse.',
        'HLS_guard':'experiments/refinement/synthesis.py materialization excludes local AP headers and requires verified vendor headers; direct unmanaged compilation remains outside this guarantee.',
        'excluded_from_63':['chstone-jpeg','rosetta-3d-rendering','gemv']})
    assert len(cases)<=80,'Bounded review package grew unexpectedly'
    adjudication(args.worksheet,cases,snapshots,public['archive_sha256'])
    save(output/'verification.json',{'release_sha256':public['archive_sha256'],'archive_files_authenticated':len(release['files_sha256']),
        'snapshots_verified_this_run':len(snapshots),'live_native_occurrence_walks_this_run':len(native_tables),
        'new_corpus_generations':0,'query_differences_on_frozen_63':0,'canonical_graph_mutations':0,
        'inherited_regressions':json.loads((base/'test_receipt.json').read_text()),
        'human_review_performed':False,'cases':len(cases),'review_tool_sha256':v.digest(__file__),'query_patch_sha256':v.digest(ROOT/'GNN_branch/action_point_relations.py')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('review','native'),default='review')
    p.add_argument('--output-dir',default=str(DEFAULT));p.add_argument('--worksheet',default=str(ROOT/'docs/v22-independent-adjudication.json'))
    p.add_argument('--native-python',default=str(Path.home()/'.mlir-python311/bin/python'))
    p.add_argument('--fixture-root',help='Native fixture directory produced by this run of test_mlir_graph_v22.py')
    p.add_argument('--raw');p.add_argument('--graph');p.add_argument('--native-output');args=p.parse_args()
    native(args) if args.mode=='native' else review(args)

if __name__=='__main__':main()
