"""Authenticate the frozen 63-program cohort, regenerate V2.2 and audit evidence.

The driver delegates graph construction, semantic queries and compaction. No
target, QoR, synthesis or training inputs participate in the representation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import sys
import tempfile

from GNN_branch import mlir_graph_v2 as v2
from GNN_branch.action_point_relations import action_point_views, get_loop_semantics, get_action_point_relations
from GNN_branch.mlir_semantics_v22 import fingerprint, canonical_analysis

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'GNN_branch/MLIR_graphs/v22_corpus/validated_rc3_release'
ARCHIVE=ROOT/'GNN_branch/MLIR_graphs/salt-hls-graphs-v21-20261027.tar.gz'
CODE=('GNN_branch/mlir_graph_gen.py','GNN_branch/mlir_graph_v2.py','GNN_branch/mlir_semantics.py',
      'GNN_branch/mlir_semantics_v22.py','GNN_branch/action_point_relations.py',
      'experiments/structural_audit/finalize_v22.py')


def save(path,data,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,sort_keys=True,indent=2)+'\n'
    if immutable and path.exists() and path.read_text()!=text:raise ValueError('Frozen contract changed: '+str(path))
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(text);temporary.replace(path)


def seal_protocol(path,contract):
    """Keep initial Git receipts; authenticate actual inputs/code on resume."""
    contract=dict(contract)
    if Path(path).exists():
        original=json.loads(Path(path).read_text())
        for key in ('git_commit','worktree_status'):contract[key]=original[key]
    save(path,contract,True)


def baseline(output):
    """The published archive, not the changing working-tree source inventory."""
    release=json.loads((ROOT/'docs/graph-release-v21.json').read_text())
    if v2.digest(ARCHIVE)!=release['archive_sha256']:raise ValueError('Frozen V2.1 archive hash mismatch')
    directory=Path(output)/'baseline'
    if not directory.exists():
        with tarfile.open(ARCHIVE,'r:gz') as bundle:
            if any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in bundle):
                raise ValueError('Unsafe baseline archive member')
            bundle.extractall(directory)
    if v2.digest(directory/'release.json')!=release['content_manifest_sha256']:raise ValueError('Baseline index mismatch')
    content=json.loads((directory/'release.json').read_text())
    for name,digest in content['files_sha256'].items():
        if v2.digest(directory/name)!=digest:raise ValueError('Baseline artifact changed: '+name)
    if len(content['graphs'])!=63 or len({r['snapshot_id'] for r in content['graphs']})!=63:
        raise ValueError('Exactly 63 distinct original snapshots are required')
    return directory,content


def rc2_baseline(output):
    """Authenticate the preserved RC2 release independently of local caches."""
    contract=json.loads((ROOT/'docs/graph-release-v22.json').read_text())
    archive=ROOT/contract['archive']
    if v2.digest(archive)!=contract['archive_sha256']:raise ValueError('RC2 archive changed')
    directory=Path(output)/'rc2_reference'
    if not directory.exists():
        with tarfile.open(archive,'r:gz') as bundle:
            if any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in bundle):
                raise ValueError('Unsafe RC2 archive')
            bundle.extractall(directory)
    content=json.loads((directory/'release.json').read_text())
    for name,digest in contract['files_sha256'].items():
        if v2.digest(directory/name)!=digest:raise ValueError('RC2 input changed: '+name)
    records=json.loads((directory/'validation.json').read_text())
    if len(records)!=63:raise ValueError('Incomplete RC2 reference')
    return directory,records


def statistics(graph):
    edges=list(graph.edges(data=True));analysis=graph.graph['compiler_analysis'];deps=analysis['dependences']
    coverage=graph.graph['action_point_coverage']['records']
    counts=dict(nodes=len(graph),edges=len(edges),registered=sum(r['metadata_registered'] for r in coverage),
        active=sum(r['compiler_matched'] for r in coverage),
        unresolved_registrations=sum(r['metadata_registered'] and not r['compiler_matched'] for r in coverage),
        source_only=sum(r['source_labeled'] and not r['metadata_registered'] for r in coverage),
        access_status=dict(Counter(e['evidence_status'] for _,_,e in edges if e['relation_type']=='memory_access')),
        dependence_status=dict(Counter(e['evidence_status'] for _,_,e in edges if e['relation_type']=='depends_on')),
        dependence_reasons=dict(Counter(r['reason'] for r in deps if r.get('fallback_required'))),
        native_dependence_outcomes=dict(Counter(r['result'] for r in deps)),
        native_dependence_reasons=dict(Counter(r['reason'] for r in deps)),
        native_negative_conclusions=sum(r['result']=='no_dependence' or r['reason']=='proven_no_alias' for r in deps),
        duplicate_native_queries=len(deps)-len({fingerprint(d) for d in deps}),
        distinct_access_pairs=len({(fingerprint(d['source']),fingerprint(d['target']),d['kind']) for d in deps}),
        alias_kinds=dict(Counter(e['alias_kind'] for _,_,e in edges if e['relation_type']=='alias')),
        relation_types=dict(Counter(e['relation_type'] for _,_,e in edges)),
        cfg_successors=sum(e['relation_type']=='control_flow' and e.get('legacy_evidence',{}).get('role')=='block_successor' for _,_,e in edges))
    return counts


def details(graph):
    """Comprehensive classifications plus a predeclared deterministic review sample."""
    roots=defaultdict(list)
    for u,v,e in graph.edges(data=True):
        if e['relation_type']=='resolves_to':roots[u].append({'root':v,'status':e['evidence_status']})
    accesses=[dict(operation=u,operation_kind=graph.nodes[u]['text'],operand=e['memory_operand'],
        access=e['access'],root=v,status=e['evidence_status'],reason=graph.nodes[e['memory_operand']].get('root_resolution_diagnostic'),
        pointer_chain=e.get('access_indices',{}).get('pointer_expression_chain'),source_analysis=e['source_analysis'])
        for u,v,e in graph.edges(data=True) if e['relation_type']=='memory_access']
    access_roots=defaultdict(set)
    for a in accesses:access_roots[a['operation']].add(a['root'])
    dep_groups=Counter();strata=defaultdict(list)
    for index,dep in enumerate(graph.graph['compiler_analysis']['dependences']):
        source='op:'+dep['source']['function']+':op'+str(dep['source']['ordinal'])
        target='op:'+dep['target']['function']+':op'+str(dep['target']['ordinal'])
        scope=tuple(graph.nodes[source].get('loop_context',[]))
        group=(dep['result'],dep['reason'],dep['kind'],tuple(sorted(access_roots[source])),
               tuple(sorted(access_roots[target])),scope)
        dep_groups[group]+=1
        stratum=(dep['result'],dep['reason'],dep['kind'])
        strata[stratum].append((fingerprint(dep),index,dep))
    samples=[dict(stratum=list(key),native_query_index=i,native_record=d,review_status='machine_assisted_draft_not_independently_adjudicated')
        for key,items in sorted(strata.items()) for _,i,d in sorted(items)[:2]]
    invocations=[];cache={}
    for _,point in graph.nodes(data=True):
        if point['category']!='action_point' or point['action_kind']!='loop':continue
        for row in get_action_point_relations(graph,point['action_id'],'all',_access_cache=cache):
            if row['relation_type']=='memory_access' and row['access_path']=='through_call':
                invocations.append({k:row.get(k) for k in ('action_point','neighbor','source_operation','access',
                    'evidence_status','memory_root','call_chain','callsite_bindings','invocation_context','concrete','root_candidates_complete')})
    return {'accesses':accesses,'dependence_groups':[{'result':k[0],'reason':k[1],'kind':k[2],
        'source_roots':k[3],'target_roots':k[4],'source_loop_scope':k[5],'count':v} for k,v in sorted(dep_groups.items())],
        'deterministic_stratified_review':samples,'stratified_review_rule':'two smallest SHA-256 native query fingerprints per (result, reason, RAW/WAR/WAW); no QoR',
        'dependence_limit':'Native first-successful-depth analysis; not exhaustive all-depth dependence proof',
        'interprocedural_accesses':invocations,
        'unresolved_registrations':[r for r in graph.graph['action_point_coverage']['records'] if r['metadata_registered'] and not r['compiler_matched']]}


def worker(args):
    directory=Path(args.baseline_root);content=json.loads((directory/'release.json').read_text())
    row=next(r for r in content['graphs'] if r['snapshot_id']==args.snapshot_id)
    for name in (row['source'],row['metadata'],row['raw_mlir'],row['canonical']):
        if v2.digest(directory/name)!=content['files_sha256'][name]:raise ValueError('Input identity mismatch: '+name)
    old=v2.read_graph(directory/row['canonical'])
    provenance={**old.graph['provenance'],'parent_v21_canonical_sha256':v2.digest(directory/row['canonical']),
        'frozen_release_sha256':v2.digest(directory/'release.json'),
        'generation_mode':'frozen_verified_raw_MLIR; original frontend is not rerun'}
    from mlir._mlir_libs import _mlir
    provenance['mlir_python_capi_sha256']=v2.digest(Path(_mlir.__file__).parent/'libMLIRPythonCAPI.so')
    graph,timing=v2.generate(directory/row['source'],directory/row['metadata'],directory/row['raw_mlir'],provenance=provenance,revision='2.2')
    if args.repeat_native:
        # A separate interpreter and native MLIR context expose pointer-order
        # nondeterminism that two serial walks of one context can conceal.
        with tempfile.TemporaryDirectory(prefix='rc3_native_repeat_') as temporary:
            target=Path(temporary)/'repeat.json'
            command=[sys.executable,'-B','-m','experiments.structural_audit.finalize_v22',
                '--mode','worker','--baseline-root',str(directory),'--snapshot-id',row['snapshot_id'],
                '--graph-output',str(target)]
            repeat=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=args.timeout)
            if repeat.returncode:raise ValueError('Independent native generation failed: '+repeat.stderr[-2500:])
            if v2.canonical_json(v2.graph_payload(graph))!=v2.canonical_json(v2.graph_payload(v2.read_graph(target))):
                raise ValueError('Independent native generation differs')
    v2.write_graph(graph,args.graph_output)
    print(json.dumps({'validation':v2.validate(graph),'native_correspondence':graph.graph['native_correspondence'],
        'timing':timing,'repeat_native':'independent_interpreters_identical' if args.repeat_native else 'not_run'}))


def compare(old,new):
    """No uncertain edge is removed/promoted by this extension."""
    if canonical_analysis(old.graph['compiler_analysis'])!=new.graph['compiler_analysis']:
        raise ValueError('Pinned native outcomes changed without a compiler migration')
    def dep_set(g):return {(u,v,e['dependence_kind'],e['evidence_status'],fingerprint(e['source_analysis']))
        for u,v,e in g.edges(data=True) if e['relation_type']=='depends_on'}
    if dep_set(old)!=dep_set(new):raise ValueError('A native dependence was lost, fabricated or promoted')
    old_access={(u,e['memory_operand'],e['access']) for u,_,e in old.edges(data=True) if e['relation_type']=='memory_access'}
    new_access={(u,e['memory_operand'],e['access']) for u,_,e in new.edges(data=True) if e['relation_type']=='memory_access'}
    if old_access!=new_access:raise ValueError('Concrete access occurrence coverage changed')
    previous_proven={(u,v,e['access']) for u,v,e in old.edges(data=True) if e['relation_type']=='memory_access' and e['evidence_status']=='proven'}
    current_proven={(u,v,e['access']) for u,v,e in new.edges(data=True) if e['relation_type']=='memory_access' and e['evidence_status']=='proven'}
    if not previous_proven<=current_proven:raise ValueError('Previously proven concrete storage relation was lost')
    old_targets={u:v for u,v,e in old.edges(data=True) if e['relation_type']=='targets'}
    new_targets={u:v for u,v,e in new.edges(data=True) if e['relation_type']=='targets'}
    removed={u:v for u,v in old_targets.items() if u not in new_targets}
    if any(old_targets[u]!=v for u,v in new_targets.items() if u in old_targets):raise ValueError('An action target was silently reassigned')
    return {'removed_unsupported_targets':removed,'newly_bound_actions':{u:v for u,v in new_targets.items() if u not in old_targets},'native_dependence_outcomes':'identical',
        'native_view_fact_order':'canonical SSA identity order; original multiset and multiplicities unchanged',
        'concrete_access_occurrences':'identical','uncertain_dependency_promotions':0,
        'explanation':'Stricter source/allocation identity rejects overlap-only action binding; local pointer/conditional chains preserve storage identity without claiming dependence.'}


def queries_equal(canonical,compact):
    from experiments.structural_audit.compare_v2 import _relations
    if v2.graph_payload(v2.expand_compact(compact))!=v2.graph_payload(canonical):raise ValueError('Compaction is not lossless')
    if _relations(canonical)!=_relations(compact) or action_point_views(canonical)!=action_point_views(compact):
        raise ValueError('Public action-point queries differ')
    for _,a in canonical.nodes(data=True):
        if a['category']=='action_point' and a['action_kind']=='loop':
            if get_loop_semantics(canonical,a['action_id'])!=get_loop_semantics(compact,a['action_id']):raise ValueError('Loop semantics differ')


def generate(args):
    from experiments.refinement.relations import compiler_tools
    output=Path(args.output_dir).resolve();base,content=baseline(output)
    rc2_dir,rc2_records=rc2_baseline(output)
    tools=compiler_tools(args.build,args.native_python)
    for key in ('cgeist_sha256','analysis_extension_sha256','lock_file_sha256'):
        if tools[key]!=content['compiler'][key]:raise ValueError('V2.1 pinned compiler identity changed: '+key)
    contract={'schema':v2.SCHEMA_V22,'semantic_revision':'rc3',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'worktree_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'rc2_archive_sha256':json.loads((ROOT/'docs/graph-release-v22.json').read_text())['archive_sha256'],'baseline_archive_sha256':v2.digest(ARCHIVE),
        'snapshots':[r['snapshot_id'] for r in content['graphs']], 'toolchain':tools,
        'code_sha256':{p:v2.digest(ROOT/p) for p in CODE},'repeat_native':args.repeat_native,
        'mlir_python_capi_sha256':v2.digest(Path(tools['pythonpath'])/'mlir/_mlir_libs/libMLIRPythonCAPI.so')}
    seal_protocol(output/'protocol.json',contract)
    path=output/'validation.json';results=json.loads(path.read_text()) if path.exists() else {}
    for i,row in enumerate(content['graphs'],1):
        if any(v2.digest(ROOT/p)!=digest for p,digest in contract['code_sha256'].items()):
            raise ValueError('Implementation changed during corpus generation; use a fresh output directory')
        key=row['snapshot_id'];stem=row['kernel']+'-'+key[:12]
        canonical=output/'canonical'/f'{stem}.json';compact_path=output/'compact'/f'{stem}.json'
        previous=results.get(key)
        if previous and previous.get('status')=='pass':
            if v2.digest(canonical)!=previous['canonical_sha256'] or v2.digest(compact_path)!=previous['compact_sha256']:
                raise ValueError('Completed release artifact was modified: '+stem)
            continue
        start=time.monotonic();record={'kernel':row['kernel'],'snapshot_id':key}
        try:
            command=[tools['python'],'-B','-m','experiments.structural_audit.finalize_v22','--mode','worker',
                '--baseline-root',str(base),'--snapshot-id',key,'--graph-output',str(canonical)]
            if args.repeat_native:command+=['--repeat-native']
            result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':tools['pythonpath'],'PYTHONHASHSEED':'0'},
                capture_output=True,text=True,timeout=args.timeout)
            log=output/'logs'/f'{stem}.log';log.parent.mkdir(exist_ok=True);log.write_text(result.stdout+'\n'+result.stderr)
            record.update(command=command,log=str(log))
            if result.returncode:raise RuntimeError(result.stderr[-3500:])
            native=json.loads(result.stdout.splitlines()[-1]);graph=v2.read_graph(canonical);old=v2.read_graph(base/row['canonical'])
            compact=v2.compact_view(graph);queries_equal(graph,compact);v2.write_graph(compact,compact_path)
            comparison=compare(old,graph)
            rc2=v2.read_graph(rc2_dir/'canonical'/Path(rc2_records[key]['canonical']).name)
            rc2_comparison=compare(rc2,graph)
            if rc2_comparison['removed_unsupported_targets']:raise ValueError('RC2 verified target was lost')
            record.update(status='pass',source_sha256=row['source_sha256'],metadata_sha256=row['metadata_sha256'],
                canonical=str(canonical),compact=str(compact_path),canonical_sha256=v2.digest(canonical),compact_sha256=v2.digest(compact_path),
                before=statistics(old),rc2_before=statistics(rc2),after=statistics(graph),comparison=comparison,rc2_comparison=rc2_comparison,native=native,
                serialization='deterministic',canonical_compact='exact_reconstruction_and_queries_identical')
            save(output/'details'/f'{stem}.json',details(graph))
        except (OSError,ValueError,KeyError,RuntimeError,subprocess.TimeoutExpired) as exc:
            record.update(status='fail',error=str(exc))
        record['runtime_seconds']=time.monotonic()-start;results[key]=record;save(path,results)
        print(f'{i}/63 {stem}: {record["status"]}',flush=True)
    summarize(output)


def summarize(output):
    output=Path(output);results=json.loads((output/'validation.json').read_text());good=[r for r in results.values() if r['status']=='pass']
    def totals(phase):
        scalar=Counter();nested=defaultdict(Counter)
        for row in good:
            for key,value in row[phase].items():
                if isinstance(value,dict):nested[key].update(value)
                elif isinstance(value,int):scalar[key]+=value
        return {**dict(scalar),**{k:dict(v) for k,v in nested.items()}}
    receipts=json.loads((output/'test_receipt.json').read_text()) if (output/'test_receipt.json').exists() else {}
    gates={'A':'pass' if len(good)==63 else 'blocked','B':'pass_mechanical_oracles; semantic_fixture_receipt_required',
        'C':'pending_independent_human_adjudication_of_real_program_worksheets','D':'pass' if sum(r['after']['registered'] for r in good)==606 else 'blocked',
        'E':'pending_regression_receipt_and_protected_artifact_verification'}
    if receipts.get('v22_native_fixture_tests'):
        gates['B']='pass_native_occurrence_checks_and_adversarial_fixtures' if receipts['v22_native_fixture_tests'].get('failed')==0 else 'blocked_fixture_tests'
    if receipts.get('mandatory_worktree_regressions'):
        gates['E']='partial_unrelated_recovery_draft_failures' if receipts['mandatory_worktree_regressions'].get('failed') else 'pass_regressions_and_protected_artifact_hashes'
    summary={'status':'V2.2 RC3 — independent evidence or regressions remain outstanding','expected':63,'passed':len(good),
        'failed':len(results)-len(good),'pending':63-len(results),'before':totals('before'),'rc2_before':totals('rc2_before'),'after':totals('after'),'gates':gates,
        'native_toolchain_changed':False,'independent_review_note':'Mechanical native occurrence checks are exhaustive for supported function-body entities; stratified dependence examples and real-source annotations remain unreviewed drafts.'}
    save(output/'comparison.json',summary)
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['kernel','snapshot_id','status','before','after','native','comparison','error']);writer.writeheader()
        for r in results.values():writer.writerow({k:json.dumps(r[k],sort_keys=True) if isinstance(r.get(k),dict) else r.get(k) for k in writer.fieldnames})
    lines=['# V2.2 RC3 semantic audit','',summary['status'],'',
        f'{len(good)}/63 snapshots passed mechanical checks; {summary["failed"]} failed; {summary["pending"]} pending.',
        '', '## Aggregate comparison','', '| Property | Frozen V2.1 | V2.2 |','|---|---:|---:|']
    for key in ('nodes','edges','registered','active','unresolved_registrations','source_only','access_status','dependence_status','dependence_reasons','native_negative_conclusions','duplicate_native_queries','cfg_successors'):
        lines.append(f'| {key} | {summary["before"].get(key)} | {summary["after"].get(key)} |')
    lines+=['','## Findings and limits','',
        'External Kalman L7 has no unique source-bound storage target. L8 was mapped onto the native L6 allocation by shared use lines; V2.2 rejects that mapping. Surviving scalar computations do not prove a separate allocation or compiler elimination.',
        'LLVM GEP/cast/memref-pointer chains retain their base storage and exact index operands. Pointer loads requiring reaching-store/call-context analysis remain uncertain. A root does not establish element overlap.',
        'Pinned Python lacks Operation.successors. V2.2 uses mlirOperationGetSuccessor and checks native CFG occurrences independently.',
        'All native dependence records, positive edges, uncertain edges and negative conclusions are preserved. AES uncertainty is not deduplicated or sampled away. The native analyzer stops at the first successful dependence depth; additional carrying loops remain unverified.',
        'details/ contains comprehensive dependence classifications, root diagnostics, unresolved registrations and deterministic two-per-stratum draft review records. These drafts are not reviewed ground truth.',
        'Numerical template spellings are lexical evidence, not AST-verified or hardware-exact Vitis semantics. No target, objective, QoR or pragma configuration enters generation.',
        '', '## Release gates','',json.dumps(gates,indent=2),'',
        'No GNN encoder, training, LLM inference, HLS synthesis or refinement behavior changed. V2.1 remains the frozen baseline; this candidate is not labeled final.']
    (output/'V22_SEMANTIC_AUDIT.md').write_text('\n'.join(lines)+'\n')
    rc3_reports(output,results,summary)
    return summary


def rc3_reports(output,results,summary):
    """Expose exact witnesses; an automated worksheet is never human review."""
    from GNN_branch.mlir_semantics import definitions,operands
    base,cohort=baseline(output);rc2_dir,previous=rc2_baseline(output)
    inputs={r['snapshot_id']:r for r in cohort['graphs']}
    bindings=[];pointers=[];invocations=[];aes={};worksheets=[]
    for ident,r in results.items():
        if r['status']!='pass':continue
        g=v2.read_graph(r['canonical']);old=v2.read_graph(rc2_dir/'canonical'/Path(previous[ident]['canonical']).name)
        row=inputs[ident];source=(base/row['source']).read_text().splitlines()
        new_coverage={a['action_id']:a for a in g.graph['action_point_coverage']['records']}
        for action in old.graph['action_point_coverage']['records']:
            if not action['metadata_registered'] or action['compiler_matched']:continue
            binding=action.get('source_binding') or {};line=binding.get('source_line',0)
            candidates=[]
            for n,a in g.nodes(data=True):
                if a.get('text') not in ('memref.alloc','memref.alloca','memref.get_global','llvm.alloca'):continue
                if a.get('op_uid','').split(':op')[0]!=binding.get('function'):continue
                values=[v for _,v,e in g.out_edges(n,data=True) if e['relation_type']=='defines']
                candidates.append({'operation':n,'kind':a['text'],'source_line':a.get('source_line'),
                    'native_attributes':a.get('mlir_attrs'),'values':values,
                    'root_evidence':[(v,t,e['evidence_status']) for v in values for _,t,e in g.out_edges(v,data=True) if e['relation_type']=='resolves_to'],
                    'rejection':'different compiler action/declaration' if a.get('source_line')!=line else 'inspect exact annotation/uniqueness'})
            bindings.append({'kernel':r['kernel'],'snapshot_id':ident,'action_id':action['action_id'],
                'source_declaration':source[line-1] if line and line<=len(source) else None,'source_line':line,
                'source_sha256':row['source_sha256'],'raw_mlir_sha256':cohort['files_sha256'][row['raw_mlir']],
                'raw_mlir':row['raw_mlir'],'compiler_objects':candidates,'final_record':new_coverage[action['action_id']],
                'transformation_provenance':'frozen IR only; no elimination receipt or diagnostic recompilation substitutes for it'})
        for op,root,e in old.edges(data=True):
            if e['relation_type']!='memory_access' or e['evidence_status']!='not_established':continue
            value=e['memory_operand'];chain=[];pending=[value];seen=set()
            while pending:
                v=pending.pop()
                if v in seen:continue
                seen.add(v);defs=definitions(g,v)
                for defining,_ in defs:
                    args=operands(g,defining)
                    chain.append({'value':v,'operation':defining,'kind':g.nodes[defining]['text'],
                        'location':g.nodes[defining].get('location'),'operands':args})
                    pending.extend(a for a in args.values() if g.nodes[a].get('is_memory'))
            new=[{'root':target,'status':a['evidence_status']} for _,target,a in g.out_edges(op,data=True) if a['relation_type']=='memory_access' and a['memory_operand']==value]
            pointers.append({'kernel':r['kernel'],'snapshot_id':ident,'operation':op,'pointer_operand':value,
                'access':e['access'],'native_effect':e['source_analysis'],'definition_chain':chain,
                'reaching_definition_evidence':g.nodes[value].get('loaded_pointer_evidence'),
                'diagnostic':g.nodes[value].get('root_resolution_diagnostic'),'rc3':new})
        diagnostic=json.loads((output/'details'/(Path(r['canonical']).stem+'.json')).read_text())
        invocations.append({'kernel':r['kernel'],'snapshot_id':ident,'through_call_accesses':diagnostic['interprocedural_accesses']})
        for sample in diagnostic['deterministic_stratified_review']:
            worksheets.append({'kernel':r['kernel'],'snapshot_id':ident,**sample,'human_reviewed':False,
                'independent_oracle':'native analysis record plus independent occurrence-level walk; semantic adjudication pending'})
        if r['kernel']=='chstone-aes':
            failures=[d for d in g.graph['compiler_analysis']['dependences'] if d['reason']=='affine_analysis_failure']
            def chain(operation):
                blocks=[]
                while operation and operation in g:
                    a=g.nodes[operation]
                    if a.get('block') is not None:blocks.append(a['block'])
                    parent=a.get('parent_operation')
                    if parent and g.nodes[parent].get('text')=='func.func':break
                    operation=parent
                return blocks
            groups=Counter();examples={}
            for d in failures:
                u='op:'+d['source']['function']+':op'+str(d['source']['ordinal'])
                v='op:'+d['target']['function']+':op'+str(d['target']['ordinal'])
                left,right=chain(u),chain(v);common=bool(left and right and left[-1]==right[-1])
                key=(d['source']['function'],common,d['kind']);groups[key]+=1;examples.setdefault(key,{'native':d,'source_blocks':left,'target_blocks':right})
            aes={'uncertainty':r['after']['dependence_reasons'],'affine_failures':len(failures),
                'no_common_affine_scope_block':sum(count for (_,common,_),count in groups.items() if not common),
                'groups':[{'function':k[0],'common_block':k[1],'kind':k[2],'count':v,'example':examples[k]} for k,v in sorted(groups.items())],
                'reference':'pinned AffineAnalysis.cpp getCommonBlockInAffineScope 322-353 and dependence checker 621-624',
                'proof_boundary':'mechanical precondition localization, not an exact native failure-site receipt; unknown is not no-dependence',
                'native_depth_limit':'first successful dependence depth only; all records and distinct RAW/WAR/WAW preserved'}
    save(output/'details/action_binding_audit.json',bindings)
    save(output/'details/pointer_root_audit.json',pointers)
    save(output/'details/interprocedural_audit.json',invocations)
    save(output/'details/AES_failure_contexts.json',aes)
    save(output/'details/independent_review_worksheet.json',worksheets)
    lines=['# RC3 action-binding audit','','One row per unresolved RC2 registration. Compiler object lists, exact attributes, SSA/root witnesses and raw-IR hashes are in `details/action_binding_audit.json`. No shape-only or shared-use-line mapping is accepted. No elimination is inferred.','',
        '| Kernel / snapshot | Action / declaration | RC3 disposition |','|---|---|---|']
    for b in bindings:
        lines.append(f"| {b['kernel']} / {b['snapshot_id'][:12]} | {b['action_id']} line {b['source_line']}: `{b['source_declaration'].strip()}` | {'matched' if b['final_record']['compiler_matched'] else 'unresolved; no exact surviving declaration/annotation target'} |")
    lines+=['','Frozen lowering does not retain an independently identifiable allocation for these declarations. Surviving scalar calculations are not allocation identity or proof of elimination. A future frontend action-preservation experiment must use separate compilation provenance.']
    (output/'V22_ACTION_BINDING_AUDIT.md').write_text('\n'.join(lines)+'\n')
    lines=['# RC3 pointer-root audit','','The six original unresolved occurrences are traced in `details/pointer_root_audit.json`. Each concrete access remains represented. A formal pointer-containing cell has no complete local reaching-store set. Resolving its contents requires invocation-aware memory-state analysis; an actual/formal pointer binding alone does not prove what its contents hold.','',
        '| Snapshot | Access | RC3 root statuses |','|---|---|---|']
    for p in pointers:lines.append(f"| {p['snapshot_id'][:12]} | {p['operation']} ({p['access']}) | {','.join(a['status'] for a in p['rc3'])} |")
    lines+=['','Supported new analysis: nonescaping allocation-local exact-address pointer cells, same native block order and complete structured conditional branches. Overwritten stores are handled. Helper escape, uncertain aliases and unsupported CFG paths keep the set incomplete. This is not a general reaching-definition or pointer-analysis engine.']
    (output/'V22_POINTER_ROOT_AUDIT.md').write_text('\n'.join(lines)+'\n')
    (output/'V22_INTERPROCEDURAL_AUDIT.md').write_text('''# RC3 interprocedural audit

`details/interprocedural_audit.json` retains concrete operation, root, access mode, invocation chain and exact binding witnesses for every derived through-call action relationship. Possible alternatives are substituted lazily per used formal, never as a Cartesian product of all arguments. Candidate-set completeness is separate from certainty of any one root.

Native adversarial fixtures cover unique and ambiguous caller roots, incomplete branches, nested helpers, separate/repeated calls, return alternatives, recursion, external calls and helper-written pointer cells. Distinct caller arrays stay distinct. LUD L2–L5 is verified only in its exact `lud_diagonal:op5` invocation; the empty helper context does not inherit caller roots. Exact test results are in the regression receipt. This is representative validation, not exhaustive interprocedural proof.

Previously generated RC2 graphs have no `semantic_revision=rc3` marker and retain historical query behavior. New RC3 graphs opt into set-valued substitutions; canonical facts are not rewritten to make query joins easier.
''')
    fields=['kernel','snapshot_id','status','rc2_nodes','rc3_nodes','rc2_edges','rc3_edges','rc2_active','rc3_active','rc2_access_status','rc3_access_status','rc2_dependence_status','rc3_dependence_status','reason']
    with (output/'V22_RC2_VS_RC3_COMPARISON.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results.values():
            record={k:r.get(k) for k in ('kernel','snapshot_id','status')}
            for phase,key in (('rc2','rc2_before'),('rc3','after')):
                for field in ('nodes','edges','active','access_status','dependence_status'):record[phase+'_'+field]=json.dumps(r.get(key,{}).get(field),sort_keys=True)
            record['reason']='Native outcomes/occurrences unchanged; evidence-first action resolution; call-scoped possible-root queries; exact local pointer-store evidence; descriptor-offset tag clarification' if r['status']=='pass' else r.get('error')
            writer.writerow(record)
    receipt=json.loads((output/'test_receipt.json').read_text()) if (output/'test_receipt.json').exists() else {}
    (output/'V22_REGRESSION_REPORT.md').write_text('# RC3 regression report\n\n'+json.dumps(receipt,indent=2)+'\n\nJPEG: same-translation-unit wrapper respects static ReadBuf linkage; table names match the benchmark. Rosetta: V2.1 remains 11/12, explicitly selected RC3 restores the static z_buffer target from compiler declaration/global-root evidence. Neither recovery workload is added to the frozen 63.\n')
    report=(output/'V22_SEMANTIC_AUDIT.md').read_text()
    report+='\n## RC2 versus RC3\n\n'+json.dumps({'RC2':summary['rc2_before'],'RC3':summary['after']},indent=2)+'\n'
    report+='\n## Evidence levels\n\nAll supported native occurrences are mechanically cross-checked. Native-parsed handwritten IR has independently specified assertions; compiled C++ fixtures and real frozen IR provide complementary evidence. No record has human adjudication. The deterministic stratified worksheets remain pending independent review, so Gate C stays open. No publication precision/recall or exhaustive memory proof is claimed.\n'
    report+='\n## AES\n\n'+json.dumps({k:v for k,v in aes.items() if k!='groups'},indent=2)+'\n'
    (output/'V22_RC3_SEMANTIC_AUDIT.md').write_text(report)


def package(output):
    """Publish an explicitly labeled RC package; never overwrite V2.1."""
    output=Path(output);summary=summarize(output)
    if summary['passed']!=63:raise ValueError('A 63-snapshot package requires all mechanical validations')
    base,content=baseline(output);results=json.loads((output/'validation.json').read_text())
    files={}
    for row in content['graphs']:
        r=results[row['snapshot_id']]
        for kind in ('canonical','compact'):
            path=Path(r[kind])
            if v2.digest(path)!=r[kind+'_sha256']:raise ValueError('Completed graph changed')
            files[kind+'/'+path.name]=path
        for name in (row['source'],row['metadata'],row['raw_mlir'],*row['headers']):files[name]=base/name
    for name in CODE:files['code/'+name]=ROOT/name
    for name in ('tests/test_mlir_graph_v22.py','tests/test_mlir_graph_v2.py','tests/test_v21_refinement.py',
                 'docs/structural-graph-v22.md','toolchain/lock.env','toolchain/patches/0009-mailohls-analysis-python.patch'):
        files['provenance/'+name]=ROOT/name
    for name in ('protocol.json','validation.json','comparison.json','comparison.csv','test_receipt.json'):files[name]=output/name
    for path in sorted(output.glob('V22_*')):files[path.name]=path
    for path in sorted((output/'logs').glob('regression*.log')):files['logs/'+path.name]=path
    for path in (output/'details').glob('*.json'):files['details/'+path.name]=path
    release={'schema':v2.SCHEMA_V22,'status':summary['status'],'graph_count':63,
        'baseline_archive_sha256':v2.digest(ARCHIVE),'files_sha256':{name:v2.digest(path) for name,path in sorted(files.items())},
        'toolchain':json.loads((output/'protocol.json').read_text())['toolchain']}
    save(output/'release.json',release,True);files['release.json']=output/'release.json'
    check=output/'checksums.sha256';check.write_text(''.join(v2.digest(p)+'  '+n+'\n' for n,p in sorted(files.items())))
    files['checksums.sha256']=check
    archive=ROOT/'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz';temporary=archive.with_suffix('.tmp')
    with temporary.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as bundle:
            for name,path in sorted(files.items()):
                info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o644
                with path.open('rb') as f:bundle.addfile(info,f)
    if archive.exists() and v2.digest(archive)!=v2.digest(temporary):raise ValueError('Refusing to overwrite a different V2.2 RC archive')
    temporary.replace(archive)
    save(ROOT/'docs/graph-release-v22-rc3.json',{**release,'archive':str(archive.relative_to(ROOT)),'archive_sha256':v2.digest(archive)},True)
    (ROOT/'docs/graph-release-v22-rc3.sha256').write_text(v2.digest(archive)+'  '+str(archive.relative_to(ROOT))+'\n')
    print(archive, v2.digest(archive))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('diagnose','generate','summarize','package','worker'),default='diagnose')
    parser.add_argument('--output-dir',default=str(DEFAULT));parser.add_argument('--build',default=str(Path.home()/'tools/Polygeist/build-mailohls-assertions'))
    parser.add_argument('--native-python',default=str(Path.home()/'.mlir-python311/bin/python'))
    parser.add_argument('--timeout',type=int,default=600);parser.add_argument('--repeat-native',action='store_true')
    parser.add_argument('--baseline-root');parser.add_argument('--snapshot-id');parser.add_argument('--graph-output')
    args=parser.parse_args()
    if args.mode=='worker':worker(args)
    elif args.mode=='generate':generate(args)
    elif args.mode=='summarize':print(json.dumps(summarize(args.output_dir),indent=2))
    elif args.mode=='package':package(args.output_dir)
    else:
        directory,content=baseline(args.output_dir);print('Authenticated',len(content['graphs']),'snapshots:',directory)


if __name__=='__main__':main()
