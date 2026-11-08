#!/usr/bin/env python3
"""Authenticate the frozen RC3 cohort and summarize compiler-only labels.

No model inference or training is performed. Reads the committed RC3 release
archive and the frozen V4b source/measurement membership audit; writes a compact
per-program manifest. Unknown evidence is kept as unknown, never a negative.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, re, tarfile
from pathlib import Path

ARCHIVE_DEFAULT = Path('GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz')
AUDIT_DEFAULT = Path('experiments/v22_gnn_training/data-audit.json')

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def load_json(tar, name): return json.load(tar.extractfile(name))
def count_values(xs): return dict(sorted(collections.Counter(xs).items()))

def kernel_family_map(split):
    """Mirror the frozen V4b family prefix assignment over measured kernels."""
    result={}
    for split_name in ('train','val','test'):
        for family in split.get(split_name+'_families',[]):
            norm=re.sub(r'[^a-z0-9]','',family.lower())
            for kernel in split.get(split_name+'_kernels',[]):
                if re.sub(r'[^a-z0-9]','',kernel.lower()).startswith(norm):
                    if kernel in result and result[kernel]!=(family,split_name):
                        raise ValueError(f'Ambiguous frozen family assignment for {kernel}')
                    result[kernel]=(family,split_name)
    return result

def graph_kernel_name(member):
    stem=Path(member).stem
    return re.sub(r'-[0-9a-f]{12}$','',stem)

def summarize_graph(name, raw, g, cohort):
    md=g['metadata']; prov=md.get('provenance',{}); nodes={n['id']:n for n in g['nodes']}; edges=g['edges']
    target={e['source']:e['target'] for e in edges if e['relation_type']=='targets' and e.get('evidence_status')=='proven'}
    action_records=md.get('action_point_coverage',{}).get('records',[])
    actions=[]
    for r in sorted(action_records,key=lambda x:int(x['action_id'][1:])):
        aid=r['action_id']; tgt=target.get(r.get('action_point_node'))
        node=nodes.get(tgt,{})
        is_loop=(node.get('category')=='operation' and bool(node.get('is_loop')))
        loop_edges=[e for e in edges if e['relation_type']=='loop_carried' and e.get('carrying_loop')==tgt and e.get('evidence_status')=='proven'] if is_loop else []
        accesses=[]
        if not is_loop and tgt:
            root_nodes={tgt}
            root_nodes |= {e['target'] if e['source']==tgt else e['source'] for e in edges if e['relation_type'] in ('resolves_to','alias') and tgt in (e['source'],e['target'])}
            accesses=[e for e in edges if e['relation_type']=='memory_access' and e.get('evidence_status')=='proven' and (e['source'] in root_nodes or e['target'] in root_nodes)]
        actions.append({
            'action_id':aid,'kind':r.get('source_binding',{}).get('kind'),
            'source_labeled':bool(r.get('source_labeled')),'metadata_registered':bool(r.get('metadata_registered')),
            'compiler_matched':bool(r.get('compiler_matched')),'compiler_eliminated':bool(r.get('compiler_eliminated')),
            'target_node':tgt,'target_category':node.get('category'),
            'loop_depth':node.get('loop_depth') if is_loop else None,
            'trip_count':r.get('compiler_proven_trip_count') if is_loop else None,
            'trip_count_provenance':r.get('trip_count_provenance') if is_loop else None,
            'proven_loop_carried_ssa_witnesses':len(loop_edges) if is_loop else None,
            'loop_carried_known_status':'positive_proven_only' if loop_edges else ('unknown_no_negative_inference' if is_loop else 'not_applicable'),
            'array_source_shape':r.get('source_binding',{}).get('array_dimensions'),
            'proven_access_witnesses':len(accesses) if not is_loop else None,
        })
    loops=[n for n in nodes.values() if n.get('category')=='operation' and n.get('is_loop')]
    mem=[n for n in nodes.values() if n.get('category')=='memory_object']
    ssa=[n for n in nodes.values() if n.get('category')=='ssa_value']
    proven_lc=[e for e in edges if e['relation_type']=='loop_carried' and e.get('evidence_status')=='proven']
    deps=[e for e in edges if e['relation_type']=='depends_on']
    accesses=[e for e in edges if e['relation_type']=='memory_access']
    affine_ops=[n for n in nodes.values() if n.get('category')=='operation' and n.get('mlir_attrs',{}).get('map') is not None]
    return {
      'graph_member':name,'graph_member_sha256':sha(raw),'graph_identity':md.get('graph_identity'),
      'source_sha256':prov.get('source_sha256'),'metadata_sha256':prov.get('metadata_sha256'),
      'mlir_sha256':prov.get('mlir_sha256'),'semantic_revision':md.get('semantic_revision'),
      'family_split':cohort.get('family'),'family_id':cohort.get('family_id',cohort.get('kernel_family')),
      'v4b_kernel':cohort.get('kernel'),'measured_rows':cohort.get('measured_rows'),
      'program':{'nodes':len(nodes),'edges':len(edges),'node_categories':count_values(n.get('category') for n in nodes.values()),
        'edge_relations':count_values(e.get('relation_type') for e in edges),
        'evidence_status':count_values(e.get('evidence_status','unknown') for e in edges),
        'action_count':len(action_records),'matched_actions':sum(bool(r.get('compiler_matched')) for r in action_records),
        'array_actions':sum(r.get('source_binding',{}).get('kind')=='array' for r in action_records),
        'loop_actions':sum(r.get('source_binding',{}).get('kind')=='loop' for r in action_records),
        'compiler_loops':len(loops),'memory_objects':len(mem),'ssa_values':len(ssa),
        'proven_loop_carried_edges':len(proven_lc),'proven_dependences':sum(e.get('evidence_status')=='proven' for e in deps),
        'possible_dependences':sum(e.get('evidence_status')=='possible' for e in deps),
        'not_established_dependences':sum(e.get('evidence_status')=='not_established' for e in deps),
        'memory_access_status':count_values(e.get('evidence_status','unknown') for e in accesses),
        'memory_access_roles':count_values(e.get('access','unknown') for e in accesses),
        'ops_with_affine_map':len(affine_ops)},
      'actions':actions,
      'label_policy':{'loop_carried_ssa':'positive only when explicit proven loop_carried edge carries on action target; absent is unknown',
        'dependence':'only explicit depends_on witnesses are labeled; missing witness is unknown, not independent',
        'trip_count':'known only when action coverage has compiler_proven_trip_count and provenance',
        'access_role':'only memory_access edge access/status fields',
        'shape':'source dimensions and compiler memory shape kept separately'}
    }

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,default=ARCHIVE_DEFAULT); p.add_argument('--audit',type=Path,default=AUDIT_DEFAULT); p.add_argument('--out',type=Path,default=Path('experiments/architecture_representation_audit/results/cohort_manifest.json')); a=p.parse_args()
    archive_bytes=a.archive.read_bytes(); archive_sha=sha(archive_bytes)
    audit=json.loads(a.audit.read_text()); cohorts={r['graph_member']:r for r in audit['kernels']}
    frozen_split=json.loads(Path('mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    family_map=kernel_family_map(frozen_split)
    with tarfile.open(a.archive,'r:gz') as tf:
        release=load_json(tf,'release.json'); validation=load_json(tf,'validation.json')
        if release.get('schema')!='salt-hls-mlir-structural-v2.2' or release.get('graph_count')!=63: raise ValueError('Unexpected archive release contract')
        members=set(tf.getnames()); canonical=sorted(n for n in members if n.startswith('canonical/') and n.endswith('.json'))
        if len(canonical)!=release['graph_count']: raise ValueError('Canonical count differs from release manifest')
        rows=[]; missing_cohort=[]
        for name in canonical:
            info=tf.getmember(name); raw=tf.extractfile(info).read(); g=json.loads(raw)
            expected=release['files_sha256'].get(name)
            if not expected or sha(raw)!=expected: raise ValueError(f'Archive member authentication failed: {name}')
            cohort=cohorts.get(name)
            if cohort:
                if cohort.get('graph_member_sha256')!=sha(raw): raise ValueError(f'Frozen data-audit graph hash mismatch: {name}')
                if cohort.get('source_sha256')!=g['metadata'].get('provenance',{}).get('source_sha256'): raise ValueError(f'Source identity mismatch: {name}')
                if cohort.get('graph_identity')!=g['metadata'].get('graph_identity'): raise ValueError(f'Graph identity mismatch: {name}')
                kernel=graph_kernel_name(name)
                family=family_map.get(kernel)
                if family is None: raise ValueError(f'Graph kernel is not assigned by frozen family split: {kernel}')
                cohort={**cohort,'kernel':kernel,'family_id':family[0],'family':family[1]}
            else: missing_cohort.append(name)
            rows.append(summarize_graph(name,raw,g,cohort or {}))
    matched=[r for r in rows if r['family_split'] in ('train','val','test')]
    split_counts=count_values(r['family_split'] for r in matched)
    summary={'authenticated_graphs':len(rows),'measured_exact_source_matches':len(matched),'unmatched_release_graphs':len(missing_cohort),
      'split_graph_counts':split_counts,'split_family_counts':audit['dataset']['split_family_counts'],
      'graph_size_nodes':{'min':min(r['program']['nodes'] for r in rows),'median':sorted(r['program']['nodes'] for r in rows)[len(rows)//2],'max':max(r['program']['nodes'] for r in rows)},
      'graph_size_edges':{'min':min(r['program']['edges'] for r in rows),'median':sorted(r['program']['edges'] for r in rows)[len(rows)//2],'max':max(r['program']['edges'] for r in rows)},
      'action_counts':count_values(str(r['program']['action_count']) for r in matched),
      'label_coverage':{
        'loop_actions':sum(r['program']['loop_actions'] for r in matched),
        'trip_count_known_loop_actions':sum(a['trip_count'] is not None for r in matched for a in r['actions']),
        'recurrence_proven_positive_loop_actions':sum(bool(a['proven_loop_carried_ssa_witnesses']) for r in matched for a in r['actions'] if a['target_category']=='operation'),
        'loop_actions_without_proven_witness_are_unknown_not_negative':sum(a['target_category']=='operation' and not a['proven_loop_carried_ssa_witnesses'] for r in matched for a in r['actions']),
        'proven_dependence_edges':sum(r['program']['proven_dependences'] for r in matched),
        'possible_dependence_edges':sum(r['program']['possible_dependences'] for r in matched),
        'not_established_dependence_edges':sum(r['program']['not_established_dependences'] for r in matched),
        'memory_access_edges':sum(sum(r['program']['memory_access_status'].values()) for r in matched),
        'affine_map_operations':sum(r['program']['ops_with_affine_map'] for r in matched),
      },
      'diversity':{'families':audit['dataset']['split_family_counts'],'graph_release_variants':len(rows),'exact_source_measured_kernels':len(matched),
        'distinct_action_count_values':sorted({r['program']['action_count'] for r in matched}),
        'compiler_loop_count_range':[min(r['program']['compiler_loops'] for r in matched),max(r['program']['compiler_loops'] for r in matched)],
        'memory_object_count_range':[min(r['program']['memory_objects'] for r in matched),max(r['program']['memory_objects'] for r in matched)]},
      'provenance':{'archive_sha256':archive_sha,'release_archive_sha256':release.get('baseline_archive_sha256'),
        'release_member_manifest_sha256':sha(json.dumps(release['files_sha256'],sort_keys=True,separators=(',',':')).encode()),
        'frozen_family_split_sha256':audit['dataset']['split_sha256'],'frozen_data_audit_sha256':sha(a.audit.read_bytes()),
        'v22_semantic_revision':'rc3','split_policy':'existing V4b family split; no program family crosses train/val/test'}}
    out={'schema':'salt-hls-architecture-representation-cohort-v1','summary':summary,'graphs':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
