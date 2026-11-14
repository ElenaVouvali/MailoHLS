"""Strict all-kernel historical graph audit, then one train-only GMT fit/export.

An explicitly named original FiLM bank is allowed as a separate fallback cohort.
No intersection, empty memory or evidence truncation can pass this preparation.
"""
import argparse
import collections
import json
from pathlib import Path
import torch

from experiments.film_gmt_stage2_final.bank_contract import sha,write,ROOT,ORIGINAL_BANK_SHA


def audit_one(kernel,entry,runtime,hist,graphs):
    from experiments.historical_gmt_stage2_ab import prepare_banks as prep
    from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import cohort_action_mapping,make_evidence
    source=ROOT/entry['files']['source']['path']
    metadata=ROOT/entry['files']['metadata']['path']
    graphfile=prep.choose_graph(kernel,graphs)
    graph,pack=hist.export_graph(graphfile,kernel,runtime)
    provenance=pack['graph_provenance']
    if provenance['source_sha256']!=sha(source) or provenance['action_sha256']!=sha(metadata):
        raise ValueError('Historical graph source/metadata differs from canonical; RC3 hashes cannot replace it')
    generator=runtime[-1]['authenticated_training_files']['mlir_graph_gen.py']['sha256']
    if provenance['generator_sha256']!=generator:
        raise ValueError('Graph generator differs from the historical GNN fitting contract')
    mapping=cohort_action_mapping(graph,pack)
    from Preprocessing.create_jsonl import load_actions,source_with_placeholders
    template=source_with_placeholders(source,load_actions(kernel,source.read_text()))
    prep.validate_mapping({'kernel_name':kernel,'source_file':str(source),'input':template},pack,mapping,hist,graph)
    for m in mapping:
        if int(m['action'][1:])-1!=m['absolute_memory_slot']:
            raise ValueError('Absolute Lk action slot changed')
    data,evidence=make_evidence(graph,pack,mapping,graphfile)
    # Source-only array pseudo-nodes cannot stand in for actual compiler evidence.
    for i,m in enumerate(mapping):
        genuine=[r for r in evidence['records'] if r['action']==i and
                 (r['kind']=='edge' or r['src']!=m['compiler_node_id'])]
        if not genuine:raise ValueError('Action-only compiler evidence for '+m['action'])
    present=data['route'].any(-1)
    info={'graph_sha256':sha(graphfile),'source_sha256':sha(source),'metadata_sha256':sha(metadata),
          'historical_graph_provenance':provenance,'action_mapping':mapping,
          'actions':int(data['active'].sum()),'evidence_records':len(evidence['records']),
          'role_counts':dict(collections.Counter(r['identity'].get('role','node') for r in evidence['records'])),
          'per_action_evidence':evidence['counts'],'memory_tokens':64+int(present.sum()),
          'active_action_anchor_sha256':__import__('hashlib').sha256(data['anchors'].contiguous().numpy().tobytes()).hexdigest()}
    return data,pack,info


def main(a):
    from experiments.historical_gmt_stage2_ab import prepare_banks as prep
    from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import find_train_names,MAX_EVIDENCE_RECORDS
    from LLM_branch.common.mailohls_contract import family_id_from_kernel_name as family
    if a.out.exists():raise FileExistsError('Fresh memory-preparation output required')
    a.out.mkdir(parents=True);torch.set_num_threads(4)
    require=sha(a.bank/'manifest.json')
    if require!=ORIGINAL_BANK_SHA:raise ValueError('This fallback must be the exact selected FiLM 31+2 bank')
    manifest=json.loads((a.bank/'manifest.json').read_text())
    for name,digest in manifest['outputs_sha256'].items():
        if sha(a.bank/name)!=digest:raise ValueError('Selected bank changed')
    data={fold:[json.loads(l) for l in (a.bank/(fold+'_sft.jsonl')).open()] for fold in ['train','val']}
    folds={fold:{e['row']['kernel_name'] for e in rows} for fold,rows in data.items()}
    if (len(folds['train']),len(folds['val']))!=(31,2):raise ValueError('Original fallback cohort differs')
    canonical={e['kernel']:e for e in json.loads((ROOT/'Data/manifest.json').read_text())['kernels']}
    hist,runtime=prep.historical_runtime(a.assets)
    write(a.out/'historical_identity.json',runtime[-1])
    train_info=runtime[-1]['authenticated_training_files']['experiment_split.json']
    hist_split=json.loads(Path(train_info['path']).read_text())
    gnn_train={family(k) for k in find_train_names(hist_split)}
    val_families={family(k) for k in folds['val']}
    if gnn_train & val_families:raise ValueError('Validation families participated in GNN fitting')
    receipt={'schema':'film-gmt-final-memory-preparation-v1','cohort':'original-authenticated-FiLM-31+2',
        'source_bank_manifest_sha256':require,'max_evidence_records':MAX_EVIDENCE_RECORDS,
        'historical_gnn_training_family_overlap_with_val':[],
        'historical_gnn_selection_used_held_out_ranking':True,
        'gnn_sha256':runtime[-1]['checkpoint_sha256'],'largest_cfd_first':{},'kernels':{},'errors':{},
        'expected_train_kernels':sorted(folds['train']),'expected_val_kernels':sorted(folds['val']),
        'locked_test_measurements_read':False}
    cohort={};packs={};hashes={}
    # Two largest CFD graph inputs first, including source-only excluded cases.
    largest=sorted(a.graphs.glob('*cfd*.gexf'),key=lambda p:p.stat().st_size,reverse=True)[:2]
    order=[p.stem for p in largest]+sorted(set.union(*folds.values())-{p.stem for p in largest})
    for k in order:
        try:
            d,p,info=audit_one(k,canonical[k],runtime,hist,a.graphs)
            if k in {x.stem for x in largest}:receipt['largest_cfd_first'][k]={'status':'PASS',**info}
            if k in set.union(*folds.values()):
                cohort[k]=d;packs[k]=p;hashes[k]=info['graph_sha256'];receipt['kernels'][k]=info
            print(k,'PASS',info['evidence_records'],'records',info['memory_tokens'],'tokens',flush=True)
        except Exception as error:
            info={'status':'FAIL','error':type(error).__name__+': '+str(error)}
            if k in {x.stem for x in largest}:receipt['largest_cfd_first'][k]=info
            if k in set.union(*folds.values()):receipt['errors'][k]=info['error']
            print(k,info,flush=True)
    required=set.union(*folds.values())
    receipt['status']='QUALIFIED_CPU' if set(cohort)==required else 'NOT_READY'
    write(a.out/'graph_audit.json',receipt)
    if set(cohort)!=required:raise RuntimeError('Original 31+2 fallback has graph blockers; no subset or GMT training permitted')
    # Snapshot source-qualified tensor inputs for independent audit/reproduction.
    torch.save({'cohort':cohort,'packs':{k:{n:v for n,v in p.items() if n not in ('static','node_states')} for k,p in packs.items()}},a.out/'qualified_cpu_inputs.pt')
    if not a.train_gmt:return
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Selected GPU unavailable')
    checkpoint,state=prep.checkpoint_from_train_only({k:cohort[k] for k in folds['train']},a.out,600,123,a.device)
    if set(state['train_kernels']+state['internal_dev_kernels'])!=folds['train']:
        raise ValueError('GMT training/selection family coverage differs')
    if {family(k) for k in state['train_kernels']} & {family(k) for k in state['internal_dev_kernels']}:
        raise ValueError('GMT internal development split is not family-disjoint')
    export=prep.build_banks(cohort,packs,hashes,checkpoint,a.out,a.bank/'train_sft.jsonl',ROOT/'mailohls_runs/mailohls_final_family_split_s123.json',123)
    receipt.update(status='GMT_EXPORTED',gmt_checkpoint_sha256=sha(checkpoint),gmt_updates=600,
                   train_kernels=state['train_kernels'],internal_dev_kernels=state['internal_dev_kernels'],
                   gmt_best_step=state['best_step'],export=export)
    write(a.out/'PREPARATION.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('/home/elvouvali/MailoHLS'))
    p.add_argument('--train-gmt',action='store_true');p.add_argument('--device',default='cuda')
    main(p.parse_args())
