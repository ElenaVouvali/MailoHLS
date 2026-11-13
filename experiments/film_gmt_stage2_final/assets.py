"""Seal the explicit fallback bank and verify every exported memory identity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch

from experiments.film_gmt_stage2_final.bank_contract import ROOT,sha,write,ORIGINAL_BANK_SHA,PARENT_REGISTRY_SHA


def memory_receipt(folder,bank):
    folder=Path(folder);prep=folder.parent
    audit=json.loads((prep/'graph_audit.json').read_text())
    trained=json.loads((prep/'PREPARATION.json').read_text())
    manifest=json.loads((folder/'memory_manifest.json').read_text())
    if audit['errors'] or trained['gmt_updates']!=600:raise ValueError('Memory was not fully qualified and trained')
    state=torch.load(prep/'gmt_selected.pt',map_location='cpu',weights_only=True)
    if int(state['steps'])!=600:raise ValueError('GMT updates differ')
    selected=set(state['train_kernels']+state['internal_dev_kernels'])
    if selected!=set(audit['expected_train_kernels']):raise ValueError('GMT fit or selection used another cohort')
    from LLM_branch.common.mailohls_contract import family_id_from_kernel_name as family
    if {family(k) for k in selected} & {family(k) for k in audit['expected_val_kernels']}:
        raise ValueError('GMT validation-family leakage')
    protected={str(p.resolve()):sha(p) for p in [prep/'gmt_selected.pt',prep/'gmt_history.json',
        prep/'historical_identity.json',prep/'graph_audit.json',prep/'PREPARATION.json',folder/'memory_manifest.json']}
    anchor_receipt=prep/'original_cache_anchors.json'
    if anchor_receipt.exists():
        pinned=json.loads(anchor_receipt.read_text())
        protected[str(anchor_receipt.resolve())]=sha(anchor_receipt)
        protected[pinned['cache_manifest']]=pinned['cache_manifest_sha256']
        for entry in pinned['kernels'].values():protected[entry['original_file']]=entry['original_file_sha256']
    info={}
    for filename,digest in manifest['files'].items():
        file=folder/filename
        if sha(file)!=digest:raise ValueError('Memory hash mismatch: '+filename)
        protected[str(file.resolve())]=digest
        p=torch.load(file,map_location='cpu',weights_only=True)
        original_file=prep/'memory_A'/filename
        base=torch.load(original_file,map_location='cpu',weights_only=True)
        protected[str(original_file.resolve())]=sha(original_file)
        if not torch.equal(p['node_embs'][:64],base['node_embs']):raise ValueError('Historical 64 action anchors changed')
        if not torch.equal(p['action_relation_mask'][:64,:64],base['action_relation_mask']):raise ValueError('Original action routing changed')
        if not torch.isfinite(p['node_embs']).all():raise ValueError('Nonfinite GMT memory')
        k=filename.removesuffix('.memory.pt')
        raw=p['node_embs'][:64].contiguous().numpy().tobytes()
        if hashlib.sha256(raw).hexdigest()!=audit['kernels'][k]['active_action_anchor_sha256']:
            raise ValueError('Action anchors do not match authenticated graph inference')
        events=p['evidence_metadata']
        if len(events)!=p['node_embs'].shape[0]-64 or not events:raise ValueError('Empty or malformed evidence memory')
        counts={}
        for i,event in enumerate(events):
            a=int(event['action_slot']);g=event['group']
            if g not in ('memory','compute_dependence','control') or int(event['evidence_count'])<=0:
                raise ValueError('Invalid compiler role summary')
            if not bool(p['action_relation_mask'][a,64+i]):raise ValueError('Evidence is not routed to its action')
            counts[a]=counts.get(a,0)+1
        if max(counts.values())>3 or len(counts)!=int(base['node_embs_mask'].sum()):
            raise ValueError('Incomplete or repeated action/group summaries')
        info[k]={'source_sha256':audit['kernels'][k]['source_sha256'],
            'metadata_sha256':audit['kernels'][k]['metadata_sha256'],
            'graph_sha256':audit['kernels'][k]['graph_sha256'],
            'action_anchor_sha256':hashlib.sha256(raw).hexdigest(),
            'memory_sha256':digest,'evidence_records':audit['kernels'][k]['evidence_records'],
            'memory_tokens':len(p['node_embs']),'active_actions':len(counts)}
    return {'schema':'film-gmt-final-assets-v1','cohort':'original-film-31+2',
        'bank_manifest_sha256':sha(Path(bank)/'manifest.json'),
        'protected_files':protected,'kernels':info,'gmt_updates':600,'gmt_best_step':state['best_step'],
        'gmt_training_kernels':state['train_kernels'],'gmt_internal_dev_kernels':state['internal_dev_kernels'],
        'gnn_checkpoint_sha256':manifest['gnn_checkpoint_sha256'],
        'historical_action_anchors_bit_identical':True,'locked_test_measurements_read':False}


def verify_receipt(path,bank):
    obj=json.loads(Path(path).read_text())
    if obj['schema']!='film-gmt-final-assets-v1' or obj['bank_manifest_sha256']!=sha(Path(bank)/'manifest.json'):
        raise ValueError('Final memory/bank receipt differs')
    for file,digest in obj['protected_files'].items():
        if sha(file)!=digest:raise ValueError('Qualified structural artifact changed: '+file)
    identity_path=next(Path(p) for p in obj['protected_files'] if Path(p).name=='historical_identity.json')
    identity=json.loads(identity_path.read_text())
    if sha(identity['checkpoint_path'])!=identity['checkpoint_sha256']:
        raise ValueError('Historical GNN weights changed')
    for record in identity['authenticated_training_files'].values():
        if sha(record['path'])!=record['sha256']:raise ValueError('Historical GNN fitting artifact changed')
    assets=Path(identity['checkpoint_path']).parents[3]
    norm=assets/'artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_normalization_stats.json'
    if sha(norm)!=identity['normalization_sha256']:raise ValueError('Historical normalization changed')
    sources={e['kernel']:e for e in json.loads((ROOT/'Data/manifest.json').read_text())['kernels']}
    for k,record in obj['kernels'].items():
        if sha(assets/'GNN_branch/MLIR_graphs'/(k+'.gexf'))!=record['graph_sha256']:
            raise ValueError('Historical compiler graph changed: '+k)
        if sha(ROOT/sources[k]['files']['source']['path'])!=record['source_sha256']:
            raise ValueError('Historical memory source changed: '+k)
        if sha(ROOT/sources[k]['files']['metadata']['path'])!=record['metadata_sha256']:
            raise ValueError('Historical memory action metadata changed: '+k)
    return obj


def snapshot(a):
    if a.out.exists():raise FileExistsError('Fresh snapshot output required')
    if sha(a.bank/'manifest.json')!=ORIGINAL_BANK_SHA:raise ValueError('Wrong selected Stage-1 bank')
    cp=json.loads((a.film/'receipt.json').read_text());protocol=json.loads((a.film.parent/'protocol.json').read_text())
    if cp['arm']!='B' or cp['step']!=600 or cp['bank_manifest_sha256']!=ORIGINAL_BANK_SHA:
        raise ValueError('Wrong selected FiLM checkpoint')
    if sha(a.parent/'training_contract.json')!=protocol['parent_contract_sha256']:
        raise ValueError('Stage-1 parent training contract changed')
    if sha(a.parent/'directive_domain_registry.json')!=PARENT_REGISTRY_SHA:
        raise ValueError('Stage-1 parent domains changed')
    m=json.loads((a.bank/'manifest.json').read_text())
    old_manifest=ROOT/'Data/provenance/gnosis_17_recovery_20261113/before/canonical_manifest.json'
    if sha(old_manifest)!=m['inputs_sha256']['Data/manifest.json']:
        raise ValueError('Original Stage-1 canonical registry not authenticated')
    a.out.mkdir(parents=True)
    for name,digest in m['outputs_sha256'].items():
        if sha(a.bank/name)!=digest:raise ValueError('Original Stage-1 bank changed')
        shutil.copyfile(a.bank/name,a.out/name)
    shutil.copyfile(a.bank/'manifest.json',a.out/'manifest.json')
    old={e['kernel']:e for e in json.loads(old_manifest.read_text())['kernels']}
    for fold in ['train','val']:
        rows=[json.loads(line) for line in (a.out/(fold+'_sft.jsonl')).open()]
        for k in {e['row']['kernel_name'] for e in rows}:
            for record in old[k]['files'].values():
                if sha(ROOT/record['path'])!=record['sha256']:
                    raise ValueError('Selected Stage-1 source identity changed: '+k)
    result=memory_receipt(a.memory,a.out)
    write(a.out/'final_assets.json',result)
    print(json.dumps({'status':'FALLBACK_ASSETS_QUALIFIED','memory_files':len(result['kernels']),
                      'bank_manifest_sha256':sha(a.out/'manifest.json')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bank','parent','film','memory','out'):p.add_argument('--'+name,type=Path,required=True)
    snapshot(p.parse_args())
