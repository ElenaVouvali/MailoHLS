"""Re-export one frozen GMT using byte-exact original historical action caches.

No training and no old-file overwrite. Preserve fresh-inference receipts and
record their floating-point differences rather than calling allclose bit equality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import torch

from experiments.film_gmt_stage2_final.bank_contract import ROOT,sha,write
from experiments.historical_gmt_stage2_ab.prepare_banks import build_banks


def tensor_sha(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main(a):
    if a.out.exists():raise FileExistsError('Original-cache re-export must be fresh')
    source=a.prepared.resolve();cache=a.cache.resolve()
    identity=json.loads((source/'historical_identity.json').read_text())
    cache_manifest=cache/'memory_manifest.json'
    rel='artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned/memory_manifest.json'
    pinned=subprocess.check_output(['git','show',identity['source_revision']+':'+rel],cwd=ROOT)
    if hashlib.sha256(pinned).hexdigest()!=sha(cache_manifest):raise ValueError('Historical cache manifest differs from pinned revision')
    cm=json.loads(cache_manifest.read_text())
    if cm['gnn_checkpoint_sha256']!=identity['checkpoint_sha256'] or cm['gnn_contract_sha256']!=identity['contract_sha256']:
        raise ValueError('Original-cache GNN identity differs')
    if cm['multiscale']['normalization_stats_sha256']!=identity['normalization_sha256']:
        raise ValueError('Original-cache normalization differs')
    saved=torch.load(source/'qualified_cpu_inputs.pt',map_location='cpu',weights_only=True)
    cohort=saved['cohort'];packs=saved['packs']
    audit=json.loads((source/'graph_audit.json').read_text())
    receipt={'schema':'film-gmt-original-cache-anchor-reexport-v1','source_preparation':str(source),
        'source_preparation_sha256':sha(source/'PREPARATION.json'),
        'source_training_inputs_sha256':sha(source/'qualified_cpu_inputs.pt'),
        'cache_manifest':str(cache_manifest),'cache_manifest_sha256':sha(cache_manifest),
        'cache_manifest_pinned_revision':identity['source_revision'],
        'gmt_retrained':False,'kernels':{}}
    for k,d in cohort.items():
        file=cache/(k+'.memory.pt');p=torch.load(file,map_location='cpu',weights_only=True)
        original=p['node_embs']
        if original.shape!=(64,128) or original.dtype!=torch.float32:raise ValueError('Wrong original action matrix')
        if p['source_gexf_sha256']!=audit['kernels'][k]['graph_sha256'] or p['gnn_checkpoint_sha256']!=identity['checkpoint_sha256']:
            raise ValueError('Original cached vectors belong to another graph or checkpoint')
        if not p['disable_pragma_injection'] or p['embedding_mode']!='multiscale::jkn+centered_conv1':
            raise ValueError('Original cache is not the pinned pragma-free multiscale representation')
        if not torch.equal(p['node_embs_mask'],d['active']) or not torch.equal(p['action_relation_mask'],d['base_relation']):
            raise ValueError('Original action mask/routing differs')
        if not torch.isfinite(original).all() or not torch.allclose(original,d['anchors'],rtol=1e-5,atol=1e-6):
            raise ValueError('Original cache cannot be corroborated by frozen-GNN reconstruction')
        receipt['kernels'][k]={'original_file':str(file),'original_file_sha256':sha(file),
            'original_anchor_sha256':tensor_sha(original),'recomputed_anchor_sha256':tensor_sha(d['anchors']),
            'max_reconstruction_abs_delta':float((original-d['anchors']).abs().max()),
            'graph_sha256':p['source_gexf_sha256'],'reconstruction_tolerance':{'rtol':1e-5,'atol':1e-6}}
        d['anchors']=original.clone();packs[k]['node_embs']=original.clone()
        audit['kernels'][k]['recomputed_action_anchor_sha256']=audit['kernels'][k]['active_action_anchor_sha256']
        audit['kernels'][k]['active_action_anchor_sha256']=tensor_sha(original)
        audit['kernels'][k]['anchor_source']='original-pinned-historical-cache'
    a.out.mkdir(parents=True)
    for name in ['gmt_selected.pt','gmt_history.json','historical_identity.json']:shutil.copyfile(source/name,a.out/name)
    write(a.out/'graph_audit.json',dict(audit,anchor_reexport_receipt='original_cache_anchors.json'))
    write(a.out/'original_cache_anchors.json',receipt)
    hashes={k:e['graph_sha256'] for k,e in audit['kernels'].items()}
    info=build_banks(cohort,packs,hashes,a.out/'gmt_selected.pt',a.out,a.bank/'train_sft.jsonl',ROOT/'mailohls_runs/mailohls_final_family_split_s123.json',123)
    for k,e in receipt['kernels'].items():
        p=torch.load(a.out/'memory_B'/(k+'.memory.pt'),map_location='cpu',weights_only=True)
        if tensor_sha(p['node_embs'][:64])!=e['original_anchor_sha256']:
            raise ValueError('Original cached anchor bits were changed during export')
    preparation=json.loads((source/'PREPARATION.json').read_text())
    preparation.update(export=info,kernels=audit['kernels'],original_cache_anchors=receipt,
                       gmt_checkpoint_sha256=sha(a.out/'gmt_selected.pt'))
    write(a.out/'PREPARATION.json',preparation)
    print(json.dumps({'status':'EXACT_ORIGINAL_ANCHORS_EXPORTED','kernels':len(cohort),
                      'gmt_retrained':False,'gmt_checkpoint_sha256':sha(a.out/'gmt_selected.pt')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['prepared','cache','bank','out']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
