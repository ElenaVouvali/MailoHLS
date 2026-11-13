#!/usr/bin/env python3
"""Historical frozen-GNN action memory vs GMT summaries for *matched* Stage-2.

Must run from an authenticated MailoHLS worktree with historical GMT branch code.
No compiler graphs, source actions or measurements are fabricated. No test labels.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

ROOT = Path.cwd().resolve()


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jwrite(p: Path, obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,p)


def torchsave(p: Path, obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError(f'Refusing overwrite: {p}')
    tmp=p.with_suffix(p.suffix+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,p)


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def historical_runtime(assets_root):
    from experiments.historical_mlir_gmt_gemv import historical as hist
    hist.ASSETS=assets_root
    hist.CKPT=assets_root/hist.CKPT_REL/'val_structural_rank_model_state_dict.pth'
    if not hist.CKPT.is_file():
        raise FileNotFoundError('Historical epoch-9 checkpoint absent: '+str(hist.CKPT))
    runtime=hist.load()
    return hist,runtime


def choose_graph(kernel, graph_dir):
    # Historical compiler graph has a known, source-local filename; no arbitrary fuzzy matching.
    options=[graph_dir/(kernel+'.gexf'),graph_dir/(kernel.replace('_','-')+'.gexf')]
    found=list(dict.fromkeys(x.resolve() for x in options if x.is_file()))
    if len(found)!=1:
        raise FileNotFoundError(f'{kernel}: expected one exact historical compiler graph, got {found}')
    return found[0]


def validate_mapping(row, pack, mapping, hist, graph):
    src=Path(row['source_file'])
    if not src.is_absolute():src=(ROOT/src).resolve()
    if not src.is_file():raise FileNotFoundError(f'Authenticated source missing: {src}')
    got=digest(src)
    expected=pack['graph_provenance'].get('source_sha256')
    if got!=expected:
        raise ValueError(f'{row["kernel_name"]}: historical graph/source SHA mismatch: {got} != {expected}')
    if not pack['disable_pragma_injection']:raise ValueError('Candidate-conditioned historical memory')
    active=pack['node_embs_mask']
    if pack['node_embs'].shape!=(64,128) or active.shape!=(64,):
        raise ValueError('Wrong historical memory shape')
    mapped={a['action'] for a in mapping}
    from LLM_branch.common import directive_domains as domains
    registry=domains.tcad_site_domains(row['kernel_name'],row['input'],ROOT/'Data/ApplicationDataset')
    actual=set(registry.actions)
    if actual!=mapped:
        raise ValueError(f'Absolute registered source actions differ from historical graph: {sorted(actual)} vs {sorted(mapped)}')
    if len(mapped)!=int(active.sum()):raise ValueError('Missing historical action')
    if not torch.isfinite(pack['node_embs']).all():raise ValueError('Nonfinite historical memory')
    return got


def checkpoint_from_train_only(cohort, directory, steps, seed, gpu):
    from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import (
        HistoricalRoleGMT,loss_fn,device_data)
    if len(cohort)<5:raise RuntimeError('At least 5 authenticated TRAIN kernels required for GMT')
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=HistoricalRoleGMT().to(gpu)
    optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)
    # Internal train-only holdout by *source family*, never Stage-2 val/test.
    groups=defaultdict(list)
    from LLM_branch.common.mailohls_contract import family_id_from_kernel_name
    for k in cohort:groups[family_id_from_kernel_name(k)].append(k)
    fams=sorted(groups)
    if len(fams)<4:raise RuntimeError('Not enough TRAIN families for internal GMT validation')
    rng=random.Random(seed);rng.shuffle(fams)
    dev=set(groups[fams[0]])
    train=sorted(set(cohort)-dev)
    best=float('inf');selected=None;best_step=None
    hist=[]
    for step in range(1,steps+1):
        k=train[(step-1)%len(train)]
        model.train();optim.zero_grad(set_to_none=True)
        d=device_data(cohort[k],gpu)
        loss,parts=loss_fn(model,d,drop=.10)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite GMT auxiliary loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad) or float(grad)<=0:raise RuntimeError('GMT has no finite gradients')
        optim.step()
        if step%max(1,steps//20)==0 or step==1 or step==steps:
            model.eval()
            with torch.inference_mode():
                vals=[]
                for name in sorted(dev):
                    sample=device_data(cohort[name],gpu)
                    _,parts_val=loss_fn(model,sample,drop=0.)
                    vals.append(parts_val['reconstruction'])
            value=sum(vals)/len(vals)
            if value<best:
                best=value;best_step=step
                selected={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
            hist.append({'step':step,'train_kernel':k,'loss':float(loss),'grad_norm':float(grad),
                         'val_reconstruction_mse':value,**parts})
            print(f'[GMT] step={step} loss={float(loss):.6f} train-only-dev={value:.6f}',flush=True)
    if selected is None:raise RuntimeError('No GMT checkpoint selected')
    state={'schema':'mailohls-historical-gmt-train-only-v1','model':selected,
           'steps':steps,'best_step':best_step,'train_kernels':train,'internal_dev_kernels':sorted(dev),
           'seed':seed,'architecture':'HistoricalRoleGMT','base_gnn_frozen':True}
    save=directory/'gmt_selected.pt'
    torchsave(save,state)
    jwrite(directory/'gmt_history.json',hist)
    return save,state


def build_banks(cohort, packs, graph_hashes, checkpoint, out, dataset, split, seed):
    from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import HistoricalRoleGMT,device_data,GROUPS
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    gmt=HistoricalRoleGMT().eval()
    gmt.load_state_dict(saved['model'],strict=True)
    A=out/'memory_A';B=out/'memory_B'
    for directory in (A,B):directory.mkdir(exist_ok=False)
    hashes_A={};hashes_B={};info={}
    with torch.inference_mode():
        for k in sorted(cohort):
            d=cohort[k]
            p=packs[k]
            base=d['anchors'].float().contiguous()
            mask=d['active'].bool().contiguous()
            original_relation=d['base_relation'].bool().contiguous()
            if base.shape!=(64,128) or not torch.equal(base,p['node_embs'].float()):
                raise ValueError(f'{k}: original action vectors changed')
            if not torch.equal(mask,p['node_embs_mask']) or not torch.equal(original_relation,p['action_relation_mask']):
                raise ValueError(f'{k}: action identity/routing changed')
            common={'gnn_dim':128,'max_slots':64,'disable_pragma_injection':True,
                    'gnn_checkpoint_sha256':p['gnn_checkpoint_sha256'],
                    'action_relation_schema':'mailohls-action-relations-v1',
                    'historical_graph_sha256':graph_hashes[k],
                    'evidence_is_compiler_only':True}
            pack_A=dict(common,node_embs=base,node_embs_mask=mask,action_relation_mask=original_relation,
                        slot_ids=torch.arange(1,65),labels=[i+1 if mask[i] else -1 for i in range(64)],
                        memory_interface='action_slots_v1',embedding_mode='historical-original-multiscale')
            summary,present,_,att=gmt(device_data(d,'cpu'),return_attention=True)
            keys=base.clone().tolist()  # build exactly once; preserve every action row
            keymask=mask.tolist();slots=[]
            for pos,action_slot in enumerate(d['action_slots'].tolist()):
                for gi,name in enumerate(GROUPS):
                    if bool(present[pos,gi]):
                        keys.append(summary[pos,gi].float().tolist());keymask.append(True)
                        slots.append({'action_slot':int(action_slot),'group':name,
                                      'evidence_count':int(d['route'][pos,gi].sum())})
            key=torch.tensor(keys,dtype=torch.float32)
            valid=torch.tensor(keymask,dtype=torch.bool)
            if not torch.equal(key[:64],base):raise AssertionError('GMT modified original 64 actions')
            rel=torch.zeros(len(key),len(key),dtype=torch.bool)
            rel[:64,:64]=original_relation
            for idx,s in enumerate(slots):rel[s['action_slot'],64+idx]=True
            rel &=valid[:,None]&valid[None,:]
            # Existing dynamic-memory interface is also used for RC3 evidence,
            # but these packs have a separate authenticated HISTORICAL schema.
            # Only the *tensor interface*, not graph provenance, is shared.
            pack_B=dict(common,node_embs=key,node_embs_mask=valid,action_relation_mask=rel,
                        action_slot_mask=mask,action_slot_count=64,
                        memory_interface='rc3_action_evidence_c_v1',
                        schema='mailohls-historical-gmt-action-evidence-v1',
                        evidence_start_slot=64,evidence_metadata=slots,
                        embedding_mode='historical-original-plus-3-typed-gmt',
                        gmt_checkpoint_sha256=digest(checkpoint))
            fA=A/(k+'.memory.pt');fB=B/(k+'.memory.pt')
            torchsave(fA,pack_A);torchsave(fB,pack_B)
            hashes_A[fA.name]=digest(fA);hashes_B[fB.name]=digest(fB)
            info[k]={'action_count':int(mask.sum()),'additional_keys':int(len(key)-64),
                     'original_hash':digest(fA),'gmt_memory_hash':digest(fB),
                     'max_keys_total':int(len(key))}
    for directory,files,kind in [(A,hashes_A,'A_original'),(B,hashes_B,'B_GMT')]:
        jwrite(directory/'memory_manifest.json',{'schema':'mailohls-matched-historical-gmt-stage2-v1',
             'kind':kind,'action_relation_schema':'mailohls-action-relations-v1',
             'gnn_checkpoint_sha256':packs[next(iter(packs))]['gnn_checkpoint_sha256'],
             'gmt_checkpoint_sha256':digest(checkpoint) if kind=='B_GMT' else None,
             'dataset_sha256':digest(dataset),'split_sha256':digest(split),
             'files':files,'kernel_shapes':info})
    return info


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--split',type=Path,required=True)
    ap.add_argument('--graphs',type=Path,required=True)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--min-train-families',type=int,default=8)
    ap.add_argument('--preflight-only',action='store_true',help='Authenticate historical cohort on CPU without training')
    ap.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Run output exists. Never overwrite: '+str(args.out))
    args.out.mkdir(parents=True)
    if not args.preflight_only and args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable on the explicitly selected GPU')
    if args.steps<40:raise ValueError('At least 40 GMT warmstart steps required')
    torch.set_num_threads(4)
    hist,runtime=historical_runtime(args.assets.resolve())
    from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import make_evidence,cohort_action_mapping
    from LLM_branch.common.mailohls_contract import family_id_from_kernel_name
    dataset=read_jsonl(args.dataset)
    split=json.loads(args.split.read_text())
    ids_train=set(split['train_jsonl_idx']);ids_val=set(split['val_jsonl_idx'])
    if ids_train & ids_val or max(ids_train|ids_val,default=-1)>=len(dataset):
        raise RuntimeError('Stage-2 split invalid')
    by_k=defaultdict(list)
    for i,row in enumerate(dataset):
        if i not in ids_train|ids_val:continue
        by_k[row['kernel_name']].append((i,row))
    errors={};cohort={};packs={};hashes={};graph_audits={}
    for k,entries in sorted(by_k.items()):
        try:
            gp=choose_graph(k,args.graphs)
            graph,pack=hist.export_graph(gp,k,runtime)
            mapping=cohort_action_mapping(graph,pack)
            src_hash=validate_mapping(entries[0][1],pack,mapping,hist,graph)
            evid,meta=make_evidence(graph,pack,mapping,gp)
            if not any(bool(x) for x in evid['route'].flatten()):raise ValueError('No compiler evidence')
            for _,row in entries[1:]:
                src=Path(row['source_file']);src=src if src.is_absolute() else ROOT/src
                if digest(src)!=src_hash:raise ValueError('A kernel has multiple source identities')
            cohort[k]=evid;packs[k]=pack;hashes[k]=digest(gp)
            graph_audits[k]={'graph_sha256':hashes[k],'source_sha256':src_hash,
                'actions':int(evid['active'].sum()),'evidence':int(evid['route'].shape[-1]),
                'split':'train' if entries[0][0] in ids_train else 'val'}
        except Exception as e:
            errors[k]=f'{type(e).__name__}: {e}'
    train_k={k for k in cohort if by_k[k][0][0] in ids_train}
    val_k={k for k in cohort if by_k[k][0][0] in ids_val}
    train_f={family_id_from_kernel_name(k) for k in train_k}
    val_f={family_id_from_kernel_name(k) for k in val_k}
    if len(train_f)<args.min_train_families or len(val_f)<2:
        jwrite(args.out/'FAILED_COVERAGE.json',{'train_families':len(train_f),'val_families':len(val_f),
                                               'errors':errors,'available':graph_audits})
        raise RuntimeError('Insufficient *matched historical* train/val families; see FAILED_COVERAGE.json')
    if train_f&val_f:raise RuntimeError('Family leakage in Stage-2 split')
    # Historical GNN checkpoint itself may have seen validation kernels; reject.
    from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import find_train_names
    training_info=runtime[-1]['authenticated_training_files']['experiment_split.json']
    hist_split=json.loads(Path(training_info['path']).read_text())
    pretrained={str(x).lower().replace('-','_') for x in find_train_names(hist_split)}
    leakage=[k for k in val_k if k.lower().replace('-','_') in pretrained]
    pretrain_families={family_id_from_kernel_name(k) for k in pretrained}
    family_leakage=sorted(val_f & pretrain_families)
    if leakage or family_leakage:
        jwrite(args.out/'FAILED_LEAKAGE.json',{'historical_pretrained_overlap':leakage,
                                                'historical_pretrained_family_overlap':family_leakage})
        raise RuntimeError('Historical GNN has seen Stage-2 validation kernels/families; cannot claim independent generalization')
    if args.preflight_only:
        jwrite(args.out/'PREFLIGHT.json',{'status':'CPU_COHORT_AUTHENTICATED_NOT_TRAINED',
            'gnn_sha256':runtime[-1]['checkpoint_sha256'],'strict_load':runtime[-1]['strict_load'],
            'train_families':sorted(train_f),'validation_families':sorted(val_f),
            'graph_audits':graph_audits,'skipped_incompatible':errors})
        print('CPU historical cohort preflight passed',flush=True)
        return
    # Filter only to authenticated intersection. No fabricated/zero-mem substitutions.
    order=[i for i in sorted(ids_train) if dataset[i]['kernel_name'] in train_k]
    order +=[i for i in sorted(ids_val) if dataset[i]['kernel_name'] in val_k]
    filtered=[dataset[i] for i in order]
    dpath=args.out/'stage2_matched.jsonl';dpath.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in filtered))
    trcount=sum(i in ids_train for i in order)
    spath=args.out/'stage2_matched_split.json'
    jwrite(spath,dict(train_jsonl_idx=list(range(trcount)),
                     val_jsonl_idx=list(range(trcount,len(order))),test_jsonl_idx=[],seed=args.seed,
                     train_families=sorted(train_f),validation_families=sorted(val_f),test_families=[],
                     source_split='Unchanged original family identity; historical-source-matched subset'))
    ckpt,_=checkpoint_from_train_only({k:cohort[k] for k in train_k},args.out,args.steps,args.seed,args.device)
    # Save GPU memory by exporting frozen GMT keys on CPU.
    torch.cuda.empty_cache()
    info=build_banks(cohort,packs,hashes,ckpt,args.out,dpath,spath,args.seed)
    jwrite(args.out/'PREPARATION.json',{
        'status':'READY_FOR_MATCHED_STAGE2','gnn_sha256':runtime[-1]['checkpoint_sha256'],
        'gmt_sha256':digest(ckpt),'train_kernels':sorted(train_k),'validation_kernels':sorted(val_k),
        'train_families':sorted(train_f),'validation_families':sorted(val_f),
        'dataset_sha256':digest(dpath),'split_sha256':digest(spath),'skipped_incompatible':errors,
        'original_and_gmt_kernels':info,'graph_audits':graph_audits,
        'limitations':['GMT pretrained on compiler reconstruction, not HLS QoR','No joint GMT/decoder finetuning',
                       'SFT validation does not prove measured feasible regret','Pinned historical encoder may not be final production model']})
    print(json.dumps({'READY_FOR_MATCHED_STAGE2':True,'train_families':len(train_f),
                      'validation_families':len(val_f),'train_rows':trcount,'val_rows':len(filtered)-trcount,
                      'GMT_checkpoint':str(ckpt)},indent=2))

if __name__=='__main__':main()
