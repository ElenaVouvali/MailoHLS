"""Complete compiler-fact GMT pretraining on the user-selected A2.

This stage trains an informative compiler memory interface. It is explicitly
separate from downstream supervised decoder alignment and its causal evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiments.gemv_structural_explainability.memory_alignment import to_device
from experiments.gemv_structural_explainability.constants import TYPES
from experiments.gemv_structural_explainability.final_artifacts_refinement.model import SemanticMemoryAlignment,SCHEMA
from experiments.gemv_structural_explainability.final_artifacts_refinement.export import SHA,sha,dump
from experiments.gemv_structural_explainability.final_artifacts_refinement.facts import FACT_NAMES


def load_data(raw):
    manifest=json.loads((raw/'manifest.json').read_text())
    if manifest['checkpoint_sha256']!=SHA:raise ValueError('Refinement must use FINAL_ARTIFACTS GNN')
    packs={}
    for kernel,record in manifest['packs'].items():
        path=raw/record['file']
        if sha(path)!=record['sha256']:raise ValueError('Frozen bank changed')
        packs[kernel]=torch.load(path,map_location='cpu',weights_only=False)
    split=manifest['split']
    train=[k for k in split['train_kernels'] if k in packs]
    val=[k for k in split['val_kernels'] if k in packs]
    if set(train)&set(val) or 'gemv' in train+val or len(train)<10 or len(val)<3:
        raise ValueError('Invalid multi-kernel training/development separation')
    return manifest,packs,train,val


def fit_scales(model,packs,train):
    values=[[],[]];masks=[[],[]]
    for kernel in train:
        p=packs[kernel]
        for k in p['active'][0].nonzero().flatten().tolist():
            for role in range(2):
                selected=p['routes'][0,k]&p['valid'][0]&(p['is_edge'][0] if role else ~p['is_edge'][0])
                if not selected.any():continue
                t=model.targets(p,0,k,role)
                values[role].append(t['numeric']);masks[role].append(t['known'])
    for role in range(2):
        value=torch.stack(values[role]);mask=torch.stack(masks[role])
        mean=(value*mask).sum(0)/mask.sum(0).clamp_min(1)
        var=((value-mean).square()*mask).sum(0)/mask.sum(0).clamp_min(1)
        model.fact_mean[role]=mean;model.fact_scale[role]=var.sqrt().clamp_min(.05)


@torch.no_grad()
def evaluate(model,packs,kernels,device):
    model.eval();losses=[];tp=fp=fn=0;numeric_error=0.;numeric_count=0;roles=0
    per_kernel={}
    for kernel in kernels:
        out=model(to_device(packs[kernel],device))
        losses.append(float(out['auxiliary_loss']))
        ktp=kfp=kfn=0
        for d in out['diagnostics']:
            nt=len(TYPES);nf=len(FACT_NAMES)//2
            pred=d['prediction'][:nt].sigmoid()>=.5;gold=d['target']['presence'].bool()
            ktp+=int((pred&gold).sum());kfp+=int((pred&~gold).sum());kfn+=int((~pred&gold).sum())
            decoded=d['prediction'][2*nt:2*nt+nf]*model.fact_scale[d['role']]+model.fact_mean[d['role']]
            known=d['target']['known']
            numeric_error+=float(((decoded-d['target']['numeric']).abs()*known).sum())
            numeric_count+=int(known.sum());roles+=1
        tp+=ktp;fp+=kfp;fn+=kfn
        per_kernel[kernel]=dict(record_presence_f1=2*ktp/max(1,2*ktp+kfp+kfn),loss=losses[-1])
    return dict(loss=float(np.mean(losses)),record_presence_micro_f1=2*tp/max(1,2*tp+fp+fn),
        numeric_mean_log_feature_error=numeric_error/max(1,numeric_count),active_role_queries=roles,
        per_kernel=per_kernel)


def save_checkpoint(path,model,optimizer,scheduler,epoch,steps,best,stale,history,identity):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        epoch=epoch,steps=steps,best=best,stale=stale,history=history,identity=identity,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),temporary)
    temporary.replace(path)


def train(args):
    from experiments.final_architecture_20261109.run import preflight
    receipt=preflight(args.gpu_uuid,minimum=14000)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    manifest,packs,train_kernels,val_kernels=load_data(args.raw)
    # No allocation during CPU ingestion. Recheck ownership immediately before
    # claiming the selected A2, never preempting another workload.
    receipt=preflight(args.gpu_uuid,minimum=14000)
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise FileExistsError('Fresh run or explicit resume required')
    args.out.mkdir(parents=True,exist_ok=True)
    model=SemanticMemoryAlignment();fit_scales(model,packs,train_kernels)
    identity=dict(schema=SCHEMA,gnn_checkpoint_sha256=SHA,raw_manifest_sha256=sha(args.raw/'manifest.json'),
        train_kernels=train_kernels,val_kernels=val_kernels,seed=args.seed,
        epochs=args.epochs,patience=args.patience,lr=args.lr,batch_kernels=4,
        fact_names=FACT_NAMES,training_objective='record presence BCE + log count Huber + train-standardized numeric Huber + 0.25 knownness MSE + 0.05 role diversity hinge',
        fact_channel_denoising_probability=.15,gemv_fitted=False,gnn_retrained=False,
        decoder_alignment_complete=False,preflight=receipt)
    dump(args.out/'protocol.json',identity)
    device=torch.device('cuda:0');model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=.5,min_lr=1e-6)
    start=0;steps=0;best=float('inf');stale=0;history=[];gradient_rows=[]
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        for key in ('schema','gnn_checkpoint_sha256','raw_manifest_sha256','seed'):
            if state['identity'][key]!=identity[key]:raise ValueError('Resume identity mismatch: '+key)
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);start=state['epoch']+1;steps=state['steps']
        best=state['best'];stale=state['stale'];history=state['history']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
    else:dump(args.out/'initial_validation.json',evaluate(model,packs,val_kernels,device))
    started=time.monotonic()
    for epoch in range(start,args.epochs):
        model.train();order=list(train_kernels);random.shuffle(order);losses=[]
        for offset in range(0,len(order),4):
            batch=order[offset:offset+4];optimizer.zero_grad(set_to_none=True)
            loss=sum(model(to_device(packs[k],device))['auxiliary_loss'] for k in batch)/len(batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite compiler alignment loss')
            loss.backward()
            gradient_rows.append(dict(step=steps,query_l2=float(model.queries.grad.norm()),
                fact_projection_l2=float(model.fact_projection[0].weight.grad.norm()),
                edge_projection_l2=float(model.edge[0].weight.grad.norm())))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();steps+=1;losses.append(float(loss.detach()))
        report=evaluate(model,packs,val_kernels,device)
        improved=report['loss']<best-1e-4
        stale=0 if improved else stale+1;best=min(best,report['loss'])
        history.append(dict(epoch=epoch,steps=steps,train_loss=float(np.mean(losses)),
            validation=report,lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started))
        scheduler.step(report['loss'])
        save_checkpoint(args.out/'last.pt',model,optimizer,scheduler,epoch,steps,best,stale,history,identity)
        if improved:save_checkpoint(args.out/'selected.pt',model,optimizer,scheduler,epoch,steps,best,stale,history,identity)
        dump(args.out/'curves.json',history)
        print(json.dumps(history[-1]),flush=True)
        if stale>=args.patience:break
    with (args.out/'gradient_norms.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(gradient_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(gradient_rows)
    selected=torch.load(args.out/'selected.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'],strict=True)
    report=evaluate(model,packs,val_kernels,device)
    dump(args.out/'completion.json',dict(status='COMPILER_FACT_REFINEMENT_TRAINED',
        selected_checkpoint_sha256=sha(args.out/'selected.pt'),selected_epoch=selected['epoch'],
        selected_updates=selected['steps'],total_updates=steps,validation=report,
        seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        decoder_alignment_complete=False,gnn_checkpoint_sha256=SHA,
        qualification='Compiler-fact supervision only; LLM informativeness requires downstream alignment and intervention tests'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu-uuid',default='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--resume',type=Path);args=p.parse_args();train(args)


if __name__=='__main__':main()
