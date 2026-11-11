"""Reproducible CLI. Missing compute produces an explicit incomplete receipt."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch
from experiments.gemv_structural_explainability.audit import (
    ROOT,GEMV,GNN,GNN_SHA,STAGE1,STAGE1_SHA,authenticate,load_encoder,
    export_pack,coverage,sha,write_json)

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT/'artifacts/gemv_structural_explainability/raw'


def preflight(out):
    receipt = dict(host=socket.gethostname(),utc=__import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(),python=sys.executable,
        torch=torch.__version__,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gnn_available=GNN.is_file(),stage1_available=(STAGE1/'adapter_model.safetensors').is_file())
    for name,args in [('gpu_inventory',['--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu']),
                      ('gpu_processes',['--query-compute-apps=gpu_uuid,pid,process_name,used_memory'])]:
        proc = subprocess.run(['nvidia-smi',*args,'--format=csv'],capture_output=True,text=True)
        receipt[name] = dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
    if receipt['stage1_available']:
        receipt['stage1_sha256'] = sha(STAGE1/'adapter_model.safetensors')
        if receipt['stage1_sha256']!=STAGE1_SHA: raise ValueError('Stage1 identity changed')
    receipt['learned_alignment_status']='NOT_STARTED'
    write_json(out/'preflight.json',receipt)
    return receipt


def write_bank(raw_dir,kernel,pack,manifest):
    raw_dir.mkdir(parents=True,exist_ok=True)
    path = raw_dir/(kernel+'.pt')
    if path.exists(): raise FileExistsError('Refusing overwrite: '+str(path))
    torch.save(pack,path)
    manifest['packs'][kernel] = dict(file=path.name,sha256=sha(path),graph_identity=pack['graph_identity'])
    # The production loader sees authenticated action anchors. Dynamic evidence
    # and summaries are computed by integration.py, never detached/cached.
    bank = raw_dir/'memory';bank.mkdir(exist_ok=True)
    loader = dict(schema='salt-gemv-alignment-anchor-bank-v1',representation_schema=manifest['representation_schema'],
        node_embs=pack['action'][0],node_embs_mask=pack['active'][0],max_slots=64,gnn_dim=128,
        disable_pragma_injection=True,slot_ids=torch.arange(1,65),
        action_relation_mask=torch.eye(64,dtype=torch.bool)&pack['active'][0,:,None]&pack['active'][0,None,:],
        gnn_checkpoint_sha256=GNN_SHA)
    torch.save(loader,bank/(kernel+'.memory.pt'))


def export(args):
    from experiments.gemv_structural_explainability.explain import embedding_metrics
    started = time.monotonic()
    torch.set_num_threads(4)
    graph,relations,actions = authenticate()
    model,schema,identity = load_encoder(args.gnn)
    pack,first,jk = export_pack(graph,model,schema)
    manifest = dict(schema='salt-gemv-alignment-raw-bank-v1',gnn_sha256=GNN_SHA,
        representation_schema=model.schema,edge_dim=pack['edge_dim'],packs={},
        encoder=identity,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        graph_file_sha256=sha(GEMV/'gemv.canonical.json'),
        source_sha256=graph.graph['provenance']['source_sha256'],
        learned_alignment_status='NOT_TRAINED',physical_qor_improvement='NOT EVALUATED')
    write_bank(args.raw_dir,'gemv',pack,manifest)
    if args.graphs_dir:
        if not args.cohort_audit: raise ValueError('--graphs-dir requires --cohort-audit')
        from GNN_branch.mlir_graph_v2 import read_graph
        audit = json.loads(args.cohort_audit.read_text())
        kernels = set(audit['one_kernel_per_training_family'].values())|set(audit['splits']['validation']['kernels'])
        split = json.loads((ROOT/'mailohls_runs/mailohls_final_family_split_s123.json').read_text())
        if 'gemv' in kernels or not kernels <= set(split['train_kernels']+split['val_kernels']):
            raise ValueError('GEMV/locked-test contamination in alignment cohort')
        manifest['cohort_audit_sha256'] = sha(args.cohort_audit)
        manifest['split_sha256'] = sha(ROOT/'mailohls_runs/mailohls_final_family_split_s123.json')
        for kernel in sorted(kernels):
            path = args.graphs_dir/(kernel+'.json')
            g = read_graph(path)
            if g.graph['graph_identity']!=audit['inventory'][kernel]['graph']['graph_identity']:
                raise ValueError('Cohort graph identity mismatch')
            p,_,_ = export_pack(g,model,schema)
            write_bank(args.raw_dir,kernel,p,manifest)
    write_json(args.raw_dir/'manifest.json',manifest)
    write_json(args.raw_dir/'memory/memory_manifest.json',manifest)
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'action_provenance.json',actions)
    write_json(args.out/'evidence_catalog.json',dict(records=pack['records'],coverage=coverage(pack),
        status_counts=__import__('collections').Counter(r['status'] for r in pack['records']),
        interpretation='Accessibility only; no learned retrieval measured'))
    metrics = embedding_metrics(pack,first,jk)
    write_json(args.out/'embedding_metrics.json',metrics)
    import numpy as np
    np.savez_compressed(args.out/'action_embeddings.npz',action=pack['action'][0].numpy(),
        active=pack['active'][0].numpy(),first=first.numpy(),jk=jk.numpy(),
        active_vectors=pack['action'][0,pack['active'][0]].numpy(),
        slot_ids=np.arange(64),active_slot_ids=np.flatnonzero(pack['active'][0].numpy()),
        active_action_ids=np.array([r['action'] for r in actions]))
    torch.save(dict(action_memory=pack['action'][0],valid_mask=pack['active'][0],
        slot_ids=torch.arange(64),gnn_sha256=GNN_SHA,graph_identity=graph.graph['graph_identity'],
        tensor_schema=schema,layout=['first_layer:0:64','jk_max:64:128']),args.out/'action_memory.pt')
    with (args.out/'active_action_embeddings.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['action','slot','construct','source_file','source_line',*['d'+str(i) for i in range(128)]])
        for record in actions:
            writer.writerow([record['action'],record['slot'],record['construct'],record['source_path'],record['line'],
                             *pack['action'][0,record['slot']].tolist()])
    manifest.update(export_seconds=time.monotonic()-started,raw_manifest_sha256=sha(args.raw_dir/'manifest.json'),
        raw_dir=str(args.raw_dir),nodes=66,edges=106,authenticated_actions=5,
        evidence_records=len(pack['records']),coverage=coverage(pack),
        source_only_actions=0,unresolved_actions=0,raw_checkpoint_bytes=Path(args.gnn).stat().st_size)
    write_json(args.out/'manifest.json',manifest)
    print(json.dumps(dict(exported=len(manifest['packs']),evidence=len(pack['records']),
                          results=str(args.out)),indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','export','figures'))
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--raw-dir',type=Path,default=DEFAULT_RAW)
    p.add_argument('--gnn',type=Path,default=GNN)
    p.add_argument('--graphs-dir',type=Path)
    p.add_argument('--cohort-audit',type=Path)
    args = p.parse_args()
    if args.stage=='preflight': preflight(args.out)
    elif args.stage=='export': export(args)
    elif args.stage=='figures':
        from experiments.gemv_structural_explainability.visualize import generate
        generate(args.out,HERE/'figures')


if __name__=='__main__': main()
