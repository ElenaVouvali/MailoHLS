"""Verify evaluation/export/XAttn from an immutable intermediate checkpoint.

This does not select checkpoints or perform optimizer updates. It lets the
latency pipeline gate be exercised while the declared training run continues.
"""
import argparse
import json
from pathlib import Path
import torch
from experiments.paper_gnn.data import PaperData, sha
from experiments.paper_gnn.train import backend, export, gpu_receipt
from experiments.paper_gnn.compatibility import check


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-gpu',default='V100')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    receipt=gpu_receipt(args.expected_gpu)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    data=PaperData(args.out);Model,_=backend()
    gpu_receipt(args.expected_gpu)
    first=next(iter(data.tensors.values()));model=Model(first.x.shape[1],first.edge_attr.shape[1]).cuda()
    if state['schema']!=data.schema:raise ValueError('Checkpoint tensor schema differs')
    model.load_state_dict(state['model'],strict=True)
    export(model,data,args.out,args.checkpoint,state['protocol']['objective'],state['protocol'])
    result=check(args.out/'memory',args.out/'xattn_check.json',args.expected_gpu)
    (args.out/'pipeline_receipt.json').write_text(json.dumps({'status':'intermediate_pipeline_passed',
        'checkpoint_sha256':sha(args.checkpoint),'epoch':state['epoch'],'updates':state['steps'],
        'gpu':receipt,'xattn_checks':len(result['checks']),'training_complete':False,
        'checkpoint_selection_changed':False},indent=2)+'\n')


if __name__=='__main__':main()
