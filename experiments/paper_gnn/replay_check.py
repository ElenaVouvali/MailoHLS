"""Exact disposable optimizer/RNG replay; no evaluation labels or released updates."""
import argparse
import copy
import json
from pathlib import Path
import random
import torch
from GNN_branch import mlir_graph_v2 as compiler
from experiments.paper_gnn.data import tensorize, sha
from experiments.paper_gnn.train import backend, gpu_receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-gpu',default='V100');a=p.parse_args()
    receipt=gpu_receipt(a.expected_gpu);torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    graph=tensorize(compiler.read_graph(a.graph),state['schema']['vocabulary']).cuda()
    Model,seed_all=backend();seed_all(123,False)
    # Fixed synthetic inputs/loss exercise replay mechanics, not QoR accuracy.
    candidates=torch.zeros(2,64,8,device='cuda');candidates[1,0,0]=1
    context=torch.zeros(1,5,device='cuda');snapshots=[];losses=[];samples=[]
    for _ in range(2):
        model=Model(graph.x.shape[1],graph.edge_attr.shape[1]).cuda()
        model.load_state_dict(state['model'],strict=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-5,weight_decay=1e-4)
        # Adam's CPU step counters can alias a deserialized state dictionary.
        # Independent replay trials must each receive an untouched copy.
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        sampler=random.Random();sampler.setstate(state['sampler_rng']);samples.append(sampler.sample(range(32),8))
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=model(graph,candidates,context).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        snapshots.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()});losses.append(float(loss.detach()))
    identical=all(torch.equal(snapshots[0][k],snapshots[1][k]) for k in snapshots[0])
    if not identical or samples[0]!=samples[1] or losses[0]!=losses[1]:raise ValueError('Optimizer/RNG replay differs')
    a.output.write_text(json.dumps({'status':'passed','exact_parameter_replay':True,'sampler_replay':True,
        'checkpoint_sha256':sha(a.checkpoint),'graph_sha256':sha(a.graph),'gpu':receipt,
        'synthetic_loss':losses,'released_optimizer_updates':0,'evaluation_labels_used':False,
        'scope':'same-device replay; not a cross-GPU bitwise guarantee or QoR result'},indent=2)+'\n')


if __name__=='__main__':main()
