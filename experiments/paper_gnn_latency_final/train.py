"""Paper protocol entry for the maintained train_GNN/Net stack (no sweeps)."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F
from GNN_branch.paper_protocol import OBJECTIVES, evaluate, objective_log2, ranking_loss, feasibility
from experiments.paper_gnn.data import ROOT, PaperData, fingerprint, sha


def backend():
    """Load the existing model with an isolated, explicit historical config."""
    sys.path.insert(0,str(ROOT/'GNN_branch'))
    saved=sys.argv
    sys.argv=['paper_gnn','--dataset','mlir','--num_layers','4','--D','64',
              '--target','perf','area','--experiment_name','paper_gnn_backend_v1']
    try:
        config=importlib.import_module('config')
        config.FLAGS.gae_T=False; config.FLAGS.gae_P=False
        config.FLAGS.num_layers=4; config.FLAGS.gnn_layer_after_MLP=1
        config.FLAGS.dropout=0.1; config.FLAGS.jkn_mode='max'
        config.FLAGS.graph_attention_heads=1; config.FLAGS.graph_layer_norm=False
        from model import PaperRC3Net
        from utils import set_reproducible_seed
    finally: sys.argv=saved
    return PaperRC3Net, set_reproducible_seed


def atomic_save(payload,path):
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary); os.replace(temporary,path)


def write_json(payload,path):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n'); os.replace(temporary,path)


def gpu_receipt(expected):
    import subprocess
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; refusing simulated training')
    name=torch.cuda.get_device_name(0)
    if expected not in name: raise RuntimeError('Expected '+expected+'; got '+name)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not uuid.startswith('GPU-'):
        raise RuntimeError('Select the GPU by its explicit preflight UUID')
    for line in processes.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if fields and fields[0]==uuid and int(fields[1])!=os.getpid():
            raise RuntimeError('Selected GPU has another compute process; refusing launch')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu','--format=csv'],text=True)
    return {'torch':torch.__version__,'cuda':torch.version.cuda,'device':name,
            'visible_devices':uuid,'inventory':inventory,'compute_processes':processes}


def development(model,data,objective,device):
    model.eval(); rows=[]
    with torch.no_grad():
        for key,records in sorted(data.groups.items()):
            if records[0]['split']!='val': continue
            graph=data.tensors[key[0]].to(device)
            _,static=model.encode(graph)
            predictions=[]
            for start in range(0,len(records),16):
                chunk=records[start:start+16]
                output=model.from_static(graph,torch.stack([r['candidate'] for r in chunk]).to(device),data.context(chunk[0],device),static)
                predictions.extend((output.cpu()*data.std+data.mean).tolist())
            for r,pred in zip(records,predictions):
                counts=[max(0.,math.expm1(min(80.,p))) for p in pred[2:]]
                for fraction in (1.,0.5):
                    rows.append({'split':'val','family':r['family'],'kernel':r['kernel'],
                        'device':r['device'],'clock':r['clock'],'budget_id':str(fraction),
                        'design_id':r['design_id'],'measured_success':True,'resources':r['resources'],
                        'predicted_resources':counts,'budget':[c*fraction for c in r['capacities']],
                        'log_latency':r['target'][0],'log_area':r['target'][1],
                        'predicted_log_latency':pred[0],'predicted_log_area':pred[1],
                        'timing_qualified':'unknown','measured_resource_precision':'rounded_percentages'})
    return evaluate(rows,objective),rows


def selection(report, policy='original'):
    if policy == 'ranking':
        from experiments.paper_gnn_latency_final.selection_policy import ranking_score
        return ranking_score(report)
    m=report['family_macro']
    return (m['feasible_selection_rate'] or 0.,
            -m['feasible_regret_at_1'] if m['feasible_regret_at_1'] is not None else -math.inf,
            m['kendall_tau_b'] if m['kendall_tau_b'] is not None else -math.inf,
            -m['log_qor_mae'] if m['log_qor_mae'] is not None else -math.inf)


def export(model,data,out,checkpoint,objective,protocol):
    from GNN_branch.action_point_relations import action_point_views
    model.eval(); raw={}
    with torch.no_grad():
        for kernel,graph in sorted(data.tensors.items()):
            raw[kernel]=model.memory(graph.to('cuda')).cpu()
            expected={int(label[1:])-1 for label in data.actions[kernel]}
            if expected!=set(graph.action_slots.cpu().tolist()): raise ValueError('Source action identity mismatch')
            probe=graph.clone(); probe.target_condition=torch.randn(1,5,device='cuda'); probe.candidates=torch.randn(64,8,device='cuda'); probe.measured_qor=torch.randn(6,device='cuda'); probe.resource_budget=torch.rand(4,device='cuda')
            if not torch.equal(raw[kernel],model.memory(probe).cpu()): raise ValueError('Static memory leakage: '+kernel)
    # Equal kernels when fitting normalization, exclusively training families.
    rms=[]
    for begin in (0,64):
        rms.append(math.sqrt(sum(float(raw[k][data.tensors[k].action_slots.cpu(),begin:begin+64].square().mean())
            for k in data.split['train_kernels'] if data.tensors[k].action_slots.numel()) /
            sum(bool(data.tensors[k].action_slots.numel()) for k in data.split['train_kernels'])))
    if min(rms)<=0 or not all(math.isfinite(x) for x in rms): raise ValueError('Invalid memory normalization')
    stats={'schema':'salt-paper-memory-normalization-v1','layout':['conv_1','jk_max'],
           'rms':rms,'fitted_kernels':data.split['train_kernels'],'split_sha256':data.identity['split_sha256'],
           'rule':'per-component train-kernel-balanced RMS then divide sqrt(2); no target or QoR statistics'}
    bank=out/'memory'; bank.mkdir(exist_ok=True)
    write_json(stats,bank/'normalization.json')
    manifest={'schema':'salt-paper-structural-memory-bank-v1','representation_schema':model.schema,
        'objective':objective,'gnn_checkpoint_sha256':sha(checkpoint),'feature_schema_sha256':fingerprint(data.schema),
        'protocol_sha256':fingerprint(protocol),'normalization_sha256':sha(bank/'normalization.json'),
        'layout':['conv_1:0:64','jk_max:64:128'],'gnn_dim':128,'files':{},
        'fusion_checkpoint_policy':'fresh objective-specific adapter required; no historical fusion weights accepted'}
    for kernel,memory in raw.items():
        graph=data.tensors[kernel]; active=torch.zeros(64,dtype=torch.bool); active[graph.action_slots.cpu()]=True
        normalized=memory.clone(); normalized[:,:64]/=rms[0]*math.sqrt(2); normalized[:,64:]/=rms[1]*math.sqrt(2)
        relations=action_point_views(data.graphs[kernel],max_slots=64)
        # Full relation metadata is preserved even for relations without a legacy bit.
        relation_mask=torch.tensor(relations['connectivity'],dtype=torch.bool) if 'connectivity' in relations else None
        if relation_mask is None:
            relation_mask=torch.eye(64,dtype=torch.bool)
        relation_mask &= active[:,None]&active[None,:]
        bound=torch.zeros(64,dtype=torch.bool)
        for node,a in data.graphs[kernel].nodes(data=True):
            if a['category']=='action_point' and any(e['relation_type']=='targets' for _,_,e in data.graphs[kernel].out_edges(node,data=True)):
                bound[int(a['action_id'][1:])-1]=True
        pack={'action_bound_mask':bound,'action_unbound_mask':active & ~bound,'schema':'salt-paper-action-memory-v1','representation_schema':model.schema,
            'node_embs':normalized,'node_embs_mask':active,'gnn_dim':128,
            'max_slots':64,'disable_pragma_injection':True,
            'embedding_mode':'static_pre_npt::conv1_jkmax_rms_v1',
            'action_relation_schema':'salt-paper-rc3-action-relations-v1',
            'labels':torch.tensor([i+1 if active[i] else -1 for i in range(64)]),
            'slot_ids':torch.arange(1,65),'action_relation_mask':relation_mask,
            'compiler_relations':relations,'graph_sha256':data.graph_hashes[kernel],
            'feature_schema_sha256':fingerprint(data.schema),'gnn_checkpoint_sha256':sha(checkpoint),
            'normalization_sha256':sha(bank/'normalization.json'),'objective':objective}
        path=bank/(kernel+'.memory.pt'); atomic_save(pack,path); manifest['files'][path.name]=sha(path)
    write_json(manifest,bank/'memory_manifest.json')
    return manifest


def run(args):
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume: raise ValueError('Existing run requires --resume')
    if args.resume_from and ((out/'last.pt').exists() or args.resume or args.selection_policy != 'ranking'):
        raise ValueError('Selection migration requires fresh output and ranking policy')
    receipt=gpu_receipt(args.expected_gpu); write_json(receipt,out/'gpu_receipt.json')
    data=PaperData(out)
    write_json(data.identity,out/'data_identity.json')
    Model,seed_all=backend(); seed_all(args.seed,False)
    from model import within_kernel_delta_pair_loss
    import hashlib
    def model_hash():
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    torch.set_num_threads(4)
    first=next(iter(data.tensors.values()))
    model=Model(first.x.shape[1],first.edge_attr.shape[1]).cuda()
    initial_hash=model_hash()
    model.eval()
    with torch.no_grad():
        graph=first.clone().cuda(); before=model.memory(graph)
        first_layer,jk=model.encode(graph)
        if not torch.equal(before[graph.action_slots],torch.cat((first_layer[graph.action_nodes],jk[graph.action_nodes]),-1)):
            raise ValueError('Memory layout differs from first-layer | JK')
        graph.target_condition=torch.randn(1,5,device='cuda')
        graph.candidates=torch.randn(64,8,device='cuda'); graph.measured_qor=torch.randn(6,device='cuda')
        graph.resource_budget=torch.rand(4,device='cuda')
        if not torch.equal(before,model.memory(graph)): raise ValueError('Candidate/target/label leakage')
        candidate=torch.zeros(2,64,8,device='cuda'); candidate[1,63,0]=1
        head_values=model(graph,candidate,torch.zeros(1,5,device='cuda'))
        singles=torch.cat([model(graph,c.unsqueeze(0),torch.zeros(1,5,device='cuda')) for c in candidate])
        if not torch.allclose(head_values,singles,atol=1e-6,rtol=1e-6): raise ValueError('Training/export inference prediction parity failed')
        if torch.equal(head_values[0],head_values[1]): raise ValueError('Complete candidate side channel is inactive')
    write_json({'schema':model.schema,'memory_dim':128,'static_layers':4,'post_directive_layers':1,
                'layout_exact':True,'batched_single_inference_parity':True,'extra_candidate_target_qor_budget_fields_ignored':True,
                'unbound_candidate_values_reach_prediction_head':True},out/'encoder_checks.json')
    protocol={'arm':args.arm,'initial_weights_sha256':initial_hash,'delta_policy':'unstandardized log2 latency; same kernel/device/clock; unique physical designs; SmoothL1 beta .5; 5 percent ties; epochs 1-3 zero, epoch4 .025, epoch5+ .05','objective':args.objective,'seed':args.seed,'schema':model.schema,
        'data_identity_sha256':fingerprint(data.identity),'epochs':40,'lr':3e-5,
        'loss':'0.5*(smooth_l1_latency_area + smooth_l1_selected_objective) + 0.1*smooth_l1_resource + 0.05*feasible_objective_ranking',
        'architecture':{'pre_directive_transformer_layers':4,'hidden_dim':64,'attention_heads':1,
            'jumping_knowledge':'max','memory_layout':['conv_1','jk_max'],'memory_dim':128,
            'post_directive_transformer_layers':1,'activation':'elu','target_context':'heads_only',
            'hardware_messages':False,'film':False},
        'relative_tie':0.05,'budget_views':[1.,0.5],'samples_per_kernel_per_epoch':128,'batch_candidates':16,
        'selection':'family_macro feasible_selection_rate DESC, conditional regret ASC, tau_b DESC, log2 MAE ASC',
        'patience':8,'ranking_weight':0.05,'resource_weight':0.1,'dropout':0.1,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'GNN_branch/model.py', ROOT/'GNN_branch/train_GNN.py',ROOT/'GNN_branch/paper_protocol.py',
             ROOT/'experiments/paper_gnn/data.py',ROOT/'GNN_branch/nn_att.py',ROOT/'GNN_branch/config.py',
             ROOT/'GNN_branch/utils.py',ROOT/'Preprocessing/create_jsonl.py',ROOT/'Preprocessing/data_preprocess.py',Path(__file__)]}}
    if args.selection_policy == 'ranking':
        from experiments.paper_gnn_latency_final.selection_policy import RANKING_POLICY
        protocol['selection'] = RANKING_POLICY
    write_json(protocol,out/'protocol.json')
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4,factor=0.5)
    start=0; best=None; stale=0; steps=0; curves=[]
    rng=random.Random(args.seed)
    if args.resume or args.resume_from:
        state=torch.load(args.resume_from or out/'last.pt',map_location='cpu',weights_only=False)
        if args.resume_from:
            from experiments.paper_gnn_latency_final.selection_policy import migrate_resume
            state=migrate_resume(state,protocol,args.resume_from,out)
        if state['protocol']!=protocol: raise ValueError('Resume protocol changed')
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng']); rng.setstate(state['sampler_rng'])
        start=state['epoch']+1; best=state['best']; stale=state['stale']; curves=state['curves']; steps=state['steps']
        if args.resume_from:
            # Re-evaluate the available revised best before continuing; preserve
            # all saved RNG states so this audit cannot change training exposure.
            rng_cpu=torch.get_rng_state(); rng_cuda=torch.cuda.get_rng_state_all()
            selected=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(selected['model'],strict=True)
            report,rows=development(model,data,args.objective,'cuda')
            write_json(report,out/'selected_metrics.json')
            with (out/'selected_predictions.jsonl').open('w') as f:
                for row in rows: f.write(json.dumps(row)+'\n')
            model.load_state_dict(state['model'],strict=True)
            torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state_all(rng_cuda)
            atomic_save(state,out/'last.pt')
        print(json.dumps({'status':'resumed','start_epoch_one_based':start+1,'preserved_optimizer_steps':steps,'selection':protocol['selection'],'stale':stale}),flush=True)
    train_groups=defaultdict(list)
    for key,records in data.groups.items():
        if records[0]['split']=='train': train_groups[key[0]].append(key)
    mean=data.mean.cuda(); std=data.std.cuda()
    for epoch in range(start,40):
        model.train(); totals=[0.,0.,0.,0.,0.]; pair_counts={'rank_full':0,'rank_half':0,'delta':0,'near_ties':0}; sampling=[]; gradients=[]; delta_weight=0.05*min(1.,max(0.,(epoch-2)/2)) if args.arm=='B' else 0.; kernels=sorted(train_groups); rng.shuffle(kernels)
        begun=time.time()
        for kernel in kernels:
            # Equal number of updates per independent source kernel. Device and
            # clock contexts rotate deterministically without budget replication.
            contexts=train_groups[kernel]
            for update in range(8):
                key=contexts[(epoch*8+update)%len(contexts)]; pool=data.groups[key]
                indices=rng.sample(range(len(pool)),min(16,len(pool))); records=[pool[i] for i in indices]
                graph=data.tensors[kernel].to('cuda')
                candidate=torch.stack([r['candidate'] for r in records]).cuda()
                y=torch.tensor([r['target'] for r in records],device='cuda')
                pred=model(graph,candidate,data.context(records[0],'cuda'))
                base_regression=F.smooth_l1_loss(pred[:,:2],((y-mean)/std)[:,:2],beta=0.5)
                resource=F.smooth_l1_loss(pred[:,2:],((y-mean)/std)[:,2:],beta=0.5)
                actual=pred*std+mean
                objective_pred=objective_log2(actual[:,0],actual[:,1],args.objective)
                truth=objective_log2(y[:,0],y[:,1],args.objective)
                # Specialize regression as well as ranking while retaining
                # both authentic latency/area measurements in the same heads.
                objective_std=std[0] if args.objective=='PARETO_LATENCY' else std[1] if args.objective=='PARETO_AREA' else data.adp_std.to('cuda')
                selected_regression=F.smooth_l1_loss(objective_pred/objective_std,truth/objective_std,beta=0.5)
                regression=(base_regression+selected_regression)/2
                ranks=[]
                for fraction in (1.,0.5):
                    budget=[c*fraction for c in records[0]['capacities']]
                    feasible=[feasibility(r['resources'],budget,measured_success=True) for r in records]
                    ranks.append(ranking_loss(objective_pred,truth,[key]*len(records),feasible,[r['design_id'] for r in records]))
                rank=torch.stack(ranks).mean()
                latency_delta=y[:,0,None]-y[None,:,0]
                upper=torch.triu(torch.ones_like(latency_delta,dtype=torch.bool),diagonal=1)
                valid_pairs=upper & (latency_delta.abs()>math.log2(1.05))
                pair_counts['delta']+=int(valid_pairs.sum())
                pair_counts['near_ties']+=int((upper & ~valid_pairs).sum())
                for name,fraction in (('rank_full',1.),('rank_half',.5)):
                    eligible=torch.tensor([feasibility(r['resources'],[c*fraction for c in r['capacities']],measured_success=True) is True for r in records],device='cuda')
                    pair_counts[name]+=int((valid_pairs & eligible[:,None] & eligible[None,:]).sum())
                delta=within_kernel_delta_pair_loss(actual[:,0],y[:,0],[str(key)]*len(records),tie_epsilon=math.log2(1.05),beta=.5) if valid_pairs.any() else actual.sum()*0
                if update==0:
                    norms={}
                    parameters=[p for p in model.parameters() if p.requires_grad]
                    for name,component in (('qor',regression),('resource',resource),('ranking',rank),('delta',delta)):
                        gs=torch.autograd.grad(component,parameters,retain_graph=True,allow_unused=True)
                        norms[name]=math.sqrt(sum(float(g.detach().square().sum()) for g in gs if g is not None))
                    gradients.append({'kernel':kernel,'context':key,'component_gradient_norms':norms})
                sampling.append({'kernel':kernel,'device':key[1],'clock':key[2],'distinct_configurations':len(records),'design_ids':[r['design_id'] for r in records]})
                loss=regression+0.1*resource+0.05*rank+delta_weight*delta
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                totals=[a+float(b.detach()) for a,b in zip(totals,(loss,regression,resource,rank,delta))]; steps+=1
        if pair_counts['rank_full']+pair_counts['rank_half']==0 or not any(g['component_gradient_norms']['ranking']>0 for g in gradients): raise RuntimeError('Ranking supervision failed qualification')
        if delta_weight>0 and (pair_counts['delta']==0 or not any(g['component_gradient_norms']['delta']>0 for g in gradients)): raise RuntimeError('Delta supervision failed qualification')
        write_json({'epoch':epoch,'delta_weight':delta_weight,'unique_kernels':len(kernels),'physical_contexts':len({(r['kernel'],r['device'],r['clock']) for r in sampling}),'pair_counts':pair_counts,'batches':sampling,'gradient_samples':gradients},out/('sampling_epoch_%02d.json'%epoch))
        report,rows=development(model,data,args.objective,'cuda'); score=selection(report,args.selection_policy)
        improved=best is None or score>tuple(best)
        if improved: best=score; stale=0
        else: stale+=1
        scheduler.step(report['family_macro']['log_qor_mae'])
        curves.append({'epoch':epoch,'steps':steps,'loss':totals[0]/(8*len(kernels)),
            'qor_loss':totals[1]/(8*len(kernels)),'resource_loss':totals[2]/(8*len(kernels)),
            'rank_loss':totals[3]/(8*len(kernels)),'delta_loss':totals[4]/(8*len(kernels)),'delta_weight':delta_weight,'valid_rank_pairs':pair_counts['rank_full']+pair_counts['rank_half'],'valid_delta_pairs':pair_counts['delta'],'seconds':time.time()-begun,**report['family_macro']})
        state={'protocol':protocol,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'sampler_rng':rng.getstate(),'epoch':epoch,'steps':steps,'best':best,'stale':stale,'curves':curves,
            'data_mean':data.mean,'data_std':data.std,'schema':data.schema}
        atomic_save(state,out/'last.pt'); write_json(curves,out/'curves.json')
        if improved:
            atomic_save(state,out/'selected.pt'); write_json(report,out/'selected_metrics.json')
            with (out/'selected_predictions.jsonl').open('w') as f:
                for row in rows: f.write(json.dumps(row)+'\n')
        print(json.dumps(curves[-1]),flush=True)
        if stale>=8: break
    selected=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'],strict=True)
    manifest=export(model,data,out,out/'selected.pt',args.objective,protocol)
    write_json({'status':'trained_exported','objective':args.objective,'checkpoint_sha256':sha(out/'selected.pt'),
        'selected_epoch':selected['epoch'],'updates':selected['steps'],'total_updates':steps,
        'memory_manifest_sha256':sha(out/'memory/memory_manifest.json'),'locked_test_evaluated':False,
        'downstream_superiority_established':False},out/'completion.json')
    return 0


def main():
    p=argparse.ArgumentParser(); p.add_argument('--objective',choices=OBJECTIVES,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=123)
    p.add_argument('--resume',action='store_true'); p.add_argument('--resume-from',type=Path)
    p.add_argument('--selection-policy',choices=['original','ranking'],default='original')
    p.add_argument('--arm',choices=['A','B'],required=True); p.add_argument('--expected-gpu',default='A30'); args=p.parse_args()
    return run(args)


if __name__=='__main__': raise SystemExit(main())
