"""Small authenticated CPU diagnostics; never calls production training or HLS."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch
from scipy.stats import spearmanr, kendalltau

from GNN_branch import mlir_graph_v2 as program, target_graph_v22 as overlay
from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch.action_point_relations import get_loop_semantics
from LLM_branch.common import directive_domains as domains
from LLM_branch.common.directive_domains import tcad
from experiments.refinement import configurations as configurations
from Preprocessing.data_preprocess import preprocess_kernel, load_action_definitions, UTILIZATION_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ('machsuite-stencil2d', 'machsuite-gemm-ncubed', 'machsuite-spmv-ellpack')
FAMILIES = ('machsuite_stencil2d', 'machsuite_gemm', 'machsuite_spmv_ellpack')
AUDIT_GRAPHS = ('spcl_example_03-fdea84258953.json', 'rodinia_lud_1_tiling_0-e35f0c4cd5d4.json',
                'serrano-kalman-filter-96b4d608969f.json')
DEVICE = {'xczu7ev-ffvc1156-2-e':'zcu104', 'xcu200-fsgd2104-2-e':'u200'}
SEED = 123
CANDIDATE_DIM = 8


def save(path, obj, immutable=False):
    text = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists() and path.read_text() != text:
        raise ValueError('Sealed experiment differs; use a fresh output directory: '+str(path))
    path.write_text(text)


def load_program(release, filename):
    # Uses exact external manifest membership and byte hash, not name similarity.
    return overlay.authenticated_input(Path(release)/'canonical'/filename)


def assess():
    old = json.loads((ROOT/'docs/target-aware-mlir-v22/final-verification.json').read_text())
    for name, expected in old['implementation']['source_sha256'].items():
        if overlay.file_hash(ROOT/'GNN_branch'/name) != expected: raise ValueError('Historical source receipt changed: '+name)
    archive = ROOT/'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
    if overlay.file_hash(archive) != old['archive_sha256']: raise ValueError('RC3 release changed')
    registry = overlay.load_registry()
    if overlay.fingerprint(registry) != old['registry_sha256']: raise ValueError('Historical profile registry changed')
    return {'source_receipts_match':True, 'archive_sha256':old['archive_sha256'],
            'registry_sha256':overlay.fingerprint(registry),
            'registry_file_sha256':overlay.file_hash(overlay.PROFILES),
            'historical_validation':'63/63 inherited; not repeated',
            'independent_human_semantic_review':'pending',
            'python':__import__('sys').version, 'torch':torch.__version__,
            'pyg':__import__('torch_geometric').__version__, 'cuda_available':torch.cuda.is_available()}


def audit(release, output):
    rows = []
    for filename in AUDIT_GRAPHS:
        base, auth = load_program(release,filename)
        compact, cauth = overlay.authenticated_input(Path(release)/'compact'/filename)
        previous = None
        for target, clock in [('zcu104',10),('u200',10),('u50',10),('zcu104',6.67)]:
            began = time.perf_counter()
            enriched = overlay.enrich(base,target,clock,authentication=auth)
            before = program.canonical_json(program.graph_payload(base))
            data = tensors.tensorize(enriched)
            other = tensors.tensorize(overlay.enrich(compact,target,clock,authentication=cauth))
            for key,value in data:
                if torch.is_tensor(value) and not torch.equal(value,other[key]): raise ValueError('Compact tensor differs: '+key)
            repeated = tensors.tensorize(enriched)
            for key,value in data:
                if torch.is_tensor(value) and not torch.equal(value,repeated[key]): raise ValueError('Nondeterministic tensor: '+key)
            compiler_mask = data.node_type < len(program.NODE_KINDS)
            if previous is not None:
                for key in ('x','node_type','op_type','value_type','root_type'):
                    if not torch.equal(data[key][compiler_mask],previous[key][compiler_mask]):
                        raise ValueError('Target changed compiler features: '+key)
            previous = data
            if before != program.canonical_json(program.graph_payload(base)): raise ValueError('Program mutated')
            rows.append({'graph':filename,'target':target,'clock_ns':clock,'nodes':data.num_nodes,
                'message_edges':data.num_edges,'negative_edges':data.negative_edge_index.shape[1],
                'node_dim':tensors.NODE_DIM,'edge_dim':tensors.EDGE_DIM,'tensor_bytes':tensors.tensor_bytes(data),
                'registered':int(data.registered_mask.sum()),'matched':int(data.matched_mask.sum()),
                'unresolved':int(data.unresolved_mask.sum()),'source_only':int(data.source_only_mask.sum()),
                'relation_types_present':sorted(set(tensors.RELATIONS[i] for i in data.edge_type.tolist())),
                'seconds':time.perf_counter()-began,'status':'pass'})
    result = {'schema':tensors.SCHEMA,'rows':rows,'node_categories':tensors.NODE_TYPES,
              'relation_categories':tensors.RELATIONS,'node_numeric':tensors.NUMERICAL,
              'edge_numeric':tensors.EDGE_NUMERICAL,'all_checks_passed':True}
    save(output/'tensor_audit.json',result)
    return result


def _distance(a,b):
    return float(torch.norm(a-b)/(torch.norm(a)+1e-12))


def smoke(release, output):
    torch.manual_seed(SEED)
    base, auth = load_program(release,AUDIT_GRAPHS[0])
    encoder = tensors.SmallGraphEncoder().eval()
    memories = []; ablated=[]; distances=[]
    contexts = [('zcu104',10),('u200',10),('u50',10),('zcu104',6.67)]
    for target,clock in contexts:
        data = tensors.tensorize(overlay.enrich(base,target,clock,authentication=auth))
        with torch.no_grad():
            memories.append(encoder.memories(data)[0])
            ablated.append(encoder.memories(tensors.without_hardware_messages(data))[0])
    for i,(target,clock) in enumerate(contexts[1:],1):
        assert _distance(memories[0],memories[i])>0
        assert torch.equal(ablated[0],ablated[i])
        distances.append({'target':target,'clock_ns':clock,'normalized_action_l2':_distance(memories[0],memories[i]),
                          'without_hardware_messages_l2':_distance(ablated[0],ablated[i])})
    # One-variable counterfactuals: no device-ID embedding or late conditioning.
    # Counterfactual uses explicit budget for an installed device in the actual graph;
    # capacity-only changes are made to derived tensor features, never to vendor facts.
    data=tensors.tensorize(overlay.enrich(base,'zcu104',10,authentication=auth))
    dsp_node=data.audit['node_ids'].index('hardware:resource:DSP')
    action_node=data.audit['node_ids'].index('action:L5')
    capacity_only=data.clone()
    at=2*tensors.NUMERICAL.index('physical_capacity')
    capacity_only.x[dsp_node,at]=math.log2(1+864)/24.
    budget=tensors.tensorize(overlay.enrich(base,'zcu104',10,budgets={'DSP':864},authentication=auth))
    with torch.no_grad():
        original=encoder(data)[action_node]
        cap_dist=_distance(original,encoder(capacity_only)[action_node])
        budget_dist=_distance(original,encoder(budget)[action_node])
    gradients=[]
    for connected in (True,False):
        d=data.clone() if connected else tensors.without_hardware_messages(data)
        d.x=d.x.detach().clone().requires_grad_(True)
        encoder.zero_grad()
        action=encoder(d)[action_node]
        loss=(action*torch.linspace(0.1,1.,128)).sum()
        loss.backward()
        hardware=d.node_type>=len(program.NODE_KINDS)
        assert torch.isfinite(d.x.grad).all()
        grads=[p.grad for p in encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        hw_norm=float(d.x.grad[hardware].abs().sum())
        assert hw_norm>0 if connected else hw_norm==0
        gradients.append({'hardware_edges':connected,'hardware_gradient_l1':hw_norm,
                          'dsp_capacity_gradient_abs':float(abs(d.x.grad[dsp_node,at])),
                          'model_gradients_finite':True})
    assert cap_dist>0 and budget_dist>0
    result={'shape':list(memories[0].shape),'context_distances':distances,
        'capacity_counterfactual':{'synthetic_tensor_only':True,'DSP_capacity':1728,'changed_to':864,'L5_l2':cap_dist},
        'budget_counterfactual':{'physical_DSP_unchanged':1728,'available_DSP':864,'L5_l2':budget_dist},
        'gradients':gradients,'hardware_edge_ablation_passed':True,
        'interpretation':'Untrained numerical information path only; not learned hardware reasoning'}
    save(output/'message_smoke.json',result);return result


def _candidate(row, mapping, actions, slots=64):
    values={}
    for column,definition in mapping.items():
        label=definition.action_id; action=actions[label]
        token=str(row.get(column,'')).lower().strip()
        parsed={kind:'0' for kind in domains.KINDS[action.kind]}
        if action.kind=='array': parsed['ARRAY_T']='none'
        if token:
            if action.kind=='loop' and token.startswith('pipeline_'):
                if token!='pipeline_1': raise ValueError('Unsupported requested II')
                parsed['PIPE']='1'
            elif action.kind=='loop' and token.startswith('unroll_'): parsed['UNROLL']=token.split('_')[1]
            elif action.kind=='array' and token.startswith('complete_'):
                parsed.update(ARRAY_T='complete',ARRAY_D=token.split('_')[1])
            elif action.kind=='array' and token.startswith(('cyclic_','block_')):
                p=token.split('_');parsed.update(ARRAY_T=p[0],ARRAY_F=p[1],ARRAY_D=p[2])
            else: raise ValueError('Unsupported measured token: '+token)
        for kind,value in parsed.items():values['AUTO{_'+kind+'_'+label+'}']=str(value)
    tcad.validate_assignments(values,actions)
    p=torch.zeros(slots,CANDIDATE_DIM)
    for label,action in actions.items():
        slot=int(label[1:])-1; get=lambda kind:values['AUTO{_'+kind+'_'+label+'}']
        if action.kind=='loop':p[slot,0]=float(get('PIPE'));p[slot,1]=math.log2(1+int(get('UNROLL')))/12
        else:
            p[slot,2+('none','complete','block','cyclic').index(get('ARRAY_T'))]=1
            p[slot,6]=math.log2(1+int(get('ARRAY_F')))/12;p[slot,7]=int(get('ARRAY_D'))/8
    return values,p


def prepare(release,output,per_context):
    split_path=ROOT/'mailohls_runs/mailohls_final_family_split_s123.json'
    split=json.loads(split_path.read_text())
    assert set(FAMILIES)<=set(split['train_families'])
    observations=[];provenance=[];graphs={}
    applications=pd.read_csv(ROOT/'Data/ApplicationInformation.csv').set_index('app_name')
    for kernel,family in zip(KERNELS,FAMILIES):
        source_dir=ROOT/'Data/ApplicationDataset'/kernel
        source=source_dir/applications.loc[kernel,'file_name'];metadata=source_dir/'kernel_info.txt'
        # Filename narrows inventory only; exact source, metadata and header hashes authenticate the match.
        found=[]
        for path in (Path(release)/'canonical').glob(kernel+'-*.json'):
            graph,auth=load_program(release,path.name)
            if graph.graph['provenance']['source_sha256']==overlay.file_hash(source) and graph.graph['provenance']['metadata_sha256']==overlay.file_hash(metadata):
                for name,expected in graph.graph['provenance']['local_header_sha256'].items():
                    assert overlay.file_hash(source_dir/name)==expected
                found.append((graph,auth,path.name))
        if len(found)!=1:raise ValueError('Ambiguous/missing authenticated measured-source graph: '+kernel)
        graph,auth,filename=found[0];graphs[kernel]=(graph,auth)
        mapping=load_action_definitions(kernel)
        actions=domains.load_source_actions(kernel,ROOT/'Data/ApplicationDataset')
        if {d.action_id for d in mapping.values()}!=set(actions):
            raise ValueError('Incomplete measured-action correspondence; do not invent unmapped directive values: '+kernel)
        for label,action in list(actions.items()):
            if action.kind=='loop': actions[label]=replace(action,trip_count=get_loop_semantics(graph,label)['compiler_proven_trip_count'])
        csv_path=ROOT/'Data/CSVS'/(kernel+'.csv')
        frame,stats=preprocess_kernel(csv_path,'gnn','xczu7ev-ffvc1156-2-e',10.,0.1,1.,all_targets=True)
        groups={};excluded=Counter()
        for _,row in frame.iterrows():
            if row['Device'] not in DEVICE or float(row['Clock_Period_nsec']) not in (3.33,5.,10.):continue
            try:values,candidate=_candidate(row,mapping,actions)
            except ValueError as exc:excluded[str(exc).split(':')[0]]+=1;continue
            target=DEVICE[row['Device']];clock=float(row['Clock_Period_nsec'])
            signature=overlay.fingerprint(values)
            key=(target,clock)
            record={'kernel':kernel,'family':family,'target':target,'device':row['Device'],'clock_ns':clock,
                'configuration':values,'configuration_sha256':signature,'version':row['Version'],
                'replicate_count':int(row['Replicate_Count']),
                'latency_ms':float(row['Latency_msec']),
                'resource_fractions':[float(row[col])/100 for col in UTILIZATION_COLUMNS],
                'partition':'diagnostic_5ns' if clock==5. else 'train'}
            record['design_point_sha256']=overlay.fingerprint([kernel,target,clock,signature])
            groups.setdefault(key,[]).append(record)
        for key,rows in sorted(groups.items()):
            observations.extend(sorted(rows,key=lambda r:hashlib.sha256((str(SEED)+r['configuration_sha256']).encode()).hexdigest())[:per_context])
        provenance.append({'kernel':kernel,'family':family,'split_membership':'train','graph':filename,
            'source_sha256':overlay.file_hash(source),'metadata_sha256':overlay.file_hash(metadata),
            'csv_sha256':overlay.file_hash(csv_path),
            'mapping_sha256':overlay.file_hash(ROOT/'Data/ApplicationAPLMapping'/(kernel+'.txt')),
            'preprocessing':stats,'tcad_exclusions':dict(excluded),
            'eligible_by_context':{str(k):len(v) for k,v in groups.items()}})
    points=[r['design_point_sha256'] for r in observations]
    assert len(set(points))==len(points)
    train={r['design_point_sha256'] for r in observations if r['partition']=='train'}
    test={r['design_point_sha256'] for r in observations if r['partition']!='train'}
    assert train.isdisjoint(test)
    protocol={'schema':tensors.SCHEMA,'seed':SEED,'kernels':KERNELS,'per_context':per_context,
              'family_split_sha256':overlay.file_hash(split_path),'provenance':provenance,
              'selection':'lowest SHA256(seed + canonical assignment hash), after fixed preprocessing and source-aware TCAD legality',
              'observations_sha256':overlay.fingerprint(observations),
              'clock_split':'3.33/10 train; 5 ns held out for final diagnostic; no model selection',
              'measurement_boundary':'Repository CSV/APL/source lineage verified; CSV rows lack per-design source/vendor-header authentication receipts.'}
    save(output/'data_protocol.json',protocol,True)
    save(output/'observations.json',observations,True)
    return observations,graphs,protocol


class CandidatePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph=tensors.SmallGraphEncoder()
        self.kind=nn.Embedding(3,4)
        self.local=nn.Sequential(nn.Linear(128+CANDIDATE_DIM+5,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(128+32+tensors.CONTEXT_DIM,64),nn.ReLU(),nn.Linear(64,5))

    def forward(self,batch,candidates,context_index,mode):
        memory,pooled=self.graph.memories(batch)
        memory=memory[context_index]
        kind=self.kind(batch.action_kind_slot[context_index])
        matched=batch.matched_mask[context_index].float().unsqueeze(-1)
        local=self.local(torch.cat([memory,candidates,kind,matched],dim=-1))
        registered=batch.registered_mask[context_index].float().unsqueeze(-1)
        local=(local*registered).sum(1)/registered.sum(1).clamp(min=1)
        context=batch.hardware_context[context_index] if mode=='numerical' else torch.zeros_like(batch.hardware_context[context_index])
        return self.head(torch.cat([pooled[context_index],local,context],dim=-1))


def tensors_for_rows(rows,graphs):
    contexts=sorted({(r['kernel'],r['target'],r['clock_ns']) for r in rows});data=[]
    for kernel,target,clock in contexts:
        graph,auth=graphs[kernel]
        data.append(tensors.tensorize(overlay.enrich(graph,target,clock,authentication=auth)))
    candidates=[]
    for r in rows:
        p=torch.zeros(64,CANDIDATE_DIM)
        # Candidate values were semantically validated before the sealed sampling protocol.
        for key,value in r['configuration'].items():
            kind,label=domains.DIRECTIVE_RE.fullmatch(key).group(2,3);s=int(label[1:])-1
            if kind=='PIPE':p[s,0]=float(value)
            elif kind=='UNROLL':p[s,1]=math.log2(1+int(value))/12
            elif kind=='ARRAY_T':p[s,2+('none','complete','block','cyclic').index(value)]=1
            elif kind=='ARRAY_F':p[s,6]=math.log2(1+int(value))/12
            elif kind=='ARRAY_D':p[s,7]=int(value)/8
        candidates.append(p)
    index=torch.tensor([contexts.index((r['kernel'],r['target'],r['clock_ns'])) for r in rows])
    y=torch.tensor([[math.log1p(r['latency_ms']*1000)]+r['resource_fractions'] for r in rows],dtype=torch.float32)
    return data,torch.stack(candidates),index,y


def metrics(rows,predicted):
    predicted=predicted.detach().cpu().numpy()
    true=np.array([[r['latency_ms']]+r['resource_fractions'] for r in rows])
    latency=np.expm1(predicted[:,0])/1000
    resources=predicted[:,1:]
    errors=np.abs(latency-true[:,0])/true[:,0]
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r['kernel'],r['target'],r['clock_ns']),[]).append(i)
    details=[]
    for key,indices in groups.items():
        a=true[indices,0];b=latency[indices]
        rho=float(spearmanr(a,b)[0]) if len(set(a))>1 and len(set(b))>1 else None
        tau=float(kendalltau(a,b)[0]) if len(set(a))>1 and len(set(b))>1 else None
        best=float(np.min(a));chosen=int(np.argmin(b))
        details.append({'kernel':key[0],'target':key[1],'clock_ns':key[2],'n':len(indices),
            'latency_mape_pct':float(np.mean(np.abs(b-a)/a)*100),'spearman':rho,'kendall':tau,
            'selected_candidate_latency_regret_pct':float((a[chosen]-best)/best*100)})
    return {'latency_mape_pct':float(errors.mean()*100),
        'resource_mae_percentage_points':dict(zip(('BRAM','DSP','FF','LUT'),(np.abs(resources-true[:,1:]).mean(0)*100).tolist())),
        'macro_context_spearman':float(np.mean([r['spearman'] for r in details if r['spearman'] is not None])),
        'macro_context_selected_regret_pct':float(np.mean([r['selected_candidate_latency_regret_pct'] for r in details])),
        'measured_full_capacity_feasible':len(rows),
        'predicted_resource_feasible':int(((resources>=0)&(resources<=1)).all(1).sum()),
        'negative_latency_predictions':int((latency<0).sum()),'contexts':details}


def train_pair(rows,graphs,steps,output,label):
    training=[r for r in rows if r['partition']=='train']
    diagnostic=[r for r in rows if r['partition']!='train']
    data,p,index,y=tensors_for_rows(training,graphs)
    mean=y.mean(0);std=y.std(0).clamp(min=1e-4); target=(y-mean)/std
    evaluation=tensors_for_rows(diagnostic,graphs) if diagnostic else None
    results={}
    for mode in ('numerical','enriched'):
        torch.manual_seed(SEED)
        model=CandidatePredictor()
        count=sum(p.numel() for p in model.parameters())
        batch=Batch.from_data_list([tensors.without_hardware_messages(d) if mode=='numerical' else d for d in data])
        optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
        curve=[];started=time.perf_counter();gradient_norms=None
        model.train()
        for step in range(steps+1):
            optimizer.zero_grad()
            prediction=model(batch,p,index,mode)
            loss=torch.mean((prediction-target)**2)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            curve.append({'step':step,'normalized_mse':float(loss.detach())})
            if step==steps:break
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():raise ValueError('Nonfinite gradient')
            if step==0:
                gradient_norms={name:sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
                    for name,params in [('graph',list(model.graph.parameters())),('candidate_head',list(model.local.parameters()))]}
                assert all(v>0 for v in gradient_norms.values())
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred=model(batch,p,index,mode)*std+mean
            training_metrics=metrics(training,pred)
            if evaluation:
                ed,ep,ei,ey=evaluation
                eb=Batch.from_data_list([tensors.without_hardware_messages(d) if mode=='numerical' else d for d in ed])
                final=metrics(diagnostic,model(eb,ep,ei,mode)*std+mean)
            else:final=None
        result={'parameters':count,'optimizer':'Adam lr=0.003; full batch; no scheduler/selection',
            'steps':steps,'initial_loss':curve[0]['normalized_mse'],'final_loss':curve[-1]['normalized_mse'],
            'loss_ratio':curve[-1]['normalized_mse']/curve[0]['normalized_mse'],
            'training':training_metrics,'held_out_5ns':final,'initial_gradients_l1':gradient_norms,
            'candidate_prediction_std_log_latency':float(pred[:,0].std()),
            'seconds':time.perf_counter()-started,'curve':curve}
        results[mode]=result
        save(output/(label+'_'+mode+'.json'),result)
    return results


def pilot(release,output,per_context,overfit_steps,train_steps):
    observations,graphs,protocol=prepare(release,output,per_context)
    training=[r for r in observations if r['partition']=='train']
    first=next(r for r in training if r['kernel']==KERNELS[0])
    tiny=[r for r in training if (r['kernel'],r['target'],r['clock_ns'])==(first['kernel'],first['target'],first['clock_ns'])][:8]
    if len(tiny)<4:raise ValueError('Insufficient authenticated tiny configurations')
    overfit=train_pair(tiny,graphs,overfit_steps,output,'overfit')
    passed=all(r['loss_ratio']<0.25 and r['training']['macro_context_spearman']>0.7 and r['candidate_prediction_std_log_latency']>1e-4 for r in overfit.values())
    transfer=train_pair(observations,graphs,train_steps,output,'transfer') if passed else None
    result={'observations':len(observations),'train':len(training),'diagnostic_5ns':len(observations)-len(training),
            'overfit_count':len(tiny),'overfit_passed':passed,'overfit':overfit,'transfer':transfer,
            'protocol':protocol,'interpretation':'Single-seed within-training-family clock interpolation; not held-out-kernel/device generalization'}
    save(output/'pilot.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','audit','smoke','pilot','all'),default='audit')
    parser.add_argument('--release-dir',default=str(ROOT/'experiments/target_graph_readiness/results/release'))
    parser.add_argument('--output-dir',default=str(ROOT/'experiments/target_graph_readiness/results/pilot_s123'))
    parser.add_argument('--per-context',type=int,default=16)
    parser.add_argument('--overfit-steps',type=int,default=250)
    parser.add_argument('--train-steps',type=int,default=180)
    args=parser.parse_args()
    if not 4<=args.per_context<=32 or not 1<=args.overfit_steps<=300 or not 1<=args.train_steps<=300:
        parser.error('Bounded pilot only: 4..32/configurations/context and 1..300 steps')
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    contract={'assessment':assess(),'seed':SEED,'per_context':args.per_context,'overfit_steps':args.overfit_steps,
        'train_steps':args.train_steps,'overfit_gate':'loss ratio <0.25, Spearman >0.7, nonconstant predictions',
        'architecture':'two TransformerConv layers, hidden 64, two averaged heads, 128-D action projection; shared candidate head; equal allocated parameters',
        'model_arms':{'numerical':'hardware messages disabled; numerical context enters prediction head',
                      'enriched':'hardware messages enabled; late context zeroed'},
        'code_sha256':{name:overlay.file_hash(ROOT/name) for name in ['GNN_branch/target_graph_tensor_v22.py','experiments/target_graph_readiness/experiment.py']}}
    save(output/'protocol.json',contract,True)
    if not (Path(args.release_dir)/'release.json').is_file():
        raise ValueError('Authenticate/extract the frozen archive with the documented project-Python command first')
    result={'assessment':contract['assessment']}
    if args.mode=='prepare':
        observations,_,data_protocol=prepare(args.release_dir,output,args.per_context)
        result['data']={'rows':len(observations),'protocol':data_protocol}
    if args.mode in ('audit','all'):result['tensorization']=audit(args.release_dir,output)
    if args.mode in ('smoke','all'):result['messages']=smoke(args.release_dir,output)
    if args.mode in ('pilot','all'):result['learning']=pilot(args.release_dir,output,args.per_context,args.overfit_steps,args.train_steps)
    save(output/(args.mode+'_results.json'),result)
    print(json.dumps({'mode':args.mode,'output':str(output),'status':'completed','learning_overfit_passed':result.get('learning',{}).get('overfit_passed')}))


if __name__=='__main__':main()
