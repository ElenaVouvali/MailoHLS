"""CPU-only replay of the authenticated V1 implementation on mapped development candidates."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    hist=args.root/'historical_v1';run=hist/'run';provenance=run/'provenance'
    manifest=json.loads((hist/'inputs.json').read_text())
    for name,digest in manifest.items():
        if sha(args.root/name)!=digest:raise ValueError('Historical input changed '+name)
    contract=json.loads((run/'gnn_checkpoint_contract.json').read_text());sidecar=json.loads((run/'val_structural_rank_model_state_dict.pth.json').read_text())
    ckpt=run/'val_structural_rank_model_state_dict.pth'
    assert sha(ckpt)==sidecar['checkpoint_sha256']
    assert sha(run/'gnn_checkpoint_contract.json')==sidecar['contract_sha256']
    assert sha(provenance/'feature_schema.json')==contract['feature_schema']['sha256']
    for filename,rec in json.loads((provenance/'provenance_manifest.json').read_text())['files'].items():
        if (provenance/filename).is_file():assert sha(provenance/filename)==rec['sha256']
    cache_auth=json.loads((Path(__file__).parent/'historical_cache_authentication.json').read_text())
    for folder,field in [('points','design_point_tensor_tree'),('graphs','static_graph_tensor_tree')]:
        assert cache_auth[folder]['sha256']==contract['training_artifacts'][field]['sha256']
        for rec in cache_auth[folder]['files']:
            path=hist/folder/rec['file']
            if path.exists():assert sha(path)==rec['sha256']
    runtime=hist/'runtime';code=runtime/'GNN_branch';code.mkdir(parents=True,exist_ok=True)
    for name in ('config.py','model.py','utils.py','nn_att.py','saver.py','mlir_data.py','reference_delta.py'):
        shutil.copy2(provenance/name,code/name)
    sys.path.insert(0,str(code));sys.argv=['legacy_replay','--dataset','mlir','--num_layers','4','--D','64','--target','perf','area','--experiment_name','v1_latency_replay']
    import torch
    torch.set_num_threads(4)
    import config
    for name,value in contract['resolved_flags'].items():setattr(config.FLAGS,name,value)
    for name,value in contract['model_init_flags'].items():setattr(config.FLAGS,name,value)
    config.FLAGS.device='cpu';config.FLAGS.experiment_name='v1_latency_replay'
    import model as historical_model
    import mlir_data as historical_data
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import networkx as nx
    from Preprocessing.create_jsonl import build_examples,application_sources
    from experiments.paper_gnn.data import fingerprint,RESOURCE_ORDER
    from GNN_branch.target_graph_v22 import load_registry
    examples,_,_=build_examples(hist/'tables');by_row={(r['kernel_name'],r['preprocessed_row']):r for r in examples}
    model=historical_model.Net(contract['model_init']['in_channels'],edge_dim=contract['model_init']['edge_dim'],target_stats=contract['target_stats'],resource_stats=contract['resource_stats'])
    state=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(state,strict=True);model.eval()
    registry=load_registry();caps=[float(registry['profiles']['zcu104']['resources'][n]['value']) for n in RESOURCE_ORDER]
    sources=application_sources({k for k,i in by_row});checks=[];rows=[];epsilon=contract['resolved_flags']['epsilon']
    for graphpath in sorted((hist/'graphs').glob('*.pt')):
        kernel=graphpath.stem;graph=torch.load(graphpath,map_location='cpu',weights_only=False);points=torch.load(hist/'points'/graphpath.name,map_location='cpu',weights_only=False)
        if graph['graph_provenance']['source_sha256']!=sha(sources[kernel]):raise ValueError('Historical source identity differs')
        original=nx.read_gexf(hist/'gexf'/(kernel+'.gexf'));ordered,_=historical_data._node_id_order(original)
        samples=[];identities=[];rejected=0;scope_reconstruction_differences=0
        for i,key in enumerate(points['keys']):
            row=by_row.get((kernel,int(key.split('_')[1])))
            if row is None:raise ValueError('Unmapped cached row')
            if row['device']!='xczu7ev-ffvc1156-2-e' or row['clock_period']!=10.:raise ValueError('Wrong physical target')
            assignment={}
            for line in row['target'].splitlines():
                lhs,value=line.split('=',1);lhs=lhs.strip().upper()
                if kernel=='machsuite-sort-radix' and lhs.endswith('_L6}'):continue
                assignment[lhs]=value.strip().lower()
            point={k[5:-1]:(v if '_ARRAY_T_' in k else int(v)) for k,v in assignment.items()}
            flat=historical_data.point_to_ordered_values(point);di=int(points['directive_indices'][i]);saved=points['pragmas'][di].float()
            if not torch.equal(torch.tensor(flat+[0.]*(len(saved)-len(flat))),saved):raise ValueError('Cached directive vector mismatch '+kernel+':'+key)
            masks=historical_data.build_scope_masks_and_dynamic_pragmas(original,ordered,point)
            if not torch.equal(masks['X_pragma_per_node'].float(),points['X_pragma_per_node'][di].float()):scope_reconstruction_differences+=1  # Cached tensors themselves match the original complete training-tree hash; current GEXF need not reproduce the old ordering.
            measured=math.exp(math.log(2)*float(points['perf'][i]))-epsilon
            if not math.isclose(measured,row['latency'],rel_tol=2e-6,abs_tol=1e-6):raise ValueError('Cached measured latency mismatch')
            measured_res=[row[c] for c in ('bram_util_%','dsp_util_%','ff_util_%','lut_util_%')]
            if not torch.allclose(points['resource_util'][i].float(),torch.tensor(measured_res)/100,atol=1e-6,rtol=1e-6):raise ValueError('Cached resource values mismatch')
            kwargs={k:v.float() if torch.is_tensor(v) and k!='edge_index' else v for k,v in graph.items() if k=='x' or k=='edge_index' or k=='edge_attr' or k.startswith('X_')}
            kwargs.update(kernel_loss_weight=torch.ones(1),kernel=kernel,gname=kernel,graph_name=kernel,key=key,X_pragma_per_node=points['X_pragma_per_node'][di].float(),pragmas=points['pragmas'][di].float().unsqueeze(0),perf=points['perf'][i].view(1).float(),area=points['area'][i].view(1).float(),resource_util=points['resource_util'][i].view(1,4).float(),target_group=points['target_groups'][i],target_condition=points['target_condition'][i].view(1,5).float())
            samples.append(Data(**kwargs));identities.append((row,assignment))
        preds=[];label_invariance=False
        with torch.no_grad():
            for batch in DataLoader(samples,batch_size=8,shuffle=False):
                output=model(batch)[0]
                if not label_invariance:
                    changed=batch.clone();changed.perf.zero_();changed.area.zero_();changed.resource_util.zero_();check=model(changed)[0]
                    for name in ('perf','area','resource_log1p'):
                        if not torch.equal(output[name],check[name]):raise ValueError('Historical label leakage')
                    label_invariance=True
                for i in range(len(output['perf'])):
                    latency_ms=2**float(output['perf'][i])-epsilon
                    if latency_ms<=0:raise ValueError('Historical inverse transform produces nonpositive latency')
                    preds.append((math.log2(latency_ms),[max(0.,math.expm1(float(v))) for v in output['resource_log1p'][i]]))
        for (r,assignment),(prediction,res) in zip(identities,preds):
            for fraction in (1.,.5):
                rows.append({'split':'val','family':__import__('LLM_branch.common.mailohls_contract',fromlist=['family_id_from_kernel_name']).family_id_from_kernel_name(kernel),'kernel':kernel,'device':r['device'],'clock':10.,'budget_id':str(fraction),'design_id':fingerprint(assignment),'measured_success':True,'resources':[p*c/100 for p,c in zip([r[c] for c in ('bram_util_%','dsp_util_%','ff_util_%','lut_util_%')],caps)],'predicted_resources':[p*c for p,c in zip(res,caps)],'budget':[c*fraction for c in caps],'log_latency':math.log2(r['latency']),'log_area':math.log2(max(r['area'],.0625)),'predicted_log_latency':prediction,'predicted_log_area':0.,'timing_qualified':'unknown','measured_resource_precision':'rounded_percentages'})
        checks.append({'kernel':kernel,'candidates':len(samples),'exact_cached_flat_directives':True,'exact_training_tree_authenticated_cached_scope_inputs':True,'current_gexf_reconstruction_differences':scope_reconstruction_differences,'latency_resource_labels_authenticated':True,'label_invariance':label_invariance,'graph_sha256':sha(graphpath),'points_sha256':sha(hist/'points'/graphpath.name)})
        print(json.dumps(checks[-1]),flush=True)
    with (hist/'replayed_predictions.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    (hist/'replay_receipt.json').write_text(json.dumps({'status':'passed','device':'cpu','checkpoint_sha256':sha(ckpt),'contract_sha256':sha(run/'gnn_checkpoint_contract.json'),'source_commit':contract['git_commit'],'checks':checks,'target_policy':'single-reference ZCU104 10ns only; no claims for other targets','predicted_resource_units':'fractions from historical log1p resource heads times authenticated capacity','transform_correction':'historical predicts log2(ms+1e-6); inverse then log2(ms) for shared evaluation','locked_test_labels_read':False},indent=2)+'\n')

if __name__=='__main__':main()
