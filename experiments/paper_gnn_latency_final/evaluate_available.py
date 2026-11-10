"""CPU matched evaluation of every currently available paper checkpoint."""
import argparse
import json
import os
from pathlib import Path
import torch
from experiments.paper_gnn.data import PaperData,sha,fingerprint
from experiments.paper_gnn.train import backend,write_json
from experiments.paper_gnn_latency_final.train import development
from experiments.paper_gnn_latency_final.evaluation_cohort import include_radix_ignoring_l6
from experiments.paper_gnn_latency_final.legacy_tensorize import tensorize as legacy_tensorize
from GNN_branch.paper_protocol import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'available_checkpoint_evaluation';out.mkdir(exist_ok=True)
    assert not torch.cuda.is_available();torch.set_num_threads(4);os.environ['SALT_CORRECTED_COHORT']=str(a.root/'cohort/corrected_cohort_v1')
    data=PaperData(out);amendment=include_radix_ignoring_l6(data,a.root/'cohort/corrected_cohort_v1');write_json(amendment,out/'evaluation_amendment.json');Model,_=backend();current=data.tensors
    preliminary=Path(os.environ.get('SALT_PRELIMINARY_LATENCY',str(a.root/'reference_preliminary/selected.pt' if (a.root/'reference_preliminary/selected.pt').exists() else a.root.parent/'paper_gnn_20261110/latency/selected.pt')))
    paths={'A':a.root/'A/selected.pt','preliminary':preliminary}
    if (a.root/'B/selected.pt').exists():paths['B']=a.root/'B/selected.pt'
    for name,path in paths.items():
        state=torch.load(path,map_location='cpu',weights_only=False);data.tensors={k:legacy_tensorize(g,state['schema']['vocabulary']) for k,g in data.graphs.items()} if name=='preliminary' else current;data.mean=state['data_mean'];data.std=state['data_std'];first=next(iter(data.tensors.values()));model=Model(first.x.shape[1],first.edge_attr.shape[1]);model.load_state_dict(state['model'],strict=True)
        report,rows=development(model,data,'PARETO_LATENCY','cpu')
        for row in rows:
            if row['kernel']=='machsuite-sort-radix':row['ignored_actions']=['L6']
        identities=sorted({(r['kernel'],r['device'],r['clock'],r['design_id']) for r in rows});report['candidate_identity_sha256']=fingerprint(identities);report['unique_observed_configurations']=len(identities);report['checkpoint_sha256']=sha(path)
        write_json(report,out/(name+'_all_contexts.json'))
        single=evaluate([r for r in rows if r['device']=='xczu7ev-ffvc1156-2-e' and r['clock']==10.],'PARETO_LATENCY');write_json(single,out/(name+'_zcu104_10ns.json'))
        with (out/(name+'_predictions.jsonl')).open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        print(json.dumps({'model':name,'checkpoint_sha256':sha(path),'unique_observed_configurations':len(identities),'metrics':report['family_macro'],'families':report['families']}),flush=True)
