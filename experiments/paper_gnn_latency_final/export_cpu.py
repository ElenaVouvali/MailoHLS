"""Export an already trained arm without requiring or allocating a GPU."""
import argparse
import json
import os
from pathlib import Path
import torch
from experiments.paper_gnn.data import PaperData,sha
from experiments.paper_gnn.train import backend,write_json
from experiments.paper_gnn_latency_final.memory_export import export

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',default='A');a=p.parse_args();out=a.root/a.arm
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty'
    os.environ['SALT_CORRECTED_COHORT']=str(a.root/'cohort/corrected_cohort_v1');torch.set_num_threads(4)
    data=PaperData(out);Model,_=backend();first=next(iter(data.tensors.values()));model=Model(first.x.shape[1],first.edge_attr.shape[1])
    state=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
    export(model,data,out,out/'selected.pt','PARETO_LATENCY',state['protocol'])
    last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    write_json({'status':'trained_exported','objective':'PARETO_LATENCY','checkpoint_sha256':sha(out/'selected.pt'),'selected_epoch':state['epoch'],'updates':state['steps'],'total_updates':last['steps'],'memory_manifest_sha256':sha(out/'memory/memory_manifest.json'),'locked_test_evaluated':False,'downstream_superiority_established':False,'export_only_recovery':True,'export_device':'cpu','export_implementation_sha256':sha(Path(__file__).parent/'memory_export.py'),'additional_training_steps':0},out/'completion.json')
