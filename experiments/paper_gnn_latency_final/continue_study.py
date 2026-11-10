"""Complete stopped A export and run B once, preserving all optimizer work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from experiments.paper_gnn.data import ROOT,PaperData,sha
from experiments.paper_gnn.train import backend,write_json
from experiments.paper_gnn_latency_final.memory_export import export


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    os.environ.update(CUDA_VISIBLE_DEVICES='GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f',PYTHONHASHSEED='123',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',SALT_CORRECTED_COHORT=str(root/'cohort/corrected_cohort_v1'))
    journal=[]
    def note(event,**kw):
        journal.append(dict(event=event,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw));write_json(journal,root/'continuation_journal.json')
    if not (root/'A/completion.json').exists():
        # Recover export from frozen selected weights only. No optimizer steps.
        from experiments.paper_gnn.train import gpu_receipt
        write_json(gpu_receipt('V100'),root/'A/export_gpu_receipt.json')
        data=PaperData(root/'A');Model,_=backend();first=next(iter(data.tensors.values()));torch.set_num_threads(4)
        model=Model(first.x.shape[1],first.edge_attr.shape[1]).cuda()
        state=torch.load(root/'A/selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
        export(model,data,root/'A',root/'A/selected.pt','PARETO_LATENCY',state['protocol'])
        last=torch.load(root/'A/last.pt',map_location='cpu',weights_only=False)
        write_json({'status':'trained_exported','objective':'PARETO_LATENCY','checkpoint_sha256':sha(root/'A/selected.pt'),'selected_epoch':state['epoch'],'updates':state['steps'],'total_updates':last['steps'],'memory_manifest_sha256':sha(root/'A/memory/memory_manifest.json'),'locked_test_evaluated':False,'downstream_superiority_established':False,'export_only_recovery':True,'export_implementation_sha256':sha(Path(__file__).parent/'memory_export.py')},root/'A/completion.json')
        note('A_export_recovered_without_training',total_optimizer_steps=last['steps'])
        del model,data;torch.cuda.empty_cache()
    # Subsequent GPU subprocesses must not overlap this controller CUDA context.
    # Export/XAttn and B run in separate processes through the next phase.
    command=[sys.executable,'-u','-m','experiments.paper_gnn_latency_final.continue_study','--root',str(root),'--workers-only']
    os.execv(sys.executable,command)


def workers(root):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f',PYTHONHASHSEED='123',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',SALT_CORRECTED_COHORT=str(root/'cohort/corrected_cohort_v1'))
    def execute(command,log):
        with log.open('x') as f:subprocess.run([sys.executable,*command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for arm in ('A','B'):
        if arm=='B':
            execute(['-u','-m','experiments.paper_gnn_latency_final.train_arm','--arm','B','--objective','PARETO_LATENCY','--seed','123','--expected-gpu','V100','--output',str(root/'B')],root/'B.log')
        execute(['-m','experiments.paper_gnn.compatibility','--bank',str(root/arm/'memory'),'--output',str(root/arm/'xattn_check.json'),'--expected-gpu','V100'],root/(arm+'_xattn.log'))
    execute(['-u','-m','experiments.paper_gnn_latency_final.finalize','--root',str(root)],root/'finalize.log')

if __name__=='__main__':
    if '--workers-only' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers-only',action='store_true');a=p.parse_args();workers(a.root)
    else:main()
