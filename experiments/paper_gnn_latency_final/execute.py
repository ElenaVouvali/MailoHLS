"""Run the two prespecified arms sequentially; no Stage-1 access or promotion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.paper_gnn.data import ROOT,sha
from experiments.paper_gnn.train import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--expected-gpu',required=True);a=p.parse_args()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,PYTHONHASHSEED='123',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',SALT_CORRECTED_COHORT=str(a.root/'cohort/corrected_cohort_v1'))
    frozen=json.loads((a.root/'cohort/candidate_pool_manifest.json').read_text())
    journal=[]
    def note(event,**kw):
        journal.append(dict(event=event,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw));write_json(journal,a.root/'execution_journal.json')
    for arm in ('A','B'):
        assert sha(a.root/'cohort/candidate_pool.jsonl')==frozen['candidate_pool_sha256']
        assert sha(a.root/'cohort/corrected_cohort_v1/preprocessing_manifest.json')==frozen['corrected_manifest_sha256']
        command=[sys.executable,'-u','-m','experiments.paper_gnn_latency_final.train','--arm',arm,'--objective','PARETO_LATENCY','--seed','123','--expected-gpu',a.expected_gpu,'--output',str(a.root/arm)]
        note('training_start',arm=arm,command=command,gpu_uuid=a.gpu_uuid)
        with (a.root/(arm+'.log')).open('x') as log:
            subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        note('training_export_complete',arm=arm,completion=json.loads((a.root/arm/'completion.json').read_text()))
        command=[sys.executable,'-m','experiments.paper_gnn.compatibility','--bank',str(a.root/arm/'memory'),'--output',str(a.root/arm/'xattn_check.json'),'--expected-gpu',a.expected_gpu]
        with (a.root/(arm+'_xattn.log')).open('x') as log:subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        note('xattn_passed',arm=arm)
    pa=json.loads((a.root/'A/protocol.json').read_text());pb=json.loads((a.root/'B/protocol.json').read_text())
    assert pa['initial_weights_sha256']==pb['initial_weights_sha256']
    assert pa['data_identity_sha256']==pb['data_identity_sha256']
    note('both_arms_completed',identical_fresh_initialization=True,identical_corrected_data=True)
    command=[sys.executable,'-u','-m','experiments.paper_gnn_latency_final.finalize','--root',str(a.root)]
    with (a.root/'finalize.log').open('x') as log:subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    note('final_evaluation_complete')

if __name__=='__main__':main()
