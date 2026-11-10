"""Run exactly two qualified matched arms when the V100 is stably idle."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiments.paper_gnn.data import ROOT,sha
from experiments.paper_gnn.train import write_json

GPU='GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,PYTHONHASHSEED='123',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',SALT_CORRECTED_COHORT=str(root/'cohort/corrected_cohort_v1'),SALT_PRELIMINARY_LATENCY=str(root.parent.parent/'paper_gnn_20261110/latency/selected.pt'))
    assert json.loads((root/'input_compatibility_probe/input_checks.json').read_text())['status']=='passed'
    assert json.loads((root/'input_compatibility_probe/xattn_cpu_check.json').read_text())['status']=='passed'
    frozen=json.loads((root/'cohort/candidate_pool_manifest.json').read_text())
    assert sha(root/'cohort/candidate_pool.jsonl')==frozen['candidate_pool_sha256']
    journal=[]
    def note(event,**kw):
        journal.append({'event':event,'time_utc':datetime.now(timezone.utc).isoformat(),**kw});write_json(journal,root/'execution_journal.json')
    def idle_wait(phase):
        quiet=None
        while True:
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
            occupied=[line for line in output.splitlines() if line.split(',')[0].strip()==GPU]
            if occupied:quiet=None
            elif quiet is None:quiet=time.monotonic()
            ready=quiet is not None and time.monotonic()-quiet>=90
            write_json({'status':'stable_idle_gpu' if ready else 'waiting_for_stably_idle_gpu','phase':phase,'time_utc':datetime.now(timezone.utc).isoformat(),'gpu_uuid':GPU,'other_compute_processes':occupied,'quiet_seconds':0 if quiet is None else time.monotonic()-quiet,'required_quiet_seconds':90,'no_processes_killed':True,'trained_final_arms':[arm for arm in ('A','B') if (root/arm/'completion.json').exists()]},root/'resource_status.json')
            if ready:return
            time.sleep(15)
    for arm in ('A','B'):
        while not (root/arm/'completion.json').exists():
            idle_wait('train_'+arm)
            command=[sys.executable,'-u','-m','experiments.paper_gnn_latency_final.train_arm','--arm',arm,'--objective','PARETO_LATENCY','--seed','123','--expected-gpu','V100','--output',str(root/arm)]
            note('training_start',arm=arm,command=command,wrapper_sha256=sha(Path(__file__).parent/'train_arm.py'))
            log=root/(arm+'.log')
            with log.open('x') as f:proc=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if proc.returncode==0:break
            text=log.read_text()
            if 'Selected GPU has another compute process; refusing launch' in text and not (root/arm/'last.pt').exists():
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                preserved=root/'blocked_zero_step_attempts'/(arm+'_'+stamp);preserved.mkdir(parents=True)
                shutil.move(log,preserved/log.name)
                for name in ('gpu_receipt.json','data_identity.json'):
                    if (root/arm/name).exists():shutil.copy2(root/arm/name,preserved/name)
                note('launch_safely_refused_before_optimizer',arm=arm,receipt=str(preserved));continue
            note('training_failed',arm=arm,returncode=proc.returncode);raise RuntimeError('Training stopped; preserve states and inspect '+str(log))
        note('training_export_completed',arm=arm,completion=json.loads((root/arm/'completion.json').read_text()))
        # Full registered-action coverage is exercised on CPU while GPU is free
        # for the next arm; this uses the actual maintained XAttn module.
        command=[sys.executable,'-u','-m','experiments.paper_gnn_latency_final.check_bank_cpu','--bank',str(root/arm/'memory'),'--output',str(root/arm/'xattn_cpu_check.json')]
        cpuenv=env.copy();cpuenv['CUDA_VISIBLE_DEVICES']=''
        with (root/(arm+'_xattn_cpu.log')).open('x') as f:subprocess.run(command,cwd=ROOT,env=cpuenv,stdout=f,stderr=subprocess.STDOUT,check=True)
        note('full_source_action_xattn_passed',arm=arm)
    idle_wait('final_evaluation_export')
    with (root/'finalize.log').open('x') as f:subprocess.run([sys.executable,'-u','-m','experiments.paper_gnn_latency_final.finalize','--root',str(root)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    note('study_complete',completion=json.loads((root/'completion.json').read_text()))

if __name__=='__main__':main()
