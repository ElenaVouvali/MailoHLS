"""Evaluate the completed ranking-policy continuations, preserving original runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.paper_gnn.data import ROOT
from experiments.paper_gnn.train import write_json
from experiments.paper_gnn_latency_final.resume_ranking import GPU


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    original=parser.parse_args().root.resolve();root=original/'ranking_selection_v2'
    for name in ('cohort','evaluation_amendment','historical_v1','reference_preliminary'):
        path=root/name
        if not path.exists():path.symlink_to(original/name,target_is_directory=True)
    status=root/'evaluation_status.json'
    while not all((root/arm/'xattn_cpu_check.json').exists() for arm in ('A','B')):
        training=json.loads((root/'resource_status.json').read_text())
        if training['phase']=='training_failed':
            write_json(dict(status='blocked_by_training_failure',training=training),status)
            return 1
        write_json(dict(status='waiting_for_both_trained_arms'),status)
        time.sleep(15)
    while True:
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        occupied=[line for line in processes.splitlines() if line.split(',')[0].strip()==GPU]
        if not occupied:break
        write_json(dict(status='waiting_for_idle_gpu',other_compute_processes=occupied),status)
        time.sleep(15)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             PYTHONHASHSEED='123',SALT_PRELIMINARY_LATENCY=str(root/'reference_preliminary/selected.pt'))
    write_json(dict(status='evaluating'),status)
    with (root/'finalize.log').open('a') as log:
        proc=subprocess.run([sys.executable,'-u','-m','experiments.paper_gnn_latency_final.finalize',
                             '--root',str(root),'--selection-policy','ranking'],cwd=ROOT,env=env,
                            stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        write_json(dict(status='evaluation_failed',returncode=proc.returncode),status)
        return proc.returncode
    cpuenv=dict(env,CUDA_VISIBLE_DEVICES='')
    with (root/'selected_xattn_cpu.log').open('a') as log:
        proc=subprocess.run([sys.executable,'-u','-m','experiments.paper_gnn_latency_final.check_bank_cpu',
                             '--bank',str(root/'selected_export/memory'),
                             '--output',str(root/'selected_export/full_source_xattn_cpu_check.json')],
                            cwd=ROOT,env=cpuenv,stdout=log,stderr=subprocess.STDOUT)
    write_json(dict(status='completed' if proc.returncode==0 else 'full_source_xattn_failed',
                    returncode=proc.returncode),status)
    return proc.returncode


if __name__=='__main__':raise SystemExit(main())
