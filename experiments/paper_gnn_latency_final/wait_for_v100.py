"""Wait safely for the original study GPU; never terminate another workload."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.paper_gnn.train import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();uuid='GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f'
    while True:
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        occupied=[line for line in processes.splitlines() if line.split(',')[0].strip()==uuid]
        ready=(a.root/'A/completion.json').exists() and not occupied
        write_json({'status':'safe_gpu_available' if ready else 'waiting_for_idle_gpu','time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu_uuid':uuid,'other_processes':occupied,'A_export_ready':(a.root/'A/completion.json').exists(),'no_processes_killed':True,'B_training_completed':False},a.root/'resource_status.json')
        if ready:break
        time.sleep(30)
    os.execv(sys.executable,[sys.executable,'-u','-m','experiments.paper_gnn_latency_final.continue_study','--root',str(a.root),'--workers-only'])
