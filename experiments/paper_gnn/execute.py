"""Sequential latency -> area -> ADP execution with receipts and GPU gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.paper_gnn.data import ROOT

A30='GPU-ac0112df-7098-6c59-5c4f-a57fa666f808'
V100='GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f'


def idle(uuid):
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return not any(line.split(',')[0].strip()==uuid for line in output.splitlines())


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait-existing-latency',action='store_true'); args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    journal=[]
    def note(event):
        journal.append({'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event})
        (args.root/'execution_journal.json').write_text(json.dumps(journal,indent=2)+'\n')
    for index,(tag,objective) in enumerate(zip(('latency','area','adp'),('PARETO_LATENCY','PARETO_AREA','PARETO_ADP'))):
        out=args.root/tag
        if tag=='latency' and args.wait_existing_latency and not (out/'completion.json').exists():
            launch=json.loads((args.root/'latency_launch.json').read_text()); pid=launch['pid']
            while not (out/'completion.json').exists():
                try: os.kill(pid,0)
                except ProcessLookupError: raise RuntimeError('Latency worker stopped without completion; inspect latency.log')
                time.sleep(20)
        if not (out/'completion.json').exists():
            if not idle(A30): raise RuntimeError('A30 has another workload; refusing launch')
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=A30;env['PYTHONHASHSEED']=str(123+index)
            command=[sys.executable,'-u','GNN_branch/train_GNN.py','--paper-workflow',
                     '--objective',objective,'--output',str(out),'--seed',str(123+index)]
            if (out/'last.pt').exists():command.append('--resume')
            note({'event':'training_start','objective':objective,'command':command,'gpu_uuid':A30})
            with (args.root/(tag+'.log')).open('a') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        completion=json.loads((out/'completion.json').read_text())
        if completion['status']!='trained_exported': raise ValueError('Unqualified training completion')
        # Prefer idle V100 for real XAttn checks; A30 is an explicit fallback
        # for the compatibility gate if V100 has another owner's workload.
        check_uuid=V100 if idle(V100) else A30
        if not idle(check_uuid):raise RuntimeError('No idle GPU for XAttn checks')
        check_path=out/('xattn_v100.json' if check_uuid==V100 else 'xattn_a30.json')
        if not check_path.exists():
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=check_uuid
            command=[sys.executable,'-m','experiments.paper_gnn.compatibility','--bank',str(out/'memory'),
                     '--output',str(check_path),'--expected-gpu','V100' if check_uuid==V100 else 'A30']
            note({'event':'xattn_check_start','objective':objective,'command':command,'gpu_uuid':check_uuid})
            with (args.root/(tag+'_xattn.log')).open('a') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        note({'event':'objective_complete','objective':objective,'training':completion,'compatibility':str(check_path)})
    note({'event':'all_objectives_trained_exported_checked','locked_test_evaluated':False,
          'downstream_qor_improvement_established':False})


if __name__=='__main__':main()
