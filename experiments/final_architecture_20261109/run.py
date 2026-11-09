"""Portable guarded orchestration; preserves historical launcher defaults."""
import argparse,json,os,socket,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXPECTED='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'

def preflight(uuid,minimum=24000):
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
 rows=[x.split(', ') for x in inventory.strip().splitlines()]; gpu=next((x for x in rows if x[0]==uuid),None)
 if gpu is None:raise RuntimeError('Selected physical UUID not found')
 if any(x.split(',')[0].strip()==uuid for x in processes.splitlines()):raise RuntimeError('Selected GPU occupied: '+processes)
 if int(gpu[2])<minimum or int(gpu[3])>10:raise RuntimeError('Insufficient safe headroom: '+str(gpu))
 return {'host':socket.gethostname(),'uuid':uuid,'logical_cuda_device':0,'gpu':gpu,'processes':processes,'python':sys.executable}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','prepare','screen','baseline','static_smoke','static_train','train_finalists','synthesize','freeze_report']);p.add_argument('--execute',action='store_true');p.add_argument('--gpu-uuid');p.add_argument('--stage1',type=Path);p.add_argument('--data-dir',type=Path);p.add_argument('--memory-dir',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--steps',type=int,default=500);p.add_argument('--resume',type=Path);p.add_argument('--minimum-free-mib',type=int,default=24000);a=p.parse_args()
 if a.stage in ('audit','prepare'):
  cmd=[sys.executable,'-m','experiments.final_architecture_20261109.'+a.stage]
 elif a.stage=='screen':
  cmd=[sys.executable,'-m','pytest','-p','no:cacheprovider','experiments/v22_gnn_training/v4b/test_v4b.py','LLM_branch/tests/test_structural_xattn.py','experiments/final_architecture_20261109/tests','-q']
 elif a.stage=='baseline':
  if not all((a.stage1,a.data_dir,a.run_dir)):p.error('baseline needs --stage1 --data-dir --run-dir')
  if hashlib.sha256((a.stage1/'adapter_model.safetensors').read_bytes()).hexdigest()!=EXPECTED:raise ValueError('V4 adapter identity mismatch')
  cmd=[sys.executable,'-m','LLM_branch.inference.eval_stage1_stage2_stage3','--stage','stage1','--adapter_dir',str(a.stage1),'--directive_domain_registry_json',str(a.stage1/'directive_domain_registry.json'),'--input_jsonl',str(a.data_dir/'requests.jsonl'),'--output_jsonl',str(a.run_dir/'baseline.jsonl'),'--score_reduction','mean','--debug_topk','3','--num_samples','1']
 elif a.stage in ('static_smoke','static_train'):
  if not all((a.stage1,a.data_dir,a.memory_dir,a.run_dir)):p.error('static training needs --stage1 --data-dir --memory-dir --run-dir')
  if hashlib.sha256((a.stage1/'adapter_model.safetensors').read_bytes()).hexdigest()!=EXPECTED:raise ValueError('V4 adapter identity mismatch')
  for f in (a.data_dir/'stage2_pilot.jsonl',a.data_dir/'split.json',a.memory_dir/'memory_manifest.json'):
   if not f.is_file():raise FileNotFoundError(f)
  from experiments.rc3_stage2_multikernel.train import command
  smoke=a.stage=='static_smoke'
  cmd=command('smoke' if smoke else 'pilot',a.data_dir,a.memory_dir,a.run_dir)
  def setarg(flag,value):cmd[cmd.index(flag)+1]=str(value)
  setarg('--init_adapter_dir',a.stage1);setarg('--directive_domain_registry_json',a.stage1/'directive_domain_registry.json')
  cmd+=['--batch_size','1','--grad_accum','8'];setarg('--max_steps',1 if smoke else a.steps);setarg('--epochs',100)
  for flag in ('--eval_steps','--save_steps','--selection_eval_steps'):setarg(flag,1 if smoke else 50)
  setarg('--initial_state_reference',a.run_dir.parent/('initial_layer32.json' if smoke else 'initial_layers8_16_24_32.json'))
  if a.resume:cmd+=['--resume_from_checkpoint',str(a.resume)]
 elif a.stage=='train_finalists':
  p.error('BLOCKED: no convergence-qualified static baseline or qualified matched target cohort yet. See REPRODUCIBILITY.md; do not auto-launch architecture arms.')
 else:
  p.error('Use the reviewed commands in REPRODUCIBILITY.md; this stage has no executable qualified arm yet')
 print('COMMAND:',__import__('shlex').join(cmd))
 if not a.execute:return
 if a.stage=='baseline' and (a.run_dir/'baseline.jsonl').exists():raise FileExistsError('Baseline output already exists')
 if a.stage in ('static_smoke','static_train') and a.run_dir.exists() and any(a.run_dir.iterdir()) and not a.resume:raise FileExistsError('Nonempty training run needs explicit --resume')
 env=os.environ.copy()
 if a.stage in ('baseline','static_smoke','static_train'):
  if not a.gpu_uuid:p.error('GPU execution requires explicit --gpu-uuid from current preflight')
  receipt=preflight(a.gpu_uuid,a.minimum_free_mib);env['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
  smoke=subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'],env=env,check=True,capture_output=True,text=True)
  a.run_dir.mkdir(parents=True,exist_ok=True);(a.run_dir/('preflight_resume.json' if a.resume else 'preflight.json')).write_text(json.dumps({**receipt,'cuda_smoke':smoke.stdout},indent=2))
 subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
