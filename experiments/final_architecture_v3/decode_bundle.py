"""Checksum-verified portable continuation; sequential real constrained decoding.
The bundle contains immutable checkpoints and frozen requests, never a backbone.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from experiments.final_architecture_20261109.run import preflight,EXPECTED

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True);p.add_argument('--memory-dir',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--max-gpu-seconds',type=int,default=3600);a=p.parse_args()
 manifest=json.loads((a.bundle/'manifest.json').read_text())
 for name,digest in manifest['files'].items():
  if sha(a.bundle/name)!=digest:raise ValueError('Transfer checksum mismatch: '+name)
 if sha(a.stage1/'adapter_model.safetensors')!=EXPECTED:raise ValueError('Wrong frozen V4 parent')
 mem=json.loads((a.memory_dir/'memory_manifest.json').read_text())
 if mem['gnn_checkpoint_sha256']!='ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162':raise ValueError('Wrong GNN')
 for label in ('stage2','sft','dpo'):
  contract=json.loads((a.bundle/label/'training_contract.json').read_text())
  if contract['structural']['memory_manifest_sha256']!=sha(a.memory_dir/'memory_manifest.json'):raise ValueError('Memory manifest differs')
 print('Four models, identical nine requests, mean RHS production decoding; max GPU seconds',a.max_gpu_seconds,flush=True)
 if not a.execute:return
 a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic();previous_charge=sum(json.loads(f.read_text()).get('runtime_seconds',json.loads(f.read_text()).get('runtime_seconds_total_so_far',0)) for f in a.out.glob('*_receipt.json'))
 for label,stage,ckpt in [('stage3','stage3',a.bundle/'dpo'),('stage2','stage2',a.bundle/'stage2'),('sft','stage2',a.bundle/'sft'),('stage1','stage1',a.stage1)]:
  target=a.out/(label+'.jsonl');receipt=a.out/(label+'_receipt.json')
  if receipt.exists():
   saved=json.loads(receipt.read_text());assert sha(target)==saved['output_sha256'];continue
  if target.exists():raise RuntimeError('Partial output preserved; inspect before explicit recovery')
  seconds=a.max_gpu_seconds-previous_charge-(time.monotonic()-started)
  if seconds<=0:raise TimeoutError('Reserved GPU time exhausted')
  # NVML utilization can lag after the prior child releases its context. Wait, never bypass or preempt.
  for retry in range(16):
   try:check=preflight(a.gpu_uuid);break
   except RuntimeError:
    if retry==15:raise
    time.sleep(2)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid,TOKENIZERS_PARALLELISM='false');arm_started=time.monotonic()
  cmd=[sys.executable,'-m','LLM_branch.inference.eval_stage1_stage2_stage3','--stage',stage,'--adapter_dir',str(ckpt),'--directive_domain_registry_json',str(ckpt/'directive_domain_registry.json'),'--input_jsonl',str(a.bundle/'requests.jsonl'),'--output_jsonl',str(target),'--score_reduction','mean','--debug_topk','8','--num_samples','1']
  if stage!='stage1':cmd+=['--memory_dir',str(a.memory_dir)]
  with (a.out/(label+'.log')).open('a') as log:subprocess.run(cmd,env=env,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=seconds)
  if len(target.read_text().splitlines())!=manifest['source_requests']:raise ValueError('Incomplete decoding')
  receipt.write_text(json.dumps({'preflight':check,'command':cmd,'output_sha256':sha(target),'request_sha256':sha(a.bundle/'requests.jsonl'),'runtime_seconds':time.monotonic()-arm_started,'runtime_seconds_total_so_far':previous_charge+time.monotonic()-started,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'qualification':'Research only; no physical improvement implied'},indent=2)+'\n')
  print('[DONE]',label,flush=True)
if __name__=='__main__':main()
