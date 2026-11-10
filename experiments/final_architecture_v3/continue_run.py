"""Sequential continuation: completed corrected baseline -> matched Stage3 -> real decoding.
Dry-run by default. Never starts a second training job while the baseline is active.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-log',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--memory-dir',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 print('Await complete baseline, then select its saved best checkpoint under frozen validation proxy; no HLS/test selection.',flush=True)
 if not a.execute:return
 a.out.mkdir(parents=True,exist_ok=True);journal=a.out/'orchestration.jsonl'
 def record(stage,status,**extra):
  with journal.open('a') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,'status':status,**extra})+'\n');f.flush();os.fsync(f.fileno())
 baseline_started=(a.baseline/'preflight.json').stat().st_mtime
 # Conservative inherited charge: verified runtimes plus one hour for earlier GPU loads/controls.
 prior_seconds=5777.3496+355.5889+3600+450+133
 deadline=baseline_started+8*3600-prior_seconds
 def remaining(cap):
  seconds=int(deadline-time.time())
  if seconds<=0:raise TimeoutError('Aggregate GPU budget exhausted, including auxiliary A2 and inherited reserve')
  return min(cap,seconds)
 started=time.monotonic();record('wait_baseline','RUNNING',aggregate_deadline_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(deadline)),prior_gpu_charge_seconds=prior_seconds)
 while True:
  log=a.baseline_log.read_text(errors='replace') if a.baseline_log.exists() else ''
  if '[DONE] Saved LoRA + STRUCTURAL xattn adapters to:' in log:break
  if 'Traceback (most recent call last)' in log:raise RuntimeError('Corrected baseline failed; no dependent experiment launched')
  if time.monotonic()-started>7200 or time.time()>=deadline:raise TimeoutError('Baseline completion wait cap reached')
  time.sleep(10)
 from .export_stage2 import export
 parent=export(a.baseline/'best_custom_stage1',a.baseline/'directive_domain_registry.json',a.out/'stage2_selected_export')
 for name in ('training_contract.json','structural_xattn.pt','adapter_model.safetensors'):
  if not (parent/name).is_file():raise FileNotFoundError(parent/name)
 record('baseline','PASS',parent=str(parent),parent_contract_sha256=sha(parent/'training_contract.json'),parent_fusion_sha256=sha(parent/'structural_xattn.pt'))
 # Allow the completed baseline process to release its context. No process is killed.
 time.sleep(10)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb',TOKENIZERS_PARALLELISM='false')
 pref=a.out/'preferences';cmd=[sys.executable,'-m','experiments.final_architecture_v3.train_preferences','--parent',str(parent),'--pairs',str(a.pairs),'--memory-dir',str(a.memory_dir),'--out',str(pref),'--steps','34']
 if not (pref/'result.json').exists():
  if pref.exists():
   checkpoints=[x for x in pref.glob('*/checkpoint-*') if (x/'progress.pt').exists()]
   if checkpoints:cmd+=['--resume',str(max(checkpoints,key=lambda x:({'sft':0,'dpo':1,'dpo_anchor':2}[x.parent.name],int(x.name.split('-')[-1]))))]
   elif any(pref.iterdir()):raise RuntimeError('Inspect incomplete preference load/cache before explicit recovery; no silent optimizer reset')
  record('stage3','RUNNING',command=cmd)
  with (a.out/'preferences.log').open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining(7200))
  record('stage3','PASS',result_sha256=sha(pref/'result.json'))
 # Same source/request cohort, all budgets full; no inherited U20050% ambiguity.
 requests=a.out/'requests.jsonl'
 if not requests.exists():
  rows=[json.loads(s) for s in a.pilot.read_text().splitlines()];selected=[]
  for kernel in ('machsuite-viterbi','rodinia_pathfinder_0_baseline_0'):
   for device,clock in (('xczu7ev-ffvc1156-2-e',10.),('xczu7ev-ffvc1156-2-e',5.),('xcu200-fsgd2104-2-e',10.)):
    hits=[r for r in rows if r['kernel_name']==kernel and r['device']==device and float(r['selected_clock_period'])==clock and float(r['resource_budget_frac'])==1.]
    if len(hits)!=1:raise ValueError('Ambiguous frozen inference request')
    row=dict(hits[0],context_id=f'{kernel}:{device}:{clock}');selected.append(row)
  requests.write_text(''.join(json.dumps(r)+'\n' for r in selected))
 results=json.loads((pref/'result.json').read_text());models={'stage2':('stage2',parent)}|{arm:('stage3' if arm.startswith('dpo') else 'stage2',Path(rec['checkpoint'])) for arm,rec in results['arms'].items()}
 for label,(stage,checkpoint) in models.items():
  output=a.out/(label+'_predictions.jsonl')
  if output.exists():
   if len(output.read_text().splitlines())!=6:raise RuntimeError('Partial decoding requires inspected resume; refusing overwrite')
   continue
  cmd=[sys.executable,'-m','LLM_branch.inference.eval_stage1_stage2_stage3','--stage',stage,'--adapter_dir',str(checkpoint),'--memory_dir',str(a.memory_dir),'--directive_domain_registry_json',str(checkpoint/'directive_domain_registry.json'),'--input_jsonl',str(requests),'--output_jsonl',str(output),'--score_reduction','mean','--debug_topk','3','--num_samples','1']
  from experiments.final_architecture_20261109.run import preflight
  preflight(env['CUDA_VISIBLE_DEVICES']);record('decode_'+label,'RUNNING',command=cmd,input_sha256=sha(requests))
  with (a.out/(label+'_decode.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining(3600))
  record('decode_'+label,'PASS',output_sha256=sha(output))
 record('continuation','PASS',qualification='Research execution only; no DPO default or hardware-quality promotion')
if __name__=='__main__':main()
