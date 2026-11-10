"""Bounded sequential real-decoder inference for corrected S2-50/S2-150/S3.
No checkpoint retuning, synthesis or production promotion. Completed arms resume
by checksum; partial per-case journals are preserved for explicit recovery.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .run import sha,dump,ADAPTER_SHA
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--worker');p.add_argument('--max-gpu-seconds',type=int,default=6000);a=p.parse_args();models=json.loads(a.models.read_text());prepared=json.loads((a.prepared/'manifest.json').read_text());assert sha(a.prepared/'requests.jsonl')==prepared['requests_sha256']
 for label,rec in models.items():
  checkpoint=Path(rec['checkpoint']);contract=json.loads((checkpoint/'training_contract.json').read_text());assert sha(checkpoint/'adapter_model.safetensors')==ADAPTER_SHA;assert sha(checkpoint/'structural_xattn.pt')==rec['fusion_sha256'];assert sha(checkpoint/'training_contract.json')==rec['contract_sha256'];assert contract['stage']==rec['stage'] and contract['objective']=='PARETO_LATENCY';assert contract['research_supervision']['policy']=='stage1-action-normalized-values-v1'
  from LLM_branch.common.memory_extension import validate_heldout_extension
  cases=[json.loads(s) for s in (a.prepared/'requests.jsonl').read_text().splitlines()];validate_heldout_extension(a.memory,contract['structural']['memory_manifest_sha256'],[(r['kernel_name'],r['input']) for r in cases])
 if not a.worker:
  print('Three frozen fusion policies;18 identical GAN requests each. S3 parent is S2-50, not S2-150. No synthesis.',flush=True)
  if not a.execute:return
  a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic();prior=sum(json.loads(p.read_text())['runtime_seconds'] for p in a.out.glob('*/execution.json'))
  for label in models:
   output=a.out/label
   if (output/'execution.json').exists():
    receipt=json.loads((output/'execution.json').read_text());assert sha(output/'predictions.jsonl')==receipt['predictions_sha256'];continue
   if output.exists():raise RuntimeError('Partial model output preserved: inspect its journal before an explicit missing-case recovery')
   remain=a.max_gpu_seconds-prior-(time.monotonic()-started)
   if remain<=0:raise TimeoutError('Reserved inference GPU budget exhausted')
   cmd=[sys.executable,'-m','experiments.gan_stage1_tcad.run_fused','--models',str(a.models),'--prepared',str(a.prepared),'--memory',str(a.memory),'--out',str(a.out),'--gpu-uuid',a.gpu_uuid,'--worker',label]
   with (a.out/(label+'.log')).open('a') as log:subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=remain)
   print('[DONE]',label,flush=True)
  return
 rec=models[a.worker];checkpoint=Path(rec['checkpoint']);out=a.out/a.worker
 from experiments.final_architecture_20261109.run import preflight
 for retry in range(16):
  try:receipt=preflight(a.gpu_uuid);break
  except RuntimeError:
   if retry==15:raise
   time.sleep(2)
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);dump(out/'preflight.json',receipt)
 from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
 original=inference.predict_case;started=time.monotonic()
 def save_case(**kwargs):
  row=original(**kwargs)
  with (out/'completed_cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  return row
 inference.predict_case=save_case
 command=[sys.argv[0],'--stage',rec['stage'],'--adapter_dir',str(checkpoint),'--directive_domain_registry_json',str(checkpoint/'directive_domain_registry.json'),'--application_dataset_dir',str(a.prepared/'sources'),'--memory_dir',str(a.memory),'--allow_memory_extension','--input_jsonl',str(a.prepared/'requests.jsonl'),'--output_jsonl',str(out/'predictions.jsonl'),'--score_reduction','mean','--debug_topk','16','--num_samples','1','--print_predictions'];sys.argv=command
 try:inference.main()
 finally:inference.predict_case=original
 rows=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()];assert len(rows)==18;assert sha(checkpoint/'adapter_model.safetensors')==ADAPTER_SHA;assert sha(checkpoint/'structural_xattn.pt')==rec['fusion_sha256']
 import torch,transformers
 dump(out/'execution.json',{'status':'PASS','label':a.worker,'model':rec,'command':command,'prepared_manifest_sha256':sha(a.prepared/'manifest.json'),'memory_manifest_sha256':sha(a.memory/'memory_manifest.json'),'runtime_seconds':time.monotonic()-started,'GPU':receipt,'script_sha256':sha(Path(__file__)),'predictions_sha256':sha(out/'predictions.jsonl'),'completed_cases':18,'checkpoint_unchanged':True,'torch':torch.__version__,'transformers':transformers.__version__,'cuda_runtime':torch.version.cuda,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'optimizer_updates':0,'new_HLS_calls':0,'qualification':'Research inference only; does not promote unqualified Stage3 or establish physical QoR.'})
if __name__=='__main__':main()
