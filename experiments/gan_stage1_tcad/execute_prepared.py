"""Portable checksum-verified execution of a sealed GAN request bundle."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from .run import ADAPTER_SHA,sha,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);a=p.parse_args();m=json.loads((a.prepared/'manifest.json').read_text())
 assert sha(a.checkpoint/'adapter_model.safetensors')==ADAPTER_SHA==m['checkpoint_sha256'];assert sha(a.checkpoint/'training_contract.json')==m['contract_sha256'];assert sha(a.checkpoint/'directive_domain_registry.json')==m['registry_sha256'];assert sha(a.prepared/'requests.jsonl')==m['requests_sha256'];assert sha(a.prepared/'sources/GAN/GAN_placeholders.cpp')==m['source_sha256'];assert sha(a.prepared/'sources/GAN/kernel_info.txt')==m['metadata_sha256']
 for name,digest in m['header_sha256'].items():assert sha(a.prepared/'sources/GAN'/name)==digest
 if a.out.exists():raise FileExistsError('Preserve completed/partial output; fresh directory required')
 from experiments.final_architecture_20261109.run import preflight
 receipt=preflight(a.gpu_uuid);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid;a.out.mkdir(parents=True);dump(a.out/'preflight.json',receipt)
 from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
 original=inference.predict_case;started=time.monotonic()
 def checkpoint_case(**kwargs):
  r=original(**kwargs)
  with (a.out/'completed_cases.jsonl').open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
  return r
 inference.predict_case=checkpoint_case
 command=[sys.argv[0],'--stage','stage1','--adapter_dir',str(a.checkpoint),'--directive_domain_registry_json',str(a.checkpoint/'directive_domain_registry.json'),'--application_dataset_dir',str(a.prepared/'sources'),'--input_jsonl',str(a.prepared/'requests.jsonl'),'--output_jsonl',str(a.out/'predictions.jsonl'),'--score_reduction','mean','--debug_topk','16','--num_samples','1','--print_predictions'];sys.argv=command
 try:inference.main()
 finally:inference.predict_case=original
 assert sha(a.checkpoint/'adapter_model.safetensors')==ADAPTER_SHA
 dump(a.out/'execution.json',{'status':'PASS','checkpoint':str(a.checkpoint),'runtime_seconds':time.monotonic()-started,'GPU':receipt,'command':command,'prepared_manifest_sha256':sha(a.prepared/'manifest.json'),'predictions_sha256':sha(a.out/'predictions.jsonl'),'script_sha256':sha(Path(__file__)),'completed_cases':len((a.out/'completed_cases.jsonl').read_text().splitlines()),'checkpoint_unchanged':True,'new_HLS_calls':0,'optimizer_updates':0})
if __name__=='__main__':main()
