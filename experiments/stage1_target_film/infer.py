"""Versioned target-adaptation loader around the unchanged production Stage1 decoder."""
import argparse,json,os,sys,time
from pathlib import Path
import torch
from peft import PeftModel
from LLM_branch.common import device_adaptation as da
from .model import ActionFiLM,features,kind_id,install
from .train import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--adaptation',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 c=json.loads((a.adaptation/'adaptation_contract.json').read_text())
 if c['schema']!='stage1-target-adaptation-v1' or c['parent_sha256']!=sha(a.parent/'adapter_model.safetensors'):raise ValueError('Parent lineage mismatch')
 if c['lora_sha256']!=sha(a.adaptation/'device_adapt/adapter_model.safetensors') or c['film_sha256']!=sha(a.adaptation/'film.pt'):raise ValueError('Adaptation weights changed')
 from experiments.final_architecture_20261109.run import preflight
 for attempt in range(16):
  try:receipt=preflight(a.gpu_uuid,minimum=10000);break
  except RuntimeError:
   if attempt==15:raise
   time.sleep(2)
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid;a.out.mkdir(parents=True)
 from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
 from LLM_branch.common.mailohls_contract import RESOURCE_KEYS
 originals={k:getattr(inf,k) for k in ('load_stage_model','predict_case','get_rhs_candidates_for_lhs','score_rhs_candidate_suffix')};state={};current={};started=time.monotonic()
 def load(args,tok):
  base=inf.load_base_model(args.model,revision=args.model_revision,compute_dtype=torch.float16);base.resize_token_embeddings(len(tok));model=PeftModel.from_pretrained(base,str(a.parent),adapter_name='mailohls_base',is_trainable=False);da.isolate_parent_token_serialization(model);model.load_adapter(str(a.adaptation/'device_adapt'),adapter_name='device_adapt',is_trainable=False);da.activate(model,trainable=False)
  if da.frozen_parent_hash(model)!=c['parent_tensor_sha256']:raise ValueError('Frozen parent tensor mismatch')
  film=ActionFiLM(model.config.hidden_size).to(next(model.parameters()).device);film.load_state_dict(torch.load(a.adaptation/'film.pt',map_location=next(model.parameters()).device,weights_only=True));film.eval();s,handle=install(model,film);state.update(s=s,film=film);model.eval();return model
 def predict(**kw):
  row=kw['case'].platform_row;budget=row['resource_budget_frac'];part=row['device'];clock=row.get('selected_clock_period',row.get('clock_period'))
  current['context']=features({'request':{'physical_context':['GAN','GAN','PARETO_LATENCY',part,clock],'budget':dict.fromkeys(('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut'),budget)}})
  result=originals['predict_case'](**kw)
  with (a.out/'completed_cases.jsonl').open('a') as f:f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
  return result
 def candidates(kernel_name,lhs,registry):current['kind']=kind_id(lhs);return originals['get_rhs_candidates_for_lhs'](kernel_name,lhs,registry)
 def suffix(**kw):
  ids=kw['base_input_ids'];tok=kw['tok'];n=len(tok(kw['candidate_text'],add_special_tokens=False)['input_ids']);base=ids.shape[1];kinds=torch.zeros(1,base+n,dtype=torch.long,device=ids.device);kinds[:,base-1:base+n-1]=current['kind'];state['s']['context']=current['context'].to(ids.device)[None,None,:];state['s']['kinds']=kinds if c['film_enabled'] else None
  try:return originals['score_rhs_candidate_suffix'](**kw)
  finally:state['s']['kinds']=None
 inf.load_stage_model=load;inf.predict_case=predict;inf.get_rhs_candidates_for_lhs=candidates;inf.score_rhs_candidate_suffix=suffix
 sys.argv=[sys.argv[0],'--stage','stage1','--adapter_dir',str(a.parent),'--directive_domain_registry_json',str(a.parent/'directive_domain_registry.json'),'--application_dataset_dir',str(a.prepared/'sources'),'--input_jsonl',str(a.prepared/'requests.jsonl'),'--output_jsonl',str(a.out/'predictions.jsonl'),'--score_reduction','mean','--debug_topk','16','--num_samples','1']
 try:inf.main()
 finally:
  for k,v in originals.items():setattr(inf,k,v)
 write(a.out/'execution.json',{'schema':'target-adaptation-inference-v1','status':'PASS','completed_cases':len((a.out/'completed_cases.jsonl').read_text().splitlines()),'predictions_sha256':sha(a.out/'predictions.jsonl'),'adaptation_contract_sha256':sha(a.adaptation/'adaptation_contract.json'),'prepared_manifest_sha256':sha(a.prepared/'manifest.json'),'parent_sha256':sha(a.parent/'adapter_model.safetensors'),'runtime_seconds':time.monotonic()-started,'preflight':receipt,'new_HLS_calls':0})
if __name__=='__main__':main()
