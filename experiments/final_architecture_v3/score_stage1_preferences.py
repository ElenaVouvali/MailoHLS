"""Auxiliary A2 baseline score/mask diagnostics on sealed non-test preferences.
No optimization and no cross-GPU reference-cache reuse. Teacher-forced local
continuation averages are diagnostics, not normalized configuration likelihoods.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from LLM_branch.train import train_SFT_xattn_new as production
from .train_preferences import encode,rows,sha
from .preferences_objective import selected_logps
from experiments.final_architecture_20261109.run import preflight
GPU='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35'
V4=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')
def main():
 p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print('A2 frozen V4 preference score diagnostics; no updates');return
 if a.out.exists():raise FileExistsError(a.out)
 receipt=preflight(GPU,14000);os.environ['CUDA_VISIBLE_DEVICES']=GPU;a.out.mkdir(parents=True);start=time.monotonic();parent=json.loads((V4/'training_contract.json').read_text());old=inference.BitsAndBytesConfig
 def fp16(**kw):kw['bnb_4bit_compute_dtype']=torch.float16;return old(**kw)
 inference.BitsAndBytesConfig=fp16
 def diagnostic(**kwargs):
  model,tok=kwargs['model'],kwargs['tok'];model.requires_grad_(False);model.eval();registry=kwargs['directive_domain_registry'];device=inference.get_first_real_device(model);allrecords={}
  for fold in ('train','val'):
   items=encode(rows(a.pairs/(fold+'_pairs.jsonl')),tok,parent,registry);records=[]
   for item in items:
    pair=item['pair'];record={'pair_id':pair['pair_id'],'family':pair['request']['family'],'context':pair['request']['physical_context'],'latency_gap':pair['latency_gap']}
    for side in ('preferred','rejected'):
     sample=item[side];ids=sample['input_ids'].to(device)
     with torch.no_grad():
      output=model(input_ids=ids[None],attention_mask=sample['attention_mask'][None].to(device));total,sft,n=selected_logps(output.logits,sample['labels'].to(device),sample['token_weights'].to(device))
      pack=production.build_deterministic_rhs_pack(pair['row']['input'],pair[side+'_target'],tok,supervise_eos=False,directive_domain_registry=registry,kernel_name=pair['row']['kernel_name']);offset=len(ids)-len(pack.input_ids);groups={}
      for pos,key in enumerate(pack.site_keys):
       if key is not None and pack.labels[pos]!=-100:groups.setdefault(key,[]).append(offset+pos)
      fieldrecords=[]
      for key,positions in groups.items():
       indices=torch.tensor(positions,device=device);lp=torch.nn.functional.log_softmax(output.logits[0,indices-1].float(),-1).gather(-1,ids[indices,None]).squeeze(-1);fieldrecords.append({'field':key,'rhs':pair[side]['assignment'][key],'continuation_tokens':len(positions),'mean_continuation_logp':float(lp.mean())})
      record[side]={'sum':float(total),'mean':float(total)/n,'action_normalized':-float(sft),'selected_field_continuation_mean':float(np.mean([r['mean_continuation_logp'] for r in fieldrecords])),'tokens':n,'nonforced_fields':len(groups),'completion_tokens':len(ids)-int(sample['routing_start_idx']),'complete_partitions':sum(k.startswith('AUTO{_ARRAY_T') and v=='complete' for k,v in pair[side]['assignment'].items()),'fields':fieldrecords};del output
    records.append(record)
   allrecords[fold]=records;(a.out/(fold+'_scores.json')).write_text(json.dumps(records,indent=2)+'\n');print('[BASELINE SCORES]',fold,len(records),flush=True)
  summaries={}
  for fold,records in allrecords.items():
   metrics={}
   for convention in ('sum','mean','action_normalized','selected_field_continuation_mean'):
    margins=np.array([r['preferred'][convention]-r['rejected'][convention] for r in records]);correlations={}
    for name,values in {'token_count_difference':[r['preferred']['tokens']-r['rejected']['tokens'] for r in records],'field_count_difference':[r['preferred']['nonforced_fields']-r['rejected']['nonforced_fields'] for r in records],'completion_length_difference':[r['preferred']['completion_tokens']-r['rejected']['completion_tokens'] for r in records],'complete_partition_difference':[r['preferred']['complete_partitions']-r['rejected']['complete_partitions'] for r in records],'latency_gap':[r['latency_gap'] for r in records]}.items():
     correlations[name]=float(np.corrcoef(margins,values)[0,1]) if np.std(values)>0 and np.std(margins)>0 else None
    metrics[convention]={'accuracy':float(np.mean(margins>0)),'mean_margin':float(margins.mean()),'pearson_correlations':correlations,'family_accuracy':{f:float(np.mean([r['preferred'][convention]>r['rejected'][convention] for r in records if r['family']==f])) for f in sorted({r['family'] for r in records})}}
   summaries[fold]=metrics
  report={'schema':'v3-stage1-score-conventions-v1','checkpoint_sha256':sha(V4/'adapter_model.safetensors'),'pair_manifest_sha256':sha(a.pairs/'manifest.json'),'hardware':receipt,'runtime_seconds':time.monotonic()-start,'precision':'NF4 FP16; auxiliary A2; not a V100S Stage3 reference cache','results':summaries,'limitations':['same sealed128/54 pair cohort, not guaranteed identical to historical136 pair identities','unadjusted correlations do not prove causality; family/request views correlated','selected_field_continuation_mean uses teacher-forced actual RHS, not alternative candidate rankings or a whole-configuration probability','resource proxies and missing timing remain unqualified','no locked test or optimizer updates']};(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');return {'context_id':kwargs['case'].platform_row['context_id'],'diagnostic':'frozen-score audit only'}
 inference.predict_case=diagnostic;sys.argv=[sys.argv[0],'--stage','stage1','--adapter_dir',str(V4),'--directive_domain_registry_json',str(V4/'directive_domain_registry.json'),'--input_jsonl',str(a.request),'--output_jsonl',str(a.out/'completion.jsonl')];inference.main()
if __name__=='__main__':main()
