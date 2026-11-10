"""CPU-only encoding/length audit using the actual Stage2 tokenizer and SFTDataset."""
import argparse,collections,json
from pathlib import Path
from types import SimpleNamespace
from LLM_branch.train import train_DPO_harp_xattn as dpo,train_SFT_xattn_new as production
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from .train_preferences import encode,rows,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();parent=dpo.load_stage2_contract(str(a.parent));args=SimpleNamespace(objective=None,model=None,model_revision=None,seed=123,stage2_seed=123,stage2_adapter_dir=str(a.parent));dpo.apply_parent_contract(production,args,parent);tok=dpo.build_tokenizer(production,args,parent);registry=inference.load_directive_domain_registry(str(a.parent/'directive_domain_registry.json'));report={'schema':'v3-real-mask-audit-v1','parent_contract_sha256':sha(a.parent/'training_contract.json'),'splits':{}}
 for fold in ('train','val'):
  encoded=encode(rows(a.pairs/(fold+'_pairs.jsonl')),tok,parent,registry);records=[]
  for item in encoded:
   r={'pair_id':item['pair']['pair_id'],'family':item['pair']['request']['family'],'encoding_sha256':item['encoding_sha256'],'latency_gap':item['pair']['latency_gap'],'clock':item['pair']['request']['physical_context'][4],'device':item['pair']['request']['physical_context'][3]}
   for side in ('preferred','rejected'):
    s=item[side];mask=s['labels'].ne(-100);assignment=item['pair'][side]['assignment'];r[side]={'supervised_value_tokens':int(mask.sum()),'completion_tokens':len(s['input_ids'])-int(s['routing_start_idx']),'weight_sum':float(s['token_weights'].sum()),'complete_partitions':sum(k.startswith('AUTO{_ARRAY_T') and v=='complete' for k,v in assignment.items())}
   records.append(r)
  less=sum(r['preferred']['supervised_value_tokens']<r['rejected']['supervised_value_tokens'] for r in records);greater=sum(r['preferred']['supervised_value_tokens']>r['rejected']['supervised_value_tokens'] for r in records);ties=len(records)-less-greater
  report['splits'][fold]={'pairs':len(records),'shorter_preferred':less,'longer_preferred':greater,'equal_score_lengths':ties,'length_only_accuracy_nontied':less/(less+greater) if less+greater else None,'length_only_half_credit_ties':(less+.5*ties)/len(records),'records':records}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({k:{n:x for n,x in v.items() if n!='records'} for k,v in report['splits'].items()},indent=2))
if __name__=='__main__':main()
