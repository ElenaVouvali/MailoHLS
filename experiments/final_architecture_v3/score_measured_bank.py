"""Frozen teacher-forced likelihood diagnostics on authenticated complete designs.
Not constrained generation, a new predictor, or a replacement for physical QoR.
"""
import argparse,json,os,time,copy
from pathlib import Path
from types import SimpleNamespace
import torch
from LLM_branch.train import train_DPO_harp_xattn as dpo,train_SFT_xattn_new as production
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from .supervision import corrected
from .preferences_objective import selected_logps
from experiments.final_architecture_20261109.run import preflight

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--memory-dir',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print('Frozen3-policy measured-bank score diagnostic; no optimization');return
 if a.out.exists():raise FileExistsError(a.out)
 receipt=preflight(a.gpu_uuid);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid;started=time.monotonic();a.out.mkdir(parents=True)
 parent=dpo.load_stage2_contract(str(a.bundle/'stage2'));args=SimpleNamespace(objective=None,model=None,model_revision=None,seed=123,stage2_seed=123,parent_contract=parent,stage2_adapter_dir=str(a.bundle/'stage2'),gradient_checkpointing=False,train_lora_dpo=False,train_xattn_dpo=False,train_attn_gate_dpo=False,train_ff_gate_dpo=False,train_special_token_embeddings=False)
 dpo.apply_parent_contract(production,args,parent);tok=dpo.build_tokenizer(production,args,parent);registry=inference.load_directive_domain_registry(str(a.bundle/'stage2/directive_domain_registry.json'));cases=[json.loads(s) for s in a.cases.read_text().splitlines()]
 with corrected(production):dataset=production.SFTDataset([r['row'] for r in cases],tok,parent['max_length'],directive_domain_registry=registry,supervise_eos=False,input_only_special_ids=parent['special_token_ids'])
 assert len(dataset.samples)==len(cases)
 for case,sample in zip(cases,dataset.samples):
  assert production.build_rhs_map_from_target(case['row']['target']),'Empty RHS map: use native auto{...} serialization'
  if case['row'].get('avail_bram')==46:
   prompt=tok.decode(sample['input_ids'][:int(sample['routing_start_idx'])]);assert 'BRAM_18K=46 ' in prompt,'Binding budget was not preserved in actual prompt'
 model=dpo.build_structural_model(production,args,tok,trainable=False);model.requires_grad_(False);model.eval();device=next(model.parameters()).device;bank=inference.load_memory_bank(str(a.memory_dir),expected_mem_dim=128,expected_max_slots=64);records=[]
 for mode,checkpoint in [('stage1_gateoff','stage2'),('stage2','stage2'),('stage3','dpo')]:
  production.load_partial_structural_xattn(model,str(a.bundle/checkpoint/'structural_xattn.pt'),'SCORE_'+mode)
  if mode=='stage1_gateoff':
   for name,pv in model.named_parameters():
    if name.endswith('attn_gate'):pv.data.zero_()
  with torch.no_grad():
   for case,sample in zip(cases,dataset.samples):
    kernel=case['row']['kernel_name'];kv,mask,relation=inference.get_real_memory_pack_for_kernel(bank,kernel,64,128,'compiler_relational');model.condition_structural_memory(kv.to(device),mask.to(device),action_relation_mask=relation.to(device));ids=sample['input_ids'].to(device)
    with torch.autocast('cuda',dtype=torch.float16):output=model(input_ids=ids[None],attention_mask=sample['attention_mask'].to(device)[None],routing_start_idx=sample['routing_start_idx'].to(device)[None],xattn_apply_mask=sample['xattn_apply_mask'].to(device)[None])
    total,nll,count=selected_logps(output.logits,sample['labels'].to(device),sample['token_weights'].to(device));records.append({'mode':mode,'case_id':case['case_id'],'request_id':case['request_id'],'kernel':kernel,'sum_value_logp':float(total),'mean_value_logp':float(total)/count,'action_normalized_logp':-float(nll),'nonforced_value_tokens':count,'measured':case['measured'],'scope':case['scope']});model.clear_structural_memory();del output
  print('[SCORED]',mode,len(cases),flush=True)
 result={'schema':'v3-measured-bank-frozen-score-v1','hardware':receipt,'runtime_seconds':time.monotonic()-started,'records':records,'scope':'Teacher-forced complete-design summed/normalized nonforced-RHS likelihood; each completion has its own causal prefix. Not natural generation, not normalized production whole-design probability, no optimizer update. Binding-budget view is bank-derived OOD diagnostic.','cases_sha256':__import__('hashlib').sha256(a.cases.read_bytes()).hexdigest()};(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
