"""No-training real-LM B48 ablation, adapter checks, causal/cache and precision traces."""
import argparse,json,os,time,collections
from pathlib import Path
import torch
from peft import PeftModel
from LLM_branch.common import device_adaptation as da
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from LLM_branch.train import train_SFT_xattn_new as prod
from experiments.final_architecture_v3.supervision import corrected
from experiments.final_architecture_v3.preferences_objective import selected_logps
from experiments.stage1_target_film.model import ActionFiLM,features,kind_id,install
from experiments.stage1_target_film.train import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--limit',type=int);p.add_argument('--training-precision-control',action='store_true');a=p.parse_args()
 from experiments.final_architecture_20261109.run import preflight
 receipt=preflight(a.gpu_uuid,minimum=10000);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);c=json.loads((a.checkpoint/'adaptation_contract.json').read_text());parent=json.loads((a.parent/'training_contract.json').read_text());assert c['parent_sha256']==sha(a.parent/'adapter_model.safetensors')=='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164';assert c['lora_sha256']==sha(a.checkpoint/'device_adapt/adapter_model.safetensors') and c['film_sha256']==sha(a.checkpoint/'film.pt')
 tok=inf.load_checkpoint_tokenizer(str(a.parent),parent);reg=inf.load_directive_domain_registry(str(a.parent/'directive_domain_registry.json'));pairs=[json.loads(s) for s in (a.pairs/'val_pairs.jsonl').read_text().splitlines()];assert len(pairs)==54
 if a.limit:pairs=pairs[:a.limit]
 base=inf.load_base_model(parent['model'],revision=parent['model_revision'],compute_dtype=torch.float16);base.resize_token_embeddings(len(tok));model=PeftModel.from_pretrained(base,str(a.parent),adapter_name='mailohls_base',is_trainable=False);da.isolate_parent_token_serialization(model);model.load_adapter(str(a.checkpoint/'device_adapt'),adapter_name='device_adapt',is_trainable=False);da.activate(model,trainable=False);model.eval()
 if a.training_precision_control:
  from peft import prepare_model_for_kbit_training
  model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False);da.activate(model,trainable=False);model.eval()
 device=next(model.parameters()).device;film=ActionFiLM(model.config.hidden_size).to(device);film.load_state_dict(torch.load(a.checkpoint/'film.pt',map_location=device,weights_only=True));film.eval();state,handle=install(model,film);frozen=da.frozen_parent_hash(model);assert frozen==c['parent_tensor_sha256'];assert not any(v.requires_grad for v in model.parameters());numerics=[];mode='off';kind_stats=collections.defaultdict(list)
 def observe(module,args,output):
  h,z,k=args;select=k.gt(0)
  if not select.any():return
  source=h[select];ab=film.net(torch.cat((z.expand(*k.shape,-1)[select],film.kind(k[select])),dim=-1));gamma,beta=ab.chunk(2,-1);delta=film.gate.tanh()*(.1*gamma.tanh()*torch.nn.functional.layer_norm(source.float(),(film.width,))+.1*beta.tanh());effective=(source.float()+delta).half().float()-source.half().float();actual=output[select].float()-source.float()
  for i,kind in enumerate(k[select].tolist()):
   rec={'kind':kind,'mode':mode,'hidden_dtype':str(h.dtype),'relative_modulation':float(delta[i].norm()/source[i].float().norm().clamp_min(1e-12)),'effective_relative_modulation':float(actual[i].norm()/source[i].float().norm().clamp_min(1e-12)),'fp16_unchanged_fraction':float(effective[i].eq(0).float().mean()),'fp16_effective_intended_norm_ratio':float(effective[i].norm()/delta[i].norm().clamp_min(1e-12)),'zero_after_fp16':bool(effective[i].eq(0).all()),'nonfinite':not bool(torch.isfinite(delta[i]).all())};numerics.append(rec)
 film.register_forward_hook(observe)
 def encode(pair,side):
  row=dict(pair['row'],target=pair[side+'_target'])
  with corrected(prod):sample=prod.SFTDataset([row],tok,parent['max_length'],directive_domain_registry=reg,supervise_eos=False,input_only_special_ids=parent['special_token_ids']).samples[0]
  pack=prod.build_deterministic_rhs_pack(row['input'],prod.reorder_target_by_source_order(row['input'],row['target']),tok,directive_domain_registry=reg,kernel_name=row['kernel_name']);offset=len(sample['input_ids'])-len(pack.input_ids);kinds=torch.zeros_like(sample['input_ids']);legacy=torch.zeros_like(kinds)
  for j,key in enumerate(pack.site_keys):
   if key and offset+j>0:
    legacy[offset+j-1]=kind_id(key)
    if sample['labels'][offset+j]!=-100:kinds[offset+j-1]=kind_id(key)
  return sample,kinds,legacy
 results=[];started=time.monotonic()
 with torch.no_grad():
  for ix,pair in enumerate(pairs):
   entries={};logit_effect=[]
   for side in ('preferred','rejected'):
    sample,kinds,legacy=encode(pair,side);ids=sample['input_ids'].to(device)[None];ctx=features(pair).to(device)[None,None];stored={}
    for mode in ('off','value','legacy'):
     state['context']=ctx;state['kinds']=None if mode=='off' else (kinds if mode=='value' else legacy).to(device)[None];output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False);_,nll,_=selected_logps(output.logits,sample['labels'].to(device),sample['token_weights'].to(device));entries[side+'_'+mode]=float(-nll);mask=kinds.to(device).gt(0);stored[mode]=output.logits[0,mask].float().cpu();del output
    diff=stored['value']-stored['off'];logit_effect.append({'side':side,'max_absolute':float(diff.abs().max()),'rms':float(diff.square().mean().sqrt()),'value_vs_legacy_max':float((stored['value']-stored['legacy']).abs().max())})
    if ix==0 and side=='preferred':
     # Native full-prefix suffix score versus a one-position cached real LM head.
     pos=int(torch.nonzero(kinds)[0]);n=pos+1;prefix=ids[:,:n];state['kinds']=kinds[:n].to(device)[None];full=model(input_ids=prefix,use_cache=True);state['kinds']=kinds[:n-1].to(device)[None];pre=model(input_ids=prefix[:,:-1],use_cache=True);state['kinds']=kinds[n-1:n].to(device)[None];cached=model(input_ids=prefix[:,-1:],past_key_values=pre.past_key_values,attention_mask=torch.ones_like(prefix),use_cache=True);cache_max=float((full.logits[:,-1].float()-cached.logits[:,-1].float()).abs().max());state['kinds']=None
     # Parent-only vs simultaneous adapters, identical source/prefix, no FiLM.
     together=model(input_ids=prefix,use_cache=False).logits[:,-1].float();model.base_model.set_adapter('mailohls_base');parent_only=model(input_ids=prefix,use_cache=False).logits[:,-1].float();da.activate(model,trainable=False);adapter_max=float((together-parent_only).abs().max());assert da.frozen_parent_hash(model)==frozen
     write(a.out/'integrity.json',{'training_precision_control':a.training_precision_control,'parameter_dtype_counts':dict(collections.Counter(str(p.dtype) for p in model.parameters())),'active_adapters':model.active_adapters,'parent_hash_verified':True,'parent_tensor_sha256':frozen,'any_trainable_model_parameters':any(v.requires_grad for v in model.parameters()),'residual_on_off_max_logit_change':adapter_max,'real_cached_vs_full_max_logit_change':cache_max,'scope':'one actual first RHS causal position; no blanket exact-cache-equivalence claim','checkpoint_hashes':c,'preflight':receipt})
   result={'pair_id':pair['pair_id'],'family':pair['request']['family'],'context':pair['request'],'scores':entries,'logit_effects':logit_effect};results.append(result)
   with (a.out/'pairs.jsonl').open('a') as f:f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
   print('[DEV]',ix+1,entries,flush=True)
 assert da.frozen_parent_hash(model)==frozen
 summary={}
 for condition in ('off','value','legacy'):
  family={}
  for f in sorted({r['family'] for r in results}):
   selected=[r for r in results if r['family']==f];family[f]={'n':len(selected),'ranking_accuracy':sum(r['scores']['preferred_'+condition]>r['scores']['rejected_'+condition] for r in selected)/len(selected)}
  summary[condition]={'family':family,'family_macro_accuracy':sum(v['ranking_accuracy'] for v in family.values())/len(family),'accuracy':sum(r['scores']['preferred_'+condition]>r['scores']['rejected_'+condition] for r in results)/len(results)}
 write(a.out/'summary.json',{'conditions':summary,'pairs':len(pairs),'runtime_seconds':time.monotonic()-started,'pair_file_sha256':sha(a.pairs/'val_pairs.jsonl'),'new_HLS_calls':0,'training_updates':0});write(a.out/'numerics.json',{'gate':float(film.gate),'effective_gate':float(film.gate.tanh()),'positions':numerics})
if __name__=='__main__':main()
