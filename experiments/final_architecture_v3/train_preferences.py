"""Bounded Stage3 reference-DPO versus chosen-only SFT; one shared NF4 backbone.
Uses the existing Stage3 model builder, Stage2 tokenizer, dataset, routing and
structural exporter. Exact frozen reference is cached before any optimizer update.
"""
import argparse,copy,hashlib,json,math,os,random,shutil,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from LLM_branch.train import train_DPO_harp_xattn as dpo,train_SFT_xattn_new as production
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from LLM_branch.common import frozen_stage1
from .preferences_objective import selected_logps,dpo_loss
from .supervision import corrected
from experiments.final_architecture_20261109.run import preflight

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def append(p,obj):
 with Path(p).open('a') as f:f.write(json.dumps(obj)+'\n');f.flush();os.fsync(f.fileno())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def encode(pairs,tok,parent,registry):
 records=[dict(p['row'],target=p[side+'_target']) for p in pairs for side in ('preferred','rejected')]
 with corrected(production):ds=production.SFTDataset(records,tok,parent['max_length'],directive_domain_registry=registry,supervise_eos=False,input_only_special_ids=parent['special_token_ids'])
 result=[]
 for n,pair in enumerate(pairs):
  a,b=ds.samples[2*n:2*n+2];ra=int(a['routing_start_idx']);rb=int(b['routing_start_idx']);assert ra==rb and torch.equal(a['input_ids'][:ra],b['input_ids'][:rb])
  result.append({'pair':pair,'preferred':a,'rejected':b,'encoding_sha256':dpo.canonical_json_sha256([{k:s[k].tolist() for k in ('input_ids','labels','token_weights','xattn_apply_mask','routing_start_idx')} for s in (a,b)])})
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--memory-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=34);p.add_argument('--resume',type=Path);p.add_argument('--gpu-uuid',default='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb');a=p.parse_args()
 if a.out.exists() and not a.resume:raise FileExistsError(a.out)
 receipt=preflight(a.gpu_uuid);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid;a.out.mkdir(parents=True,exist_ok=True)
 parent=dpo.load_stage2_contract(str(a.parent));args=SimpleNamespace(objective=None,model=None,model_revision=None,seed=123,stage2_seed=123,parent_contract=parent,stage2_adapter_dir=str(a.parent),gradient_checkpointing=True,train_lora_dpo=False,train_xattn_dpo=True,train_attn_gate_dpo=True,train_ff_gate_dpo=False,train_special_token_embeddings=False)
 dpo.apply_parent_contract(production,args,parent);tok=dpo.build_tokenizer(production,args,parent)
 manifest=json.loads((a.pairs/'manifest.json').read_text());memory_manifest=a.memory_dir/'memory_manifest.json'
 if sha(memory_manifest)!=parent['structural']['memory_manifest_sha256']:raise ValueError('Parent/memory contract differs')
 for fold in ('train','val'):
  if sha(a.pairs/(fold+'_pairs.jsonl'))!=manifest['pair_sha256'][fold]:raise ValueError('Preference file changed')
 encoded={k:encode(rows(a.pairs/(k+'_pairs.jsonl')),tok,parent,inference.load_directive_domain_registry(str(a.parent/'directive_domain_registry.json'))) for k in ('train','val')}
 if not all(encoded.values()):raise ValueError('Missing training or independent development pairs')
 bank=inference.load_memory_bank(str(a.memory_dir),expected_mem_dim=128,expected_max_slots=64)
 model=dpo.build_structural_model(production,args,tok,trainable=True);device=next(model.parameters()).device
 # The same FP32 trainable projections are used for reference cache and both arms.
 for name,param in model.named_parameters():
  if param.requires_grad:param.data=param.data.float()
 frozen=frozen_stage1.frozen_stage1_hashes(model);frozen_stage1.assert_stage1_frozen(model)
 if frozen['combined_sha256']!=parent['structural']['frozen_stage1_sha256']:raise ValueError('Loaded Stage1 differs from authenticated Stage2 parent')
 initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};assert initial
 def score(sample,kernel):
  kv,mask,relation=inference.get_real_memory_pack_for_kernel(bank,kernel,64,128,'compiler_relational');model.condition_structural_memory(kv.to(device),mask.to(device),action_relation_mask=relation.to(device))
  ids=sample['input_ids'].to(device)
  with torch.autocast('cuda',dtype=torch.float16):
   output=model(input_ids=ids[None],attention_mask=sample['attention_mask'].to(device)[None],routing_start_idx=sample['routing_start_idx'].to(device)[None],xattn_apply_mask=sample['xattn_apply_mask'].to(device)[None])
  values=selected_logps(output.logits,sample['labels'].to(device),sample['token_weights'].to(device))
  if not torch.is_grad_enabled():model.clear_structural_memory()
  return values
 identity={'schema':'v3-stage3-reference-cache-v1','parent_contract_sha256':sha(a.parent/'training_contract.json'),'parent_structural_sha256':sha(a.parent/'structural_xattn.pt'),'frozen_stage1':frozen,'memory_manifest_sha256':sha(memory_manifest),'pair_files':manifest['pair_sha256'],'encoding_sha256':dpo.canonical_json_sha256({k:[r['encoding_sha256'] for r in v] for k,v in encoded.items()}),'mask':'summed causal nonforced value tokens, no delimiters/singletons/prompt','precision':'NF4/FP16 backbone; FP32 selected log-softmax and sums; same trainable FP32 fusion for policy/ref','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 write(a.out/'preflight.json',receipt)
 cachefile=a.out/'reference.json';model.eval()
 if cachefile.exists():
  cache=json.loads(cachefile.read_text())
  if cache['identity']!=identity:raise ValueError('Reference cache identity mismatch')
 else:
  cache={'identity':identity,'scores':{}}
  with torch.no_grad():
   for fold,items in encoded.items():
    records=[]
    for item in items:
     kernel=item['pair']['row']['kernel_name'];x,sx,nx=score(item['preferred'],kernel);y,sy,ny=score(item['rejected'],kernel)
     records.append({'pair_id':item['pair']['pair_id'],'family':item['pair']['request']['family'],'chosen':float(x),'rejected':float(y),'chosen_nll':float(sx),'rejected_nll':float(sy),'chosen_tokens':nx,'rejected_tokens':ny})
    cache['scores'][fold]=records;print('[REFERENCE]',fold,len(records),flush=True)
  write(cachefile,cache)
 with torch.no_grad():
  first=encoded['train'][0];x,_,_=score(first['preferred'],first['pair']['row']['kernel_name']);assert abs(float(x)-cache['scores']['train'][0]['chosen'])<=1e-5,'Zero-update reference parity failed'
 rng=random.Random(123);order=list(range(len(encoded['train'])));rng.shuffle(order);schedule=[order[i%len(order)] for i in range(a.steps*4)]
 write(a.out/'schedule.json',schedule)
 def evaluate(fold):
  model.eval();records=[]
  with torch.no_grad():
   for item,ref in zip(encoded[fold],cache['scores'][fold]):
    kernel=item['pair']['row']['kernel_name'];x,sx,nx=score(item['preferred'],kernel);y,sy,ny=score(item['rejected'],kernel)
    records.append({**ref,'chosen':float(x),'rejected':float(y),'chosen_nll':float(sx),'rejected_nll':float(sy),'margin':float(x-y),'reference_corrected_margin':float(x-y)-ref['chosen']+ref['rejected'],'mean_token_margin':float(x)/nx-float(y)/ny,'action_normalized_margin':float(sy-sx),'ranking_correct':bool(x>y)})
  return records
 def save(folder,arm,step,optimizer,scaler):
  folder.mkdir(parents=True,exist_ok=True);torch.save(production.get_structural_xattn_state_dict(model),folder/'structural_xattn.pt')
  for name in ('adapter_model.safetensors','adapter_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','directive_domain_registry.json'):
   shutil.copyfile(a.parent/name,folder/name)
  contract=copy.deepcopy(parent);contract['stage']='stage3' if arm=='dpo' else 'stage2';contract['parent_stage2']={'contract_sha256':identity['parent_contract_sha256'],'structural_xattn_sha256':identity['parent_structural_sha256']}
  contract['stage3_preference']={'schema':'mailohls-stage3-preference-v1','algorithm':'reference_dpo' if arm=='dpo' else 'chosen_only_SFT_control','beta':.1,'sft_alpha':0.,'logp_reduction':'sum','reference_cache_sha256':sha(cachefile),'successful_updates':step,'trainable':'fusion/gates only','qualification':'DPO_REQUIRED_BUT_NOT_QUALIFIED','pair_manifest_sha256':sha(a.pairs/'manifest.json'),'limits':manifest['resource_qualification']}
  write(folder/'training_contract.json',contract)
  torch.save({'step':step,'arm':arm,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'reference_cache_sha256':sha(cachefile)},folder/'progress.pt')
 summaries={}
 resume=torch.load(a.resume/'progress.pt',map_location='cpu',weights_only=False) if a.resume else None
 for arm in ('sft','dpo'):
  if resume and resume['arm']=='dpo' and arm=='sft':continue
  for n,pv in model.named_parameters():
   if n in initial:pv.data.copy_(initial[n].to(pv.device))
  random.seed(123);np.random.seed(123);torch.manual_seed(123)
  groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.endswith('attn_gate')],'lr':2e-5},{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.endswith('attn_gate')],'lr':1e-5}];optimizer=torch.optim.AdamW(groups,weight_decay=0.);scaler=torch.amp.GradScaler('cuda');start=0
  if resume and resume['arm']==arm:
   if resume['reference_cache_sha256']!=sha(cachefile):raise ValueError('Resume reference changed')
   production.load_partial_structural_xattn(model,str(a.resume/'structural_xattn.pt'),'RESUME');optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler']);start=resume['step'];random.setstate(resume['rng_python']);np.random.set_state(resume['rng_numpy']);torch.set_rng_state(resume['rng_torch']);torch.cuda.set_rng_state_all(resume['rng_cuda'])
  folder=a.out/arm;folder.mkdir(exist_ok=True);started=time.monotonic()
  if start==0:save(folder/'checkpoint-0',arm,0,optimizer,scaler)
  successful=start;attempts=0
  while successful<a.steps and attempts<2*a.steps:
   model.train()
   for module in model.modules():
    if isinstance(module,torch.nn.Dropout):module.eval()
   optimizer.zero_grad(set_to_none=True);losses=[]
   for idx in schedule[successful*4:(successful+1)*4]:
    item=encoded['train'][idx];ref=cache['scores']['train'][idx];kernel=item['pair']['row']['kernel_name'];x,sx,nx=score(item['preferred'],kernel)
    if arm=='dpo':
     y,sy,ny=score(item['rejected'],kernel);loss=dpo_loss(x,y,torch.tensor(ref['chosen'],device=device),torch.tensor(ref['rejected'],device=device),.1)
    else:loss=sx
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preference loss')
    scaler.scale(loss/4).backward();model.clear_structural_memory();losses.append(float(loss.detach()))
   scaler.unscale_(optimizer);grad={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
   norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);before=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped=scaler.get_scale()<before;attempts+=1
   if not skipped:successful+=1
   append(folder/'training.jsonl',{'step':successful,'attempt':attempts,'loss':float(np.mean(losses)),'grad_norm':float(norm),'gradient_norms':grad,'gates':{n:float(p.detach().tanh()) for n,p in model.named_parameters() if n.endswith('attn_gate')},'optimizer_step_skipped':skipped,'elapsed_seconds':time.monotonic()-started})
   print('[PREFERENCE]',arm,successful,float(np.mean(losses)),flush=True)
   if successful in (1,5,10,20,a.steps) and not skipped:save(folder/('checkpoint-'+str(successful)),arm,successful,optimizer,scaler)
  frozen_stage1.assert_frozen_stage1_unchanged(model,frozen)
  evaluations={fold:evaluate(fold) for fold in ('train','val')}
  for fold,records in evaluations.items():write(folder/(fold+'_scores.json'),records)
  summaries[arm]={'successful_updates':successful,'attempts':attempts,'runtime_seconds':time.monotonic()-started,'ranking_accuracy':{fold:sum(r['ranking_correct'] for r in v)/len(v) for fold,v in evaluations.items()},'frozen_stage1_verified':True,'checkpoint':str(folder/('checkpoint-'+str(successful)))};write(folder/'result.json',summaries[arm])
 write(a.out/'result.json',{'arms':summaries,'reference_accuracy':{fold:sum(r['chosen']>r['rejected'] for r in v)/len(v) for fold,v in cache['scores'].items()},'qualification':'DPO_REQUIRED_BUT_NOT_QUALIFIED pending constrained generation and physical validation','peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
if __name__=='__main__':main()
