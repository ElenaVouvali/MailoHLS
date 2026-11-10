"""Matched residual-LoRA / residual-LoRA+FiLM; action-SFT plus same-request rank."""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from peft import PeftModel,LoraConfig,prepare_model_for_kbit_training
from LLM_branch.common import device_adaptation as da
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from LLM_branch.train import train_SFT_xattn_new as prod
from experiments.final_architecture_v3.supervision import corrected
from experiments.final_architecture_v3.preferences_objective import selected_logps
from .model import ActionFiLM,features,kind_id,install

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--steps',type=int,default=64);p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 from experiments.final_architecture_20261109.run import preflight
 receipt=preflight(a.gpu_uuid);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
 if a.out.exists() and not a.resume:raise FileExistsError(a.out)
 a.out.mkdir(parents=True,exist_ok=True)
 if sha(a.parent/'adapter_model.safetensors')!='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164':raise ValueError('Wrong V4 parent')
 contract=json.loads((a.parent/'training_contract.json').read_text());tok=inf.load_checkpoint_tokenizer(str(a.parent),contract);registry=inf.load_directive_domain_registry(str(a.parent/'directive_domain_registry.json'))
 pairs={fold:[json.loads(s) for s in (a.pairs/(fold+'_pairs.jsonl')).read_text().splitlines()] for fold in ('train','val')}
 assert {x['request']['family'] for x in pairs['train']}.isdisjoint({x['request']['family'] for x in pairs['val']})
 if a.smoke:pairs={k:v[:2] for k,v in pairs.items()}
 encoded={}
 for fold,rows in pairs.items():
  encoded[fold]=[]
  for row in rows:
   samples=[]
   for side in ('preferred','rejected'):
    example=dict(row['row'],target=row[side+'_target'])
    with corrected(prod):ds=prod.SFTDataset([example],tok,contract['max_length'],directive_domain_registry=registry,supervise_eos=False,input_only_special_ids=contract['special_token_ids'])
    sample=ds.samples[0];pack=prod.build_deterministic_rhs_pack(example['input'],prod.reorder_target_by_source_order(example['input'],example['target']),tok,directive_domain_registry=registry,kernel_name=example['kernel_name']);offset=len(sample['input_ids'])-len(pack.input_ids);assert sample['input_ids'][offset:].tolist()==pack.input_ids
    kinds=torch.zeros(len(sample['input_ids']),dtype=torch.long)
    for j,key in enumerate(pack.site_keys):
     if key is not None and offset+j>0:kinds[offset+j-1]=kind_id(key)
    assert (kinds[sample['xattn_apply_mask'].bool()]>0).all();sample['film_kinds']=kinds;samples.append(sample)
   encoded[fold].append((row,samples))
 random.seed(123);torch.manual_seed(123)
 base=inf.load_base_model(contract['model'],revision=contract['model_revision']);base.resize_token_embeddings(len(tok));base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
 model=PeftModel.from_pretrained(base,str(a.parent),adapter_name='mailohls_base',is_trainable=False);da.isolate_parent_token_serialization(model)
 cfg=json.loads((a.parent/'adapter_config.json').read_text());model.add_adapter('device_adapt',LoraConfig(r=cfg['r'],lora_alpha=cfg['lora_alpha'],target_modules=cfg['target_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'));da.activate(model,trainable=True);model.enable_input_require_grads();assert da.trainables_contract(model)['trainable_numel']>0
 frozen=da.frozen_parent_hash(model);device=next(model.parameters()).device
 film=ActionFiLM(model.config.hidden_size).to(device);state,handle=install(model,film)
 initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};film_initial=copy.deepcopy(film.state_dict());order=list(range(len(encoded['train'])));random.Random(123).shuffle(order);schedule=[order[i%len(order)] for i in range(a.steps*2)]
 write(a.out/'protocol.json',{'parent_sha256':sha(a.parent/'adapter_model.safetensors'),'parent_contract_sha256':sha(a.parent/'training_contract.json'),'data_sha256':{f:sha(a.pairs/(f+'_pairs.jsonl')) for f in pairs},'schedule':schedule,'seed':123,'steps':a.steps,'effective_batch':2,'lr':2e-5,'lambda_rank':.1,'tau':1.,'selection':'validation family-macro action-normalized pair accuracy, then chosen NLL; no GAN','preflight':receipt,'trainable_inventory':da.trainables_contract(model),'code_files':{str(Path(__file__).name):sha(__file__), 'model.py':sha(Path(__file__).with_name('model.py'))},'supervision':'paired complete measured preferred references; proxy feasibility, no timing claim; each completion own prefix'})
 def score(sample,row,active):
  state['context']=features(row).to(device)[None,None,:];state['kinds']=sample['film_kinds'].to(device)[None] if active else None
  with torch.autocast('cuda',dtype=torch.float16):o=model(input_ids=sample['input_ids'].to(device)[None],attention_mask=sample['attention_mask'].to(device)[None],use_cache=False)
  _,nll,_=selected_logps(o.logits,sample['labels'].to(device),sample['token_weights'].to(device));return -nll
 def evaluate(active):
  model.eval();film.eval();result=[]
  with torch.no_grad():
   for row,(pos,neg) in encoded['val']:
    x,y=score(pos,row,active),score(neg,row,active);result.append({'pair_id':row['pair_id'],'family':row['request']['family'],'margin':float(x-y),'chosen_nll':float(-x),'correct':bool(x>y)})
  families=sorted({r['family'] for r in result});accuracy=sum(sum(r['correct'] for r in result if r['family']==f)/sum(r['family']==f for r in result) for f in families)/len(families)
  return accuracy,sum(r['chosen_nll'] for r in result)/len(result),result
 resume_state=torch.load(a.resume/'progress.pt',map_location='cpu',weights_only=False) if a.resume else None
 for arm in ('A','B'):
  if a.resume and arm<a.resume.parent.name:continue
  folder=a.out/arm;folder.mkdir(exist_ok=True);random.seed(123);torch.manual_seed(123)
  for n,pv in model.named_parameters():
   if n in initial:pv.data.copy_(initial[n].to(pv.device))
  film.load_state_dict(film_initial);active=arm=='B';parameters=[pv for pv in model.parameters() if pv.requires_grad]+(list(film.parameters()) if active else []);optimizer=torch.optim.AdamW(parameters,lr=2e-5,weight_decay=0.);scaler=torch.amp.GradScaler('cuda');best=(-1.,-float('inf'));started=time.monotonic();successful=0
  if a.resume and arm==a.resume.parent.name:
   from safetensors.torch import load_file
   from peft import set_peft_model_state_dict
   set_peft_model_state_dict(model,load_file(str(a.resume/'device_adapt/adapter_model.safetensors')),adapter_name='device_adapt');film.load_state_dict(torch.load(a.resume/'film.pt',map_location=device,weights_only=True));optimizer.load_state_dict(resume_state['optimizer']);scaler.load_state_dict(resume_state['scaler']);successful=resume_state['step'];torch.set_rng_state(resume_state['rng']);torch.cuda.set_rng_state_all(resume_state['cuda_rng'])
   selected=json.loads((folder/'selected.json').read_text());best=(selected['macro_pair_accuracy'],-selected['chosen_nll'])
  def save(step):
   dest=folder/('checkpoint-'+str(step));dest.mkdir(exist_ok=True);model.save_pretrained(dest,selected_adapters=['device_adapt'],save_embedding_layers=False);torch.save(film.state_dict(),dest/'film.pt');torch.save({'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},dest/'progress.pt');write(dest/'adaptation_contract.json',{'schema':'stage1-target-adaptation-v1','arm':arm,'parent':str(a.parent),'parent_sha256':sha(a.parent/'adapter_model.safetensors'),'parent_tensor_sha256':frozen,'steps':step,'film_enabled':active,'frozen_parent_verified':da.frozen_parent_hash(model)==frozen,'lora_sha256':sha(dest/'device_adapt/adapter_model.safetensors'),'film_sha256':sha(dest/'film.pt')});return dest
  for step in range(successful,a.steps+1):
   if step in (0,16,32,48,a.steps):
    accuracy,nll,records=evaluate(active);dest=save(step);write(dest/'validation.json',{'macro_pair_accuracy':accuracy,'chosen_nll':nll,'records':records});key=(accuracy,-nll)
    if key>best:best=key;write(folder/'selected.json',{'checkpoint':str(dest),'step':step,'macro_pair_accuracy':accuracy,'chosen_nll':nll})
    print('[VAL]',arm,step,accuracy,nll,flush=True)
   if step==a.steps:break
   model.train();film.train();optimizer.zero_grad(set_to_none=True);losses=[]
   for ix in schedule[step*2:(step+1)*2]:
    row,(pos,neg)=encoded['train'][ix];x=score(pos,row,active);y=score(neg,row,active);loss=-x+.1*F.softplus(-(x-y));scaler.scale(loss/2).backward();losses.append({'sft':float(-x.detach()),'rank':float(F.softplus(-(x-y)).detach())})
   scaler.unscale_(optimizer);grads={n:float(p.grad.norm()) for n,p in film.named_parameters() if p.grad is not None};norm=torch.nn.utils.clip_grad_norm_(parameters,1.);before=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped=scaler.get_scale()<before
   if skipped:raise FloatingPointError('Skipped optimizer update; preserve checkpoint and diagnose')
   successful+=1
   with (folder/'training.jsonl').open('a') as f:f.write(json.dumps({'step':successful,'losses':losses,'grad_norm':float(norm),'film_gradients':grads,'gate':float(film.gate.tanh()),'elapsed_seconds':time.monotonic()-started})+'\n');f.flush();os.fsync(f.fileno())
   print('[TRAIN]',arm,successful,losses,grads,flush=True)
  assert da.frozen_parent_hash(model)==frozen
  write(folder/'completed.json',{'updates':successful,'runtime_seconds':time.monotonic()-started,'parent_unchanged':True,'physical_HLS_calls':0})
if __name__=='__main__':main()
