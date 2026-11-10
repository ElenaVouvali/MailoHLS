"""Auxiliary frozen Stage1 representation capture; no directive predictions claimed.
Uses official loader/tokenizer/prompt and selected positions only. Dedicated A2
execution is explicitly tagged and NF4 computation forced to FP16, as on V100S.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from LLM_branch.train import train_SFT_xattn_new as production
from .supervision import corrected
from .preferences_objective import selected_logps
from experiments.final_architecture_20261109.run import preflight
UUID='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35'
V4=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print('Capture one ZCU104/10ns/full source representative per kernel on dedicated A2, frozen Stage1, FP16 NF4, layers8/16/24/32.');return
 receipt=preflight(UUID,minimum=14000);a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=UUID
 expected='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'
 if sha(V4/'adapter_model.safetensors')!=expected:raise ValueError('V4 identity mismatch')
 rows=[json.loads(s) for s in a.pilot.read_text().splitlines()];selected={}
 for r in rows:
  if r['device']=='xczu7ev-ffvc1156-2-e' and float(r['selected_clock_period'])==10. and float(r['resource_budget_frac'])==1.:selected.setdefault(r['kernel_name'],r)
 contexts=[dict(r,context_id='probe:'+k+':zcu104:10:full') for k,r in sorted(selected.items())];requests=a.out/'requests.jsonl'
 text=''.join(json.dumps(r)+'\n' for r in contexts)
 if requests.exists() and requests.read_text()!=text:raise ValueError('Probe requests changed')
 requests.write_text(text);parent=json.loads((V4/'training_contract.json').read_text());registry=inference.load_directive_domain_registry(str(V4/'directive_domain_registry.json'));old_config=inference.BitsAndBytesConfig
 def fp16(**kwargs):kwargs['bnb_4bit_compute_dtype']=torch.float16;return old_config(**kwargs)
 inference.BitsAndBytesConfig=fp16
 def capture(**kwargs):
  model,tok,case=kwargs['model'],kwargs['tok'],kwargs['case'];kernel=case.kernel_name;dest=a.out/(kernel+'.json')
  if dest.exists():return {'context_id':case.platform_row['context_id'],'status':'preserved_capture'}
  row=selected[kernel];model.requires_grad_(False);model.eval()
  with corrected(production):dataset=production.SFTDataset([row],tok,parent['max_length'],directive_domain_registry=registry,supervise_eos=False,input_only_special_ids=parent['special_token_ids'])
  sample=dataset.samples[0];ids=sample['input_ids'];target_ids={tok.convert_tokens_to_ids('<L'+str(i)+'>'):i for i in range(1,65)};slots={};action=None
  for i,t in enumerate(ids.tolist()):
   if i<int(sample['routing_start_idx']):continue
   if t in target_ids:action=target_ids[t]
   if action is not None and int(sample['labels'][i])!=-100:slots.setdefault(action,i-1)
  positions=list(slots.values());actions=list(slots);cache={};handles=[];attr=inference.infer_decoder_layers_attr_name(model);layers=model.get_submodule(attr);assert len(layers)==32 and model.config.hidden_size==4096
  for layer in (8,16,24,32):
   block=layers[layer-1];data={};cache[layer]=data
   def attention_boundary(m,args,data=data):data['post_self_attention_residual']=args[0][0,positions].detach().float().cpu().clone()
   def decoder_boundary(m,args,output,data=data):data['post_decoder_residual']=output[0][0,positions].detach().float().cpu().clone()
   handles.extend([block.post_attention_layernorm.register_forward_pre_hook(attention_boundary),block.register_forward_hook(decoder_boundary)])
  head={}
  def head_boundary(m,args):head['final_normalized']=args[0][0,positions].detach().float().cpu().clone()
  handles.append(model.get_output_embeddings().register_forward_pre_hook(head_boundary));device=inference.get_first_real_device(model)
  def forward():
   with torch.no_grad():
    output=model(input_ids=ids[None].to(device),attention_mask=sample['attention_mask'][None].to(device));score=selected_logps(output.logits,sample['labels'].to(device),sample['token_weights'].to(device))[0].detach().cpu();del output;return score
  traced=forward()
  for h in handles:h.remove()
  untraced=forward();assert abs(float(traced-untraced))<=1e-5
  records=[];vectors=[]
  for layer,data in cache.items():
   for boundary,values in data.items():
    for i,slot in enumerate(actions):records.append({'action':'L'+str(slot),'position':positions[i],'layer':layer,'boundary':boundary,'tensor_index':len(vectors)});vectors.append(values[i])
  for i,slot in enumerate(actions):records.append({'action':'L'+str(slot),'position':positions[i],'layer':32,'boundary':'final_normalized','tensor_index':len(vectors)});vectors.append(head['final_normalized'][i])
  tensorfile=a.out/(kernel+'.pt');torch.save(vectors,tensorfile);graph=a.graphs/(kernel+'.json')
  report={'schema':'v3-stage1-selected-representations-v1','kernel':kernel,'family':row['_family'],'split':row['_quality_split'],'records':records,'source_prompt_sha256':hashlib.sha256(row['input'].encode()).hexdigest(),'encoding_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),'graph_sha256':sha(graph),'tensor_file':str(tensorfile),'tensor_sha256':sha(tensorfile),'hook_score_parity':True,'adapter_sha256':expected,'precision':'NF4 FP16 compute; frozen official Stage1 inference loader','hardware':receipt,'context':'ZCU104/10ns/full','prefix':'first nonforced value/action under frozen authenticated preferred whole completion; previous choices are teacher-forced','limitations':['Auxiliary A2 captures, not V100S candidate-ranking or physical performance evidence','No generated directive prediction from this callback','Compiler fact recoverability is not decision relevance']};dest.write_text(json.dumps(report,indent=2)+'\n');print('[CAPTURE]',kernel,len(actions),flush=True)
  return {'context_id':case.platform_row['context_id'],'status':'captured','actions':len(actions)}
 inference.predict_case=capture;sys.argv=[sys.argv[0],'--stage','stage1','--adapter_dir',str(V4),'--directive_domain_registry_json',str(V4/'directive_domain_registry.json'),'--input_jsonl',str(requests),'--output_jsonl',str(a.out/'completion.jsonl'),'--num_samples','1']
 inference.main();receipt.update(runtime_seconds=time.monotonic()-started,request_sha256=sha(requests),kernels=len(contexts),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),compute_dtype='float16',auxiliary_only=True)
 (a.out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
