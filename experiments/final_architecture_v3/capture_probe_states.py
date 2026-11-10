"""Selected-position Stage1/fused states from identical teacher-forced prefixes.
No cross-kernel semantic-generalization claim is made by extraction itself.
"""
import hashlib,json
from pathlib import Path
import torch

def capture(model,tok,encoded,score,out):
 out=Path(out)
 if out.exists():return
 first={}
 for fold,items in encoded.items():
  for item in items:first.setdefault(item['pair']['row']['kernel_name'],(fold,item))
 gates=[p for n,p in model.named_parameters() if n.endswith('attn_gate')];saved=[p.detach().clone() for p in gates];records=[];vectors=[];model.eval()
 target_ids={tok.convert_tokens_to_ids('<L'+str(i)+'>'):i for i in range(1,65)}
 try:
  for kernel,(fold,item) in sorted(first.items()):
   sample=item['preferred'];slots={};action=None;start=int(sample['routing_start_idx'])
   for i,token in enumerate(sample['input_ids'].tolist()):
    if i<start:continue
    if token in target_ids:action=target_ids[token]
    if action is not None and int(sample['labels'][i])!=-100:slots.setdefault(action,i-1)
   if not slots:continue
   positions=list(slots.values());actions=list(slots)
   for mode in ('stage1_gate_off','static_fused'):
    with torch.no_grad():
     for p,value in zip(gates,saved):p.copy_(torch.zeros_like(value) if mode=='stage1_gate_off' else value)
    captures={};handles=[]
    for name,module in model.named_modules():
     if not name.endswith('gated_cross_attn_layer'):continue
     layer=int(name.split('.layers.')[1].split('.')[0])+1;data={};captures[layer]=data
     def pre(m,args,data=data):data['before']=args[0][0,positions].detach().float().cpu().clone()
     def post(m,args,result,data=data):data['after']=result[0,positions].detach().float().cpu().clone()
     handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
    with torch.no_grad():traced=score(sample,kernel)[0]
    for handle in handles:handle.remove()
    with torch.no_grad():untraced=score(sample,kernel)[0]
    if abs(float(traced-untraced))>1e-6:raise ValueError('Probe hook changed selected causal score')
    for layer,data in captures.items():
     for index,action in enumerate(actions):
      vid=len(vectors);vectors.append({'before':data['before'][index],'after':data['after'][index]});records.append({'kernel':kernel,'family':item['pair']['request']['family'],'split':fold,'action':'L'+str(action),'mode':mode,'layer':layer,'position':positions[index],'tensor_index':vid,'encoding_sha256':item['encoding_sha256'],'context':item['pair']['request']['physical_context'],'scope':'first nonforced value per action; preferred complete teacher prefix; same source/context/prefix for gate-off and fused','hook_parity':True})
 finally:
  with torch.no_grad():
   for p,value in zip(gates,saved):p.copy_(value)
 tensors=out.with_suffix('.pt');torch.save(vectors,tensors);out.write_text(json.dumps({'schema':'v3-selected-probe-states-v1','records':records,'tensor_sha256':hashlib.sha256(tensors.read_bytes()).hexdigest(),'tensor_file':str(tensors),'families':sorted({r['family'] for r in records}),'kernels':len(first),'limitations':['teacher prefixes include previous measured preferred directives','gate-off is same frozen Stage1 with fusion disabled, not another checkpoint','same source varies across only one context here; no useful QoR established']},indent=2)+'\n')
