"""Reuse head-only counterfactuals on identical real backbone states, no labels."""
import hashlib,json
import torch
from torch.nn import functional as F
from .alignment import causal_kinds

def cache_key(ids,text,context):
 return hashlib.sha256(json.dumps([ids.tolist(),text,context.tolist()],separators=(',',':')).encode()).hexdigest()

@torch.no_grad()
def paired_scores(model,tok,raw,ctx,kind,state,film,text,base_len):
 ids=tok(text,add_special_tokens=False)['input_ids'];n=len(ids);saved=state['kinds'];capture=state.get('capture');state['kinds']=None;state['capture']=False;result={}
 try:
  for mode in ('off','value','legacy'):
   kinds=causal_kinds(tok,text,base_len,kind,raw.device,'legacy' if mode=='legacy' else 'value')
   h=raw if mode=='off' else film(raw,ctx,kinds)
   logits=model.get_output_embeddings()(h)[:,base_len-1:base_len-1+n].float();target=torch.tensor(ids,device=raw.device)[None,:,None];lp=F.log_softmax(logits,-1).gather(-1,target).squeeze(-1);result[mode]={'sum_logprob':float(lp.sum()),'mean_logprob':float(lp.mean()),'token_count':n}
   del logits,lp
 finally:state['kinds']=saved;state['capture']=capture
 return result
