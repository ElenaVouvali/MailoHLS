"""Experimental output-head FiLM; explicit causal positions, no source-token edits."""
import math,re
import torch
from torch import nn
KINDS=('PIPE','UNROLL','ARRAY_T','ARRAY_F','ARRAY_D')

def kind_id(lhs):
 m=re.fullmatch(r'AUTO\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*\}',lhs.upper())
 if not m:raise ValueError(lhs)
 return KINDS.index(m[1])+1

def features(pair):
 # Deterministic physical scaling; no fitted statistics or measured QoR inputs.
 req=pair['request'];part=req['physical_context'][3];clock=float(req['physical_context'][4])
 from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES,RESOURCE_KEYS
 caps=[DEVICE_RESOURCES[part][k] for k in RESOURCE_KEYS];budget=[req['budget'][k] for k in ('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut')]
 known={'xczu7ev-ffvc1156-2-e':0,'xcu200-fsgd2104-2-e':1}
 if part not in known:raise ValueError('Unauthenticated device: '+part)
 return torch.tensor([float(known[part]==0),float(known[part]==1),math.log(clock/10),*[math.log1p(float(v))/16 for v in caps],*[float(v) for v in budget],*[math.log1p(float(c)*float(b))/16 for c,b in zip(caps,budget)]],dtype=torch.float32)

class ActionFiLM(nn.Module):
 def __init__(self,width,context_dim=15,bottleneck=32):
  super().__init__();self.kind=nn.Embedding(6,8);self.net=nn.Sequential(nn.Linear(context_dim+8,bottleneck),nn.SiLU(),nn.Linear(bottleneck,2*width));self.gate=nn.Parameter(torch.zeros(()));self.width=width
 def forward(self,h,context,kinds):
  if h.shape[:-1]!=kinds.shape:raise ValueError('FiLM position mismatch')
  selected=kinds.gt(0)
  if not selected.any():return h
  source=h[selected];z=context.expand(*kinds.shape,-1)[selected];ab=self.net(torch.cat((z,self.kind(kinds[selected])),dim=-1));gamma,beta=ab.chunk(2,-1)
  delta=self.gate.tanh()*(.1*gamma.tanh()*nn.functional.layer_norm(source.float(),(self.width,))+.1*beta.tanh())
  result=h.clone();result[selected]=source+delta.to(source.dtype);return result

def install(model,film):
 state={'context':None,'kinds':None}
 def hook(module,args):
  h=args[0]
  if state['kinds'] is None:return args
  kinds=state['kinds']
  # Cached decode callers supply current-token kinds explicitly; no prefix inference.
  if kinds.shape[1]!=h.shape[1]:raise ValueError('Cached FiLM requires current causal position metadata')
  return (film(h,state['context'],kinds),*args[1:])
 handle=model.get_output_embeddings().register_forward_pre_hook(hook)
 return state,handle
