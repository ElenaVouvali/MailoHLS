from types import SimpleNamespace
import torch
import pytest
from LLM_branch.train.train_DPO_harp_xattn import STRUCTURALDPOTrainer
from LLM_branch.common import structural_memory
from experiments.final_architecture_v3.preferences_objective import dpo_loss,selected_logps

def test_sign_and_reference_stop_gradient():
 a=torch.tensor(-3.,requires_grad=True);b=torch.tensor(-4.,requires_grad=True);r=torch.tensor(-3.,requires_grad=True)
 loss=dpo_loss(a,b,r,torch.tensor(-4.));loss.backward()
 assert loss.item()==pytest.approx(.69314718) and a.grad<0 and b.grad>0 and r.grad is None
 # Numerical derivative agrees with autograd.
 eps=1e-3
 derivative=(dpo_loss(a.detach()+eps,b.detach(),r,torch.tensor(-4.))-dpo_loss(a.detach()-eps,b.detach(),r,torch.tensor(-4.)))/(2*eps)
 assert derivative.item()==pytest.approx(a.grad.item(),abs=1e-4)

def test_full_value_sum_and_weighted_anchor_are_distinct():
 logits=torch.tensor([[[1.,2.,3.],[3.,2.,1.],[1.,1.,1.],[2.,1.,0.]]],requires_grad=True)
 labels=torch.tensor([-100,2,0,-100]);weights=torch.tensor([0.,.5,.5,0.])
 total,sft,n=selected_logps(logits,labels,weights)
 assert n==2 and total.item()==pytest.approx(-2*sft.item());(-total).backward();assert logits.grad[0,2:].abs().sum()==0

def test_real_stage3_mixed_length_memory_padding():
 trainer=object.__new__(STRUCTURALDPOTrainer);trainer.mem_bank={};trainer.mem_dim=128;trainer.max_slots=64;trainer.structural_routing='compiler_relational'
 def get(bank,name,*a,**kw):
  size={'a':67,'b':73}[name];return torch.ones(1,size,128),torch.ones(1,size,dtype=torch.bool),torch.ones(1,size,size,dtype=torch.bool)
 trainer._sft_mod=SimpleNamespace(structural_memory_utils=SimpleNamespace(get_structural_memory_pack_for_kernel=get,pad_memory_pack_batch=structural_memory.pad_memory_pack_batch))
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
  def condition_structural_memory(self,memory,mask,**kw):self.values=(memory,mask,kw['action_relation_mask'])
 m=Model();trainer._condition_structural_memory_from_kernel_names(m,['a','b']);kv,mask,rel=m.values
 assert kv.shape==(2,73,128) and not mask[0,67:].any() and not rel[0,67:,:].any() and not rel[0,:,67:].any()
