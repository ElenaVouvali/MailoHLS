import torch
from LLM_branch.common.structural_xattn import MaskedCrossAttention

def test_readonly_observer_preserves_output_and_records_no_padding_mass():
 torch.manual_seed(123)
 block=MaskedCrossAttention(dim=32,dim_memory=12,dim_head=8,heads=2,mask_mode='segment')
 x=torch.randn(1,5,32);memory=torch.randn(1,4,12);slots=torch.tensor([[0,1,0,2,0]]);mask=torch.tensor([[True,True,True,False]])
 args=dict(placeholder_slot_ids=slots,memory_mask=mask)
 baseline=block(x,memory,**args);seen=[]
 block._diagnostic_attention_observer=lambda a,m:seen.append(a.detach().clone())
 observed=block(x,memory,**args)
 assert torch.equal(baseline,observed)
 assert len(seen)==1 and torch.equal(seen[0][...,3],torch.zeros_like(seen[0][...,3]))
 assert torch.equal(seen[0][:,:,0,:],torch.zeros_like(seen[0][:,:,0,:]))
 assert all('observer' not in key for key in block.state_dict())
