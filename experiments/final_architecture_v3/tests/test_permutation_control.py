import torch
from LLM_branch.common.structural_xattn import MaskedCrossAttention
from experiments.final_architecture_20261109.evidence_controls import changed

def test_attention_with_same_route_key_permutation_is_placebo():
 torch.manual_seed(123);module=MaskedCrossAttention(dim=16,dim_memory=8,heads=2,dim_head=4);h=torch.randn(1,2,16);kv=torch.randn(1,4,8);mask=torch.ones(1,4,dtype=torch.bool);slots=torch.ones(1,2,dtype=torch.long);relation=torch.zeros(1,4,4,dtype=torch.bool);relation[:,0,:3]=True
 def score(memory):return module(h,memory,placeholder_slot_ids=slots,memory_mask=mask,action_relation_mask=relation)
 a=score(kv);same_route=kv[:,[0,2,1,3]];different_route=kv[:,[0,3,2,1]]
 assert torch.allclose(a,score(same_route),atol=1e-6,rtol=1e-6)
 assert not torch.allclose(a,score(different_route),atol=1e-5,rtol=1e-5)

def test_cross_route_control_changes_content_not_masks_or_actions():
 torch.manual_seed(123);kv=torch.randn(68,128);mask=torch.ones(68,dtype=torch.bool);relation=torch.zeros(68,68,dtype=torch.bool);relation[0,64:66]=True;relation[1,66:68]=True
 meta=[{'evidence_type':'ssa','evidence_status':'proven','known':{}} for _ in range(4)];record={'kv':kv,'mask':mask,'relation_mask':relation,'evidence_metadata':meta}
 changed_record=changed({'kernel':record},'cross_route_evidence')['kernel']
 assert torch.equal(changed_record['kv'][:64],kv[:64]) and torch.equal(changed_record['mask'],mask) and torch.equal(changed_record['relation_mask'],relation)
 assert not torch.equal(changed_record['kv'][64:66],kv[64:66])
