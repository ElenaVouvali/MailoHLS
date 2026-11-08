"""CPU tests for the C-memory contract without loading either neural checkpoint."""
import torch
from LLM_branch.common import structural_memory as sm
from LLM_branch.common.structural_xattn import MaskedCrossAttention

def c_pack():
    slots=64; evidence=3; total=slots+evidence
    memory=torch.randn(total,128)
    mask=torch.zeros(total,dtype=torch.bool); mask[0]=True; mask[1]=True; mask[slots:slots+2]=True
    action_mask=torch.zeros(slots,dtype=torch.bool); action_mask[:2]=True
    relation=torch.zeros(total,total,dtype=torch.bool)
    relation[0,0]=True; relation[0,slots]=True
    relation[1,1]=True; relation[1,slots+1]=True
    return {'node_embs':memory,'node_embs_mask':mask,'gnn_dim':128,'max_slots':64,
      'memory_interface':'rc3_action_evidence_c_v1','action_slot_count':64,'action_slot_mask':action_mask,
      'evidence_start_slot':64,'action_relation_mask':relation,'disable_pragma_injection':True,
      'evidence_metadata':[{'evidence_type':'affine_memory_access','provenance':{'graph_identity':'g1'}}]*2}

def test_c_loader_preserves_absolute_actions_and_dynamic_key_axis(tmp_path):
    torch.save(c_pack(),tmp_path/'kernel.memory.pt')
    bank,dim=sm.load_memory_bank(str(tmp_path),expected_mem_dim=128,expected_max_slots=64)
    memory,valid,relation=sm.get_structural_memory_pack_for_kernel(bank,'kernel',64,dim,'compiler_relational')
    assert memory.shape==(1,67,128)
    assert valid.shape==(1,67) and valid.sum().item()==4
    assert relation.shape==(1,67,67)
    assert relation[0,0,64] and relation[0,1,65]
    assert not relation[0,64:].any()

def test_c_masked_attention_cannot_read_padding_or_other_action_keys(tmp_path):
    torch.manual_seed(7); pack=c_pack()
    torch.save(pack,tmp_path/'kernel.memory.pt')
    bank,_=sm.load_memory_bank(str(tmp_path),expected_mem_dim=128,expected_max_slots=64)
    memory,valid,relation=sm.get_structural_memory_pack_for_kernel(bank,'kernel',64,128,'compiler_relational')
    x=torch.randn(1,1,16,requires_grad=True)
    ids=torch.tensor([[1]]) # the source marker's forward-filled L1 action slot
    module=MaskedCrossAttention(dim=16,dim_memory=128,dim_head=8,heads=2,mask_mode='segment')
    out=module(x,memory,placeholder_slot_ids=ids,memory_mask=valid,action_relation_mask=relation)
    changed=memory.clone()
    allowed=relation[0,0]&valid[0]
    changed[:,~allowed]+=1000
    same=module(x,changed,placeholder_slot_ids=ids,memory_mask=valid,action_relation_mask=relation)
    assert torch.isfinite(out).all() and torch.allclose(out,same,atol=1e-6,rtol=1e-6)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_empty_evidence_and_inactive_absolute_slots_remain_masked(tmp_path):
    pack=c_pack(); pack['action_slot_mask'][1]=False; pack['node_embs_mask'][1]=False
    pack['action_relation_mask'][1].zero_()
    pack['action_relation_mask'][:,1].zero_()
    torch.save(pack,tmp_path/'kernel.memory.pt')
    bank,_=sm.load_memory_bank(str(tmp_path),expected_mem_dim=128,expected_max_slots=64)
    memory,valid,relation=sm.get_structural_memory_pack_for_kernel(bank,'kernel',64,128,'compiler_relational')
    assert not valid[0,1] and not relation[0,:,1].any() and not relation[0,1].any()

def test_variable_evidence_packs_batch_with_false_padding():
    a=torch.ones(66,128); b=torch.full((70,128),2.0)
    ma=torch.ones(66,dtype=torch.bool); mb=torch.ones(70,dtype=torch.bool)
    ra=torch.eye(66,dtype=torch.bool); rb=torch.eye(70,dtype=torch.bool)
    kv,mask,rel=sm.pad_memory_pack_batch([a,b],[ma,mb],[ra,rb])
    assert kv.shape==(2,70,128) and mask.shape==(2,70) and rel.shape==(2,70,70)
    assert torch.equal(kv[0,:66],a) and not mask[0,66:].any()
    assert not rel[0,66:,:].any() and not rel[0,:,66:].any()
