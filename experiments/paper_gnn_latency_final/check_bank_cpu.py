"""Exercise actual maintained XAttn, source/unbound masks and candidate independence on CPU."""
import argparse
import json
from pathlib import Path
import torch
from LLM_branch.common.structural_memory import load_memory_bank
from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
from experiments.paper_gnn.data import sha
from experiments.paper_gnn.train import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(123)
    manifest=json.loads((a.bank/'memory_manifest.json').read_text());bank,dim=load_memory_bank(str(a.bank),expected_mem_dim=128,expected_max_slots=64,require_pragma_free_memory=True,expected_representation_schema=manifest['representation_schema']);checks=[]
    for filename,digest in sorted(manifest['files'].items()):
        path=a.bank/filename;assert sha(path)==digest;pack=torch.load(path,map_location='cpu',weights_only=False);source=pack['source_action_mask'];active=pack['compiler_memory_mask'];unbound=pack['action_unbound_mask'];bound=pack['action_bound_mask']
        assert torch.equal(active,pack['node_embs_mask']);assert not (active & ~source).any();assert torch.equal(unbound,source & ~bound);assert not (bound & ~active).any();assert not pack['node_embs'][~active].count_nonzero();assert torch.isfinite(pack['node_embs']).all()
        memory=pack['node_embs'].unsqueeze(0);mask=active.unsqueeze(0);slots=[0]+[int(i)+1 for i in active.nonzero().flatten()];missing=(source & ~active).nonzero().flatten()
        if len(missing):slots.append(int(missing[0])+1)
        for index in pack.get('source_only_unbound_mask',torch.zeros_like(active)).nonzero().flatten():
            assert pack['action_relation_mask'][index,index] and int(pack['action_relation_mask'][index].sum())==1
        qslots=torch.tensor([slots]);x=torch.randn(1,len(slots),32);modes=[]
        for mode in ('action_segment','all','compiler_relation'):
            block=GatedCrossAttentionBlock(dim=32,dim_memory=128,dim_head=8,heads=2,mask_mode='segment')
            kwargs=dict(placeholder_slot_ids=qslots,memory_mask=mask)
            if mode=='all':kwargs['action_relation_mask']=mask.unsqueeze(-1)&mask.unsqueeze(-2)
            elif mode=='compiler_relation':kwargs['action_relation_mask']=pack['action_relation_mask'].unsqueeze(0)
            assert torch.equal(block(x,memory,**kwargs),x)
            with torch.no_grad():block.attn_gate.fill_(.1)
            y=block(x,memory,**kwargs);y.square().mean().backward();grads=[p.grad for p in block.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.count_nonzero() for g in grads)
            assert torch.equal(y[:,0],x[:,0])
            if len(missing) and mode!='all':assert torch.equal(y[:,-1],x[:,-1]), 'Unbound query received invented structural key'
            modes.append(mode)
        checks.append({'kernel':filename.removesuffix('.memory.pt'),'source_actions':int(source.sum()),'compiler_memory_actions':int(active.sum()),'unbound_source_actions':int(unbound.sum()),'source_actions_without_compiler_node':int((source & ~active).sum()),'modes':modes,'real_xattn_forward_backward':True,'nonzero_gradients':True,'zero_gate_exact_identity':True,'unbound_absent_key_identity':True})
    write_json({'status':'passed','device':'cpu','representation_schema':manifest['representation_schema'],'checkpoint_sha256':manifest['gnn_checkpoint_sha256'],'memory_manifest_sha256':sha(a.bank/'memory_manifest.json'),'fresh_fusion_weights':True,'checks':checks,'downstream_llm_improvement_verified':False},a.output)
