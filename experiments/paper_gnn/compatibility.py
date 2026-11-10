"""Real maintained XAttn forward/backward on each new frozen memory bank."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import torch
from LLM_branch.common.structural_memory import load_memory_bank
from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
from experiments.paper_gnn.data import sha


def check(bank_path, output, expected_gpu):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for execution receipt')
    name=torch.cuda.get_device_name(0)
    if expected_gpu not in name: raise RuntimeError('Unexpected GPU '+name)
    manifest=json.loads((bank_path/'memory_manifest.json').read_text())
    if manifest['representation_schema']!='salt-paper-rc3-transformer4-jkmax-conv1-jk-v1':
        raise ValueError('Incompatible memory representation')
    if manifest['layout']!=['conv_1:0:64','jk_max:64:128']: raise ValueError('Incompatible layout')
    for filename, digest in manifest['files'].items():
        if sha(bank_path/filename)!=digest: raise ValueError('Bank hash changed')
        pack=torch.load(bank_path/filename,map_location='cpu',weights_only=False)
        if pack['gnn_checkpoint_sha256']!=manifest['gnn_checkpoint_sha256'] or pack['feature_schema_sha256']!=manifest['feature_schema_sha256']:
            raise ValueError('Mixed checkpoint or schema bank')
        if pack['normalization_sha256']!=manifest['normalization_sha256']: raise ValueError('Mixed normalization')
        if not torch.isfinite(pack['node_embs']).all() or pack['node_embs'][~pack['node_embs_mask']].count_nonzero():
            raise ValueError('Invalid or nonzero padded memory')
    bank,dim=load_memory_bank(str(bank_path),expected_mem_dim=128,expected_max_slots=64,require_pragma_free_memory=True,
        expected_representation_schema=manifest['representation_schema'])
    torch.manual_seed(123)
    checks=[]
    for kernel in sorted({r['_source_kernel'] for r in bank.values()}):
        rec=bank[kernel] if kernel in bank else next(r for r in bank.values() if r['_source_kernel']==kernel)
        if not rec['mask'].any(): continue
        memory=rec['kv'].unsqueeze(0).cuda(); mask=rec['mask'].unsqueeze(0).cuda()
        active=rec['mask'].nonzero().flatten(); slot=int(active[0])+1
        slots=torch.tensor([[0,slot,slot,0]],device='cuda'); x=torch.randn(1,4,32,device='cuda')
        modes=[]
        for mode in ('segment','all','relation'):
            # A/B/C are visibility/routing interfaces over exactly the same
            # vectors. Each run starts with fresh fusion weights.
            block=GatedCrossAttentionBlock(dim=32,dim_memory=128,dim_head=8,heads=2,mask_mode='segment').cuda()
            # The maintained mechanism always requires action queries. Global
            # visibility is an all-active relation mask, not missing slot IDs.
            kwargs={'placeholder_slot_ids':slots,'memory_mask':mask}
            if mode=='all':kwargs['action_relation_mask']=mask.unsqueeze(-1)&mask.unsqueeze(-2)
            if mode=='relation':kwargs['action_relation_mask']=rec['relation_mask'].unsqueeze(0).cuda()
            zero=block(x,memory,**kwargs)
            if not torch.equal(zero,x): raise ValueError('Zero gate is not identity')
            with torch.no_grad():block.attn_gate.fill_(0.1)
            observed=block(x,memory,**kwargs)
            observed.square().mean().backward()
            grads=[p.grad for p in block.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
                raise ValueError('Missing/nonfinite XAttn backward gradients')
            modes.append(mode)
        checks.append({'kernel':kernel,'modes':modes,'zero_gate_exact':True,'forward_backward_finite':True})
    report={'schema':'salt-paper-xattn-check-v1','status':'passed','gpu':name,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'implementation_sha256':sha(Path(__file__)),
        'memory_loader_sha256':sha(Path(__file__).resolve().parents[2]/'LLM_branch/common/structural_memory.py'),
        'xattn_module_sha256':sha(Path(__file__).resolve().parents[2]/'LLM_branch/common/structural_xattn.py'),
        'torch':torch.__version__,'memory_manifest_sha256':sha(bank_path/'memory_manifest.json'),
        'checks':checks,'fresh_fusion_weights':True,'real_backbone_generation':False,
        'downstream_qor_improvement_established':False}
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-gpu',default='V100'); a=p.parse_args()
    check(a.bank,a.output,a.expected_gpu)


if __name__=='__main__':main()
