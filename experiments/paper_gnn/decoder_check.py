"""Real decoder ABI/gradient check with a frozen small Llama and actual bank.

Random tiny decoder weights isolate interface compatibility. This is not a
pretrained Stage-1/Stage-2 quality comparison or a released optimizer update.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from LLM_branch.common.structural_memory import load_memory_bank
from LLM_branch.common.structural_xattn import (StructuralCrossAttentionMixin,
    GatedCrossAttentionBlock, extend_instance, infer_decoder_layers_attr_name)
from experiments.paper_gnn.data import sha
from experiments.paper_gnn.train import gpu_receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-gpu',default='V100');a=p.parse_args()
    receipt=gpu_receipt(a.expected_gpu);torch.set_num_threads(4);torch.manual_seed(123)
    manifest=json.loads((a.bank/'memory_manifest.json').read_text())
    bank,dim=load_memory_bank(str(a.bank),expected_mem_dim=128,expected_max_slots=64,
        require_pragma_free_memory=True,expected_representation_schema=manifest['representation_schema'])
    kernel=sorted({r['_source_kernel'] for r in bank.values() if r['mask'].any()})[0]
    rec=bank[kernel];slot=int(rec['mask'].nonzero()[0])+1
    cfg=LlamaConfig(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,attention_dropout=0.)
    base=LlamaForCausalLM(cfg).cuda().eval();fused=copy.deepcopy(base)
    extend_instance(fused,StructuralCrossAttentionMixin)
    fused.set_decoder_layers_attr_name(infer_decoder_layers_attr_name(fused))
    fused.init_structural_cross_attention(placeholder_token_ids=tuple(range(101,165)),
        lang_hidden_size=32,mem_hidden_size=128,cross_attn_every_n_layers=1,xattn_heads=2,
        xattn_dim_head=8,only_attend_immediate_memory=True,mask_mode='segment',
        structural_fusion_placement='post_self_attention_residual')
    fused.cuda().eval()
    memory=rec['kv'].unsqueeze(0).cuda();mask=rec['mask'].unsqueeze(0).cuda()
    fused.condition_structural_memory(memory,mask)
    tokens=torch.tensor([[7,100+slot,11,12,100+slot,13]],device='cuda');attention_mask=torch.ones_like(tokens)
    with torch.no_grad():
        baseline=base(input_ids=tokens,attention_mask=attention_mask).logits
        zero=fused(input_ids=tokens,attention_mask=attention_mask).logits
    error=float((baseline-zero).abs().max())
    if error>=1e-5:raise ValueError('Zero-gate real decoder logits differ')
    fused.requires_grad_(False)
    blocks=[m for m in fused.modules() if isinstance(m,GatedCrossAttentionBlock)]
    for block in blocks:
        block.requires_grad_(True)
        with torch.no_grad():block.attn_gate.fill_(0.1)
    fused.condition_structural_memory(memory,mask)
    loss=fused(input_ids=tokens,attention_mask=attention_mask,labels=tokens).loss
    loss.backward()
    grads=[p.grad for block in blocks for p in block.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
        raise ValueError('No finite real-decoder XAttn gradient')
    if any(p.grad is not None for p in base.parameters()):raise ValueError('Baseline received gradients')
    a.output.write_text(json.dumps({'schema':'salt-paper-decoder-abi-check-v1','status':'passed',
        'gpu':receipt,'bank_manifest_sha256':sha(a.bank/'memory_manifest.json'),
        'implementation_sha256':sha(Path(__file__)),'kernel':kernel,'memory_dim':dim,
        'decoder':'random initialized LlamaForCausalLM, hidden32, two decoder layers, frozen base',
        'zero_gate_max_logit_difference':error,'finite_xattn_backward':True,'fusion_blocks':len(blocks),
        'pretrained_quality_evaluation':False,'released_optimizer_updates':0},indent=2)+'\n')


if __name__=='__main__':main()
