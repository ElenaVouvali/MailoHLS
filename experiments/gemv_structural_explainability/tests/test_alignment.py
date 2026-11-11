"""Unit fixtures are not trained retrieval evidence; no figures use these outputs."""
import copy
from pathlib import Path

import pytest
import torch
import numpy._core.multiarray
from torch import nn

from experiments.gemv_structural_explainability.memory_alignment import MemoryAlignment
from experiments.gemv_structural_explainability.interventions import plan,apply,deltas
from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock


@pytest.fixture
def pack():
    torch.manual_seed(123);torch.set_num_threads(4)
    # Reuse actual frozen evidence states; tokenizer weights below are unit-test
    # initializations only, never saved as learned experimental results.
    p=Path(__file__).resolve().parents[3]/'artifacts/gemv_structural_explainability/raw/gemv.pt'
    return torch.load(p,map_location='cpu',weights_only=False)


@pytest.mark.parametrize('mode',['action','evidence','gmt'])
def test_absolute_action_prefix_and_allowed_routes(pack,mode):
    module=MemoryAlignment(pack['edge_dim'],mode)
    out=module(pack)
    assert torch.equal(out['memory'][:,:64],pack['action'])
    assert torch.equal(out['mask'][:,:64],pack['active'])
    assert not out['relation'][:,5:64].any()
    assert not out['relation'][:,64:].any()
    if mode!='action':
        n=pack['a'].shape[1]
        assert torch.equal(out['relation'][:,:64,64:64+n],pack['routes'])


def test_summary_masking_and_reconstruction_weights(pack):
    out=MemoryAlignment(pack['edge_dim'])(pack)
    n=pack['a'].shape[1]
    weights=out['summary_attention']
    for k in range(5):
        assert not weights[0,k,:,:,~pack['routes'][0,k]].any()
        torch.testing.assert_close(weights[0,k].sum(-1),torch.ones(4,2))
        assert out['relation'][0,k,64+n+2*k:64+n+2*k+2].all()
        assert out['relation'][0,k,64+n:].sum()==2
    assert not out['mask'][0,64+n+10:].any()


def test_empty_padding_capacity_and_knownness(pack):
    m=MemoryAlignment(pack['edge_dim'])
    changed={**pack,'valid':torch.zeros_like(pack['valid'])}
    out=m(changed)
    assert torch.isfinite(out['memory']).all()
    assert not out['memory'][:,64:].any()
    assert not out['mask'][:,64:].any()
    assert not out['summary_attention'].any()
    with pytest.raises(ValueError,match='capacity'):
        MemoryAlignment(pack['edge_dim'],capacity=1)(pack)
    with pytest.raises(ValueError,match='mask shape'):
        m(pack,evidence_keep=torch.ones(1,1,dtype=torch.bool))


def test_gradients_through_tokenizer_and_typed_edge_encoder(pack):
    m=MemoryAlignment(pack['edge_dim'])
    out=m(pack)
    n=pack['a'].shape[1]
    out['memory'][:,64+n:64+n+10].square().sum().backward()
    for name in ('queries','condition.weight','tokenizer.in_proj_weight','edge.0.weight',
                 'node.weight','type_embedding.weight','known.weight'):
        grad=dict(m.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0,name


def test_real_cross_attention_zero_gate_and_invalid_attention(pack):
    m=MemoryAlignment(pack['edge_dim']);out=m(pack)
    block=GatedCrossAttentionBlock(dim=32,dim_memory=128,heads=2,dim_head=8,attn_gate_init=0.)
    x=torch.randn(1,3,32);slots=torch.tensor([[0,1,5]])
    captured=[]
    block.attn._diagnostic_attention_observer=lambda a,mask:captured.append((a.detach(),mask))
    y=block(x,out['memory'],placeholder_slot_ids=slots,memory_mask=out['mask'],action_relation_mask=out['relation'])
    assert torch.equal(x,y)
    weights,allowed=captured[0]
    assert not weights.masked_select(~allowed.expand_as(weights)).any()
    assert not weights[:,:,0].any()


def test_interventions_reproducible_and_content_controls_matched(pack):
    specs=plan(pack)
    assert specs==plan(pack)
    rows=pack['records']
    for s in specs:
        for recipient,donor in s.replacements:
            assert pack['routes'][0,s.action,recipient]
            assert not pack['routes'][0,s.action,donor]
            assert all(rows[recipient][key]==rows[donor][key] for key in ('type','status','known'))
    m=MemoryAlignment(pack['edge_dim'])
    original=m(pack)
    recurrence=next(s for s in specs if s.action==4 and s.name=='remove_family:loop_carried')
    changed=apply(m,pack,recurrence)
    assert not changed['relation'][0,4,64+torch.tensor(recurrence.removed)].any()
    assert torch.equal(original['memory'][:,:64],changed['memory'][:,:64])
    assert not changed['summary_attention'][0,4,:,:,list(recurrence.removed)].any()
    assert torch.equal(apply(m,pack,recurrence)['memory'],changed['memory'])


def test_intervention_delta_uses_fixed_candidate_pair():
    effect=deltas({'a':-.5,'b':-.7,'c':-2.},{'a':-.8,'b':-.6,'c':-1.})
    assert effect['winner_changed'] and effect['winner']=='b'
    assert effect['delta_margin']==pytest.approx(-.4)
    assert effect['delta_logp']['a']==pytest.approx(-.3)
    with pytest.raises(ValueError):deltas({'a':0.},{'b':0.})


def test_causal_multitoken_suffix_includes_newline():
    from types import SimpleNamespace
    from LLM_branch.inference.eval_stage1_stage2_stage3 import score_rhs_candidate_suffix
    class Tokenizer:
        def __call__(self,text,**kwargs):return {'input_ids':[ord(c)%64 for c in text]}
    class Model(nn.Module):
        def forward(self,input_ids,**kwargs):
            positions=torch.arange(input_ids.shape[1],dtype=torch.float32)[None,:,None]
            logits=positions*torch.arange(64,dtype=torch.float32)[None,None,:]/100
            return SimpleNamespace(logits=logits)
    model=Model();tok=Tokenizer();base=torch.tensor([[1,2,3]])
    score=score_rhs_candidate_suffix(model=model,tok=tok,base_input_ids=base,
        base_attention_mask=torch.ones_like(base),candidate_text='32\n')
    ids=tok('32\n')['input_ids']
    logits=model(torch.cat((base,torch.tensor([ids])),1)).logits[0,2:5]
    expected=logits.log_softmax(-1)[torch.arange(3),torch.tensor(ids)].mean()
    assert score['token_count']==3
    assert score['mean_logprob']==pytest.approx(float(expected))
