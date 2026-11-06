import math
from collections import Counter
import pytest
import torch
from .objective import dpo_loss, selected_logps, pair_loss
from .prepare import one_pair, balance
from .common import OUT, read, rows, authenticate


def test_standard_dpo_and_frozen_reference_gradient():
    a=torch.tensor(-2.,requires_grad=True);b=torch.tensor(-4.,requires_grad=True)
    ra=torch.tensor(-2.,requires_grad=True);rb=torch.tensor(-4.,requires_grad=True)
    loss=dpo_loss(a,b,ra,rb)
    assert loss.item()==pytest.approx(math.log(2))
    loss.backward();assert a.grad<0 and b.grad>0
    assert ra.grad is None and rb.grad is None
    assert dpo_loss(a+1,b,ra,rb)<loss


def test_causal_mask_sum_and_sft_mass_are_distinct():
    torch.manual_seed(2);logits=torch.randn(1,5,7,requires_grad=True)
    labels=torch.tensor([-100,-100,2,-100,4]);weights=torch.tensor([0.,0.,.5,0.,.5])
    summed,sft,count,correct=selected_logps(logits,labels,weights)
    expected=torch.log_softmax(logits.float(),-1)[0]
    assert count==2
    torch.testing.assert_close(summed,expected[1,2]+expected[3,4])
    torch.testing.assert_close(sft,-summed/2)
    total,dpo=pair_loss(summed,summed-1,summed.detach(),(summed-1).detach(),sft)
    torch.testing.assert_close(total,sft+.2*dpo)
    total.backward()
    assert logits.grad[0,0].abs().sum()==0 and logits.grad[0,2].abs().sum()==0
    assert logits.grad[0,1].abs().sum()>0 and logits.grad[0,3].abs().sum()>0


def test_preferences_exclude_near_ties():
    req={'request_id':'x'}
    def candidate(lat,id):return {'latency':lat,'candidate_id':id}
    pair,reason,n=one_pair(req,[candidate(100,'a'),candidate(100.5,'b'),candidate(109,'c')])
    assert pair is None and reason and n==0
    pair,reason,n=one_pair(req,[candidate(100,'a'),candidate(100.5,'b'),candidate(130,'c')])
    assert reason is None and pair['preferred']['latency']<=101
    assert pair['rejected']['latency']>=pair['preferred']['latency']*1.1


def test_family_and_device_balance_not_sorted_device_truncation():
    pairs=[]
    for f in range(17):
        for device in ['a','z']:
            for clock in [3.33,5.,10.]:
                for budget in [1.,.5,.25]:
                    pairs.append(dict(pair_id=f'{f}/{device}/{clock}/{budget}',request=dict(family=str(f),
                        physical_context=['kernel','source','file',device,clock],budget=dict(b=budget))))
    from experiments.stage1_quality_sft import core
    for p in pairs:p['request']['budget']={b:next(iter(p['request']['budget'].values())) for b in core.BUDGET}
    chosen=balance(pairs)
    assert len(chosen)==136
    assert Counter(p['request']['family'] for p in chosen)=={str(f):8 for f in range(17)}
    for family in range(17):
        ds=Counter(p['request']['physical_context'][3] for p in chosen if p['request']['family']==str(family))
        assert ds=={'a':4,'z':4}


def test_trainable_lora_and_special_tokens_receive_gradients(tmp_path):
    from transformers import LlamaConfig,LlamaForCausalLM
    from peft import LoraConfig,get_peft_model,PeftModel
    torch.manual_seed(7)
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)
    config._attn_implementation='eager'
    model=get_peft_model(LlamaForCausalLM(config),LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],
        task_type='CAUSAL_LM',trainable_token_indices={'embed_tokens':[1,2]}))
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'lora_B' in n:p.fill_(.01)
    model.save_pretrained(tmp_path/'start',save_embedding_layers=False)
    base=LlamaForCausalLM(config)
    policy=PeftModel.from_pretrained(base,tmp_path/'start',is_trainable=True)
    logits=policy(input_ids=torch.tensor([[1,3,2,5,7]])).logits
    seq,sft,_,_=selected_logps(logits,torch.tensor([-100,-100,2,5,7]),torch.tensor([0.,0.,1/3,1/3,1/3]))
    loss,_=pair_loss(seq,seq*.5,seq.detach(),(seq*.5).detach(),sft)
    loss.backward()
    for marker in ['lora_','trainable_tokens_delta']:
        gradients=[p.grad for n,p in policy.named_parameters() if marker in n and p.grad is not None]
        assert gradients and sum(float(g.abs().sum()) for g in gradients)>0
    assert all(p.grad is None for n,p in policy.named_parameters() if 'lora_' not in n and 'trainable_tokens_delta' not in n)


def test_sealed_pairs_and_native_teacher_forcing():
    if not (OUT/'manifest.json').exists():pytest.skip('Preference audit not sealed yet')
    c,tok=authenticate()
    from .train import encode
    train=rows(OUT/'train_pairs.jsonl');val=rows(OUT/'validation_pairs.jsonl')
    assert {p['request']['family'] for p in train}.isdisjoint({p['request']['family'] for p in val})
    assert read(OUT/'manifest.json')['test_measurements_decoded']==0
    for p in train+val:
        assert p['preferred']['latency']<=p['best_feasible_latency']*1.01+1e-12
        assert p['rejected']['latency']>=1.1*p['preferred']['latency']-1e-12
    encoded=encode(train+val,tok,c)
    assert len(encoded)==len(train)+len(val)
    for item in encoded:
        assert item['encoding_hashes']['preferred']!=item['encoding_hashes']['rejected']
        assert sum(item['preferred']['labels'].ne(-100))>0
