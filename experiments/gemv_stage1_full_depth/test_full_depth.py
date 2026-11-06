import torch
from transformers import LlamaConfig, LlamaForCausalLM
from .common import LAYERS
from .trace import SelectedStates

def test_full_decoder_layer_schedule():
    assert LAYERS == tuple(range(1, 33))

def test_selected_hooks_are_read_only_and_capture_three_boundaries():
    torch.manual_seed(7)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4)).eval()
    ids=torch.tensor([[1,2,3,4]])
    with torch.inference_mode(): reference=model(input_ids=ids).logits.detach().clone()
    trace=SelectedStates(model,layers=(1,2,3,4))
    trace.arm({'request_id':'test','prefix_mode':'controlled'},[{'action':'L5','field':'PIPE','position_kind':'decision','token_index':3}],ids[0].tolist(),type('T',(),{'decode':lambda self,x:str(x)})())
    with torch.inference_mode(): actual=model(input_ids=ids).logits.detach().clone()
    trace.disarm();trace.close()
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    boundaries={r['boundary'] for r in trace.index}
    assert boundaries=={'post_self_attention_residual','post_decoder_residual','final_rms_normalized_lm_head_input'}
    assert len(trace.vectors)==2*4+1 and all(v.shape==(32,) for v in trace.vectors)
    assert {r['layer'] for r in trace.index if r['boundary']!='final_rms_normalized_lm_head_input'}=={1,2,3,4}
