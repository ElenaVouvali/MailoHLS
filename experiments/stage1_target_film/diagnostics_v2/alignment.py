"""Exact value-token spans in the native RHS-plus-newline candidate encoding."""
import torch
from LLM_branch.train.stage1_decision_loss import _value_positions

def value_token_positions(tok,candidate_text):
 if not candidate_text.endswith('\n'):raise ValueError('Native candidate must terminate with newline')
 value=candidate_text[:-1]  # string schema boundary, NOT a token-count assumption
 ids=tok(candidate_text,add_special_tokens=False)['input_ids']
 return ids,_value_positions(tok,value,ids)

def causal_kinds(tok,candidate_text,base_len,kind,device,policy='value'):
 ids,selected=value_token_positions(tok,candidate_text)
 mask=torch.zeros(1,base_len+len(ids),dtype=torch.long,device=device)
 for j in (range(len(ids)) if policy=='legacy' else selected):mask[0,base_len-1+j]=kind
 return mask
