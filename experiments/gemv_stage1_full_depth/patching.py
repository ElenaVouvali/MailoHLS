"""Read-only post-self-attention activation patching for matched GEMV target pairs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from LLM_branch.common import structural_xattn
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from .common import ADAPTER,DEFAULT_OUT,GPU,contract,decision_prefixes,read_json,sha,write_json
from .prepare import auth,FOCUS

LAYERS=(1,4,8,12,16,20,24,28,32)
FIELDS=['auto{_ARRAY_D_L1}','auto{_ARRAY_F_L1}','auto{_ARRAY_F_L2}','auto{_PIPE_L4}','auto{_UNROLL_L4}','auto{_PIPE_L5}','auto{_UNROLL_L5}']
PAIRS=[('device','GRID_XCU200_10P0NS_100PCT'),('clock','GRID_ZCU104_3P33NS_100PCT'),('budget','GRID_ZCU104_10P0NS_25PCT')]

def run(out):
 out=Path(out); assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla V100S-PCIE-32GB'
 c,tok=auth(); request_rows={json.loads(x)['context_id']:json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()}
 results={x['request_id']:x for x in read_json(out/'candidate_scores.json')};r1='GRID_ZCU104_10P0NS_100PCT';ref=results[r1]['assignment']
 # Load the exact Stage-1 adapter through the existing authenticated loader.
 args=SimpleNamespace(stage='stage1',model=c['model'],model_revision=c['model_revision'],no_4bit=False,device_map='auto',adapter_dir=str(ADAPTER),lora_adapter_dir='',embedding_weights_tied=c['embedding_weights_tied'])
 model=inf.load_stage_model(args,tok);model.eval();assert not getattr(model,'initialized_structural_xattn',False)
 registry=inf.load_directive_domain_registry(str(ADAPTER/'directive_domain_registry.json'))
 inf.add_missing_source_directive_domains(registry,[inf.InferenceCase('gemv',request_rows[r1]['input'],'PARETO_LATENCY',1.,0.,request_rows[r1],None)],str(Path(__file__).resolve().parent/'inputs'))
 layers_path=structural_xattn.infer_decoder_layers_attr_name(model);native=structural_xattn.getattr_recursive(model,layers_path)
 def state(rid,field,layer):
  ix=read_json(out/'hidden_states'/f'{rid}.index.json');arr=__import__('numpy').load(out/'hidden_states'/f'{rid}.npz')['vectors']
  hit=[x for x in ix if x['prefix_mode']=='controlled' and x['field']==field and x['layer']==layer and x['boundary']=='post_self_attention_residual']
  assert len(hit)==1,(rid,field,layer,len(hit));return arr[hit[0]['vector_index']]
 def ctrl(rid,field):
  return next(x for x in results[rid]['trace_records'] if x['prefix_mode']=='controlled' and x['field']==field)
 records=[]
 for kind,receiver in PAIRS:
  row=request_rows[receiver]
  prefixes={p['field']:p for p in decision_prefixes(row,tok,ref)}
  for field in FIELDS:
   ref_rec=ctrl(r1,field);recv_rec=ctrl(receiver,field)
   if set(ref_rec['active_candidates'])!=set(recv_rec['active_candidates']):
    records.append(dict(kind=kind,receiver=receiver,field=field,status='SKIPPED_DIFFERENT_CANDIDATE_DOMAIN'));continue
   score_ref={x['rhs']:float(x.get('score',x['mean_logprob'])) for x in ref_rec['scores']}
   candidates=[x['rhs'] for x in ref_rec['scores'][:2]]
   if len(candidates)<2: records.append(dict(kind=kind,receiver=receiver,field=field,status='FORCED'));continue
   prefix=prefixes[field];assert prefix['ids'] and prefix['field']==field
   base_ids=torch.tensor([prefix['ids']],dtype=torch.long,device=inf.get_first_real_device(model));kw=dict(model=model,tok=tok,base_input_ids=base_ids,base_attention_mask=torch.ones_like(base_ids),use_structural_memory=False)
   receiver_index=len(prefix['ids'])-1; assert receiver_index==len(prefix['ids'])-1
   # Recompute the untouched receiver scores with the real scorer; compare to the saved prefix replay.
   unpatched={}
   for rhs in candidates:
    stat=inf.score_rhs_candidate_suffix(**kw,candidate_text=rhs+'\n');unpatched[rhs]=float(stat['mean_logprob'])
    expected={x['rhs']:float(x.get('score',x['mean_logprob'])) for x in recv_rec['scores']}[rhs]
    if abs(unpatched[rhs]-expected)>1e-4:raise AssertionError(f'baseline score replay mismatch {receiver}:{field}:{rhs}: {unpatched[rhs]} vs {expected}')
   base_margin=unpatched[candidates[0]]-unpatched[candidates[1]]
   for layer_no in LAYERS:
    donor=torch.tensor(state(r1,field,layer_no),device=inf.get_first_real_device(model),dtype=next(model.parameters()).dtype)
    own=torch.tensor(state(receiver,field,layer_no),device=inf.get_first_real_device(model),dtype=next(model.parameters()).dtype)
    mod=native[layer_no-1].post_attention_layernorm
    def score_with(vector,rhs):
     def hook(_module,inputs):
      h=inputs[0];assert h.ndim==3 and h.shape[0]==1 and receiver_index<h.shape[1]
      patched=h.clone();patched[0,receiver_index]=vector.to(device=h.device,dtype=h.dtype)
      return (patched,*inputs[1:])
     handle=mod.register_forward_pre_hook(hook)
     try:return float(inf.score_rhs_candidate_suffix(**kw,candidate_text=rhs+'\n')['mean_logprob'])
     finally:handle.remove()
    patched={rhs:score_with(donor,rhs) for rhs in candidates}
    placebo={rhs:score_with(own,rhs) for rhs in candidates}
    pm=patched[candidates[0]]-patched[candidates[1]];qm=placebo[candidates[0]]-placebo[candidates[1]]
    records.append(dict(kind=kind,receiver=receiver,field=field,layer=layer_no,status='PATCHED',candidate_pair=candidates,reference_margin=score_ref[candidates[0]]-score_ref[candidates[1]],receiver_margin=base_margin,patched_margin=pm,placebo_margin=qm,delta_margin=pm-base_margin,placebo_delta=qm-base_margin,patched_scores=patched,placebo_scores=placebo,score_convention='production mean token logprob including serialized RHS and newline',boundary='post_self_attention_residual',donor_request=r1,receiver_prefix_mode='controlled',prefix_token_index=receiver_index))
  print('[PATCH]',kind,receiver,flush=True)
 write_json(out/'activation_patching.json',dict(schema='read-only-post-self-attention-patching-v1',gpu_uuid=GPU,checkpoint_sha256=sha(ADAPTER/'adapter_model.safetensors'),layers=list(LAYERS),fields=FIELDS,records=records,weights_modified=False,structural_memory_used=False))
 del model;torch.cuda.empty_cache()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT_OUT);run(ap.parse_args().out)
