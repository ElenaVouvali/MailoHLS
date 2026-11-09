"""Inference-only evidence interventions through the production Stage2 decoder.

The final trained checkpoint is a failure diagnostic, not a selected winner.
Masks/action slots/padding remain unchanged; each control keeps its own natural
prefix. Set SALT_CONTROL_OUTPUT to a fresh JSONL file.
"""
import copy,json,os
from pathlib import Path
import torch
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
original=inference.predict_case

def changed(bank,mode):
 result={};seen={}
 for name,rec in bank.items():
  if id(rec) not in seen:
   pack=dict(rec);kv=rec['kv'].clone();valid=torch.where(rec['mask'].bool())[0];evidence=valid[valid>=64]
   if mode=='zero_evidence':kv[evidence]=0
   elif mode=='permuted_evidence':
    gen=torch.Generator(device='cpu').manual_seed(123)
    groups={}
    metadata=rec['evidence_metadata']
    for key in evidence.tolist():
     record=metadata[key-64]
     category=(record['evidence_type'],record.get('evidence_status'),tuple(sorted(record.get('known',{}).items())),tuple(rec['relation_mask'][:64,key].bool().tolist()))
     groups.setdefault(category,[]).append(key)
    ledger=[]
    for indices in groups.values():
     order=torch.randperm(len(indices),generator=gen).tolist()
     donors=[indices[i] for i in order]
     kv[indices]=rec['kv'][donors]
     ledger.extend([{'recipient':i,'donor':j} for i,j in zip(indices,donors)])
    pack['permutation_ledger']=ledger
   else:raise ValueError(mode)
   pack['kv']=kv;seen[id(rec)]=pack
  result[name]=seen[id(rec)]
 return result

def checked(**kwargs):
 model=kwargs['model'];result=original(**kwargs);variants={'authentic':result}
 if os.environ.get('SALT_GENERATED_OUTPUT'):
  with Path(os.environ['SALT_GENERATED_OUTPUT']).open('a') as f:f.write(json.dumps(result)+'\n')
 for mode in ('zero_evidence','permuted_evidence'):
  args=dict(kwargs);args['mem_bank']=changed(kwargs['mem_bank'],mode)
  variants[mode]=original(**args)
 gates=[p for n,p in model.named_parameters() if n.endswith('attn_gate')];saved=[p.detach().clone() for p in gates]
 try:
  with torch.no_grad():
   for p in gates:p.zero_()
  variants['gate_off']=original(**kwargs)
 finally:
  with torch.no_grad():
   for p,v in zip(gates,saved):p.copy_(v)
 output=Path(os.environ['SALT_CONTROL_OUTPUT']);output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('a') as f:f.write(json.dumps({'context_id':result['context_id'],'variants':variants,'permutation_ledger':{k:v.get('permutation_ledger',[]) for k,v in changed(kwargs['mem_bank'],'permuted_evidence').items()},'scope':'natural prefixes; final checkpoint failure diagnostic; type/status/knownness/route-matched permutation; no physical QoR inference'})+'\n')
 return result
if __name__=='__main__':
 from .run import preflight
 if os.environ.get('SALT_ALLOW_RAY_SHARING')=='1':
  from experiments.rc3_stage2_multikernel.train import preflight as ray_preflight
  ray_preflight(True)
 else:preflight(os.environ['CUDA_VISIBLE_DEVICES'])
 if Path(os.environ['SALT_CONTROL_OUTPUT']).exists():raise FileExistsError('Use a fresh control output')
 inference.predict_case=checked;inference.main()
