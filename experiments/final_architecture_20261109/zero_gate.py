"""Real-decoder parity by wrapping the authenticated inference entrypoint.

Use normal stage1 CLI arguments and SALT_PARITY_MEMORY_DIR. Loads the LLM once,
then compares the first request before/after fresh layer32 zero-gate fusion.
"""
import json,os
from pathlib import Path
import torch
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
original=inference.predict_case
count=0

def checked(**kwargs):
 global count
 baseline=original(**kwargs)
 if count==0:
  model,tok=kwargs['model'],kwargs['tok']
  inference.attach_structural_modules(model,tok,128,32,4,64,1,False,'post_self_attention_residual',1.,1.)
  gates={n:float(p.item()) for n,p in model.named_parameters() if n.endswith('attn_gate')}
  assert gates and all(v==0 for v in gates.values()),gates
  bank=inference.load_memory_bank(os.environ['SALT_PARITY_MEMORY_DIR'],expected_mem_dim=128,expected_max_slots=64)
  args=dict(kwargs);args.update(stage='stage2',mem_bank=bank,mem_dim=128,max_slots=64,structural_routing='compiler_relational')
  with torch.inference_mode():fused=original(**args)
  same=baseline['candidates']==fused['candidates']
  dest=Path(os.environ['SALT_PARITY_OUTPUT']);dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_text(json.dumps({'same_candidates_and_scores_exactly':same,'gates':gates,'baseline':baseline,'zero_gate':fused},indent=2)+'\n')
  assert same,'Zero-gate real-decoder candidate/score mismatch; do not train'
 count+=1
 return baseline
if __name__=='__main__':
 from .run import preflight
 preflight(os.environ['CUDA_VISIBLE_DEVICES'])
 inference.predict_case=checked
 inference.main()
