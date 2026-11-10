"""Auxiliary A2 real-decoder evidence controls on fixed intermediate Step50.
Not a primary V100S policy-ranking or physical-QoR comparison.
"""
import argparse,json,os,sys
from pathlib import Path
import torch
from experiments.final_architecture_20261109 import evidence_controls as controls
from experiments.final_architecture_20261109.run import preflight

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--memory-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print('Supplementary A2 natural-prefix authentic/cross-route/gate-off control, two reference cases');return
 if a.out.exists():raise FileExistsError(a.out)
 receipt=preflight('GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35',14000);a.out.mkdir(parents=True);rows=[json.loads(s) for s in a.pilot.read_text().splitlines()];selected=[]
 for kernel in ('machsuite-viterbi','rodinia_pathfinder_0_baseline_0'):
  row=next(r for r in rows if r['kernel_name']==kernel and r['device']=='xczu7ev-ffvc1156-2-e' and float(r['selected_clock_period'])==10. and float(r['resource_budget_frac'])==1.);selected.append(dict(row,context_id=f'{kernel}:xczu7ev-ffvc1156-2-e:10.0'))
 (a.out/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected));(a.out/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
 old=controls.inference.BitsAndBytesConfig
 def fp16(**kw):kw['bnb_4bit_compute_dtype']=torch.float16;return old(**kw)
 controls.inference.BitsAndBytesConfig=fp16;controls.inference.predict_case=controls.checked;os.environ['SALT_CONTROL_OUTPUT']=str(a.out/'controls.jsonl');os.environ['SALT_CONTROL_MODES']='cross_route_evidence'
 sys.argv=[sys.argv[0],'--stage','stage2','--adapter_dir',str(a.parent),'--directive_domain_registry_json',str(a.parent/'directive_domain_registry.json'),'--memory_dir',str(a.memory_dir),'--input_jsonl',str(a.out/'requests.jsonl'),'--output_jsonl',str(a.out/'authentic.jsonl'),'--debug_topk','8','--num_samples','1','--score_reduction','mean'];controls.inference.main()
 records=[json.loads(s) for s in (a.out/'controls.jsonl').read_text().splitlines()];(a.out/'stage1_gateoff_reference.jsonl').write_text(''.join(json.dumps(r['variants']['gate_off'])+'\n' for r in records))
 (a.out/'result.json').write_text(json.dumps({'schema':'v3-a2-corrected-step50-evidence-controls-v1','contexts':len(records),'parent':str(a.parent),'primary_qualification':False,'within_route_shuffle':'permutation symmetry placebo','cross_route':'same type/status/knownness, content moved across different eligibility profiles; masks fixed','source_ledger':'controls.jsonl'},indent=2)+'\n')
if __name__=='__main__':main()
