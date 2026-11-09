"""Execute the retained registered latency Stage-1 through the official decoder."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MODEL=Path('/home/elvouvali/MailoHLS/mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1')
GPU='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'
HASH='a15a0af1022370b08ee1e64cb9934a12e9f1c7ef1b96aaaf4ccfc406f4e0cc86'
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-jsonl',type=Path,required=True);p.add_argument('--output-jsonl',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 assert hashlib.sha256((MODEL/'adapter_model.safetensors').read_bytes()).hexdigest()==HASH
 contract=json.loads((MODEL/'training_contract.json').read_text());registry=MODEL.parent/'directive_domain_registry.json'
 assert hashlib.sha256(registry.read_bytes()).hexdigest()==contract['directive_domain_registry_sha256']
 for line in a.input_jsonl.read_text().splitlines():
  r=json.loads(line)
  if r.get('objective',r.get('obj_mode'))!='PARETO_LATENCY':raise ValueError('latency-only selected endpoint')
  if r.get('frequency_mode')!='specified':raise ValueError('qualification fixes requested clock')
 if a.output_jsonl.exists():raise FileExistsError(a.output_jsonl)
 cmd=[sys.executable,'-m','LLM_branch.inference.eval_stage1_stage2_stage3','--stage','stage1','--adapter_dir',str(MODEL),'--directive_domain_registry_json',str(registry),'--input_jsonl',str(a.input_jsonl),'--output_jsonl',str(a.output_jsonl),'--score_reduction','mean','--debug_topk','3','--num_samples','1']
 print(' '.join(cmd),flush=True)
 if not a.execute:return
 from experiments.rc3_stage2_multikernel.train import preflight
 preflight(False);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=GPU
 a.output_jsonl.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(cmd,env=env,cwd=ROOT,check=True)
if __name__=='__main__':main()
