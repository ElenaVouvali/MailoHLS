"""Guarded wrapper around the production Stage-2 trainer."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
GPU_UUID='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'
STAGE1='/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run'
EXPECTED_STAGE1='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'
def sha(path):
 import hashlib
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def command(stage,data_dir,memory_dir,run_dir):
 smoke=stage=='smoke'; steps,layers=(5,32) if smoke else (20,8)
 eval_steps,save_steps,selection_steps=(5,5,5) if smoke else (10,10,20)
 return [sys.executable,'-m','LLM_branch.train.train_SFT_xattn_new',
  '--run_mode','single','--stage2_output_dir',str(run_dir),'--output_dir',str(run_dir),
  '--dataset',str(data_dir/'stage2_pilot.jsonl'),'--split_json',str(data_dir/'split.json'),
  '--memory_dir',str(memory_dir),'--init_adapter_dir',STAGE1,'--directive_domain_registry_json',str(Path(STAGE1)/'directive_domain_registry.json'),
  '--research_allow_stage1_subset',
  '--model','deepseek-ai/deepseek-coder-6.7b-base','--model_revision','ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912',
  '--objective','PARETO_LATENCY','--device_mode','known','--resource_budget_mode','none',
  '--directive_loss_weighting','uniform','--family_sampling_power','0',
  '--minimum_validation_families','2','--minimum_test_families','0',
  '--max_slots','64','--mem_dim','128','--every_n_layers',str(layers),
  '--xattn_heads','4','--xattn_dim_head','64','--xattn_ff_mult','1',
  '--structural_fusion_placement','post_self_attention_residual','--structural_routing','compiler_relational',
  '--initial_state_reference',str(run_dir.parent/('initial_structural_state_layer32.json' if layers==32 else 'initial_structural_state_layers8_16_24_32.json')),
  '--lr_lora','0','--lr_embed','0','--lr_xattn','1e-4','--lr_gate','2e-4','--lr_ff','0','--lr_gate_ff','0',
  '--max_steps',str(steps),'--epochs','1','--eval_steps',str(eval_steps),'--save_steps',str(save_steps),
  '--selection_eval_steps',str(selection_steps),'--seed','123',
  '--gradient_checkpointing','--num_workers','0','--candidate_loss_weight','0'] + (
      ['--research_allow_layer_smoke','--selection_num_val_kernels','1',
       '--selection_cases_per_kernel_device','1'] if smoke else
      ['--selection_num_val_kernels','2','--selection_cases_per_kernel_device','1'])
def preflight(allow_ray_sharing=False):
 try: out=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.free','--format=csv,noheader'],text=True)
 except Exception as e: raise RuntimeError(f'nvidia-smi preflight failed: {e}')
 rows=[x.strip().split(', ') for x in out.strip().splitlines()]
 target=next((r for r in rows if r[0]==GPU_UUID),None)
 if target is None or 'V100S' not in target[1]: raise RuntimeError(f'required V100S UUID unavailable; observed={rows}')
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
 target_procs=[line.strip() for line in procs.splitlines() if line.split(',')[0].strip()==GPU_UUID]
 free_mib=int(target[3].split()[0]); util=int(target[2].split()[0])
 if target_procs:
  only_ray=all('RayWorkerWrapper' in line for line in target_procs)
  if not allow_ray_sharing or not only_ray:
   raise RuntimeError('refusing active V100S processes without explicit Ray-sharing mode: '+('; '.join(target_procs)))
  if free_mib<12288 or util>=95:
   raise RuntimeError(f'V100S Ray-sharing safety threshold failed (need >=12288 MiB free and <95% utilization): {target}; processes={target_procs}')
 elif free_mib<24000 or util>10:
  raise RuntimeError(f'V100S is not sufficiently idle for exclusive run: {target}')
 return target
def main():
 p=argparse.ArgumentParser(); p.add_argument('--stage',choices=('smoke','pilot'),required=True); p.add_argument('--allow-gpu-sharing',action='store_true',help='Allow sharing only with Ray workers on the required V100S when at least 12 GiB remain free.')
 p.add_argument('--data-dir',type=Path,default=ROOT/'artifacts/rc3_stage2_multikernel/v1/audit')
 p.add_argument('--memory-dir',type=Path,default=ROOT/'artifacts/rc3_stage2_multikernel/v1/memory_c/memory')
 p.add_argument('--run-dir',type=Path,default=Path('/home/elvouvali/mailohls_runs/rc3_stage2_multikernel/v1'))
 p.add_argument('--execute',action='store_true'); a=p.parse_args()
 if sha(Path(STAGE1)/'adapter_model.safetensors')!=EXPECTED_STAGE1: raise RuntimeError('Stage-1 adapter hash mismatch')
 if not (a.data_dir/'stage2_pilot.jsonl').is_file() or not (a.data_dir/'split.json').is_file(): raise FileNotFoundError('Run audit.py first')
 if not (a.memory_dir/'memory_manifest.json').is_file(): raise FileNotFoundError('Run build_memory.py first')
 cmd=command(a.stage,a.data_dir,a.memory_dir,a.run_dir/(a.stage+'_layer32' if a.stage=='smoke' else 'pilot_layers8_16_24_32'))
 print('COMMAND: '+' '.join(cmd))
 if not a.execute: return
 gpu=preflight(a.allow_gpu_sharing); env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=GPU_UUID
 print('GPU PREFLIGHT:',gpu); a.run_dir.mkdir(parents=True,exist_ok=True)
 subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__': main()
