"""Dry-run by default; portable corrected static-C training, no historical defaults changed."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from experiments.rc3_stage2_multikernel.train import command
from experiments.final_architecture_20261109.run import preflight,EXPECTED
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','baseline'],required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--steps',type=int,default=150)
 p.add_argument('--stage1',type=Path,default=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run'))
 p.add_argument('--data-dir',type=Path,default=Path('/home/elvouvali/MailoHLS-rc3-stage2-multikernel-c/artifacts/rc3_stage2_multikernel/v1/audit'))
 p.add_argument('--memory-dir',type=Path,default=Path('/home/elvouvali/MailoHLS-rc3-stage2-multikernel-c/artifacts/rc3_stage2_multikernel/v1/memory_c/memory'))
 p.add_argument('--gpu-uuid',default='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb');a=p.parse_args()
 if hashlib.sha256((a.stage1/'adapter_model.safetensors').read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Stage1 mismatch')
 smoke=a.stage=='smoke';cmd=command('pilot',a.data_dir,a.memory_dir,a.run_dir)
 def setarg(k,v):cmd[cmd.index(k)+1]=str(v)
 setarg('--init_adapter_dir',a.stage1);setarg('--directive_domain_registry_json',a.stage1/'directive_domain_registry.json');setarg('--max_steps',1 if smoke else a.steps);setarg('--epochs',100)
 setarg('--initial_state_reference',a.run_dir.parent/'initial_v3_layers8_16_24_32.json')
 for flag in ('--eval_steps','--save_steps','--selection_eval_steps'):setarg(flag,1 if smoke else 50)
 setarg('--selection_num_val_kernels',1 if smoke else 3)
 cmd+=['--budget_target_max_duplicates','0','--batch_size','1','--grad_accum','8','--early_stopping_patience','0' if smoke else '3']
 if smoke:cmd+=['--no-eval_on_start']
 if a.resume:cmd+=['--resume_from_checkpoint',str(a.resume)]
 cmd[2]='experiments.final_architecture_v3.supervision';cmd[3:3]=['--receipt',str(a.run_dir/'corrected_supervision.json'),'--']
 print(__import__('shlex').join(cmd),flush=True)
 if not a.execute:return
 if a.run_dir.exists() and any(a.run_dir.iterdir()) and not a.resume:raise FileExistsError('Fresh run or explicit resume required')
 receipt=preflight(a.gpu_uuid);a.run_dir.mkdir(parents=True,exist_ok=True)
 receipt.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dataset_sha256=hashlib.sha256((a.data_dir/'stage2_pilot.jsonl').read_bytes()).hexdigest(),split_sha256=hashlib.sha256((a.data_dir/'split.json').read_bytes()).hexdigest(),memory_manifest_sha256=hashlib.sha256((a.memory_dir/'memory_manifest.json').read_bytes()).hexdigest())
 (a.run_dir/('resume_preflight.json' if a.resume else 'preflight.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid,TOKENIZERS_PARALLELISM='false')
 subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
