"""Complete no-training GAN ON/OFF with exact-prefix head counterfactual reuse."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from experiments.final_architecture_20261109.run import preflight

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
 while not ((root/'development/summary.json').exists() and (root/'training_precision_control/summary.json').exists()):time.sleep(15)
 devices=['GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb','GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35'];gpu=None
 while gpu is None:
  for candidate in devices:
   try:preflight(candidate,minimum=10000);gpu=candidate;break
   except RuntimeError:pass
  if gpu is None:time.sleep(15)
 parent='/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run';checkpoint='artifacts/stage1_target_film/liono_B64/B/checkpoint-48';prepared='/home/elvouvali/MailoHLS-gan-stage1-tcad-20261110/artifacts/gan_stage1_tcad/run_20261110';trace=root/'paired_scores.jsonl'
 for mode in ('on','off','legacy'):
  out=root/('gan_'+mode)
  if (out/'execution.json').exists():continue
  command=[sys.executable,'-m','experiments.stage1_target_film.infer','--parent',parent,'--adaptation',checkpoint,'--prepared',prepared,'--out',str(out),'--gpu-uuid',gpu,'--film-mode','off' if mode=='off' else 'on','--mask-policy','legacy' if mode=='legacy' else 'value']
  command+=['--paired-trace',str(trace)] if mode=='on' else ['--replay-trace',str(trace)]
  subprocess.run(command,check=True)
 print('[ABLATION_COMPLETE] corrected natural ON/OFF outputs and paired fixed-prefix scores available',flush=True)
if __name__=='__main__':main()
