"""Resume-safe post-training GAN evaluation; fails closed on busy GPU or changed files."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from .train import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--v4',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);a=p.parse_args();training=a.root/'matched64'
 while not (training/'B/completed.json').is_file():
  # A bounded wait only; training failure must be inspected, not silently retried.
  if time.time()-(a.root/'matched64.log').stat().st_mtime>1800:raise RuntimeError('Training log stale; inspect before continuation')
  time.sleep(15)
 baseline=json.loads((a.v4.parent/'execution.json').read_text());assert baseline['completed_cases']==18 and baseline['predictions_sha256']==sha(a.v4)
 manifest=json.loads((a.prepared/'manifest.json').read_text());assert sha(a.prepared/'requests.jsonl')==manifest['requests_sha256'];assert sha(a.parent/'adapter_model.safetensors')==manifest['checkpoint_sha256']
 for arm in ('A','B'):
  selected=json.loads((training/arm/'selected.json').read_text());dest=a.root/('gan_'+arm)
  if (dest/'execution.json').exists():continue
  subprocess.run([sys.executable,'-m','experiments.stage1_target_film.infer','--parent',str(a.parent),'--adaptation',selected['checkpoint'],'--prepared',str(a.prepared),'--out',str(dest),'--gpu-uuid',a.gpu_uuid],check=True)
 subprocess.run([sys.executable,'-m','experiments.stage1_target_film.report_gan','--v4',str(a.v4),'--a',str(a.root/'gan_A/predictions.jsonl'),'--b',str(a.root/'gan_B/predictions.jsonl'),'--out',str(a.root/'gan_report')],check=True)
if __name__=='__main__':main()
