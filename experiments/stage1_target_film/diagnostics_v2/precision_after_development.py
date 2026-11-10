import time,subprocess,sys
from pathlib import Path
from experiments.final_architecture_20261109.run import preflight
root=Path('artifacts/stage1_target_film/diagnostics_v2')
while not (root/'development/summary.json').exists():time.sleep(15)
while True:
 try:preflight('GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35',minimum=10000);break
 except RuntimeError:time.sleep(15)
subprocess.run([sys.executable,'-m','experiments.stage1_target_film.diagnostics_v2.run_development','--parent','/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run','--checkpoint','artifacts/stage1_target_film/liono_B64/B/checkpoint-48','--pairs','/home/elvouvali/MailoHLS-final-architecture-v3-20261110/artifacts/final_architecture_v3/preferences_v3','--out',str(root/'training_precision_control'),'--gpu-uuid','GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35','--limit','5','--training-precision-control'],check=True)
