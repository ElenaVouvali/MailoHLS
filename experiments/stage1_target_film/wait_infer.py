"""Bounded wait for an idle GPU; never share, preempt, or repair the server."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from experiments.final_architecture_20261109.run import preflight

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--max-wait-seconds',type=int,default=21600);p.add_argument('--receipt',type=Path,required=True);opts,command=p.parse_known_args()
 if command and command[0]=='--':command=command[1:]
 if not command:raise ValueError('Missing inference command')
 started=time.monotonic();last=None
 while True:
  try:gpu=preflight(opts.gpu_uuid);break
  except (RuntimeError,subprocess.CalledProcessError) as exc:
   message=str(exc)
   if message!=last:print('[WAITING_FOR_IDLE_GPU]',message,flush=True);last=message
   if time.monotonic()-started>opts.max_wait_seconds:
    opts.receipt.write_text(json.dumps({'status':'BLOCKED','reason':message,'wait_seconds':time.monotonic()-started})+'\n');return
   time.sleep(30)
 print('[GPU_IDLE] launching requested inference',flush=True)
 result=subprocess.run(command,check=False)
 opts.receipt.write_text(json.dumps({'status':'PASS' if result.returncode==0 else 'FAIL','returncode':result.returncode,'wait_seconds':time.monotonic()-started,'preflight':gpu,'command':command},indent=2)+'\n')
if __name__=='__main__':main()
