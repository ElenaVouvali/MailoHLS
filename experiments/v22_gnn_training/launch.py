"""Safely launch or resume a detached V2.2 GNN training run on an idle GPU."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.v22_gnn_training.checkpointing import active_pid, atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[2]


def _nvidia_smi(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True, stderr=subprocess.STDOUT)


def gpu_inventory():
    rows = list(csv.reader(_nvidia_smi(
        '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits').splitlines()))
    result = []
    for row in rows:
        if len(row) != 7:
            continue
        result.append({'index':int(row[0].strip()),'uuid':row[1].strip(),
                       'name':row[2].strip(),'total_mib':int(row[3].strip()),
                       'used_mib':int(row[4].strip()),'free_mib':int(row[5].strip()),
                       'utilization_percent':int(row[6].strip())})
    return result


def compute_processes():
    output = _nvidia_smi('--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                         '--format=csv,noheader,nounits').strip()
    if not output:
        return []
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) == 4:
            rows.append({'uuid':row[0].strip(),'pid':int(row[1].strip()),
                         'name':row[2].strip(),'used_mib':row[3].strip()})
    return rows


def _process_command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:
        return ''


def _ensure_no_running_run(output):
    lock_path=output/'launcher.lock'
    if lock_path.exists():
        try:
            lock=json.loads(lock_path.read_text())
        except (OSError,json.JSONDecodeError):
            raise RuntimeError('Refusing launch: launcher.lock is unreadable; inspect the run')
        pid=int(lock.get('pid',-1))
        command=_process_command(pid)
        if active_pid(pid) and ('v22_gnn_training.launch' in command or 'v22_gnn_training.train' in command):
            raise RuntimeError(f'Run already has active launcher/trainer PID {pid}')
        status_path=output/'status.json'
        if status_path.exists():
            try:
                status=json.loads(status_path.read_text())
                train_pid=int(status.get('pid',-1))
            except (OSError,ValueError,json.JSONDecodeError):
                train_pid=-1
            train_command=_process_command(train_pid)
            if active_pid(train_pid) and 'experiments.v22_gnn_training.train' in train_command:
                raise RuntimeError(f'Run has active trainer PID {train_pid} despite stale launcher lock')
        lock_path.unlink()


def _reserve_lock(output):
    lock_path=output/'launcher.lock'
    flags=os.O_CREAT|os.O_EXCL|os.O_WRONLY
    fd=os.open(str(lock_path),flags,0o600)
    with os.fdopen(fd,'w') as stream:
        json.dump({'pid':os.getpid(),'status':'launching','created_at':utc_now()},stream)
        stream.flush(); os.fsync(stream.fileno())
    return lock_path


def _worker(spec_path):
    spec=json.loads(Path(spec_path).read_text())
    output=Path(spec['output'])
    command=spec['command']
    started=time.time()
    child=None
    requested_signal={'number':None}

    def forward_signal(signum,_frame):
        requested_signal['number']=int(signum)
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    old={sig:signal.signal(sig,forward_signal) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        child=subprocess.Popen(command,cwd=spec['cwd'],env=spec['env'],stdin=subprocess.DEVNULL,
                               stdout=None,stderr=subprocess.STDOUT,close_fds=True)
        atomic_json(output/'trainer_pid.json',{'pid':child.pid,'started_at':utc_now(),
                    'command':command,'gpu_uuid':spec['gpu']['uuid'],'gpu_name':spec['gpu']['name']})
        rc=child.wait()
        result={'schema':'salt-hls-v22-launch-exit-v1','status':'exited','returncode':int(rc),
                'worker_pid':os.getpid(),'trainer_pid':child.pid,'started_unix':started,
                'finished_at':utc_now(),'forwarded_signal':requested_signal['number'],
                'gpu':spec['gpu'],'log':str(output/'run.log')}
        atomic_json(output/'exit_status.json',result)
        return int(rc)
    except BaseException as exc:
        result={'schema':'salt-hls-v22-launch-exit-v1','status':'worker_exception',
                'worker_pid':os.getpid(),'started_unix':started,'finished_at':utc_now(),
                'exception_type':type(exc).__name__,'message':str(exc),
                'forwarded_signal':requested_signal['number'],'gpu':spec.get('gpu')}
        atomic_json(output/'exit_status.json',result)
        raise
    finally:
        for sig,handler in old.items():
            signal.signal(sig,handler)
        try:
            (output/'launcher.lock').unlink()
        except FileNotFoundError:
            pass


def _parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-name',default='A2',help='Substring identifying the requested GPU model')
    parser.add_argument('--minimum-free-mib',type=int,default=6000)
    parser.add_argument('--allow-shared-gpu',action='store_true',
                        help='Permit existing compute processes; avoid unless explicitly intended')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--steps',type=int,default=8000)
    parser.add_argument('--candidates',type=int,default=8)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--log-every',type=int,default=100)
    parser.add_argument('--val-every',type=int,default=500)
    parser.add_argument('--validation-candidates',type=int,default=256)
    parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--progress-every',type=int,default=25)
    parser.add_argument('--rank-weight',type=float,default=0.1)
    parser.add_argument('--skip-test',action='store_true',
                        help='Preflight only: stop after validation selection without reading test metrics')
    return parser


def main():
    if '--worker' in sys.argv:
        index=sys.argv.index('--worker')
        spec=sys.argv[index+1]
        raise SystemExit(_worker(spec))
    args=_parser().parse_args()
    if args.minimum_free_mib<1:
        raise SystemExit('--minimum-free-mib must be positive')
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if dirty.strip():
        raise RuntimeError('Refusing detached launch from a dirty worktree; commit the exact experiment sources first')
    release=args.release_dir.expanduser().resolve()
    if not (release/'release.json').is_file():
        raise FileNotFoundError('Authenticated extracted graph release is missing: '+str(release))
    output=args.output.expanduser().resolve()
    if args.resume:
        if not output.is_dir() or not (output/'training_state.pt').is_file():
            raise FileNotFoundError('--resume requires an existing run with training_state.pt')
        _ensure_no_running_run(output)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Refusing to overwrite nonempty run directory: '+str(output))
    gpus=gpu_inventory()
    matching=[gpu for gpu in gpus if args.gpu_name.casefold() in gpu['name'].casefold()]
    if len(matching)!=1:
        raise RuntimeError(f'Expected one GPU matching {args.gpu_name!r}; found {matching}')
    gpu=matching[0]
    processes=[row for row in compute_processes() if row['uuid']==gpu['uuid']]
    if processes and not args.allow_shared_gpu:
        raise RuntimeError(f'Refusing to share requested GPU {gpu["name"]}; processes are active: {processes}')
    if gpu['free_mib']<args.minimum_free_mib:
        raise RuntimeError(f'Refusing GPU with only {gpu["free_mib"]} MiB free; minimum is {args.minimum_free_mib} MiB')
    if not processes and gpu['utilization_percent']>15 and not args.allow_shared_gpu:
        raise RuntimeError(f'Refusing apparently busy GPU at {gpu["utilization_percent"]}% utilization')
    if not args.resume:
        output.mkdir(parents=True,exist_ok=True)
    lock_path=_reserve_lock(output)
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':gpu['uuid'],'PYTHONPATH':str(ROOT),
                'PYTHONUNBUFFERED':'1','SALT_HLS_GPU_UUID':gpu['uuid']})
    train_command=[sys.executable,'-u','-m','experiments.v22_gnn_training.train',
                   '--release-dir',str(release),'--output',str(output),
                   '--steps',str(args.steps),'--candidates',str(args.candidates),
                   '--lr',str(args.lr),'--weight-decay',str(args.weight_decay),
                   '--log-every',str(args.log_every),'--val-every',str(args.val_every),
                   '--validation-candidates',str(args.validation_candidates),
                   '--patience',str(args.patience),'--checkpoint-every',str(args.checkpoint_every),
                   '--progress-every',str(args.progress_every),'--rank-weight',str(args.rank_weight),
                   '--expected-gpu',args.gpu_name]
    if args.resume:
        train_command.append('--resume')
    if args.skip_test:
        train_command.append('--skip-test')
    spec={'cwd':str(ROOT),'output':str(output),'release_dir':str(release),
          'command':train_command,'env':env,'gpu':gpu,'allow_shared_gpu':args.allow_shared_gpu,
          'launched_at':utc_now(),'branch':subprocess.check_output(
              ['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
          'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    spec_path=output/'launcher_spec.json'
    atomic_json(spec_path,spec)
    log=open(output/'run.log','ab',buffering=0)
    try:
        worker=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                                 '--worker',str(spec_path)],cwd=ROOT,env=env,
                                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                start_new_session=True,close_fds=True)
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    finally:
        log.close()
    atomic_json(lock_path,{'pid':worker.pid,'status':'running','created_at':utc_now(),
                           'gpu_uuid':gpu['uuid'],'gpu_name':gpu['name']})
    atomic_json(output/'launcher_pid.json',{'pid':worker.pid,'started_at':utc_now(),
                 'gpu':gpu,'command':train_command})
    print(json.dumps({'status':'launched','worker_pid':worker.pid,'gpu':gpu,
                      'output':str(output),'log':str(output/'run.log'),
                      'status_file':str(output/'status.json'),
                      'progress_file':str(output/'progress.jsonl'),
                      'exit_status_file':str(output/'exit_status.json'),
                      'resume_command':f'{sys.executable} -m experiments.v22_gnn_training.launch --gpu-name {args.gpu_name} --release-dir {release} --output {output} --resume'},
                      sort_keys=True),flush=True)


if __name__=='__main__':
    main()
