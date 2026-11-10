"""Continue saved arms under the user-requested ranking-first stopping policy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.paper_gnn.data import ROOT, sha
from experiments.paper_gnn.train import write_json
from experiments.paper_gnn_latency_final.selection_policy import RANKING_POLICY

GPU = 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=['A', 'B'], default=['B', 'A'])
    args = parser.parse_args()
    original = args.root.resolve()
    root = original / 'ranking_selection_v2'
    root.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONHASHSEED='123', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', SALT_CORRECTED_COHORT=str(original / 'cohort/corrected_cohort_v1'))
    policy = dict(schema='salt-latency-selection-revision-v2', selection=RANKING_POLICY,
                  authorization='User requested tau-b, MAE, regret priority and continuation of B',
                  original_policy_sha256=sha(ROOT / 'experiments/paper_gnn_latency_final/predeclared_policy.json'),
                  post_hoc_revision=True, original_runs_preserved=True, max_total_epochs=40,
                  patience=8, architecture_losses_optimizer_sampling_unchanged=True,
                  matched_comparison='Continue B first, then A under the identical revised stopping rule',
                  controller_sha256=sha(Path(__file__)))
    write_json(policy, root / 'policy_revision.json')
    journal_path = root / 'execution_journal.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else []

    def status(phase, **extra):
        receipt = dict(time_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                       gpu_uuid=GPU, policy=RANKING_POLICY, **extra)
        write_json(receipt, root / 'resource_status.json')
        write_json({**receipt, 'continuation_root': str(root)}, original / 'resource_status.json')

    for arm in args.arms:
        out = root / arm
        if not (out / 'completion.json').exists():
            while True:
                processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
                occupied = [line for line in processes.splitlines() if line.split(',')[0].strip() == GPU]
                if occupied:
                    status('waiting_for_idle_gpu', arm=arm, other_compute_processes=occupied)
                    time.sleep(15)
                    continue
                command = [sys.executable, '-u', '-m', 'experiments.paper_gnn_latency_final.train_arm',
                           '--arm', arm, '--objective', 'PARETO_LATENCY', '--seed', '123',
                           '--expected-gpu', 'V100', '--selection-policy', 'ranking', '--output', str(out)]
                command += ['--resume'] if (out / 'last.pt').exists() else ['--resume-from', str(original / arm / 'last.pt')]
                status('resuming_' + arm, arm=arm, log=str(root / (arm + '.log')))
                with (root / (arm + '.log')).open('a') as log:
                    proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                journal.append(dict(time_utc=datetime.now(timezone.utc).isoformat(), arm=arm,
                                    command=command, returncode=proc.returncode))
                write_json(journal, journal_path)
                if proc.returncode == 0:
                    break
                tail = (root / (arm + '.log')).read_text()[-4000:]
                if 'Selected GPU has another compute process; refusing launch' in tail:
                    status('launch_safely_refused', arm=arm)
                    time.sleep(15)
                    continue
                status('training_failed', arm=arm, returncode=proc.returncode)
                raise RuntimeError('Inspect ' + str(root / (arm + '.log')))
        cpuenv = dict(env, CUDA_VISIBLE_DEVICES='')
        with (root / (arm + '_xattn_cpu.log')).open('a') as log:
            subprocess.run([sys.executable, '-u', '-m', 'experiments.paper_gnn_latency_final.check_bank_cpu',
                            '--bank', str(out / 'memory'), '--output', str(out / 'xattn_cpu_check.json')],
                           cwd=ROOT, env=cpuenv, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('trained_exported_' + arm, arm=arm,
               completion=json.loads((out / 'completion.json').read_text()))
    status('both_ranking_policy_arms_trained_exported', final_evaluation_pending=True)


if __name__ == '__main__':
    main()
