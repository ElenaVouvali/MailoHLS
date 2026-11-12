"""Run development evaluation or capture reproducible execution blockers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from GNN_branch.paper_protocol import OBJECTIVES, evaluate

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(ssh_destination):
    split_path = ROOT/'mailohls_runs/mailohls_final_family_split_s123.json'
    split = json.loads(split_path.read_text())
    families = [set(split[k+'_families']) for k in ('train', 'val', 'test')]
    if any(families[i] & families[j] for i in range(3) for j in range(i)):
        raise ValueError('Family leakage in sealed split')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', ssh_destination,
               'hostname; nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv; '
               'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv']
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        remote = {'command': command, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    except subprocess.TimeoutExpired as exc:
        remote = {'command': command, 'returncode': None, 'error': str(exc)}
    deps = {name: importlib.util.find_spec(name) is not None for name in ('torch', 'torch_geometric', 'scipy')}
    paths = ['GNN_branch/model.py', 'GNN_branch/train_GNN.py', 'GNN_branch/mlir_data.py',
             'GNN_branch/build_multiscale_memory.py', 'GNN_branch/paper_protocol.py',
             'GNN_branch/MLIR_dataset/all_kernels/feature_schema.json']
    blockers = []
    if remote['returncode'] != 0:
        blockers.append('Davinci SSH preflight failed; no remote GPU inventory or training receipt')
    return {'schema': 'salt-hls-paper-gnn-preflight-v2',
            'status': 'blocked' if blockers else 'connected_gpu_inventory_captured',
            'utc': datetime.now(timezone.utc).isoformat(), 'remote': remote,
            'local_dependencies': deps, 'split_sha256': sha(split_path),
            'split_families': {k: split[k+'_families'] for k in ('train','val','test')},
            'code_sha256': {p: sha(ROOT/p) for p in paths}, 'blockers': blockers,
            'scope': 'connectivity/inventory only; actual experiment completion is recorded under each objective output',
            'locked_test_evaluated': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('preflight'); a.add_argument('--ssh', default='evouvali@davinci.microlab.ntua.gr'); a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('evaluate'); a.add_argument('--rows', type=Path, required=True)
    a.add_argument('--objective', choices=OBJECTIVES, required=True); a.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Receipt exists; choose a fresh output path')
    if args.command == 'preflight':
        report = preflight(args.ssh)
    else:
        rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
        report = evaluate(rows, args.objective)
        report['input_sha256'] = sha(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'status': report.get('status','evaluated')}))
    return 2 if report.get('status') == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
