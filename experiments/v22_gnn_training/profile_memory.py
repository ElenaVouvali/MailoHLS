"""Measure full V2.2 GNN forward/backward memory on an A2 without training data.

The zero candidate inputs are used only to size the candidate head. This
profile performs one disposable optimizer update per graph and saves no model
or QoR result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F
import torch_geometric

from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch import target_graph_v22 as overlay
from experiments.v22_gnn_training.train import (
    CANDIDATE_DEMAND_DIM,
    CANDIDATE_DIM,
    CandidateQoRModel,
    PreparedData,
    SEED,
    _fit_normalization,
    _train_step,
)

ROOT = Path(__file__).resolve().parents[2]
RELEASE_MANIFEST = ROOT / 'docs/graph-release-v22-rc3.json'
GRAPH_PREFIXES = {
    'small_stencil': 'machsuite-stencil2d-',
    'medium_gemm': 'machsuite-gemm-ncubed-',
    'largest_aes': 'chstone-aes-',
}


def profile_one(path: Path, expected_sha: str, name: str, device: torch.device):
    if overlay.file_hash(path) != expected_sha:
        raise ValueError('Graph member SHA256 mismatch: ' + str(path))
    base, authentication = overlay.authenticated_input(path)
    joint = overlay.enrich(base, 'zcu104', 10, authentication=authentication)
    started = time.perf_counter()
    cpu_data = tensors.tensorize(joint)
    tensorization_seconds = time.perf_counter() - started
    tensor_size_bytes = tensors.tensor_bytes(cpu_data)
    graph = cpu_data.to(device)

    model = CandidateQoRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    candidates = torch.zeros((8, 64, CANDIDATE_DIM), dtype=torch.float32, device=device)
    demands = torch.zeros((8, 64, CANDIDATE_DEMAND_DIM), dtype=torch.float32, device=device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before_update = torch.cuda.memory_allocated(device)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    prediction = model(graph, candidates, demands, arm='hardware')
    # A disposable scalar objective exercises activations, backward and Adam
    # state allocation. This is not a measured QoR label or training update.
    loss = F.smooth_l1_loss(prediction, torch.zeros_like(prediction))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - started

    result = {
        'case': name,
        'graph_member': str(path),
        'graph_identity': base.graph['graph_identity'],
        'graph_member_sha256': expected_sha,
        'nodes': graph.num_nodes,
        'positive_messages': graph.edge_index.shape[1],
        'candidates_per_context': int(candidates.shape[0]),
        'candidate_inputs_are_synthetic_zero_tensors_for_profiling_only': True,
        'cpu_tensor_bytes': tensor_size_bytes,
        'cpu_tensorization_seconds': tensorization_seconds,
        'forward_backward_optimizer_update_seconds': update_seconds,
        'cuda_allocated_before_update_bytes': allocated_before_update,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'finite_loss': bool(torch.isfinite(loss).item()),
        'finite_predictions': bool(torch.isfinite(prediction).all().item()),
    }
    del optimizer, model, graph, cpu_data, joint, base
    torch.cuda.empty_cache()
    return result


def profile_measured_update(release_dir: Path, audit_path: Path, device: torch.device):
    """Run one disposable labeled optimizer update per arm on real training rows."""
    dataset = PreparedData(release_dir, audit_path)
    groups = {key: rows for key, rows in dataset.groups.items()
              if rows and rows[0]['split'] == 'train'}
    if not groups:
        raise ValueError('Authenticated dataset contains no eligible training groups')
    mean, std = _fit_normalization(dataset.records)
    group_key = sorted(groups)[0]
    group = groups[group_key]
    count = min(8, len(group))
    selected = sorted(__import__('random').Random(SEED).sample(range(len(group)), count))
    outcomes = {}
    for arm in ('numerical', 'hardware'):
        torch.manual_seed(SEED)
        model = CandidateQoRModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        model.train()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        loss, regression, ranking = _train_step(
            model, optimizer, dataset, group_key, selected, 'PARETO_LATENCY', 1.0,
            mean, std, arm, device)
        torch.cuda.synchronize(device)
        outcomes[arm] = {
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'finite_loss': all(torch.isfinite(torch.tensor(value)).item()
                               for value in (loss, regression, ranking)),
            'total_loss': loss,
            'qor_loss': regression,
            'ranking_loss': ranking,
            'forward_backward_optimizer_update_seconds': time.perf_counter() - started,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        }
        del optimizer, model
        torch.cuda.empty_cache()
    return {
        'dataset_sha256': dataset.audit['dataset']['dataset_sha256'],
        'split_sha256': dataset.audit['dataset']['split_sha256'],
        'eligible_records': len(dataset.records),
        'eligible_rows_by_split': {
            name: sum(row['split'] == name for row in dataset.records)
            for name in ('train', 'val', 'test')},
        'measured_training_group': {
            'kernel': group_key[0], 'profile': group_key[1], 'clock_ns': group_key[2],
            'selected_candidate_hashes': [group[index]['config_hash'] for index in selected],
            'selected_candidate_count': count,
            'objective': 'PARETO_LATENCY', 'resource_budget_fraction': 1.0,
        },
        'arms': outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-gpu', default='A2',
                        help='Required substring in the visible CUDA device name (default: A2)')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no memory profile was produced')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose only the assigned A2 and verify its visible device name')
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(device)
    if args.expected_gpu.casefold() not in gpu_name.casefold():
        raise RuntimeError(f'Expected {args.expected_gpu!r} at visible cuda:0, found {gpu_name}')
    audit_path = args.output.with_name(args.output.stem + '.data-audit.json')
    if args.output.exists() or audit_path.exists():
        raise FileExistsError('Refusing to overwrite memory-profile output or audit')

    release_manifest = json.loads((args.release_dir / 'release.json').read_text())
    repo_manifest = json.loads(RELEASE_MANIFEST.read_text())
    archive_path = ROOT / repo_manifest['archive']
    if overlay.file_hash(archive_path) != repo_manifest['archive_sha256']:
        raise ValueError('Repository graph archive checksum mismatch')
    if release_manifest.get('files_sha256') != repo_manifest['files_sha256']:
        raise ValueError('Extracted graph release differs from the frozen RC3 manifest')

    rows = []
    oom_cases = []
    for name, prefix in GRAPH_PREFIXES.items():
        members = [member for member in repo_manifest['files_sha256']
                   if member.startswith('canonical/' + prefix) and member.endswith('.json')]
        if len(members) != 1:
            raise ValueError(f'Expected one authenticated {name} snapshot; found {members}')
        member = members[0]
        path = args.release_dir / member
        try:
            rows.append(profile_one(path, repo_manifest['files_sha256'][member], name, device))
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            oom_cases.append({'case':name, 'graph_member':member,
                              'status':'cuda_out_of_memory', 'message':str(exc)})

    measured_update = None
    if not oom_cases:
        try:
            measured_update = profile_measured_update(args.release_dir, audit_path, device)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            oom_cases.append({'case':'measured_candidate_update',
                              'status':'cuda_out_of_memory', 'message':str(exc)})

    result = {
        'schema': 'salt-hls-v22-a2-memory-preflight-v1',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'graph_archive_sha256': repo_manifest['archive_sha256'],
        'tensor_schema': tensors.SCHEMA,
        'model_schema': 'salt-hls-v22-candidate-qor-head-v1-experimental',
        'environment': {
            'torch': torch.__version__,
            'pyg': torch_geometric.__version__,
            'cuda_runtime': torch.version.cuda,
            'gpu': gpu_name,
            'gpu_total_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
            'visible_device_count': torch.cuda.device_count(),
        },
        'profiles': rows,
        'oom_cases': oom_cases,
        'measured_update_status': 'completed' if measured_update is not None else 'not_completed',
        'measured_candidate_update_preflight': measured_update,
        'data_audit_path': str(audit_path),
        'interpretation': 'One disposable forward/backward/optimizer update per graph; zero candidate input tensors only size the head. No measured labels, training checkpoint, or QoR result are produced.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'gpu': gpu_name, 'output': str(args.output),
                      'profiles': [{k: row[k] for k in ('case','cpu_tensor_bytes',
                          'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes',
                          'forward_backward_optimizer_update_seconds')} for row in rows],
                      'oom_cases': oom_cases,
                      'measured_update_status': result['measured_update_status']}, indent=2))
    if oom_cases:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
