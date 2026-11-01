"""Bounded fixed-context V4 learnability diagnostic.

This is a regression-only memorization test for the existing CandidateQoRModel.
It uses one authenticated training kernel/device/clock context; it is not a
generalization or target-transfer evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch.nn import functional as F

from experiments.v22_gnn_training import checkpointing, fast_probe, train


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _choose_spread(rows, count):
    """Deterministic complete-assignment coverage, independent of QoR labels."""
    remaining = sorted(rows, key=lambda row: row['config_hash'])
    chosen = [remaining.pop(0)]
    while remaining and len(chosen) < count:
        candidate = min(remaining, key=lambda row: (
            -min(fast_probe._assignment_distance(row['assignment'], old['assignment'])
                 for old in chosen), row['config_hash']))
        chosen.append(candidate)
        remaining.remove(candidate)
    return sorted(chosen, key=lambda row: row['config_hash'])


def _rank(y, p, fn):
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return None
    value = float(fn(y, p)[0])
    return value if math.isfinite(value) else None


def _evaluate(model, graph, candidates, demands, rows, mean, std, device):
    model.eval()
    with torch.no_grad():
        pred_z = model(graph, candidates, demands, arm='hardware')
        true_z = (torch.tensor([row['transformed'] for row in rows], dtype=torch.float32,
                               device=device) - mean) / std
        pred = torch.expm1(pred_z * std + mean).clamp_min(0).cpu().numpy()
        true = np.asarray([row['physical'] for row in rows], dtype=np.float64)
        baseline = np.broadcast_to(np.mean(true, axis=0, keepdims=True), true.shape)
        latency = true[:, 0]
        selected = int(np.argmin(pred[:, 0]))
        best = float(np.min(latency))
        return {
            'standardized_smooth_l1': float(F.smooth_l1_loss(pred_z, true_z).cpu()),
            'constant_baseline_standardized_smooth_l1': float(
                F.smooth_l1_loss(torch.zeros_like(true_z), true_z).cpu()),
            'latency_mae_ms': float(np.mean(np.abs(pred[:, 0] - latency))),
            'latency_mape': float(np.mean(np.abs(pred[:, 0] - latency) /
                                          np.maximum(latency, 1e-12))),
            'latency_spearman': _rank(latency, pred[:, 0], spearmanr),
            'latency_kendall': _rank(latency, pred[:, 0], kendalltau),
            'constant_latency_mae_ms': float(np.mean(np.abs(baseline[:, 0] - latency))),
            'predicted_best_config_hash': rows[selected]['config_hash'],
            'measured_best_config_hash': rows[int(np.argmin(latency))]['config_hash'],
            'predicted_best_is_measured_best': bool(
                rows[selected]['config_hash'] == rows[int(np.argmin(latency))]['config_hash']),
            'predicted_best_regret_fraction': float(
                (latency[selected] - best) / max(abs(best), 1e-12)),
            'predicted_latency_range_ms': [float(np.min(pred[:, 0])),
                                           float(np.max(pred[:, 0]))],
            'measured_latency_range_ms': [float(np.min(latency)), float(np.max(latency))],
        }


def run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    audit_path = output / 'data_audit.json'
    data = train.PreparedData(args.release_dir, audit_path,
                              selected_kernels=[args.kernel])
    prep_seconds = time.monotonic() - started
    if prep_seconds > args.preparation_limit_seconds:
        raise TimeoutError(f'Authenticated dataset preparation exceeded {args.preparation_limit_seconds}s')
    key = (args.kernel, args.profile, float(args.clock_ns))
    rows = data.groups.get(key, [])
    if len(rows) < 16:
        raise ValueError(f'Fixed training context has only {len(rows)} eligible measured assignments')
    chosen = _choose_spread(rows, min(max(args.candidates, 16), min(32, len(rows))))
    if len({row['config_hash'] for row in chosen}) != len(chosen):
        raise ValueError('Selected fixed-context candidates are not distinct')

    # One-time feature materialization. The same authenticated V4 interfaces
    # used by the standard trainer are called, then held in memory.
    feature_started = time.monotonic()
    graph = data.graph_tensor(chosen[0], 1.0, 'hardware').cpu()
    pairs = [data.candidate_inputs(row) for row in chosen]
    candidates = torch.stack([pair[0] for pair in pairs]).cpu()
    demands = torch.stack([pair[1] for pair in pairs]).cpu()
    mean, std = train._fit_normalization(chosen)
    feature_seconds = time.monotonic() - feature_started

    # Authenticate the exact fixed-set selection before training.
    manifest = {
        'schema': 'salt-hls-v4-fixed-context-learnability-v1',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                 cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        'graph_release_archive_sha256': data.audit['graph_release']['archive_sha256'],
        'dataset_sha256': data.audit['dataset']['dataset_sha256'],
        'split_sha256': data.audit['dataset']['split_sha256'],
        'kernel': args.kernel, 'family': chosen[0]['family'],
        'profile': args.profile, 'clock_ns': float(args.clock_ns),
        'split': chosen[0]['split'], 'selected_config_hashes': [r['config_hash'] for r in chosen],
        'selection': 'deterministic greedy maximum Hamming diversity with config-hash tie break; no QoR labels',
        'candidate_count': len(chosen), 'available_context_candidates': len(rows),
        'test_rows_loaded_for_fit_or_evaluation': False,
        'budget_fraction': 1.0, 'arm': 'hardware', 'ranking_loss_weight': 0.0,
        'normalization_fit': 'selected training records only',
        'graph_tensor_schema': fast_probe.tensors.SCHEMA,
        'candidate_demand_schema': train.action_demand.CANDIDATE_SCHEMA,
        'runtime_limits_seconds': {'preparation': args.preparation_limit_seconds,
                                   'gpu': args.gpu_limit_seconds},
        'source_files_sha256': fast_probe._source_hashes(),
        'learnability_probe_sha256': fast_probe.overlay.file_hash(Path(__file__)),
        'release_dir': str(Path(args.release_dir).resolve()),
        'data_audit_sha256': _hash(audit_path),
    }
    if prep_seconds + feature_seconds > args.preparation_limit_seconds:
        raise TimeoutError(f'Authenticated data and feature preparation exceeded {args.preparation_limit_seconds}s')
    manifest['manifest_sha256'] = fast_probe._canonical_hash(manifest)
    checkpointing.atomic_json(output / 'learnability_manifest.json', manifest)

    # Verify accelerator identity and exclusive availability before training.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one UUID-isolated CUDA device is required')
    gpu = fast_probe._verify_idle_gpu(args.gpu_name, args.gpu_uuid)
    device = torch.device('cuda:0')
    if args.gpu_name.casefold() not in torch.cuda.get_device_name(device).casefold():
        raise RuntimeError('CUDA-visible device name does not match requested accelerator')

    gpu_started = time.monotonic()
    deadline = gpu_started + args.gpu_limit_seconds
    graph, candidates, demands = (value.to(device) for value in (graph, candidates, demands))
    mean, std = mean.to(device), std.to(device)
    target_raw = torch.tensor([row['transformed'] for row in chosen], dtype=torch.float32,
                              device=device)
    target = (target_raw - mean) / std
    model = train.CandidateQoRModel().to(device)
    initial_hash = train.model_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    checkpoint_steps = {1, 10, 25, 50, 100, 200, args.updates}
    try:
        for step in range(1, args.updates + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU diagnostic wall-clock budget reached')
            model.train()
            prediction = model(graph, candidates, demands, arm='hardware')
            loss = F.smooth_l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fixed-context loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise ValueError('Nonfinite gradient norm')
            encoder_grad = sum(float(p.grad.norm()) for p in model.graph.parameters()
                               if p.grad is not None)
            head_grad = sum(float(p.grad.norm()) for p in model.head.parameters()
                            if p.grad is not None)
            optimizer.step()
            if step in checkpoint_steps:
                metric = _evaluate(model, graph, candidates, demands, chosen, mean, std, device)
                history.append({'step': step, 'training_loss': float(loss.detach()),
                                'gradient_norm_preclip': float(grad_norm),
                                'encoder_gradient_l1_of_norms': encoder_grad,
                                'head_gradient_l1_of_norms': head_grad,
                                **metric})
    except BaseException as exc:
        checkpointing.atomic_json(output / 'status.json', {
            'status': 'failed_or_interrupted', 'exception': type(exc).__name__,
            'message': str(exc), 'completed_updates': step - 1 if 'step' in locals() else 0,
            'elapsed_gpu_seconds': time.monotonic() - gpu_started,
        })
        raise

    elapsed_gpu = time.monotonic() - gpu_started
    if elapsed_gpu > args.gpu_limit_seconds:
        raise TimeoutError('GPU diagnostic exceeded its declared wall-clock budget')
    final = _evaluate(model, graph, candidates, demands, chosen, mean, std, device)
    baseline = {
        'targetwise_mean_prediction': np.mean([row['physical'] for row in chosen], axis=0).tolist(),
        'latency_mae_ms': final['constant_latency_mae_ms'],
    }
    checkpoint_path = output / 'fixed_context_hardware.pt'
    checkpointing.atomic_torch_save(checkpoint_path, {
        'schema': train.HEAD_SCHEMA, 'diagnostic_schema': manifest['schema'],
        'manifest_sha256': manifest['manifest_sha256'], 'initial_model_sha256': initial_hash,
        'model': {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        'normalization_mean': mean.detach().cpu(), 'normalization_std': std.detach().cpu(),
        'selected_config_hashes': manifest['selected_config_hashes'],
    })
    report = {
        'status': 'completed', 'manifest_sha256': manifest['manifest_sha256'],
        'preparation_seconds': prep_seconds, 'feature_preparation_seconds': feature_seconds,
        'gpu_name': torch.cuda.get_device_name(device), 'gpu_uuid': args.gpu_uuid,
        'gpu_inventory_before_launch': gpu, 'gpu_seconds': elapsed_gpu,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'updates': args.updates, 'initial_model_sha256': initial_hash,
        'final_model_sha256': train.model_state_sha256(model),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'training_loss_initial': history[0]['training_loss'],
        'training_loss_final_recorded': history[-1]['training_loss'],
        'history': history, 'final_training_set_metrics': final,
        'constant_baseline': baseline,
        'checkpoint': checkpoint_path.name, 'checkpoint_sha256': _hash(checkpoint_path),
        'interpretation': 'fixed-context training-set memorization only; no held-out or target-generalization claim',
    }
    checkpointing.atomic_json(output / 'learnability_results.json', report)
    checkpointing.atomic_json(output / 'status.json', {'status': 'completed',
                               'manifest_sha256': manifest['manifest_sha256'],
                               'completed_updates': args.updates,
                               'gpu_seconds': elapsed_gpu})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel', default='machsuite-stencil2d')
    parser.add_argument('--profile', choices=('zcu104', 'u200'), default='zcu104')
    parser.add_argument('--clock-ns', type=float, default=10.0)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpu-name', default='A2')
    parser.add_argument('--gpu-uuid', default='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35')
    parser.add_argument('--preparation-limit-seconds', type=int, default=600)
    parser.add_argument('--gpu-limit-seconds', type=int, default=1200)
    args = parser.parse_args()
    if not 16 <= args.candidates <= 32 or not 1 <= args.updates <= 1000:
        parser.error('candidate count must be 16-32 and updates must be 1-1000')
    report = run(args)
    print(json.dumps({key: report[key] for key in (
        'status', 'preparation_seconds', 'feature_preparation_seconds', 'gpu_seconds',
        'updates', 'training_loss_initial', 'training_loss_final_recorded',
        'final_training_set_metrics', 'constant_baseline', 'checkpoint_sha256')}, indent=2))


if __name__ == '__main__':
    main()
