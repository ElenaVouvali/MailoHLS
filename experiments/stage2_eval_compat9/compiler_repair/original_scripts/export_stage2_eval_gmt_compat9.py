#!/usr/bin/env python3
"""Audit and export all 9 evaluation kernels with explicit two-kernel compatibility provenance.

The authentic frozen GNN/GMT export is delegated to the unmodified original
export_stage2_eval_gmt.py. This wrapper refuses unverified compiler graph
changes and separately marks the resulting memory bank as a heterogeneous
compiler representation, not untouched historical preprocessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

KERNELS = ('GAN', 'Kalman', 'chstone-aes', 'chstone-jpeg', 'covariance',
           'jacobi-2d', 'rosetta-3d-rendering', 'syr2k', 'trmm-opt')
SEVEN = tuple(k for k in KERNELS if k not in ('Kalman', 'rosetta-3d-rendering'))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('repo', 'sources', 'graphs', 'assets-root', 'out', 'successful-graphs', 'repair-script', 'original-exporter'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    v = {k.replace('-', '_'): value.resolve() for k, value in vars(a).items()}
    receipt_path = v['graphs'] / 'EVAL_COMPAT9_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 'mailohls-stage2-eval-compat9-v1':
        raise RuntimeError('Wrong or missing versioned graph compatibility receipt')
    if receipt.get('status') != 'COMPATIBILITY_EXPERIMENT_NOT_HISTORICAL_IDENTICAL':
        raise RuntimeError('Incorrect graph compatibility status')
    if receipt['compatibility_wrapper_sha256'] != sha(v['repair_script']):
        raise RuntimeError('Graph repair source code changed since graph generation')
    assert set(receipt['graph_sha256']) == set(KERNELS)
    for k in KERNELS:
        current = v['graphs'] / (k + '.gexf')
        if sha(current) != receipt['graph_sha256'][k]:
            raise RuntimeError(f'{k}: GEXF differs from compatibility receipt')
    for k in SEVEN:
        if sha(v['graphs'] / (k + '.gexf')) != sha(v['successful_graphs'] / (k + '.gexf')):
            raise RuntimeError(f'{k}: historical successful control graph was changed')
    if v['out'].exists():
        raise FileExistsError(f'Export destination already exists: {v["out"]}')
    cmd = [sys.executable, str(v['original_exporter']),
           '--repo', str(v['repo']), '--sources', str(v['sources']),
           '--graphs', str(v['graphs']), '--assets-root', str(v['assets_root']),
           '--out', str(v['out'])]
    print('[EVAL COMPAT9] Using unmodified frozen GNN/GMT exporter', flush=True)
    subprocess.run(cmd, cwd=v['repo'], check=True)
    pass_path = v['out'] / 'EXPORT_PASS.json'
    manifest_path = v['out'] / 'memory_B/memory_manifest.json'
    if not pass_path.is_file() or not manifest_path.is_file():
        raise RuntimeError('Full exporter did not produce complete success receipt and memory_B manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest['kind'] != 'B_GMT' or set(manifest['files']) != {k+'.memory.pt' for k in KERNELS}:
        raise RuntimeError('Memory manifest does not contain exactly 9 frozen GMT memories')
    for file, expected in manifest['files'].items():
        if sha(v['out'] / 'memory_B' / file) != expected:
            raise RuntimeError(f'Invalid memory SHA: {file}')
    result = {
        'schema': 'mailohls-stage2-eval-compat9-v1',
        'status': 'FULL_NINE_COMPATIBILITY_COHORT_NOT_IDENTICAL_HISTORICAL_COMPILER',
        'all_nine_frozen_GNN_GMT_memories': True,
        'frozen_original_exporter_sha256': sha(v['original_exporter']),
        'compiler_graph_receipt_sha256': sha(receipt_path),
        'base_compiler_generator_sha256': receipt['base_generator_sha256'],
        'compatibility_wrapper_sha256': receipt['compatibility_wrapper_sha256'],
        'compiler_compatibility_variants': receipt['variants'],
        'seven_control_graphs_bitwise_unchanged': True,
        'original_export_pass_sha256': sha(pass_path),
        'memory_manifest_sha256': sha(manifest_path),
        'gnn_checkpoint_sha256': manifest['gnn_checkpoint_sha256'],
        'gmt_checkpoint_sha256': manifest['gmt_checkpoint_sha256'],
        'labels_read_for_export': False,
        'representation_shift_caveat': 'Kalman early IR and Rosetta global-root support; not directly equivalent to unchanged historical preprocessing',
    }
    (v['out'] / 'COMPAT9_EXPORT_PASS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[9/9 COMPATIBILITY GNN/GMT MEMORY PASS] Frozen weights, heterogeneous compiler input representations', flush=True)

if __name__ == '__main__':
    main()
