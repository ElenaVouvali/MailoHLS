#!/usr/bin/env python3
"""PREVIEW-ONLY, frozen historical GNN/GMT export for seven of nine locked SALT-HLS kernels.

Uses MailoHLS's authenticated historical.py exporter and the existing
prepare_banks.build_banks() for inference-only role summaries. This never fits
GNN/GMT weights, accesses labels, or changes Stage-2 model parameters.

Run from a Liono worktree with the historical checkpoint/normalization accessible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

KERNELS = (
    'GAN', 'Kalman', 'chstone-aes', 'chstone-jpeg',
    'covariance', 'jacobi-2d', 'rosetta-3d-rendering', 'syr2k', 'trmm-opt',
)


# Explicitly incomplete cohort: Kalman and Rosetta remain excluded.
PREVIEW_KERNELS = (
    'GAN', 'chstone-aes', 'chstone-jpeg', 'covariance',
    'jacobi-2d', 'syr2k', 'trmm-opt',
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_path(folder: Path) -> Path:
    for name in ('source.cpp', 'source.c'):
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(f'No canonical source.c/source.cpp in {folder}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--assets-root', type=Path,
                   default=Path('/home/elvouvali/MailoHLS'),
                   help='Original historical GNN checkpoint + normalization root')
    a = p.parse_args()
    a.repo, a.sources, a.graphs, a.assets_root = (
        x.resolve() for x in (a.repo, a.sources, a.graphs, a.assets_root)
    )
    a.out = a.out.resolve()
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite {a.out}')
    for directory in (a.repo, a.sources, a.graphs, a.assets_root):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    expected_repo = '6ee1390611c9047a4b8e2b07200d223b93ee8d16'
    import subprocess
    sha = subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != expected_repo:
        raise RuntimeError(f'Not the deployed Stage-2 code version: {sha} != {expected_repo}')
    if not (a.sources/'SOURCES_MANIFEST.json').is_file():
        raise FileNotFoundError('Sources are not staged/audited by prepare_stage2_eval_sources.sh')
    source_manifest = json.loads((a.sources/'SOURCES_MANIFEST.json').read_text())
    if set(source_manifest) != set(KERNELS):
        raise ValueError(f'Expected all 9 sources: {sorted(set(KERNELS)-set(source_manifest))}')

    sys.path.insert(0, str(a.repo))
    os.chdir(a.repo)
    import torch
    from LLM_branch.common import directive_domains as domains
    from experiments.historical_mlir_gmt_gemv import historical as hist
    from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import (
        cohort_action_mapping, make_evidence,
    )
    from experiments.historical_gmt_stage2_ab import prepare_banks as prep

    training_gmt = a.repo / 'artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors'
    original_manifest = json.loads((training_gmt/'memory_B/memory_manifest.json').read_text())
    gmt_cp = training_gmt/'gmt_selected.pt'
    if not gmt_cp.is_file():
        raise FileNotFoundError(f'Frozen GMT checkpoint not found: {gmt_cp}')
    if digest(gmt_cp) != original_manifest['gmt_checkpoint_sha256']:
        raise RuntimeError('GMT frozen checkpoint does not match original 33-kernel Stage-2 memory')
    state = torch.load(gmt_cp, map_location='cpu', weights_only=False)
    if int(state['steps']) != 600 or state['architecture'] != 'HistoricalRoleGMT':
        raise RuntimeError('GMT state does not represent the selected frozen 600-step model')
    def norm(name):
        return str(name).strip().lower().replace('_', '-')
    locked = {norm(name) for name in KERNELS} | {'serrano-kalman-filter'}
    fitted = {norm(name) for name in state['train_kernels'] + state['internal_dev_kernels']}
    if fitted & locked:
        raise RuntimeError(f'GMT checkpoint includes a locked evaluation kernel: {sorted(fitted & locked)}')

    hist.ASSETS = a.assets_root
    hist.CKPT = a.assets_root/hist.CKPT_REL/'val_structural_rank_model_state_dict.pth'
    runtime = hist.load()
    info = runtime[-1]
    if info['checkpoint_sha256'] != original_manifest['gnn_checkpoint_sha256']:
        raise RuntimeError('Historical GNN checkpoint SHA differs from Stage-2 training memory')
    generator = (a.repo/'Checkpoints/gnn_final_absolute_direct_rank_s123/'
                 'run1/provenance/mlir_graph_gen.py')
    if digest(generator) != info['authenticated_training_files']['mlir_graph_gen.py']['sha256']:
        raise RuntimeError('Evaluation graphs must be generated with historical GNN-fit generator')
    if not any(a.graphs.glob('*.gexf')):
        raise FileNotFoundError(f'No graphs in {a.graphs}: run prepare_stage2_eval_graphs.sh first')

    cohort, packs, graph_hashes, audit = {}, {}, {}, {}
    for kernel in PREVIEW_KERNELS:
        folder = a.sources/kernel
        source, metadata = source_path(folder), folder/'kernel_info.txt'
        manifest = source_manifest[kernel]
        if digest(source) != manifest['source_sha256'] or digest(metadata) != manifest['metadata_sha256']:
            raise ValueError(f'{kernel}: source bytes changed after staging')
        graph_file = a.graphs/f'{kernel}.gexf'
        if not graph_file.is_file():
            raise FileNotFoundError(f'{kernel}: no historical MLIR graph at {graph_file}')
        graph, pack = hist.export_graph(graph_file, kernel, runtime)
        gp = pack['graph_provenance']
        for key, expected in (
            ('source_sha256', digest(source)),
            ('action_sha256', digest(metadata)),
            ('generator_sha256', digest(generator)),
        ):
            if gp.get(key) != expected:
                raise RuntimeError(f'{kernel}: graph provenance {key} does not match the staged source/metadata/generator')
        if not pack['disable_pragma_injection']:
            raise RuntimeError(f'{kernel}: historical GNN memory contains directive injection')
        if pack['node_embs'].shape != (64, 128) or not torch.isfinite(pack['node_embs']).all():
            raise RuntimeError(f'{kernel}: malformed/nonfinite 64 x 128 GNN action anchors')
        mapping = cohort_action_mapping(graph, pack)
        action_set = {m['action'] for m in mapping}
        declared = domains.load_source_actions(kernel, a.sources)
        if action_set != set(declared) or action_set != set(manifest['actions']):
            raise RuntimeError(f'{kernel}: historical action IDs differ from source metadata; '
                               f'graph={sorted(action_set)} metadata={sorted(declared)}')
        if any(m['absolute_memory_slot'] != int(m['action'][1:])-1 for m in mapping):
            raise RuntimeError(f'{kernel}: action-slot renumbering is forbidden')
        for m in mapping:
            kind = declared[m['action']].kind
            if kind != m['action_type']:
                raise RuntimeError(f'{kernel}: graph/source action kind differs: {m["action"]} {kind} != {m["action_type"]}')
        # This invokes the *historical, trained* GNN once per evaluation graph,
        # then constructs role-specific evidence for the already frozen GMT.
        records, evidence = make_evidence(graph, pack, mapping, graph_file)
        if not torch.equal(records['anchors'], pack['node_embs'].float()):
            raise RuntimeError(f'{kernel}: GNN anchor bytes changed before GMT export')
        cohort[kernel], packs[kernel], graph_hashes[kernel] = records, pack, digest(graph_file)
        audit[kernel] = {
            'source_sha256':digest(source), 'metadata_sha256':digest(metadata),
            'graph_sha256':digest(graph_file), 'action_count':len(mapping),
            'evidence_count':len(evidence['records']), 'graph_provenance':gp,
            'evaluation_labels_read':False,
        }
        print(f'[GNN/GMT INPUT] {kernel}: {len(mapping)} Lk actions, {len(evidence["records"])} compiler evidence records', flush=True)

    # Build_banks() *only infers* through a frozen HistoricalRoleGMT state.
    # Its checkpoint_from_train_only() function is intentionally NOT called.
    a.out.mkdir(parents=True)
    source_only = a.out/'source_only_eval_inputs.json'
    source_only.write_text(json.dumps({'schema':'salt-evaluation-sources-only-v1',
                                       'kernels':audit},indent=2,sort_keys=True,default=str)+'\n')
    evaluation_lock = a.repo/'MailoHLS_ASPLOS_evaluation_workload_kit/EVALUATION_WORKLOAD_LOCK.json'
    memory_summary = prep.build_banks(cohort, packs, graph_hashes, gmt_cp,
                                      a.out, source_only, evaluation_lock, 123)
    manifest_path=a.out/'memory_B/memory_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if (manifest['gnn_checkpoint_sha256'] != original_manifest['gnn_checkpoint_sha256']
            or manifest['gmt_checkpoint_sha256'] != original_manifest['gmt_checkpoint_sha256']):
        raise RuntimeError('Exported memory is incompatible with frozen Stage-2 GNN/GMT')
    if set(manifest['files']) != {k+'.memory.pt' for k in PREVIEW_KERNELS}:
        raise RuntimeError('Preview output manifest does not contain exactly seven kernels')
    for filename, expected in manifest['files'].items():
        if digest(a.out/'memory_B'/filename) != expected:
            raise RuntimeError(f'Memory file SHA mismatch: {filename}')
        payload=torch.load(a.out/'memory_B'/filename,map_location='cpu',weights_only=False)
        if payload.get('memory_interface')!='rc3_action_evidence_c_v1' or not payload.get('evidence_metadata'):
            raise RuntimeError(f'Not a genuine GMT evidence memory file: {filename}')
    (a.out/'PREVIEW_7_PASS.json').write_text(json.dumps({
        'status':'PREVIEW_7_OF_9_NOT_FINAL', 'source_revision':expected_repo,
        'gnn_checkpoint_sha256':manifest['gnn_checkpoint_sha256'],
        'gmt_checkpoint_sha256':manifest['gmt_checkpoint_sha256'],
        'memory_B_manifest_sha256':digest(manifest_path),
        'kernels':memory_summary,
        'test_measurements_read':False, 'gnn_or_gmt_retrained':False,
    },indent=2,sort_keys=True,default=str)+'\n')
    print('[PREVIEW 7/9 PASS] Frozen historical GNN + frozen GMT memory; NOT final 9-kernel evaluation')
    print('[EXPORT DIR]',a.out/'memory_B')
    print('[NOTE] PREVIEW ONLY: two kernels lack authenticated graphs; do not report as 9/9.')
    print('[NOTE] Do not copy memory_A; Stage-2 inference requires memory_B.')

if __name__ == '__main__':
    main()
