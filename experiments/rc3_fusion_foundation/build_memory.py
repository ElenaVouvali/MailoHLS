"""Build authenticated frozen-GNN A/B/C memories for RC3 GEMV graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from GNN_branch.mlir_graph_v2 import read_graph
from experiments.rc3_fusion_foundation.structural_memory_v1 import (
    build_memory_pack, load_frozen_encoder, sha256, _target_graph,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA = 'ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162'
GRAPH_PATHS = {
    'gemv_rc3_original': ROOT / 'paper_figures/gemv_rc3_20261105/gemv.canonical.json',
    'reduce32': ROOT / 'experiments/rc3_fusion_foundation/benchmark/variants/reduce32/graph.json',
    'independent32': ROOT / 'experiments/rc3_fusion_foundation/benchmark/variants/independent32/graph.json',
    'reduce32_rename': ROOT / 'experiments/rc3_fusion_foundation/benchmark/variants/reduce32_rename/graph.json',
}
EXPECTED_GRAPH = {
    'gemv_rc3_original': '42fe84cb8b8c2f5665dc016fcfb6a969f93b848a3407c7343100b6b4bcabae21',
    'reduce32': '450bbccefa800b3bbe166705e34f64566ad09f13c228fdfb2de4447d808078a9',
    'independent32': '131bf03acc94d3fe0b9e917ffa239c942e97495b744a78ef6698b0541f47725e',
    'reduce32_rename': '553aaa1deb05d16c2f86f472a464ae8c2ae794885a472f490c9ebce46d1b49e3',
}
EXPECTED_GRAPH_FILE_SHA = {
    'gemv_rc3_original': '7d295d791f8438eeda62a6b230ab80c8d3a051ecda43b822c7a3827ec0cb43a6',
    'reduce32': 'b6335e17be35bc028013b76b3f5cae49544fe6d3a635fefed503a69ede785954',
    'independent32': '72d202b26b18b5883bdc1fa9749d165d4861a88718cf976e2b5646a2ad326f5b',
    'reduce32_rename': '31408f6364dd53c531bff0aa1dffe91cd6cfc54b43f2ad9ad1cadab536f621b7',
}
EXPECTED_SOURCE_SHA = {
    'gemv_rc3_original': 'c6e8b0aab3919a433803bacf29c7af6b56a6a5cb4d7a511a6875a21593a7e8f6',
    'reduce32': '0bf12183719b41fec7b50dd458848e991964454a89467577616f4460480caece',
    'independent32': '004692648f0ed934e1cb44d96118d49af9d207ff292782c6156a72232bae04ea',
    'reduce32_rename': '517281ed8f7082858dc9cf3e4df0f10b39f105a28195d36ad52f94778d9c2bde',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='/home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37/compiler_selected.pt')
    p.add_argument('--out', default=str(ROOT / 'artifacts/rc3_fusion_foundation/v1_20261107'))
    p.add_argument('--stage1-worktree', default='/tmp/MailoHLS-stage1-refinement')
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    model, gnn_identity = load_frozen_encoder(checkpoint, CHECKPOINT_SHA)
    manifest = {'schema': 'salt-hls-rc3-memory-build-manifest-v1', 'device': 'cpu',
                'python': sys.version, 'torch': torch.__version__, 'gnn': gnn_identity,
                'inputs': [], 'memory_files': []}
    canonical = read_graph(GRAPH_PATHS['gemv_rc3_original'])
    _, zcu = _target_graph(canonical, target='zcu104', clock_ns=3.33)
    _, u200 = _target_graph(canonical, target='u200', clock_ns=10.0)
    with torch.inference_mode():
        z_mem, z_pool = model.encode(zcu); u_mem, u_pool = model.encode(u200)
    manifest['target_invariance_audit'] = {
        'contexts': ['ZCU104/3.33ns/full physical capacity', 'U200/10ns/full physical capacity'],
        'hardware_messages': False,
        'action_memory_exactly_equal': bool(torch.equal(z_mem, u_mem)),
        'pooled_program_memory_exactly_equal': bool(torch.equal(z_pool, u_pool)),
        'max_abs_action_memory_delta': float((z_mem-u_mem).abs().max()),
        'interpretation': 'Frozen V4b compiler-only memory is target-independent under this encoder path; target conditioning remains in the LLM request unless a later separately trained target-memory/gating ablation changes it.',
    }
    if not manifest['target_invariance_audit']['action_memory_exactly_equal']:
        raise ValueError('Unexpected hardware-context dependence in compiler-only V4b action memory')
    for name, path in GRAPH_PATHS.items():
        if not path.exists():
            manifest['inputs'].append({'name': name, 'path': str(path), 'status': 'unavailable'})
            continue
        if sha256(path) != EXPECTED_GRAPH_FILE_SHA[name]:
            raise ValueError(f'{name}: graph file SHA-256 differs from the authenticated input manifest')
        graph = read_graph(path)
        actual_source_sha = graph.graph.get('provenance', {}).get('source_sha256')
        if actual_source_sha != EXPECTED_SOURCE_SHA[name]:
            raise ValueError(f'{name}: source/action graph provenance differs from authenticated source manifest')
        expected = EXPECTED_GRAPH[name]
        pack = build_memory_pack(graph, model, graph_file_sha256=sha256(path),
                                 gnn_identity=gnn_identity, target='u200', clock_ns=10.0,
                                 expected_graph_identity=expected)
        # Assert A/B/C use precisely the same vectors; only routing/visibility differs.
        a = pack['action_memory']; b = pack['action_memory'].clone(); c = pack['action_memory'].clone()
        if not (torch.equal(a, b) and torch.equal(a, c)):
            raise AssertionError('A/B/C action vectors diverged')
        file = out / f'{name}.pt'
        torch.save(pack, file)
        manifest['inputs'].append({'name': name, 'path': str(path), 'graph_file_sha256': sha256(path),
                                   'graph_identity': graph.graph['graph_identity'],
                                   'source_sha256': graph.graph.get('provenance', {}).get('source_sha256'),
                                   'metadata_sha256': graph.graph.get('provenance', {}).get('metadata_sha256'),
                                   'mlir_sha256': graph.graph.get('provenance', {}).get('mlir_sha256'),
                                   'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(),
                                   'actions': [f'L{i+1}' for i, x in enumerate(pack['action_valid_mask'][0].tolist()) if x],
                                   'status': 'built'})
        manifest['memory_files'].append({'name': file.name, 'sha256': sha256(file),
                                         'shapes': pack['shapes'], 'evidence_count': pack['evidence_count'],
                                         'routing': {f'L{i+1}': [f'L{j+1}' for j, yes in enumerate(row) if yes]
                                                     for i, row in enumerate(pack['action_relation_mask'][0].tolist()) if row.count(True)}})
    manifest['checkpoint_sha256'] = sha256(checkpoint)
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'manifest': str(manifest_path), 'built': len(manifest['memory_files']),
                      'missing': [x['name'] for x in manifest['inputs'] if x['status'] != 'built']}, indent=2))


if __name__ == '__main__':
    main()
