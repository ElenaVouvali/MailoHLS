"""Predeclared v2 historical-membership sensitivity, without historical labels.

Subset the frozen v2 pools, reuse identical authenticated structural features,
and refit normalization on the selected training observations only.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import torch
from experiments.v22_gnn_training import train, checkpointing
from experiments.v22_gnn_training.fast_probe import _source_hashes
from .train_probe import _json_sha, _cached_candidate_inputs, prepare_small_cohort


def cache_identity(manifest):
    return {'features': _source_hashes(), 'manifest': _json_sha(manifest),
            'preparation': hashlib.sha256((inspect.getsource(prepare_small_cohort) +
                         inspect.getsource(_cached_candidate_inputs)).encode()).hexdigest(),
            'torch': torch.__version__,
            'preferences': hashlib.sha256(Path(__file__).with_name('preferences.py').read_bytes()).hexdigest()}


def subset_manifest(parent, training_kernels):
    training = set(training_kernels)
    if not training or not training <= set(parent['training_kernels']):
        raise ValueError('Sensitivity training membership must be a nonempty subset of full training cohort')
    result = copy.deepcopy(parent)
    result.pop('manifest_sha256')
    result['training_kernels'] = sorted(training)
    result['contexts'] = {key: row for key, row in result['contexts'].items()
                          if row['split'] == 'val' or row['kernel'] in training}
    result['cohort_version'] = 'v4b-v2-historical14-membership-matched-pools'
    result['parent_manifest_sha256'] = parent['manifest_sha256']
    result['manifest_sha256'] = _json_sha(result)
    return result


def subset_prepared(parent, manifest):
    allowed = set(manifest['contexts'])
    result = dict(parent)
    groups = [g for g in parent['groups'] if g['key'] in allowed]
    if {g['key'] for g in groups} != allowed:
        raise ValueError('Subset contexts absent from authenticated full prepared cache')
    for group in groups:
        expected = manifest['contexts'][group['key']]['selected_assignment_hashes']
        if [r['config_hash'] for r in group['rows']] != expected:
            raise ValueError('Subset candidate pool changed')
    result['groups'] = groups
    identities = {identity for g in groups for identity in g['identities']}
    result['inputs'] = {key: value for key, value in parent['inputs'].items() if key in identities}
    keys = {(g['kernel'], g['profile'], g['clock']) for g in groups}
    result['graph_cache'] = {key: value for key, value in parent['graph_cache'].items() if key in keys}
    train_rows = [r for g in groups if g['split'] == 'train' for r in g['rows']]
    result['mean'], result['std'] = train._fit_normalization(train_rows)
    result['manifest'] = manifest
    result['selected_measurements'] = sum(len(g['rows']) for g in groups)
    result['preparation_seconds'] = 0.  # deterministic subset derivation, no graph rebuilding
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-cache', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Subset output must be a new directory')
    parent_manifest = json.loads(args.manifest.read_text())
    cached = torch.load(args.parent_cache, map_location='cpu', weights_only=False)
    if cached['identity'] != cache_identity(parent_manifest):
        raise ValueError('Parent cache provenance mismatch')
    from experiments.v22_gnn_training.directive_preference_probe import TRAIN_KERNELS
    child = subset_manifest(parent_manifest, TRAIN_KERNELS)
    prepared = subset_prepared(cached['prepared'], child)
    args.output_dir.mkdir(parents=True)
    checkpointing.atomic_json(args.output_dir / 'preference_cohort_manifest.json', child)
    checkpointing.atomic_json(args.output_dir / 'eligibility.json', json.loads(
        args.manifest.with_name('eligibility.json').read_text()))
    derivation = {'parent_cache_sha256': train.overlay.file_hash(args.parent_cache),
                  'parent_manifest_sha256': parent_manifest['manifest_sha256'],
                  'child_manifest_sha256': child['manifest_sha256'],
                  'policy': 'historical TRAIN_KERNELS membership; identical v2 per-context pools; no QoR membership selection',
                  'normalization': 'refitted using child training observations only',
                  'test_kernels_loaded': []}
    checkpointing.atomic_torch_save(args.output_dir / 'prepared.pt', {
        'identity': cache_identity(child), 'prepared': prepared, 'derivation': derivation})
    checkpointing.atomic_json(args.output_dir / 'derivation.json', derivation)
    print(json.dumps({'training_kernels': len(child['training_kernels']),
                      'validation_kernels': len(child['validation_kernels']),
                      'contexts': len(child['contexts']), **derivation}, indent=2))


if __name__ == '__main__':
    main()
