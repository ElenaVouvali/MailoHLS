#!/usr/bin/env python3
"""Reconstruct explicitly marked provenance for a pre-contract GNN run."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from klepto.archives import file_archive


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def canonical_sha256(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flags_klepto', required=True, type=Path)
    parser.add_argument('--last_checkpoint', required=True, type=Path)
    parser.add_argument('--feature_schema', required=True, type=Path)
    parser.add_argument('--dataset_index', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', action='append', nargs=3,
                        metavar=('TAG', 'EPOCH', 'PATH'), default=[])
    args = parser.parse_args()

    archive = file_archive(str(args.flags_klepto))
    archive.load()
    flags = archive['FLAGS']
    resolved_flags = {
        key: jsonable(value) for key, value in sorted(vars(flags).items())
    }
    records = torch.load(args.dataset_index, map_location='cpu', weights_only=False)
    val = {item.strip() for item in flags.val_kernels.split(',') if item.strip()}
    test = {item.strip() for item in flags.test_kernels.split(',') if item.strip()}
    excluded = {
        item.strip() for item in flags.development_exclude_kernels.split(',')
        if item.strip()
    }
    split = {'train': [], 'val': [], 'test': []}
    for record in records:
        kernel = record['kernel_name']
        if kernel in excluded:
            continue
        destination = 'test' if kernel in test else 'val' if kernel in val else 'train'
        split[destination].append(jsonable(record))

    last = torch.load(args.last_checkpoint, map_location='cpu', weights_only=False)
    feature_schema = json.loads(args.feature_schema.read_text(encoding='utf-8'))
    model_flags = (
        'task', 'num_layers', 'D', 'target', 'gnn_type', 'encode_edge',
        'encode_edge_position', 'dropout', 'jkn_mode', 'jkn_enable',
        'gnn_layer_after_MLP', 'node_attention',
        'node_attention_MLP', 'separate_T', 'separate_P', 'separate_pseudo',
        'separate_icmp', 'P_use_all_nodes', 'gae_T', 'gae_P', 'input_encode',
        'decoder_type',
        'decompose_targets', 'target_mode', 'standardize_targets',
        'qor_output_init_scale',
        'MLP_common_lyr', 'pragma_as_MLP', 'pragma_as_MLP_list',
        'pragma_scope', 'keep_pragma_attribute', 'pragma_order',
        'pragma_MLP_hidden_channels',
        'merge_MLP_hidden_channels', 'activation',
    )
    contract = {
        'format': 'mailohls-gnn-checkpoint-v1',
        'provenance_status': 'reconstructed',
        'git_commit': 'unknown',
        'resolved_flags': resolved_flags,
        'model_init_flags': {
            key: resolved_flags[key] for key in model_flags if key in resolved_flags
        },
        'model_init': {
            'in_channels': int(flags.num_features),
            'edge_dim': int(flags.edge_dim),
            'targets': jsonable(flags.target),
            'resource_aux_heads': False,
        },
        'feature_schema': {
            'path': str(args.feature_schema.resolve()),
            'sha256': sha256(args.feature_schema),
            'version': feature_schema.get('feature_schema_version'),
            'node_feature_dim': feature_schema.get('node_feature_dim'),
            'edge_feature_dim': feature_schema.get('edge_feature_dim'),
        },
        'dataset_manifest': {
            'path': str(args.dataset_index.resolve()),
            'sha256': sha256(args.dataset_index),
        },
        'dataset_manifest_sha256': sha256(args.dataset_index),
        'target_stats': jsonable(last['target_stats']),
        'resource_stats': None,
        'split_sha256': canonical_sha256(split),
        'reconstruction_inputs': {
            'flags_klepto_sha256': sha256(args.flags_klepto),
            'last_checkpoint_sha256': sha256(args.last_checkpoint),
        },
    }
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    contract_hash = sha256(args.output)
    for tag, epoch, checkpoint_path in args.checkpoint:
        checkpoint = Path(checkpoint_path)
        sidecar = {
            'format': 'mailohls-gnn-checkpoint-sidecar-v1',
            'provenance_status': 'reconstructed',
            'checkpoint_sha256': sha256(checkpoint),
            'contract_sha256': contract_hash,
            'checkpoint_tag': tag,
            'checkpoint_epoch': int(epoch),
        }
        Path(f'{checkpoint}.json').write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )


if __name__ == '__main__':
    main()
