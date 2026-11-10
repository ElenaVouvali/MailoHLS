"""Explicit user-directed ranking selection revision; optimizer/losses unchanged."""
import math
import shutil
from pathlib import Path

RANKING_POLICY = 'family_macro tau_b DESC, log2 MAE ASC, conditional regret ASC; feasible_selection_rate diagnostic only'


def ranking_score(report):
    m = report['family_macro']
    return tuple(sign * m[key] if m[key] is not None else -math.inf
                 for key, sign in (('kendall_tau_b', 1), ('log_qor_mae', -1),
                                   ('feasible_regret_at_1', -1)))


def migrate_resume(state, protocol, source, out):
    """Only selection policy and the audited training entrypoint may change.

    The old run stays untouched. A historical best must have available weights;
    silently substituting an unavailable intermediate checkpoint is forbidden.
    """
    import torch
    from experiments.paper_gnn.data import sha
    from experiments.paper_gnn_latency_final.train import atomic_save, write_json
    if source.parent.resolve() == out.resolve():
        raise ValueError('Policy revision requires a separate output directory')
    old = state['protocol']
    expected = dict(protocol)
    expected['selection'] = old['selection']
    expected['source_sha256'] = dict(protocol['source_sha256'])
    key = 'experiments/paper_gnn_latency_final/train.py'
    snapshot = out.parent / 'source_snapshot/train.py'
    if sha(snapshot) != old['source_sha256'][key]:
        raise ValueError('Original training implementation snapshot mismatch')
    expected['source_sha256'][key] = old['source_sha256'][key]
    if expected != old:
        raise ValueError('Unexpected change beyond selection policy/entrypoint')
    curves = state['curves']
    scores = [ranking_score({'family_macro': row}) for row in curves]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_epoch = curves[best_index]['epoch']
    if best_epoch == state['epoch']:
        selected = state
    else:
        selected = torch.load(source.parent / 'selected.pt', map_location='cpu', weights_only=False)
        if selected['epoch'] != best_epoch:
            raise ValueError('Weights for the revised historical best are unavailable')
    # Copy states without modifying their tensor, optimizer or RNG contents.
    resumed = dict(state, protocol=protocol, best=scores[best_index],
                   stale=state['epoch'] - best_epoch)
    selected = dict(selected, protocol=protocol, best=scores[best_index], stale=0)
    for file in source.parent.glob('sampling_epoch_*.json'):
        shutil.copy2(file, out / file.name)
    atomic_save(selected, out / 'selected.pt')
    write_json(curves, out / 'curves.json')
    write_json({
        'schema': 'salt-latency-selection-revision-v2',
        'basis': 'User explicitly requested tau-b, then log2 MAE, then feasible regret; feasibility diagnostic only',
        'post_hoc_policy_change': True, 'original_policy_preserved': True,
        'source_checkpoint': str(source.resolve()), 'source_checkpoint_sha256': sha(source),
        'original_protocol': old, 'revised_protocol': protocol,
        'completed_epochs_preserved': state['epoch'] + 1,
        'optimizer_steps_preserved': state['steps'],
        'historical_best_epoch_one_based': best_epoch + 1,
        'recomputed_stale_epochs': resumed['stale'],
        'preserved': ['model', 'optimizer', 'scheduler', 'torch_rng', 'cuda_rng', 'sampler_rng', 'curves'],
        'total_epoch_cap': 40, 'patience': 8,
        'architecture_loss_data_sampling_unchanged': True,
        'implementation_sha256': sha(Path(__file__)),
    }, out / 'selection_revision.json')
    return resumed
