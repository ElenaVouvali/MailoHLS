"""Recheck cached measurement membership against freshly authenticated records."""
import torch
from experiments.v22_gnn_training.train import _fit_normalization


def validate_cached_measurements(prepared, data, manifest):
    contexts=manifest['contexts']
    if {g['key'] for g in prepared['groups']} != set(contexts):
        raise ValueError('Cached context inventory differs from frozen manifest')
    fresh={(r['kernel'],r['profile'],float(r['clock']),r['config_hash']):r for r in data.records}
    expected_inputs=set(); train_rows=[]
    for group in prepared['groups']:
        context=contexts[group['key']]
        if (group['kernel'],group['profile'],group['clock'],group['split']) != (
                context['kernel'],context['device'],context['clock_ns'],context['split']):
            raise ValueError('Cached context crosses authenticated split/physical target')
        if group['split'] not in ('train','val') or 'kalman' in group['kernel'].lower():
            raise ValueError('Reserved test kernel in cache')
        if [r['config_hash'] for r in group['rows']] != context['selected_assignment_hashes']:
            raise ValueError('Cached candidate selection differs from frozen pool')
        if len(group['rows']) != len(group['identities']):
            raise ValueError('Cached rows and candidate features are misaligned')
        for row,identity in zip(group['rows'],group['identities']):
            expected=(group['kernel'],group['profile'],float(group['clock']),row['config_hash'])
            if identity != expected or expected not in fresh:
                raise ValueError('Unmeasured or misaligned cached candidate')
            actual=fresh[expected]
            for field in ('assignment','physical','transformed','capacities','preprocessed_row','split'):
                if row[field] != actual[field]:
                    raise ValueError(f'Cached measurement provenance changed: {field}')
            expected_inputs.add(identity)
            if group['split']=='train': train_rows.append(actual)
    if set(prepared['inputs']) != expected_inputs:
        raise ValueError('Cached features include undeclared candidates')
    mean,std=_fit_normalization(train_rows)
    if not torch.equal(mean,prepared['mean']) or not torch.equal(std,prepared['std']):
        raise ValueError('Cached normalization is not fitted on declared training records')
    return {'cached_measurements_reauthenticated':sum(len(g['rows']) for g in prepared['groups']),
            'normalization_reauthenticated':True,'test_kernels_loaded':[]}
