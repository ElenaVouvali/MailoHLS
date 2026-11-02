"""CPU fixed-training-context forward/backward sanity; not held-out evidence."""
import argparse
import json
from pathlib import Path
import torch
from .train_probe import _initial_models, _forward_pair, _all_parameters
from .models import hybrid_loss


def run(cache_path):
    torch.set_num_threads(4)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    prepared = cache['prepared']
    group = next(g for g in prepared['groups'] if g['split'] == 'train' and g['budget_pairs'][1.])
    a, b, _ = group['budget_pairs'][1.][0]
    graph = prepared['graph_cache'][(group['kernel'], group['profile'], group['clock'])]
    features = [prepared['inputs'][group['identities'][i]] for i in (a, b)]
    candidates = torch.stack([x[0] for x in features]); demands = torch.stack([x[1] for x in features])
    target = (torch.tensor([group['rows'][i]['transformed'] for i in (a,b)]) - prepared['mean']) / prepared['std']
    results = {}
    for arm, model in _initial_models(123, prepared['profile_feature_dim']).items():
        opt = torch.optim.AdamW(_all_parameters(model), lr=3e-4, weight_decay=1e-4)
        history = []
        for _ in range(3):
            prediction, pair = _forward_pair(arm, model, graph, candidates, demands, 1.,
                prepared['mean'], prepared['std'], prepared['profile_features'][(group['profile'], group['clock'])])
            duplicate, _ = _forward_pair(arm, model, graph, candidates, demands, .1,
                prepared['mean'], prepared['std'], prepared['profile_features'][(group['profile'], group['clock'])])
            assert torch.equal(prediction, duplicate)
            loss, _ = hybrid_loss(prediction, target, pair, [True])
            opt.zero_grad(); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(_all_parameters(model), 1.)
            assert torch.isfinite(loss) and torch.isfinite(grad) and grad > 0
            history.append({'loss': float(loss), 'gradient_norm': float(grad)})
            opt.step()
        results[arm] = {'history': history, 'parameter_count': sum(p.numel() for p in _all_parameters(model))}
    return {'schema': 'v4b-real-training-context-sanity-v2', 'purpose': 'finite loss and gradients only, not generalization',
            'context': group['key'], 'candidate_hashes': [group['rows'][i]['config_hash'] for i in (a,b)],
            'data_kind': 'authenticated measured training examples', 'device': 'cpu',
            'cache_identity': cache['identity'], 'arms': results, 'test_kernels_loaded': []}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True); p.add_argument('--output', required=True)
    args=p.parse_args()
    report=run(args.cache)
    with open(args.output, 'x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
    print(json.dumps(report['arms'], indent=2))


if __name__ == '__main__':
    main()
