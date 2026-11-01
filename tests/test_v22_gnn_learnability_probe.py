import copy

import numpy as np
import torch

from experiments.v22_gnn_training import fast_probe, target_condition_probe as probe


def _row(config_hash, values):
    return {'config_hash': config_hash, 'assignment': values}


def test_training_candidate_selection_is_deterministic_and_label_blind():
    rows = [_row(f'{i:02d}', {'L1': i % 4, 'L2': i // 4}) for i in range(40)]
    before = copy.deepcopy(rows)
    first = fast_probe._select_training_candidates(rows, 16)
    second = fast_probe._select_training_candidates(rows, 16)
    assert [r['config_hash'] for r in first] == [r['config_hash'] for r in second]
    assert rows == before


def test_candidate_only_model_accepts_v4_feature_dimensions():
    model = probe.CandidateOnlyModel()
    context = torch.zeros((1, probe.train.tensors.CONTEXT_DIM))
    candidates = torch.zeros((3, 64, probe.train.CANDIDATE_DIM))
    demand = torch.zeros((3, 64, probe.train.CANDIDATE_DEMAND_DIM))
    result = model(context, candidates, demand)
    assert result.shape == (3, 5)
    assert torch.isfinite(result).all()


def test_paired_target_contrasts_require_exact_assignment_hash():
    k1 = ('rodinia_pathfinder_0_baseline_0', 'zcu104', 3.33)
    k2 = ('rodinia_pathfinder_0_baseline_0', 'zcu104', 5.0)
    groups = {
        k1: [{'config_hash':'shared','physical':[1.,2,3,4,5]},
             {'config_hash':'left-only','physical':[2.,2,3,4,5]}],
        k2: [{'config_hash':'shared','physical':[1.2,2,3,4,5]},
             {'config_hash':'right-only','physical':[3.,2,3,4,5]}],
    }
    predictions = {(k1,'shared'):np.asarray([1.1,2,3,4,5]),
                   (k2,'shared'):np.asarray([1.25,2,3,4,5]),
                   (k1,'left-only'):np.asarray([2,2,3,4,5]),
                   (k2,'right-only'):np.asarray([3,2,3,4,5])}
    row = next(item for item in probe._target_contrasts(groups,predictions)['clock']
               if item['target_pair']=='zcu104:3.33->5ns')
    assert row['matched_assignments'] == 1
    np.testing.assert_allclose(row['mean_absolute_paired_error_by_target'][0], .05, atol=1e-8)


def test_rank_metric_is_undefined_for_constant_predictions():
    assert probe._metric_rank(probe.spearmanr, np.array([1.,2.]), np.array([4.,4.])) is None


def test_near_zero_objective_has_undefined_relative_regret():
    regret, reason = probe._relative_regret(np.array([0., 0.2]), 0.2)
    assert regret is None
    assert reason == 'near_zero_measured_optimum'
    regret, reason = probe._relative_regret(np.array([1., 2.]), 1.5)
    assert regret == .5
    assert reason is None
