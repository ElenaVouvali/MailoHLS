import numpy as np

from experiments.v22_gnn_training import target_condition_probe as probe


def test_meaningful_delta_positive_and_negative_signs():
    delta, meaningful = probe._meaningful_latency_delta(
        np.asarray([10.0, 10.0]), np.asarray([10.2, 9.8]))
    assert meaningful.tolist() == [True, True]
    assert delta[0] > 0 and delta[1] < 0


def test_zero_and_tiny_delta_are_negligible():
    _, meaningful = probe._meaningful_latency_delta(
        np.asarray([10.0, 10.0]), np.asarray([10.0, 10.00001]))
    assert meaningful.tolist() == [False, False]


def test_threshold_boundary_is_negligible():
    # At zero latency the one-cycle absolute tolerance defines the boundary.
    _, meaningful = probe._meaningful_latency_delta(np.asarray([0.0]), np.asarray([1e-5]))
    assert meaningful.tolist() == [False]
    # A binary-exact relative boundary is also excluded by the strict > test.
    _, meaningful = probe._meaningful_latency_delta(
        np.asarray([1.0]), np.asarray([1.25]), relative=.2, absolute_ms=0.0)
    assert meaningful.tolist() == [False]


def test_relative_threshold_uses_endpoint_latencies():
    # The same 50 us change clears the 1% threshold at 4 ms but not at 10 ms.
    _, meaningful = probe._meaningful_latency_delta(
        np.asarray([4.0, 10.0]), np.asarray([4.05, 10.05]))
    assert meaningful.tolist() == [True, False]


def test_contrast_row_reports_signs_errors_and_negligible_count():
    left = {key: {'physical': [value, 0, 0, 0, 0]} for key, value in
            [('positive', 10.0), ('negative', 10.0), ('zero', 10.0)]}
    right = {key: {'physical': [value, 0, 0, 0, 0]} for key, value in
             [('positive', 10.2), ('negative', 9.8), ('zero', 10.0)]}
    key1, key2 = ('k', 'd1', 10), ('k', 'd2', 10)
    predictions = {(key1, 'positive'): np.array([10., 0, 0, 0, 0]),
                   (key1, 'negative'): np.array([10., 0, 0, 0, 0]),
                   (key1, 'zero'): np.array([10., 0, 0, 0, 0]),
                   (key2, 'positive'): np.array([10.1, 0, 0, 0, 0]),
                   (key2, 'negative'): np.array([9.9, 0, 0, 0, 0]),
                   (key2, 'zero'): np.array([10., 0, 0, 0, 0])}
    result = probe._contrast_row('device', 'k', 'd1->d2',
                                 ['positive', 'negative', 'zero'], key1, key2,
                                 left, right, predictions)
    assert result['meaningful_positive_latency_delta_count'] == 1
    assert result['meaningful_negative_latency_delta_count'] == 1
    assert result['negligible_latency_delta_count'] == 1
    assert result['latency_sign_accuracy'] == 1.0
    assert np.isclose(result['latency_delta_mae_ms'], 1.0 / 15.0)
    assert result['latency_delta_rmse_ms'] > 0


def test_all_negligible_and_empty_groups_keep_sign_undefined():
    key1, key2 = ('k', 'd1', 10), ('k', 'd2', 10)
    left = {'x': {'physical': [10., 0, 0, 0, 0]}}
    right = {'x': {'physical': [10., 0, 0, 0, 0]}}
    predictions = {(key1, 'x'): np.array([10., 0, 0, 0, 0]),
                   (key2, 'x'): np.array([10., 0, 0, 0, 0])}
    result = probe._contrast_row('device', 'k', 'd1->d2', ['x'], key1, key2,
                                 left, right, predictions)
    assert result['meaningful_latency_delta_count'] == 0
    assert result['negligible_latency_delta_count'] == 1
    assert result['latency_sign_accuracy'] is None
    empty = probe._contrast_row('device', 'k', 'd1->d2', [], key1, key2,
                                {}, {}, {})
    assert empty['matched_assignments'] == 0
    assert empty['latency_sign_accuracy'] is None
    assert empty['latency_delta_mae_ms'] is None


def test_zero_optimum_has_undefined_relative_but_defined_absolute_regret():
    relative, reason = probe._relative_regret(np.asarray([0.0, 0.0, 2.0]), 2.0)
    assert relative is None
    assert reason == 'near_zero_measured_optimum'
    assert probe._absolute_regret(np.asarray([0.0, 0.0, 2.0]), 2.0) == 2.0


def test_feasible_shortlist_reports_selection_success_and_regret():
    result = probe._feasible_ranked_shortlists(
        [1., 2., 3.], [1., 2., 3.], [True, True, False], [True, True, True])
    assert result['topk']['1']['selected_is_measured_feasible'] is True
    assert result['topk']['1']['relative_measured_feasible_pool_regret'] == 0.0
    assert result['topk']['3']['shortlist_actual_feasible_count'] == 2


def test_false_feasible_top_choice_is_not_given_normal_regret():
    result = probe._feasible_ranked_shortlists(
        [1., 2.], [0., 1.], [False, True], [True, True])
    assert result['topk']['1']['selected_is_measured_feasible'] is False
    assert result['topk']['1']['relative_measured_feasible_pool_regret'] is None
    assert result['topk']['1']['regret_undefined_reason'] == (
        'shortlist_contains_no_measured_feasible_candidate')
    assert result['topk']['3']['relative_measured_feasible_pool_regret'] == 0.0


def test_empty_predicted_and_measured_feasible_pools_remain_explicit():
    none_predicted = probe._feasible_ranked_shortlists(
        [1., 2.], [1., 2.], [True, False], [False, False])
    assert none_predicted['predicted_feasible_selection_empty'] is True
    assert none_predicted['topk']['1']['regret_undefined_reason'] == (
        'no_predicted_feasible_candidate')
    none_measured = probe._feasible_ranked_shortlists(
        [1., 2.], [1., 2.], [False, False], [True, True])
    assert none_measured['topk']['1']['regret_undefined_reason'] == (
        'no_measured_feasible_candidate_in_pool')


def test_budget_independent_mode_uses_full_capacity_graph_for_every_request():
    cache = {'context': {1.0: {'hardware': 'full', 'numerical': 'full-num'},
                         .5: {'hardware': 'half', 'numerical': 'half-num'},
                         .1: {'hardware': 'tenth', 'numerical': 'tenth-num'}}}
    assert probe._graph_at_requested_budget(cache, 'context', .5, 'hardware') == 'half'
    assert probe._graph_at_requested_budget(cache, 'context', .5, 'hardware', True) == 'full'
    assert probe._graph_at_requested_budget(cache, 'context', .1, 'numerical', True) == 'full-num'
