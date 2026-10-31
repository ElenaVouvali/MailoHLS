import unittest

import torch
from torch_geometric.data import Data

from LLM_branch.common.directive_domains import tcad
from GNN_branch.target_graph_v22 import load_registry
from GNN_branch import target_graph_tensor_v22 as tensors
from experiments.v22_gnn_training.train import (
    CANDIDATE_DEMAND_DIM,
    CANDIDATE_DIM,
    CandidateQoRModel,
    candidate_vector,
    measured_objective,
    physical_targets,
    ranking_loss,
    fit_ranking_scales,
    validation_selection_score,
    _feasibility_metrics,
    _feasible_selection,
    _rank_statistic,
    _sample_identity,
    _sample_trace_sha256,
    _topk_feasible_selection,
    model_state_sha256,
)
from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import target_graph_v22 as overlay


class TrainingContractsTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_registry()['profiles']['zcu104']

    def test_targets_use_physical_capacity_and_log1p(self):
        row = {'latency': 2.0, 'bram_util_%': 50, 'dsp_util_%': 25,
               'ff_util_%': 10, 'lut_util_%': 5}
        transformed, physical = physical_targets(row, self.profile)
        self.assertAlmostEqual(physical[0], 2.0)
        self.assertAlmostEqual(physical[1], self.profile['resources']['BRAM_18K']['value'] * 0.5)
        self.assertAlmostEqual(transformed[0], torch.log1p(torch.tensor(2.)).item())
        self.assertEqual(len(physical), 5)

    def test_candidate_encoding_preserves_absolute_slots(self):
        actions = {'L2': tcad.SourceAction(kind='loop', trip_count=8)}
        values = {'AUTO{_PIPE_L2}': '1', 'AUTO{_UNROLL_L2}': '4'}
        encoded = candidate_vector(values, actions)
        self.assertEqual(encoded.shape, (64, 8))
        self.assertEqual(encoded[0].abs().sum().item(), 0.0)
        self.assertEqual(encoded[1, 0].item(), 1.0)
        self.assertAlmostEqual(encoded[1, 1].item(), torch.log2(torch.tensor(5.)).item() / 12)

    def test_objective_definitions_match_mailohls_area_and_adp(self):
        record = {'physical': [2.0, 5.0, 10.0, 15.0, 20.0],
                  'capacities': [10.0, 100.0, 100.0, 100.0]}
        self.assertAlmostEqual(measured_objective(record, 'PARETO_LATENCY'), 2.0)
        self.assertAlmostEqual(measured_objective(record, 'PARETO_AREA'), 0.2375)
        self.assertAlmostEqual(measured_objective(record, 'PARETO_ADP'), 0.475)

    def test_pairwise_loss_prefers_better_measured_candidate(self):
        # Prediction is [log-latency, log BRAM, log DSP, log FF, log LUT].
        prediction = torch.tensor([[1., 0., 0., 0., 0.],
                                   [2., 0., 0., 0., 0.]], requires_grad=True)
        records = [
            {'physical': [1.0, 0., 0., 0., 0.], 'capacities': [1., 1., 1., 1.]},
            {'physical': [4.0, 0., 0., 0., 0.], 'capacities': [1., 1., 1., 1.]},
        ]
        loss = ranking_loss(prediction, records, 'PARETO_LATENCY', [1., 1., 1., 1.])
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertGreater(prediction.grad[0, 0].item(), 0)
        self.assertLess(prediction.grad[1, 0].item(), 0)

    def test_ranking_ignores_practically_tied_pairs(self):
        prediction = torch.zeros((2, 5), requires_grad=True)
        records = [
            {'physical': [1.0, 0., 0., 0., 0.], 'capacities': [1., 1., 1., 1.]},
            {'physical': [1.02, 0., 0., 0., 0.], 'capacities': [1., 1., 1., 1.]},
        ]
        loss = ranking_loss(prediction, records, 'PARETO_LATENCY', [1., 1., 1., 1.])
        self.assertEqual(loss.item(), 0.0)

    def test_objective_score_normalization_restores_unit_consistent_rank_gradient(self):
        records = [
            {'physical':[1.0,10.,10.,10.,10.],'capacities':[100.,100.,100.,100.]},
            {'physical':[2.0,90.,90.,90.,90.],'capacities':[100.,100.,100.,100.]},
        ]
        prediction = torch.tensor([
            [torch.log1p(torch.tensor(2.)).item(), *([torch.log1p(torch.tensor(90.)).item()]*4)],
            [torch.log1p(torch.tensor(1.)).item(), *([torch.log1p(torch.tensor(10.)).item()]*4)],
        ], requires_grad=True)
        scale = {'PARETO_LATENCY':0.5,'PARETO_AREA':0.01,'PARETO_ADP':0.02}
        for objective in ('PARETO_LATENCY','PARETO_AREA','PARETO_ADP'):
            raw_prediction = prediction.detach().clone().requires_grad_(True)
            normalized_prediction = prediction.detach().clone().requires_grad_(True)
            raw = ranking_loss(raw_prediction,records,objective,[100.]*4)
            normalized = ranking_loss(normalized_prediction,records,objective,[100.]*4,
                                      objective_scales=scale)
            raw.backward(); normalized.backward()
            self.assertGreater(float(normalized_prediction.grad.norm()), 0.0)
            self.assertNotAlmostEqual(float(normalized_prediction.grad.norm()),
                                      float(raw_prediction.grad.norm()), places=6)
            self.assertTrue(torch.isfinite(normalized_prediction.grad).all())

    def test_ranking_scales_are_fit_from_only_the_supplied_training_records(self):
        train = [
            {'kernel':'k','profile':'zcu104','clock':10.,
             'physical':[1.0,10.,10.,10.,10.],'capacities':[100.,100.,100.,100.]},
            {'kernel':'k','profile':'zcu104','clock':10.,
             'physical':[2.0,90.,90.,90.,90.],'capacities':[100.,100.,100.,100.]},
        ]
        scales = fit_ranking_scales(train)
        self.assertEqual(set(scales),{'PARETO_LATENCY','PARETO_AREA','PARETO_ADP'})
        self.assertTrue(all(value > 0 for value in scales.values()))

    def test_conditioning_arms_keep_identical_late_numeric_context(self):
        data = Data()
        data.action_kind_slot = torch.zeros((1, 64), dtype=torch.long)
        data.matched_mask = torch.ones((1, 64), dtype=torch.bool)
        data.registered_mask = torch.ones((1, 64), dtype=torch.bool)
        data.hardware_context = torch.full((1, tensors.CONTEXT_DIM), 0.25)
        model = CandidateQoRModel().eval()
        memory = torch.randn((1, 64, 128))
        pooled = torch.randn((1, 128))
        candidates = torch.zeros((2, 64, CANDIDATE_DIM))
        demands = torch.zeros((2, 64, CANDIDATE_DEMAND_DIM))
        with torch.no_grad():
            numerical = model.from_memory(data, memory, pooled, candidates, demands, 'numerical')
            hardware = model.from_memory(data, memory, pooled, candidates, demands, 'hardware')
        self.assertTrue(torch.equal(numerical, hardware))

    def test_v4_schema_is_distinct_from_historical_tensor_schema(self):
        self.assertEqual(tensors.SCHEMA, 'salt-hls-target-tensors-v4-semantic-wide-affine-experimental')
        self.assertNotEqual(tensors.SCHEMA, 'salt-hls-target-tensors-v3-semantic-experimental')

    def test_hardware_edges_are_the_only_removed_tensor_messages(self):
        data = Data(
            node_type=torch.tensor([
                tensors.NODE_TYPES.index('operation'),
                tensors.NODE_TYPES.index('action_point'),
                tensors.NODE_TYPES.index('hardware_resource')]),
            edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long),
            edge_type=torch.tensor([
                tensors.RELATIONS.index('operand'),
                tensors.RELATIONS.index('reverse:operand'),
                tensors.RELATIONS.index(next(r for r in overlay.RELATIONS if r.endswith('action_resource_relevance'))),
                tensors.RELATIONS.index('reverse:' + next(r for r in overlay.RELATIONS if r.endswith('action_resource_relevance')))],
                dtype=torch.long),
            edge_attr=torch.randn((4, tensors.EDGE_DIM)),
            negative_edge_index=torch.tensor([[1], [2]], dtype=torch.long),
            negative_edge_type=torch.tensor([0]),
            negative_edge_attr=torch.randn((1, tensors.EDGE_DIM)),
        )
        numerical = tensors.without_hardware_messages(data)
        self.assertEqual(numerical.num_nodes, data.num_nodes)  # audit-only hardware nodes remain
        self.assertEqual(numerical.edge_index.shape[1], 2)
        self.assertTrue(torch.equal(numerical.edge_attr, data.edge_attr[:2]))
        self.assertTrue(torch.equal(numerical.negative_edge_index, data.negative_edge_index))
        self.assertTrue(torch.equal(numerical.negative_edge_attr, data.negative_edge_attr))

    def test_v4_hardware_features_reach_actions_only_through_hardware_edges(self):
        torch.manual_seed(123)
        model = tensors.SmallGraphEncoder().eval()
        n = 3
        relation = next(r for r in overlay.RELATIONS if r.endswith('action_resource_relevance'))
        edge_types = torch.tensor([tensors.RELATIONS.index(relation),
                                   tensors.RELATIONS.index('reverse:' + relation)])
        edge_index = torch.tensor([[2, 1], [1, 2]], dtype=torch.long)
        base = Data(
            x=torch.zeros((n, tensors.NODE_DIM)),
            node_type=torch.tensor([tensors.NODE_TYPES.index('operation'),
                                    tensors.NODE_TYPES.index('action_point'),
                                    tensors.NODE_TYPES.index('hardware_resource')]),
            op_type=torch.zeros(n, dtype=torch.long), value_type=torch.zeros(n, dtype=torch.long),
            resource_type=torch.tensor([0, 0, 1]), root_type=torch.zeros(n, dtype=torch.long),
            edge_index=edge_index, edge_type=edge_types,
            edge_attr=torch.zeros((2, tensors.EDGE_DIM)),
            action_node_index=torch.tensor([1]), action_slot=torch.tensor([0]),
            action_kind_slot=torch.zeros((1, 64), dtype=torch.long),
            registered_mask=torch.zeros((1, 64), dtype=torch.bool),
            matched_mask=torch.zeros((1, 64), dtype=torch.bool),
            unresolved_mask=torch.zeros((1, 64), dtype=torch.bool),
            source_only_mask=torch.zeros((1, 64), dtype=torch.bool),
            action_demand=torch.zeros((1, 64, tensors.ACTION_DEMAND_DIM)),
            action_semantic=torch.zeros((1, 64, tensors.SEMANTIC_EDGE_DIM)),
            action_semantic_mask=torch.zeros((1, 64), dtype=torch.bool),
        )
        base.registered_mask[0, 0] = True
        base.matched_mask[0, 0] = True
        changed = base.clone()
        changed.x[2, 0] = 4.0
        changed.x[2, 1] = 1.0

        numerical_a = tensors.without_hardware_messages(base)
        numerical_b = tensors.without_hardware_messages(changed)
        with torch.no_grad():
            memory_na, pooled_na = model.memories(numerical_a)
            memory_nb, pooled_nb = model.memories(numerical_b)
            memory_ha, _ = model.memories(base)
            memory_hb, _ = model.memories(changed)
        self.assertTrue(torch.equal(memory_na, memory_nb))
        self.assertTrue(torch.equal(pooled_na, pooled_nb))
        self.assertGreater(float((memory_ha - memory_hb).norm()), 1e-8)

    def test_hardware_relation_gradients_exist_only_when_messages_are_enabled(self):
        torch.manual_seed(91)
        model = tensors.SmallGraphEncoder().train()
        relation = next(r for r in overlay.RELATIONS if r.endswith('action_resource_relevance'))
        hardware_relation_ids = [tensors.RELATIONS.index(relation),
                                 tensors.RELATIONS.index('reverse:' + relation)]
        data = Data(
            x=torch.randn((3, tensors.NODE_DIM)),
            node_type=torch.tensor([tensors.NODE_TYPES.index('operation'),
                                    tensors.NODE_TYPES.index('action_point'),
                                    tensors.NODE_TYPES.index('hardware_resource')]),
            op_type=torch.tensor([0, 0, 0]), value_type=torch.tensor([0, 0, 0]),
            resource_type=torch.tensor([0, 0, 1]), root_type=torch.zeros(3, dtype=torch.long),
            edge_index=torch.tensor([[2, 1], [1, 2]], dtype=torch.long),
            edge_type=torch.tensor(hardware_relation_ids),
            edge_attr=torch.randn((2, tensors.EDGE_DIM)),
            action_node_index=torch.tensor([1]), action_slot=torch.tensor([0]),
            action_demand=torch.zeros((1, 64, tensors.ACTION_DEMAND_DIM)),
            action_semantic=torch.zeros((1, 64, tensors.SEMANTIC_EDGE_DIM)),
            action_semantic_mask=torch.zeros((1, 64), dtype=torch.bool),
        )
        data.registered_mask = torch.zeros((1, 64), dtype=torch.bool)
        data.registered_mask[0, 0] = True
        data.matched_mask = data.registered_mask.clone()
        data.action_kind_slot = torch.zeros((1, 64), dtype=torch.long)

        hardware_output = model(data)[data.action_node_index]
        hardware_output.square().sum().backward()
        relation_grad = model.relation.weight.grad[hardware_relation_ids]
        self.assertTrue(torch.isfinite(relation_grad).all())
        self.assertGreater(float(relation_grad.abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        numerical = tensors.without_hardware_messages(data)
        numerical_output = model(numerical)[numerical.action_node_index]
        numerical_output.square().sum().backward()
        relation_grad = model.relation.weight.grad[hardware_relation_ids]
        self.assertTrue(torch.equal(relation_grad, torch.zeros_like(relation_grad)))

    def test_initialization_and_sample_identity_are_deterministic_and_exact(self):
        torch.manual_seed(123)
        first = CandidateQoRModel()
        first_hash = model_state_sha256(first)
        torch.manual_seed(123)
        second = CandidateQoRModel()
        self.assertEqual(first_hash, model_state_sha256(second))
        group_key = ('kernel', 'zcu104', 10.0)
        records = [{'config_hash': 'cfg-a'}, {'config_hash': 'cfg-b'}]
        identity = _sample_identity(group_key, records, [0, 1], 'PARETO_LATENCY', 1.0)
        self.assertEqual(identity['candidate_hashes'], ['cfg-a', 'cfg-b'])
        self.assertEqual(_sample_trace_sha256([identity]), _sample_trace_sha256([identity]))

    def test_undefined_rank_statistics_are_not_scored_as_perfect(self):
        value, reason = _rank_statistic(__import__('scipy').stats.spearmanr,
                                        [1., 1., 1.], [1., 2., 3.])
        self.assertIsNone(value)
        self.assertEqual(reason, 'constant_measured_scores')

    def test_feasibility_metrics_keep_empty_denominators_undefined(self):
        result = _feasibility_metrics([True, False], [False, False])
        self.assertIsNone(result['precision_among_predicted_feasible'])
        self.assertEqual(result['recall_of_measured_feasible'], 0.0)
        no_reference = _feasibility_metrics([False, False], [True, False])
        self.assertIsNone(no_reference['false_infeasible_rate_among_measured_feasible'])

    def test_topk_measured_pool_reports_failures_and_success_regret(self):
        result = _topk_feasible_selection(
            predicted_score=[0.1, 0.2, 0.3, 0.4],
            measured_score=[2.0, 1.0, 1.1, 3.0],
            predicted_feasible=[True, True, False, True],
            measured_feasible=[False, True, True, False],
            k_values=(1, 3))
        self.assertFalse(result['1']['success'])
        self.assertIsNone(result['1']['measured_pool_regret_fraction'])
        self.assertTrue(result['3']['success'])
        self.assertAlmostEqual(result['3']['measured_pool_regret_fraction'], 0.0)
        self.assertEqual(result['3']['shortlist_indices'], [0, 1, 3])
        no_predictions = _topk_feasible_selection([1.], [1.], [False], [True], (1,))
        self.assertEqual(no_predictions['1']['status'], 'no_predicted_feasible_candidate')
        no_reference = _topk_feasible_selection([1.], [1.], [True], [False], (1,))
        self.assertEqual(no_reference['1']['status'], 'no_measured_feasible_reference')

    def test_single_selection_reports_false_feasibility_and_empty_pools(self):
        false_feasible = _feasible_selection([0.1, 0.2], [1.0, 2.0], [True, False], [False, True])
        self.assertEqual(false_feasible['status'], 'false_feasible_selection')
        self.assertIsNone(false_feasible['measured_pool_regret_fraction'])
        no_prediction = _feasible_selection([1.0], [1.0], [False], [True])
        self.assertEqual(no_prediction['status'], 'no_predicted_feasible_candidate')
        no_reference = _feasible_selection([1.0], [1.0], [True], [False])
        self.assertEqual(no_reference['status'], 'no_measured_feasible_reference')
        with self.assertRaises(ValueError):
            _topk_feasible_selection([float('nan')], [1.0], [True], [True])

    def test_validation_selection_prioritizes_feasibility_and_ranking(self):
        baseline = {
            'macro_feasibility_accuracy': 0.7,
            'macro_rank_spearman': 0.3,
            'macro_evaluated_pool_regret_fraction': 0.2,
            'macro_latency_mape': 0.2,
            'macro_resource_mae_fraction': 0.1,
        }
        feasible = dict(baseline, macro_feasibility_accuracy=0.9)
        ranked = dict(baseline, macro_rank_spearman=0.7)
        self.assertLess(validation_selection_score(feasible), validation_selection_score(baseline))
        self.assertLess(validation_selection_score(ranked), validation_selection_score(baseline))


if __name__ == '__main__':
    unittest.main()
