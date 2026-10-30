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
)


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
