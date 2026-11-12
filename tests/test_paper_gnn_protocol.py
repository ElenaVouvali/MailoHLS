import unittest
from GNN_branch.paper_protocol import feasibility, objective_log2, evaluate, ranking_loss


class PaperProtocolTests(unittest.TestCase):
    def test_unknown_is_not_infeasible(self):
        self.assertIsNone(feasibility([None, 1, 1, 1], [4]*4, measured_success=True))
        self.assertIsNone(feasibility([100]*4, [4]*4, measured_success=False))
        self.assertIs(feasibility([5, None, 1, 1], [4]*4, measured_success=True), False)
        self.assertIs(feasibility([4]*4, [4]*4, measured_success=True), True)

    def test_objective_conventions_are_not_invented(self):
        self.assertEqual(objective_log2(3, 4, 'PARETO_ADP'), 7)

    def test_metrics_and_locked_test(self):
        rows = []
        for family in ('a', 'b'):
            for design, latency in [('fast', 1.), ('slow', 2.)]:
                rows.append(dict(split='val', family=family, kernel=family, device='d', clock=10,
                    budget_id='full', design_id=design, measured_success=True,
                    resources=[1]*4, predicted_resources=[1]*4, budget=[4]*4,
                    log_latency=latency, log_area=1., predicted_log_latency=latency,
                    predicted_log_area=1.))
        result = evaluate(rows + [rows[0]], 'PARETO_LATENCY')
        self.assertEqual(result['family_macro'], dict(kendall_tau_b=1., feasible_regret_at_1=0., log_qor_mae=0., feasible_selection_rate=1.))
        self.assertEqual(result['coverage']['duplicate_views'], 1)
        rows[0] = {**rows[0], 'split': 'test'}
        with self.assertRaises(ValueError): evaluate(rows, 'PARETO_LATENCY')

    def test_rank_gradient_and_context_separation(self):
        import torch
        pred = torch.tensor([1., 2., 9.], requires_grad=True)
        loss = ranking_loss(pred, torch.tensor([1., 2., 0.]), [('k','d',10)]*2+[('k','d',5)], [True]*3, ['a','b','c'])
        loss.backward()
        self.assertGreater(pred.grad[0].item(), 0)
        self.assertEqual(pred.grad[2].item(), 0)


if __name__ == '__main__': unittest.main()
