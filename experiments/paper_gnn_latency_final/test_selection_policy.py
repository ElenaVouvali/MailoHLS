"""Selection ordering must not accidentally restore feasibility priority."""
import unittest
from experiments.paper_gnn_latency_final.selection_policy import ranking_score


class SelectionPolicyTests(unittest.TestCase):
    def score(self, tau=.2, mae=1., regret=.5, feasible=1.):
        return ranking_score({'family_macro': dict(kendall_tau_b=tau, log_qor_mae=mae,
                             feasible_regret_at_1=regret, feasible_selection_rate=feasible)})

    def test_tau_leads_even_when_feasibility_or_other_metrics_drop(self):
        self.assertGreater(self.score(tau=.21, mae=3., regret=4., feasible=0.), self.score())

    def test_mae_then_regret_break_exact_ties(self):
        self.assertGreater(self.score(mae=.9, regret=9.), self.score())
        self.assertGreater(self.score(regret=.4), self.score())
        self.assertEqual(self.score(feasible=0.), self.score(feasible=1.))

    def test_undefined_tau_never_beats_defined_zero(self):
        self.assertLess(self.score(tau=None), self.score(tau=0.))


if __name__ == '__main__':
    unittest.main()
