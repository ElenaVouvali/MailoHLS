import math
import unittest
import torch
from GNN_branch.paper_protocol import evaluate, ranking_loss
from experiments.paper_gnn.train import backend

class Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend()
        from model import within_kernel_delta_pair_loss
        cls.delta=staticmethod(within_kernel_delta_pair_loss)

    def test_actual_log_delta_and_near_ties(self):
        p=torch.tensor([0.,0.,0.],requires_grad=True)
        y=torch.tensor([0.,math.log2(1.01),2.])
        loss=self.delta(p,y,['k:d:10']*3,tie_epsilon=math.log2(1.05),beta=.5)
        loss.backward()
        self.assertGreater(loss.item(),0)
        self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertAlmostEqual(float(loss),float(self.delta(p+37,y+11,['k:d:10']*3,tie_epsilon=math.log2(1.05),beta=.5)),places=5)
        self.assertEqual(float(self.delta(p[:2],y[:2],['k:d:10']*2,tie_epsilon=math.log2(1.05),beta=.5)),0.)

    def test_device_clock_separation(self):
        p=torch.tensor([0.,0.,9.],requires_grad=True)
        y=torch.tensor([0.,2.,-20.])
        loss=self.delta(p,y,['k:d:10','k:d:10','k:d:5'],tie_epsilon=math.log2(1.05),beta=.5)
        loss.backward();self.assertEqual(p.grad[2],0.)

    def test_duplicate_rank_observation_ignored(self):
        a=torch.tensor([0.,0.],requires_grad=True)
        b=torch.tensor([0.,0.,999.],requires_grad=True)
        x=ranking_loss(a,torch.tensor([0.,2.]),[('k','d',10)]*2,[True]*2,['a','b'])
        y=ranking_loss(b,torch.tensor([0.,2.,1000.]),[('k','d',10)]*3,[True]*3,['a','b','a'])
        self.assertEqual(float(x),float(y))

    def row(self,name,true,pred,demand,pred_demand):
        return dict(split='val',family='f',kernel='k',device='d',clock=10,budget_id='1',design_id=name,measured_success=True,resources=[demand]*4,predicted_resources=[pred_demand]*4,budget=[100]*4,log_latency=true,predicted_log_latency=pred,log_area=0.,predicted_log_area=0.)

    def test_selection_never_uses_oracle_filter(self):
        r=evaluate([self.row('fast_infeasible',0.,0.,150.,1.),self.row('feasible',1.,1.,10.,1.)],'PARETO_LATENCY')
        self.assertEqual(r['family_macro']['feasible_selection_rate'],0.)
        self.assertIsNone(r['family_macro']['feasible_regret_at_1'])
        self.assertEqual(r['pools'][0]['selected_design'],'fast_infeasible')

    def test_impossible_pool_is_separate(self):
        r=evaluate([self.row('a',0.,0.,150.,1.)],'PARETO_LATENCY')
        self.assertIsNone(r['family_macro']['feasible_selection_rate'])
        self.assertEqual(r['impossible_pool_count'],1)

    def test_rounded_equal_capacity_is_unknown(self):
        row=self.row('a',0.,0.,100.,1.);row['measured_resource_precision']='rounded_percentages'
        r=evaluate([row],'PARETO_LATENCY')
        self.assertEqual(r['pools'][0]['selection_status'],'unknown_physical_feasibility')

if __name__=='__main__':unittest.main()
