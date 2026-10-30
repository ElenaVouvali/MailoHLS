import random
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.v22_gnn_training import checkpointing
from experiments.v22_gnn_training import train


class CheckpointingTest(unittest.TestCase):
    def test_atomic_json_and_torch_writes_leave_no_temporary_files(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            checkpointing.atomic_json(root/'status.json',{'status':'one'})
            checkpointing.atomic_json(root/'status.json',{'status':'two'})
            self.assertEqual({'status':'two'},__import__('json').loads((root/'status.json').read_text()))
            checkpointing.atomic_torch_save(root/'state.pt',{'step':3,'weight':torch.tensor([1.,2.])})
            saved=torch.load(root/'state.pt',map_location='cpu',weights_only=False)
            self.assertEqual(3,saved['step'])
            self.assertTrue(torch.equal(torch.tensor([1.,2.]),saved['weight']))
            self.assertEqual({'status.json','state.pt'}, {path.name for path in root.iterdir()})

    def test_rng_restore_reproduces_python_numpy_and_torch_sequences(self):
        random.seed(123); np.random.seed(123); torch.manual_seed(123)
        state=checkpointing.capture_rng_state()
        expected=(random.random(),np.random.random(),torch.rand(4))
        checkpointing.restore_rng_state(state)
        actual=(random.random(),np.random.random(),torch.rand(4))
        self.assertEqual(expected[0],actual[0])
        self.assertEqual(expected[1],actual[1])
        self.assertTrue(torch.equal(expected[2],actual[2]))

    def test_resume_rejects_changed_protocol_fingerprint(self):
        checkpoint={'protocol_fingerprint':'data-A-code-A'}
        checkpointing.verify_fingerprint(checkpoint,'data-A-code-A')
        with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
            checkpointing.verify_fingerprint(checkpoint,'data-A-code-B')

    def test_resume_accepts_only_its_own_live_detached_launcher_parent(self):
        import json
        import os
        with TemporaryDirectory() as temporary:
            output=Path(temporary)
            (output/'launcher.lock').write_text(json.dumps({'pid':os.getpid()}))
            with mock.patch.object(train.os,'getppid',return_value=os.getpid()):
                train._verify_resume_lock(output)
            with mock.patch.object(train.os,'getppid',return_value=os.getpid()+1):
                with self.assertRaisesRegex(RuntimeError,'still active'):
                    train._verify_resume_lock(output)

    def test_interrupted_optimizer_resume_matches_uninterrupted_sequence(self):
        def initialize():
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            model=torch.nn.Linear(3,1)
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
            return model,optimizer

        def update(model,optimizer):
            x=torch.tensor([[random.random(),np.random.random(),float(torch.rand(()))]])
            y=torch.tensor([[0.25]])
            optimizer.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean()
            loss.backward(); optimizer.step()
            return float(loss.detach())

        baseline,baseline_opt=initialize()
        baseline_losses=[update(baseline,baseline_opt) for _ in range(5)]

        interrupted,interrupted_opt=initialize()
        prefix=[update(interrupted,interrupted_opt) for _ in range(2)]
        saved_model={key:value.detach().clone() for key,value in interrupted.state_dict().items()}
        saved_optimizer=interrupted_opt.state_dict()
        saved_rng=checkpointing.capture_rng_state()

        resumed,resumed_opt=initialize()
        resumed.load_state_dict(saved_model)
        resumed_opt.load_state_dict(saved_optimizer)
        checkpointing.restore_rng_state(saved_rng)
        suffix=[update(resumed,resumed_opt) for _ in range(3)]

        self.assertEqual(baseline_losses[:2],prefix)
        self.assertEqual(baseline_losses[2:],suffix)
        for expected,actual in zip(baseline.parameters(),resumed.parameters()):
            self.assertTrue(torch.equal(expected,actual))

    def test_candidate_demand_cache_distinguishes_config_and_target_context(self):
        dataset=train.PreparedData.__new__(train.PreparedData)
        dataset.demand_cache=OrderedDict()
        dataset.graphs={'toy':('program','auth')}
        dataset.actions={'toy':{}}
        calls=[]

        def enrich(base,profile,clock,authentication=None):
            return (profile,clock)

        def demand(context,assignment,actions):
            calls.append((context,tuple(sorted(assignment.items()))))
            value=float(len(calls))
            return torch.full((64,train.CANDIDATE_DEMAND_DIM),value),None

        rows=[{'kernel':'toy','profile':'zcu104','clock':10.,'config_hash':'cfg-a','assignment':{}},
              {'kernel':'toy','profile':'u200','clock':10.,'config_hash':'cfg-a','assignment':{}},
              {'kernel':'toy','profile':'zcu104','clock':5.,'config_hash':'cfg-a','assignment':{}},
              {'kernel':'toy','profile':'zcu104','clock':10.,'config_hash':'cfg-b','assignment':{}}]
        with mock.patch.object(train.overlay,'enrich',side_effect=enrich), \
             mock.patch.object(train,'candidate_demand_tensor',side_effect=demand):
            features=[dataset.candidate_inputs(row)[1] for row in rows]
            repeated=dataset.candidate_inputs(rows[0])[1]
        self.assertEqual(4,len(calls))
        self.assertEqual([1.,2.,3.,4.],[float(value[0,0]) for value in features])
        self.assertTrue(torch.equal(features[0],repeated))


if __name__=='__main__':
    unittest.main()
