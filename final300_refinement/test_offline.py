"""Standard-library smoke tests; no GPU, SSH, synthesis or repository checkout."""
import json, pathlib, subprocess, sys, tempfile, unittest
import refine_final300 as m

PKG=pathlib.Path(__file__).parent
class Smoke(unittest.TestCase):
    def test_checkpoint_scorer_generated_and_syntax(self):
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'scorer.py'
            subprocess.run([sys.executable,str(PKG/'setup_scorer.py'),'--output',str(out)],check=True)
            self.assertIn('score_slate',out.read_text())
            compile(out.read_text(),str(out),'exec')
    def test_budget_qualifications(self):
        sample={'status':'success','latency_cycles':100,'budget_feasible':True,'timing_qualified':True}
        self.assertTrue(m.measured_ok(sample))
        for k,v in [('status','timed_out'),('latency_cycles',None),('budget_feasible',False),('timing_qualified',False)]:
            self.assertFalse(m.measured_ok({**sample,k:v}))
    def test_history_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d)
            p=root/'step_01';p.mkdir()
            m.write(p/'query.json',{'signature':'x','proposal':{'assignment':{'f':'0'},'signature':'x','proposal_type':'baseline','reference_changed_actions':[]}})
            self.assertEqual(m.histories(root,None),[])
            m.write(p/'measurement.json',{'status':'timed_out','latency_cycles':None})
            self.assertEqual(len(m.histories(root,None)),1)
            m.write(root/'identity.json',{'checkpoint':'a'})
    def test_runner_accepts_explicit_provenance(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);folder=root/'source'/'Kalman';folder.mkdir(parents=True)
            (folder/'kernel_info.txt').write_text('kalman\nL1,loop,8\nL2,array,a,1,8\n')
            source=folder/'source.c'
            source.write_text('void kalman() {\n L1: for(int i=0;i<8;i++) {}\n L2: int a[8];\n}\n')
            record={'kernel':'Kalman','checkpoint_step':300,'status':'refinement_proposal_not_synthesized',
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'device':'xczu7ev-ffvc1156-2-e','clock_period_ns':10.,'budget_percent':100.,
                'checkpoint_sha256':'dummy-for-synthetic-test-only','directives':[
                  {'action_id':'L1','kind':'loop','pipeline_ii':0,'unroll_factor':2},
                  {'action_id':'L2','kind':'array','partition_type':'cyclic','partition_factor':2,'partition_dim':1}]}
            prediction=root/'candidate.jsonl';prediction.write_text(json.dumps(record)+'\n')
            subprocess.run([sys.executable,str(PKG/'coroni_batch.py'),'prepare','--predictions',str(prediction),
                '--sources-root',str(root/'source'),'--out',str(root/'bundle'),
                '--device','zcu104','--clock-ns','10','--budget-percent','100'],check=True,capture_output=True)
            manifest=json.loads((root/'bundle'/'manifest.json').read_text())
            self.assertEqual(len(manifest['jobs']),1)
            self.assertEqual(manifest['jobs'][0]['assignment']['AUTO{_UNROLL_L1}'],'2')
            self.assertTrue((root/'bundle'/'SHA256SUMS').exists())
    def test_atomic_writes_no_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'x.json'
            m.write(p,{'i':1});self.assertEqual(m.read(p),{'i':1})
            with self.assertRaises(ValueError):m.write(p,{'NaN':float('nan')})
            self.assertEqual(m.read(p),{'i':1})
if __name__=='__main__':unittest.main()
