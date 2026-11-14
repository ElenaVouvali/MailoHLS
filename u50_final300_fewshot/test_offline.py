"""CPU-only tests for calibration shot accounting, target isolation and HLS bundling."""
import importlib.util,json,hashlib,tempfile,unittest
from pathlib import Path
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('u50_fewshot',P/'u50_fewshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('coroni_batch',P/'coroni_batch.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class FewShotTest(unittest.TestCase):
    def test_canonical_family(self):
        self.assertEqual(m.family('rodinia_pathfinder_4_doublebuffer_0'),'rodinia_pathfinder')
        self.assertEqual(m.family('machsuite-gemm-blocked'),'machsuite_gemm')
        self.assertEqual(m.family('rodinia_cfd_step_factor_4_doublebuffer_0'),'rodinia_cfd_step_factor')

    def test_measured_shot_accounting(self):
        bank={f'rodinia-test{i}-1':[
            {'target':f'auto{{_UNROLL_L1}} = {j}','latency':float(j+1)}
            for j in range(2)] for i in range(11)}
        # Family names of synthetic names differ, so each context is independent.
        pairs=m.sample_pairs(bank,8,123)
        self.assertEqual(len(pairs),4)
        self.assertEqual(sum(q['measurements_charged'] for q in pairs),8)
        self.assertEqual({q['chosen']['latency'] for q in pairs},{1.})
        self.assertEqual(m.sample_pairs(bank,8,123),pairs)
        self.assertNotEqual(m.sample_pairs(bank,8,456)[0]['kernel'],pairs[0]['kernel'])

    def test_no_false_preferences_from_ties(self):
        # Insufficient information about preference must not create a positive
        # ranking label; fall through to another independent family instead.
        bank={f'machsuite-fam{i}':[
            {'target':'auto{_UNROLL_L1} = 0','latency':10.0},
            {'target':'auto{_UNROLL_L1} = 2','latency':10.0 if i==0 else 12.0},
        ] for i in range(6)}
        pairs=m.sample_pairs(bank,8,123)
        self.assertEqual(len(pairs),4)
        self.assertTrue(all(p['relative_latency_gap'] >= .005 for p in pairs))
        self.assertTrue(all(p['chosen']['latency'] < p['rejected']['latency'] for p in pairs))
        self.assertEqual(m.sample_pairs(bank,8,123),pairs)

    def test_near_equal_latency_is_not_pretended_to_be_hard_negative(self):
        bank={f'rodinia-fam{i}':[
            {'target':'auto{_UNROLL_L1} = 0','latency':10.0},
            {'target':'auto{_UNROLL_L1} = 2','latency':10.001},
        ] for i in range(6)}
        with self.assertRaisesRegex(ValueError,'qualified measured support designs'):
            m.sample_pairs(bank,8,123)

    def test_no_duplicate_measurement(self):
        bank={'x':[{'target':'a','latency':1.} for _ in range(2)]}
        with self.assertRaises(ValueError):m.sample_pairs(bank,4,123)

    def test_u50_batch_can_prepare_without_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=root/'sources';d=sources/'GAN';d.mkdir(parents=True)
            (d/'kernel_info.txt').write_text('kernel\nL1,loop,16\n')
            c=d/'source.c';c.write_text('void kernel(int *a) {\nint i;\nL1: for (i=0;i<16;i++) {a[i]++;}\n}\n')
            row={'kernel':'GAN','device':m.U50,'clock_period_ns':10.,'budget_percent':100.,
                'source_sha256':b.sha(c),'checkpoint_step':300,'checkpoint_sha256':'a'*64,
                'status':'inference_only_not_synthesized',
                'directives':[{'action_id':'L1','kind':'loop','pipeline_ii':1,'unroll_factor':0}]}
            inp=root/'prediction.jsonl';inp.write_text(json.dumps(row)+'\n')
            from argparse import Namespace
            args=Namespace(out=str(root/'batch'),predictions=str(inp),sources_root=str(sources),
                targets_json=None,device='u50',clock_ns=10.,budget_percent=100.,require_nine=False)
            b.prepare(args)
            manifest=b.verify_bundle(root/'batch')
            self.assertEqual(len(manifest['jobs']),1)
            self.assertEqual(manifest['jobs'][0]['part'],m.U50)
            self.assertEqual(manifest['jobs'][0]['capacities'],m.CAPS)

if __name__=='__main__':unittest.main()
