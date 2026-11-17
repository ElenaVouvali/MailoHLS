"""Offline native replay and mocked synthesis governance; zero real HLS calls."""
import argparse
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import cg_bdr as cg
import runner_v3 as runner
import semantic_v3 as sem
import rank_v3 as rank


def ns(bundle, **overrides):
    values = dict(bundle=bundle, no_hls=False, approval=None, enable_one_scout=False,
                  vitis_hls=HERE/'tests/mock_vitis.py', timeout=10, min_free_gb=0, test_mock=True)
    values.update(overrides)
    return argparse.Namespace(**values)


class NativeReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        with mock.patch.object(cg,'synthesize',side_effect=AssertionError('No HLS in preparation')):
            runner.prepare(argparse.Namespace(runs_root=cls.root,kernels=runner.NAMES,
                           policy='graph',reuse_verified_baselines=True))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        quiet = mock.patch.object(cg,'emit')
        quiet.start()
        self.addCleanup(quiet.stop)

    def b(self,kernel):
        return runner.bundle(self.root,kernel)

    def clone(self,kernel):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        dest = Path(temp.name)/'bundle'
        shutil.copytree(str(self.b(kernel)),str(dest))
        return dest

    def test_native_grouping_and_jacobi_ancestors(self):
        p = cg.preview(self.b('jacobi-2d'))
        for loop in ('L3','L5'):
            self.assertEqual(p['schedule_groups'][loop]['count'],32)
            self.assertEqual(p['schedule_groups'][loop]['achieved_ii']['values'],[3]*32)
        self.assertLess(len(p['bottleneck_families']),10)
        self.assertGreater(sum(d['occurrence_count'] for d in p['bottleneck_families']),len(p['bottleneck_families']))
        self.assertEqual({tuple(c['changed_actions']) for c in p['candidates']},{('L2',),('L4',)})
        self.assertTrue(all(c['family']=='ancestor_concurrency' for c in p['candidates']))
        self.assertEqual([r['unroll_factor'] for r in cg.jload(self.b('jacobi-2d')/'manifest.json')['initial_directives']],[0,32,0,32,0])
        self.assertTrue(all(c['evidence']['source_access']['owning_loop'] in ('L3','L5') for c in p['candidates']))

    def test_all_schedule_variants_trmm_and_tradeoffs(self):
        p = cg.preview(self.b('trmm-opt'))
        g = p['schedule_groups']['L2']
        self.assertEqual(g['count'],10)
        self.assertEqual((g['achieved_ii']['min'],g['achieved_ii']['max'],g['achieved_ii']['median']),(52,61,56.5))
        self.assertIsNone(g['exact_final_ii'])
        self.assertEqual(p['candidates'],[])
        self.assertTrue(any('parallelism' in r['reason'] for r in p['rejections']))
        s = cg.preview(self.b('syr2k'))
        self.assertEqual(s['candidates'],[])
        self.assertTrue(any('optimistic' in r['reason'] for r in s['rejections']))

    def test_gan_general_banking_rule_and_kalman_abstention(self):
        p = cg.preview(self.b('GAN'))
        c = p['candidates'][0]
        self.assertEqual(c['family'],'banking')
        self.assertEqual(c['changed']['L2']['new']['partition_type'],'cyclic')
        self.assertGreaterEqual(c['changed']['L2']['new']['partition_factor'],10)
        self.assertEqual(c['evidence']['documented_stride'],1)
        self.assertEqual(cg.preview(self.b('Kalman'))['candidates'],[])
        self.assertNotIn('DEPENDENCE false',(HERE/'rank_v3.py').read_text())
        # Rename kernel metadata: proposals must not contain benchmark special cases.
        m = cg.verify_bundle(self.b('GAN'))
        m['kernel'] = 'unseen-fully-connected'
        hist = cg.jload(self.b('GAN')/'state.json')['evaluations']
        proposals,_ = rank.propose(cg.base,m,hist[0],hist)
        self.assertEqual(proposals[0]['signature'],c['signature'])

    def test_readonly_preview_summary_and_no_hls_pilot(self):
        before = {str(p.relative_to(self.root)):cg.digest(p) for p in self.root.rglob('*') if p.is_file()}
        with mock.patch('subprocess.run',side_effect=AssertionError('No process in dry-run')):
            for k in runner.NAMES:
                cg.preview(self.b(k),True)
                cg.summary(self.b(k))
            cg.run(ns(self.b('GAN'),no_hls=True))
            runner.pilot(argparse.Namespace(runs_root=self.root,kernels=runner.NAMES,mode='strict',no_hls=True))
        self.assertEqual(before,{str(p.relative_to(self.root)):cg.digest(p) for p in self.root.rglob('*') if p.is_file()})

    def test_import_charges_one_zero_new_calls_and_source_matching(self):
        for kernel in runner.NAMES:
            s = cg.summary(self.b(kernel))
            self.assertEqual((s['total_evaluations_charged'],s['actual_new_vitis_calls'],s['reused_verified_evaluations']),(1,0,1))
            hist = cg.jload(self.b(kernel)/'state.json')['evaluations']
            self.assertTrue(hist[0]['reuse']['source_matching'])
            self.assertTrue(hist[0]['reuse']['commands_matching'])
            self.assertEqual([r['best']['signature'] for r in s['incumbent_after_budget']],[hist[0]['signature']]*4)

    def test_import_mismatches_and_old_schema_rejected(self):
        b = self.clone('GAN')
        with self.assertRaisesRegex(RuntimeError,'fresh empty'):
            cg.import_baseline(b,runner.archive('GAN'))
        m = cg.jload(b/'manifest.json')
        m['schema'] = 'cg-bdr-v2-research'
        cg.jdump(b/'manifest.json',m)
        with self.assertRaisesRegex(RuntimeError,'Not a v3'):
            cg.verify_bundle(b)
        b = self.clone('GAN')
        (b/'inputs/template.cpp').write_text('// tampered\n')
        with self.assertRaisesRegex(RuntimeError,'Input hash'):
            cg.verify_bundle(b)
        b = self.clone('GAN')
        cg.jdump(b/'state.json',dict(evaluations=[],stop_reason=None))
        shutil.rmtree(str(b/'projects'))
        with self.assertRaisesRegex(RuntimeError,'context mismatch|hashes mismatch'):
            cg.import_baseline(b,runner.archive('Kalman'))

    def test_import_rejects_command_mismatch(self):
        b = self.clone('GAN')
        cg.jdump(b/'state.json',dict(evaluations=[],stop_reason=None))
        shutil.rmtree(str(b/'projects'))
        with mock.patch.object(cg,'expected_tcl',return_value='extra command\n'):
            with self.assertRaisesRegex(RuntimeError,'commands/tool flags mismatch'):
                cg.import_baseline(b,runner.archive('GAN'))
        self.assertEqual(cg.summary(b)['total_evaluations_charged'],0)

    def test_best_case_only_even_larger_worst_case_and_unknown_bounds(self):
        h = cg.jload(self.b('GAN')/'state.json')['evaluations']
        better = copy.deepcopy(h[0])
        better['result']['index'] = 2
        better['result']['metrics'].update(latency_best_cycles=2000,latency_worst_cycles=999999)
        self.assertIs(cg.best_row(h+[better]),better)
        for key,value in [('latency_best_cycles',None),('estimated_clock_ns',None),('estimated_clock_ns',float('nan'))]:
            x = copy.deepcopy(better)
            x['result']['metrics'][key] = value
            self.assertFalse(cg.eligible(x['result']['metrics']))
        m = cg.verify_bundle(self.b('GAN'))
        report = self.b('GAN')/'projects/eval-01/hls_project/solution1/syn/report/forward_propagation_csynth.xml'
        tree = ET.parse(str(report))
        tree.find('.//Best-caseLatency').text = '?'
        dest = self.root/'invalid.xml'
        tree.write(str(dest))
        self.addCleanup(dest.unlink)
        with self.assertRaisesRegex(RuntimeError,'best-case latency'):
            cg.read_report(dest,m)

    def test_approval_lock_pending_and_interrupted_charge(self):
        b = self.clone('jacobi-2d')
        with mock.patch('builtins.input',return_value='yes'),mock.patch.object(cg,'synthesize') as synth:
            with self.assertRaisesRegex(RuntimeError,'typed candidate approval'):
                cg.run(ns(b))
            synth.assert_not_called()
        p = cg.preview(b)
        token = cg.approval_token(b,p)
        with cg.writer(b):
            with self.assertRaisesRegex(RuntimeError,'One writer'):
                cg.run(ns(b,approval=token))
        with mock.patch.object(cg,'synthesize',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                cg.run(ns(b,approval=token))
        self.assertTrue((b/'pending.json').exists())
        self.assertEqual(cg.summary(b)['total_evaluations_charged'],2)
        self.assertIn('Pending',cg.preview(b)['stop_reason'])
        with mock.patch.object(cg,'synthesize',side_effect=AssertionError('Must not retry')):
            cg.run(ns(b,approval=token))

    def test_mock_synthesis_failure_charged_and_no_native_improvement(self):
        b = self.clone('GAN')
        def fixture(bundle,m,cfg,index,*unused):
            work = bundle/'projects'/('eval-%02d'%index)
            work.mkdir()
            (work/'candidate.cpp').write_text(cg.base.materialize((bundle/'inputs/template.cpp').read_text(),cfg))
            (work/'run_hls.tcl').write_text('synthetic offline fixture')
            (work/'synthesis.log').write_text('SYNTHETIC TEST FAILURE')
            return dict(index=index,work_dir=str(work.relative_to(bundle)),metrics=None,error='synthetic failure',
                        report_sha256=None,synthesis_status='failed',tool_mode='SYNTHETIC_TEST_ONLY')
        token = cg.approval_token(b,cg.preview(b))
        with mock.patch.object(cg,'synthesize',side_effect=fixture):
            cg.run(ns(b,approval=token))
        s = cg.summary(b)
        self.assertEqual((s['total_evaluations_charged'],s['actual_new_vitis_calls'],s['synthesis_failures']),(2,0,1))
        self.assertEqual(s['selected']['evaluation'],1)
        self.assertFalse((b/'pending.json').exists())
        used = cg.jload(b/'state.json')['evaluations'][-1]['signature']
        self.assertNotIn(used,[c['signature'] for c in cg.preview(b)['candidates']])

    def test_synthetic_process_boundary_offline(self):
        b = self.clone('GAN')
        cg.run(ns(b,approval=cg.approval_token(b,cg.preview(b))))
        s = cg.summary(b)
        self.assertEqual((s['total_evaluations_charged'],s['actual_new_vitis_calls'],s['synthesis_failures']),(2,0,1))
        r = cg.jload(b/'state.json')['evaluations'][-1]['result']
        self.assertTrue(r['invocation_started'])
        self.assertEqual(r['tool_mode'],'SYNTHETIC_TEST_ONLY')
        self.assertFalse((b/'pending.json').exists())

    def test_equivalent_family_penalty_and_neutral_fingerprint(self):
        b = self.b('GAN')
        m = cg.verify_bundle(b)
        hist = cg.jload(b/'state.json')['evaluations']
        p,_ = rank.propose(cg.base,m,hist[0],hist)
        unsuccessful = copy.deepcopy(hist[0])
        unsuccessful['selection'] = p[0]
        unsuccessful['signature'] = 'another-tried-signature'
        proposals,rejected = rank.propose(cg.base,m,hist[0],hist+[unsuccessful])
        self.assertEqual(proposals,[])
        self.assertTrue(any('Unsuccessful local family' in r['reason'] for r in rejected))
        self.assertEqual(rank.fingerprint(hist[0]['result']),rank.fingerprint(unsuccessful['result']))

    def test_feedback_only_has_no_graph_filter(self):
        old = runner.archive('jacobi-2d')
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        with mock.patch.object(cg.base,'graph_index',side_effect=AssertionError('Ablation cannot consult topology')), \
             mock.patch.object(cg.base,'direct_accesses',side_effect=AssertionError('No graph relation filter')):
            runner.prepare(argparse.Namespace(runs_root=root,kernels=['jacobi-2d'],policy='feedback-only',reuse_verified_baselines=True))
            p = cg.preview(runner.bundle(root,'jacobi-2d'))
        self.assertEqual({tuple(c['changed_actions']) for c in p['candidates']},{('L2',),('L4',)})
        self.assertNotIn('LLVM',json.dumps(p['candidates']))

    def test_scout_optin_one_only_and_cap(self):
        m = cg.verify_bundle(self.b('trmm-opt'))
        hist = cg.jload(self.b('trmm-opt')/'state.json')['evaluations']
        self.assertEqual(rank.propose(cg.base,m,hist[0],hist)[0],[])
        scout = rank.propose(cg.base,m,hist[0],hist,True)[0]
        self.assertEqual(len(scout),1)
        self.assertTrue(scout[0]['exploratory'])
        prior = copy.deepcopy(hist[0])
        prior['selection'] = scout[0]
        self.assertEqual(rank.propose(cg.base,m,hist[0],hist+[prior],True)[0],[])
        b = self.clone('GAN')
        # An irrevocable fourth-slot reservation blocks calls regardless of selection.
        state = cg.jload(b/'state.json')
        for i in (2,3):
            r = copy.deepcopy(state['evaluations'][0])
            r['signature'] = str(i)
            state['evaluations'].append(r)
        # Counters are independent of receipt validation and conservatively count pending.
        cg.jdump(b/'pending.json',dict(index=4,invocation_started=True))
        self.assertEqual(cg.counters(b,state)['total_evaluations_charged'],4)

    def test_source_mismatch_in_valid_new_bundle(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        src = root/'src'
        shutil.copytree(str(self.b('GAN')/'inputs'),str(src))
        (src/'template.cpp').write_text((src/'template.cpp').read_text()+'// different frozen source\n')
        initial = self.b('GAN')/'initial_prediction.json'
        dest = root/'new'
        cg.prepare(argparse.Namespace(bundle=dest,kernel='GAN',source_dir=src,template=src/'template.cpp',
                   kernel_info=src/'kernel_info.txt',graph=src/'llvm_graph.gexf',initial=initial,policy='graph'))
        with self.assertRaisesRegex(RuntimeError,'source/graph/header hashes mismatch'):
            cg.import_baseline(dest,runner.archive('GAN'))
        self.assertEqual(cg.summary(dest)['total_evaluations_charged'],0)

    def test_four_reservations_enforced_with_mocked_failures(self):
        b = self.clone('GAN')
        def fixture(bundle,m,cfg,index,*unused):
            self.assertEqual(cg.jload(bundle/'pending.json')['index'],index)
            work = bundle/'projects'/('eval-%02d'%index)
            work.mkdir()
            (work/'candidate.cpp').write_text(cg.base.materialize((bundle/'inputs/template.cpp').read_text(),cfg))
            (work/'run_hls.tcl').write_text('SYNTHETIC')
            (work/'synthesis.log').write_text('SYNTHETIC')
            return dict(index=index,work_dir=str(work.relative_to(bundle)),metrics=None,error='synthetic',
                        report_sha256=None,synthesis_status='failed',tool_mode='SYNTHETIC_TEST_ONLY')
        def proposals(base,m,incumbent,history,enable_scout):
            cfg = copy.deepcopy(m['initial_directives'])
            cfg[1]['partition_factor'] = [6,8,10][len(history)-1]
            return [dict(directives=cfg,signature=cg.signature(cfg),family='mocked_contract',changed_actions=['L2'])],[]
        with mock.patch.object(cg,'synthesize',side_effect=fixture) as synth, mock.patch.object(rank,'propose',side_effect=proposals):
            for index in (2,3,4):
                p = cg.preview(b)
                cg.run(ns(b,approval=cg.approval_token(b,p)))
                self.assertEqual(cg.summary(b)['total_evaluations_charged'],index)
            self.assertIn('exhausted',cg.preview(b)['stop_reason'])
            cg.run(ns(b,approval='yes'))
            self.assertEqual(synth.call_count,3)
        self.assertEqual(cg.summary(b)['actual_new_vitis_calls'],0)

    def test_runner_fresh_approval_per_call(self):
        previews = {k:cg.preview(self.b(k)) for k in ('GAN','jacobi-2d')}
        def response(argv):
            if argv[2]=='preview':
                return json.dumps(previews['GAN' if 'GAN_' in argv[4] else 'jacobi-2d'])
            return 'V3_RESULT\n{}\n'
        with mock.patch.object(runner,'transfer',side_effect=lambda root,k: ('controller',str(self.b(k)))), \
             mock.patch.object(runner,'remote',side_effect=response) as ssh, \
             mock.patch('builtins.input',side_effect=[cg.approval_token(self.b('GAN'),previews['GAN']),'yes']):
            runner.pilot(argparse.Namespace(runs_root=self.root,kernels=['GAN','jacobi-2d'],mode='strict',no_hls=False))
        self.assertEqual(sum(call[0][0][2]=='run' for call in ssh.call_args_list),1)

    def test_cross_version_archives_and_original_paths_unchanged(self):
        result = runner.audit()
        self.assertEqual(result['archived']['legacy']['verified_evaluations'],11)
        self.assertEqual(result['archived']['v2']['verified_evaluations'],9)
        changed = subprocess.check_output(['git','diff','--name-only','e0cf0f722e2e45ce78ca7fc2770f37e9ee08f489'],
                                         cwd=str(HERE.parents[1]),universal_newlines=True).splitlines()
        self.assertTrue(all(path.startswith('refinement/v3/') for path in changed))

    def test_warning_multiplicity_and_nonlabel_schedule_names(self):
        b = self.b('GAN')
        hist = cg.jload(b/'state.json')['evaluations']
        d = hist[0]['result']['diagnostics'][1]
        group = sem.warning_groups([d]*500)
        self.assertEqual(len(group),1)
        self.assertEqual(group[0]['occurrence_count'],500)
        m = cg.verify_bundle(b)
        repeated = copy.deepcopy(hist[0])
        repeated['result']['bottleneck_families'] = sem.warning_groups(hist[0]['result']['diagnostics']*50)
        x = rank.propose(cg.base,m,hist[0],hist)[0]
        y = rank.propose(cg.base,m,repeated,hist)[0]
        self.assertEqual([(c['signature'],c['rank_key']) for c in x],[(c['signature'],c['rank_key']) for c in y])
        row = dict(loop='L3_generated',target_ii=1,achieved_ii=5,pipeline_depth=2)
        self.assertIsNone(sem.schedule_groups([row])['L3_generated']['source_loop'])


if __name__=='__main__':
    unittest.main()
