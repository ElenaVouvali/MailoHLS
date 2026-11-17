"""Offline contract tests; the mock results are not hardware measurements."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import cg_bdr as cg
import audit_evidence
import select_initial
import networkx as nx

TEMPLATE = '''void toy() {
/*L1:*/ float A[16];
#pragma HLS array_partition variable=A type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}
float acc = 0;
/*L2:*/ for (int j=0; j<16; j++)
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
{
acc += A[j];
}
}
'''
CONFIG = [dict(action_id='L1',kind='array',variable='A',partition_type='block',partition_factor=4,partition_dim=1),
          dict(action_id='L2',kind='loop',pipeline_ii=1,unroll_factor=4)]


class ControllerTests(unittest.TestCase):
    def setUp(self):
        quiet = mock.patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'toy.cpp').write_text(TEMPLATE)
        (self.source/'kernel_info.txt').write_text('toy\nL1,array,A,1,16\nL2,loop,16\n')
        (self.source/'initial.json').write_text(json.dumps(dict(kernel='toy',objective='LATENCY',directives=CONFIG)))
        # Analysis stubs must not leak into synthesis snapshots.
        (self.source/'ap_fixed.h').write_text('ANALYSIS_ONLY_STUB')
        (self.source/'ordinary.h').write_text('// genuine project header\n')
        graph = nx.MultiDiGraph()
        graph.add_node('array',type=100,function=1,text='ARRAY_PARTITION',full_text=TEMPLATE.splitlines()[2])
        graph.add_node('loop',type=100,function=1,text='PIPELINE',full_text=TEMPLATE.splitlines()[5])
        graph.add_node('scope',type=104,function=1,text='array_scope',array_var='A')
        graph.add_node('load',type=0,function=1,text='load',full_text='%A_load = load float, float* %A')
        graph.add_edge('array','scope',flow=200)
        graph.add_edge('scope','load',flow=7)
        nx.write_gexf(graph,self.source/'toy.gexf')
        self.bundle = self.root/'bundle'
        self.prepare()

    def cli(self,*args,success=True):
        p = subprocess.run([sys.executable,str(PACKAGE/'cg_bdr.py'),*map(str,args)],
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
        if success:
            self.assertEqual(p.returncode,0,p.stdout)
        else:
            self.assertNotEqual(p.returncode,0,p.stdout)
        return p

    def prepare(self,**changes):
        opts = dict(kernel='toy',template=self.source/'toy.cpp',source_dir=self.source,
                    kernel_info=self.source/'kernel_info.txt',graph=self.source/'toy.gexf',
                    initial=self.source/'initial.json',bundle=self.bundle,part=cg.PART,
                    clock=10,budget_percent=100,max_evaluations=4,top='',allow_partial_graph=False)
        opts.update(changes)
        return cg.prepare(argparse.Namespace(**opts))

    def evaluate(self,**changes):
        opts = dict(bundle=self.bundle,vitis_hls=PACKAGE/'tests/mock_vitis.py',test_mock=True,
                    limit=1,timeout=10,min_free_gb=0,explore_on_no_diagnostic=False)
        opts.update(changes)
        cg.run(argparse.Namespace(**opts))

    def state(self):
        return cg.jload(self.bundle/'state.json')

    def test_prepare_is_offline_and_excludes_vendor_stubs(self):
        self.assertEqual(self.state()['evaluations'],[])
        self.assertFalse((self.bundle/'projects').exists())
        self.assertFalse((self.bundle/'inputs/ap_fixed.h').exists())
        self.assertTrue((self.bundle/'inputs/ordinary.h').is_file())
        m = cg.verify_bundle(self.bundle)
        self.assertEqual(set(m['compiler_graph']['anchored_actions']),{'L1','L2'})
        self.assertEqual(len(m['loop_array_relations']),1)

    def test_materialization_preserves_lines_and_disabled_directives(self):
        cfg = copy.deepcopy(CONFIG)
        cfg[0].update(partition_type='none',partition_factor=0,partition_dim=0)
        cfg[1].update(pipeline_ii=0,unroll_factor=1)
        text = cg.materialize(TEMPLATE,cfg)
        self.assertEqual(len(text.splitlines()),len(TEMPLATE.splitlines()))
        self.assertNotIn('#pragma',text)
        self.assertIn('L2: for',text)
        self.assertIn('/*L1:*/ float A',text)

    def test_placeholder_braces_are_not_loop_bodies(self):
        actions = cg.verify_bundle(self.bundle)['actions']
        spans = cg.loop_spans(TEMPLATE,actions)
        self.assertEqual(spans['L2']['end'],10)
        unbraced = TEMPLATE.replace('{\nacc += A[j];\n}', 'acc += A[j];')
        self.assertNotIn('L2',cg.loop_spans(unbraced,actions))

    def test_measured_repair_incumbent_preview_and_abstention(self):
        self.evaluate()
        self.assertEqual(len(self.state()['evaluations']),1)
        self.assertEqual(self.state()['evaluations'][0]['result']['metrics']['latency_cycles'],1000)
        before = {str(p.relative_to(self.bundle)):cg.digest(p) for p in self.bundle.rglob('*') if p.is_file()}
        proposals = cg.preview(self.bundle)
        self.assertEqual(proposals[0]['kind'],'banking_style')
        self.assertEqual(set(proposals[0]['changed']),{'L1'})
        self.assertEqual(before,{str(p.relative_to(self.bundle)):cg.digest(p) for p in self.bundle.rglob('*') if p.is_file()})
        self.evaluate()
        summary = cg.jload(self.bundle/'summary.json')
        self.assertEqual(summary['best']['best_latency_cycles'],800)
        self.assertEqual(summary['charged_evaluations'],2)
        self.evaluate()
        self.assertEqual(len(self.state()['evaluations']),2)
        self.assertEqual(self.state()['stop'],'No evidence-supported untried repair')

    def test_combined_banking_and_factor_repair(self):
        self.evaluate()
        m = cg.verify_bundle(self.bundle)
        m['_source_template_lines'] = TEMPLATE.splitlines()
        cfg = copy.deepcopy(CONFIG)
        cfg[1]['unroll_factor'] = 8
        result = self.state()['evaluations'][0]['result']
        proposals = cg.generate_repair(m,cfg,result,{cg.signature(cfg)})
        self.assertEqual(proposals[0]['kind'],'banking_style_and_factor')
        self.assertEqual(proposals[0]['directives'][0]['partition_factor'],8)

    def test_vitis_generated_array_name_requires_exact_source_access(self):
        self.evaluate()
        m = cg.verify_bundle(self.bundle)
        relation = m['loop_array_relations'][0]
        line = relation['accesses'][0]['line']
        d = dict(array='A_V_0_2',source_line=line)
        self.assertIsNotNone(cg.vitis_array_matches_source(d,relation,TEMPLATE.splitlines()))
        d['array'] = 'AnotherArray_V_0'
        self.assertIsNone(cg.vitis_array_matches_source(d,relation,TEMPLATE.splitlines()))
        d.update(array='A_V_0',source_line=1)
        self.assertIsNone(cg.vitis_array_matches_source(d,relation,TEMPLATE.splitlines()))

    def test_graph_links_required_for_array_repair(self):
        m = cg.verify_bundle(self.bundle)
        graph = copy.deepcopy(m['compiler_graph'])
        graph['array_scope_access_links'] = {}
        self.assertEqual(cg.direct_accesses(TEMPLATE,m['actions'],m['loop_spans'],graph),[])

    def test_pending_reservation_blocks_free_retry(self):
        cg.jdump(self.bundle/'pending.json',dict(index=1))
        with self.assertRaisesRegex(RuntimeError,'Pending reserved synthesis'):
            self.evaluate()
        self.assertEqual(self.state()['evaluations'],[])

    def test_tampered_snapshot_is_rejected(self):
        (self.bundle/'inputs/template.cpp').write_text(TEMPLATE+'// modified\n')
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
            self.evaluate()
        self.assertFalse((self.bundle/'pending.json').exists())

    def test_one_evaluation_cap_includes_baseline(self):
        capped = self.root/'capped'
        self.prepare(bundle=capped,max_evaluations=1)
        self.evaluate(bundle=capped)
        with self.assertRaisesRegex(RuntimeError,'budget already exhausted'):
            self.evaluate(bundle=capped)
        self.assertEqual(len(cg.jload(capped/'state.json')['evaluations']),1)

    def test_unsuccessful_tool_call_consumes_budget(self):
        failed = self.root/'failed_vitis.py'
        failed.write_text('#!/usr/bin/env python3\nraise SystemExit(1)\n')
        failed.chmod(0o755)
        self.evaluate(vitis_hls=failed)
        self.assertEqual(len(self.state()['evaluations']),1)
        self.assertEqual(self.state()['evaluations'][0]['result']['synthesis_status'],'failed')
        self.evaluate(vitis_hls=failed)
        self.assertEqual(len(self.state()['evaluations']),1)

    def test_part_and_initial_context_are_validated(self):
        args = ['prepare','--kernel','toy','--source-dir',self.source,'--template',self.source/'toy.cpp',
                '--kernel-info',self.source/'kernel_info.txt','--graph',self.source/'toy.gexf',
                '--initial',self.source/'initial.json','--bundle',self.root/'u200','--part','xcu200-fsgd2104-2-e']
        p = self.cli(*args,success=False)
        self.assertIn('only supports the ZCU104',p.stdout)
        initial = dict(kernel='toy',device='xcu200-fsgd2104-2-e',directives=CONFIG)
        (self.source/'initial.json').write_text(json.dumps(initial))
        with self.assertRaisesRegex(RuntimeError,'device conflicts'):
            self.prepare(bundle=self.root/'wrong_context')


class NativeEvidenceTests(unittest.TestCase):
    def test_native_candidate_trace_replays_the_tested_repair_policy(self):
        for bundle in (PACKAGE/'evidence').glob('*_4eval'):
            m = cg.jload(bundle/'manifest.json')
            for action in m['actions'].values():
                if action['kind']=='array':
                    action['dims'] = {int(k):v for k,v in action['dims'].items()}
            m['_source_template_lines'] = (bundle/'inputs/template.cpp').read_text().splitlines()
            history = cg.jload(bundle/'state.json')['evaluations']
            for index, candidate in enumerate(history[1:],1):
                earlier = history[:index]
                good = [r for r in earlier if cg.eligible(r['result']['metrics'])]
                incumbent = min(good,key=lambda r:r['result']['metrics']['latency_cycles'])
                proposals = cg.generate_repair(m,incumbent['directives'],incumbent['result'],
                                               {r['signature'] for r in earlier})
                self.assertTrue(proposals,bundle.name)
                self.assertEqual(proposals[0]['signature'],candidate['signature'],bundle.name)
                self.assertEqual(proposals[0]['kind'],candidate['selection']['kind'],bundle.name)

    def test_all_eleven_native_evaluations_verify_without_hls(self):
        with mock.patch.object(cg,'synthesize',side_effect=AssertionError('Audit must not invoke HLS')):
            result = audit_evidence.audit()
        self.assertEqual(result['verified_evaluations'],11)
        self.assertEqual(result['new_hls_calls'],0)
        gan = next(r for r in result['kernels'] if r['kernel']=='GAN')
        self.assertEqual((gan['initial_worst_cycles'],gan['worst_case_cycles']),(3133,1389))

    def test_native_parser_uses_worst_bound_and_checks_target(self):
        bundle = PACKAGE/'evidence/covariance_LATENCY_10ns_4eval'
        m = cg.jload(bundle/'manifest.json')
        report = bundle/'projects/eval-01/hls_project/solution1/syn/report/kernel_covariance_csynth.xml'
        metrics = cg.read_report(report,m)
        self.assertEqual(metrics['latency_cycles'],375643)
        wrong = dict(m,clock_ns=6.67)
        with self.assertRaisesRegex(RuntimeError,'context mismatch'):
            cg.read_report(report,wrong)
        self.assertFalse(cg.read_report(report,dict(m,budget_percent=1))['budget_feasible'])

    def test_prediction_selector_matches_context_and_rejects_duplicates(self):
        path = PACKAGE/'evidence_plot_rows.jsonl'
        row = select_initial.select(path,'GAN','LATENCY',cg.PART,10,100)
        self.assertEqual(row['latency'],3133)
        with self.assertRaises(ValueError):
            select_initial.select(path,'GAN','LATENCY',cg.PART,6.67,100)
        with tempfile.TemporaryDirectory() as temp:
            duplicate = Path(temp)/'duplicate.jsonl'
            duplicate.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError):
                select_initial.select(duplicate,'GAN','LATENCY',cg.PART,10,100)


if __name__ == '__main__':
    unittest.main()
