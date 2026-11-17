"""Offline v2 contracts and native evidence replay; no Vitis invocations."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(V2))
spec = importlib.util.spec_from_file_location('v2_evidence_audit',V2/'audit_evidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
cg = audit.cg
sem = cg.semantic_v2
import networkx as nx

TEMPLATE = '''void toy() {
/*L1:*/ float A[16];
#pragma HLS array_partition variable=A type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}
float acc = 0;
/*L2:*/ for (int j=0; j<16; j++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
acc += A[j];
}
}
'''
CONFIG = [dict(action_id='L1',kind='array',variable='A',partition_type='block',partition_factor=4,partition_dim=1),
          dict(action_id='L2',kind='loop',pipeline_ii=1,unroll_factor=4)]


class V2Contracts(unittest.TestCase):
    def test_loop_only_memory_repair_requires_measured_ii_stall(self):
        actions = {'L1':dict(kind='loop',bound=16)}
        config = [dict(action_id='L1',kind='loop',pipeline_ii=1,unroll_factor=4)]
        m = dict(actions=actions,compiler_graph=dict(anchored_actions=['L1']),semantic_v2={},loop_array_relations=[])
        diagnostic = dict(type='memory_ports',scheduled_loop='L1',reported_loop='L1',affected_loops=['L1'],raw='memory port warning')
        result = dict(metrics=dict(budget_feasible=True,utilization_percent={'lut':10}),diagnostics=[diagnostic],schedule_results=[])
        self.assertEqual(cg.generate_repair(m,config,result,set()),[])
        result['schedule_results'] = [dict(loop='L1',target_ii=1,achieved_ii=1,pipeline_depth=2)]
        self.assertEqual(cg.generate_repair(m,config,result,set()),[])
        result['schedule_results'][0]['achieved_ii'] = 4
        proposal = cg.generate_repair(m,config,result,set())[0]
        self.assertEqual(proposal['kind'],'loop_only_port_relief')
        self.assertLess(proposal['directives'][0]['unroll_factor'],4)

    def hierarchy(self,update='A[j] = B[j];'):
        template = '''void nested() {
/*L1:*/ for (int i=0; i<16; i++) {
/*L2:*/ for (int j=0; j<16; j++) {
'''+update+'\n}\n}\n}\n'
        actions = {'L1':dict(kind='loop',bound=16),'L2':dict(kind='loop',bound=16)}
        graph = dict(anchored_actions=['L1','L2'])
        spans = cg.loop_spans(template,actions)
        semantic = sem.analyze(template,actions,spans,graph)
        return dict(actions=actions,compiler_graph=graph,semantic_v2=semantic,loop_array_relations=[])

    def recurrence(self):
        return dict(metrics=dict(budget_feasible=True,utilization_percent={'lut':10}),
                    diagnostics=[dict(type='recurrence',scheduled_loop='L1',reported_loop='L1',raw='reported recurrence')],
                    schedule_results=[dict(loop='L1',target_ii=1,achieved_ii=4,pipeline_depth=8)])

    def test_parent_child_pipeline_relocation_has_explicit_scope_evidence(self):
        m = self.hierarchy()
        cfg = [dict(action_id='L1',kind='loop',pipeline_ii=1,unroll_factor=0),
               dict(action_id='L2',kind='loop',pipeline_ii=0,unroll_factor=0)]
        proposal = cg.generate_repair(m,cfg,self.recurrence(),set())[0]
        self.assertEqual(proposal['kind'],'parent_child_pipeline_relocation')
        self.assertEqual(proposal['changed_actions'],['L1','L2'])
        self.assertEqual([r['pipeline_ii'] for r in proposal['directives']],[0,1])
        self.assertFalse(proposal['evidence']['child_dependence_proven'])
        m['compiler_graph']['anchored_actions'] = ['L1']
        self.assertEqual(cg.generate_repair(m,cfg,self.recurrence(),set()),[])

    def test_possible_child_accumulation_blocks_pipeline_relocation(self):
        m = self.hierarchy('A[j] += B[j];')
        cfg = [dict(action_id='L1',kind='loop',pipeline_ii=1,unroll_factor=0),
               dict(action_id='L2',kind='loop',pipeline_ii=0,unroll_factor=0)]
        self.assertEqual(cg.generate_repair(m,cfg,self.recurrence(),set()),[])

    def test_semantics_separates_child_operations_from_parent(self):
        semantic = self.hierarchy()['semantic_v2']
        self.assertEqual(semantic['loop_parent']['L2'],'L1')
        self.assertEqual(semantic['loop_children']['L1'],['L2'])
        self.assertFalse(any(row['accesses'] for row in semantic['direct_statements']['L1']))
        self.assertTrue(any(row['accesses'] for row in semantic['direct_statements']['L2']))
        self.assertEqual(semantic['loop_edges'][0]['llvm_loopinfo_correspondence'],'unverified')

    def test_source_accumulation_is_never_dependence_proof(self):
        m = self.hierarchy('A[j] = A[j] + B[j];')
        hint = sem.scope_independent_hint(m['semantic_v2'],'L2')
        self.assertTrue(hint['possible_accumulations'])
        self.assertFalse(hint['independence_proven'])

    def test_unknown_clock_is_ineligible(self):
        metrics = dict(budget_feasible=True,hls_estimated_clock_within_target=None)
        self.assertFalse(cg.eligible(metrics))
        metrics['hls_estimated_clock_within_target'] = True
        self.assertTrue(cg.eligible(metrics))

    def test_one_scout_is_opt_in_and_never_repeated(self):
        m = dict(actions={'L1':dict(kind='loop',bound=16)},compiler_graph=dict(anchored_actions=['L1']),
                 semantic_v2={},loop_array_relations=[])
        cfg = [dict(action_id='L1',kind='loop',pipeline_ii=1,unroll_factor=2)]
        result = dict(metrics=dict(budget_feasible=True,utilization_percent={'lut':10},
                                  loop_summary=[dict(name='L1',latency=100)]),diagnostics=[])
        self.assertEqual(cg.generate_repair(m,cfg,result,set()),[])
        proposals = cg.generate_repair(m,cfg,result,set(),explore=True,history=[dict(selection={})])
        self.assertEqual(len(proposals),1)
        self.assertTrue(proposals[0]['exploratory'])
        self.assertEqual(cg.generate_repair(m,cfg,result,set(),explore=True,
                                          history=[dict(selection=dict(exploratory=True))]),[])


class V2BundleTests(unittest.TestCase):
    def setUp(self):
        quiet = mock.patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root/'toy.cpp').write_text(TEMPLATE)
        (self.root/'kernel_info.txt').write_text('toy\nL1,array,A,1,16\nL2,loop,16\n')
        (self.root/'initial.json').write_text(json.dumps(dict(kernel='toy',objective='LATENCY',directives=CONFIG)))
        graph = nx.MultiDiGraph()
        graph.add_node('array',type=100,function=1,text='ARRAY_PARTITION',full_text=TEMPLATE.splitlines()[2])
        graph.add_node('loop',type=100,function=1,text='PIPELINE',full_text=TEMPLATE.splitlines()[5])
        graph.add_node('scope',type=104,function=1,text='array_scope',array_var='A')
        graph.add_node('load',type=0,function=1,text='load',full_text='%A_load = load float, float* %A')
        graph.add_edge('array','scope',flow=200)
        graph.add_edge('scope','load',flow=7)
        nx.write_gexf(graph,self.root/'toy.gexf')
        self.bundle = self.root/'bundle'
        args = dict(kernel='toy',source_dir=self.root,template=self.root/'toy.cpp',kernel_info=self.root/'kernel_info.txt',
                    graph=self.root/'toy.gexf',initial=self.root/'initial.json',bundle=self.bundle,part=cg.PART,
                    clock=10,budget_percent=100,max_evaluations=4,top='',allow_partial_graph=False)
        cg.prepare(argparse.Namespace(**args))

    def evaluate(self):
        cg.run(argparse.Namespace(bundle=self.bundle,vitis_hls=V2/'tests/mock_vitis.py',test_mock=True,
                                 limit=1,timeout=10,min_free_gb=0,enable_one_scout=False))

    def test_new_bundle_seals_both_v2_modules_and_semantics(self):
        m = cg.verify_bundle(self.bundle)
        self.assertEqual(m['schema'],'cg-bdr-v2-research')
        self.assertEqual(m['semantics_sha256'],audit.SEMANTICS)
        self.assertEqual(m['objective_metric'],'best_case_latency_cycles')
        self.assertIn('direct_statements',m['semantic_v2'])

    def test_semantics_hash_tampering_is_rejected(self):
        m = cg.jload(self.bundle/'manifest.json')
        m['semantics_sha256'] = '0'*64
        cg.jdump(self.bundle/'manifest.json',m)
        with self.assertRaisesRegex(RuntimeError,'semantics code changed'):
            cg.verify_bundle(self.bundle)

    def test_v1_bundle_cannot_be_resumed_by_v2(self):
        m = cg.jload(self.bundle/'manifest.json')
        m['schema'] = 'cg-bdr-v0.1-research'
        cg.jdump(self.bundle/'manifest.json',m)
        with self.assertRaisesRegex(RuntimeError,'Unknown bundle format'):
            cg.verify_bundle(self.bundle)

    def test_end_to_end_uses_best_bound_and_preserves_worst(self):
        self.evaluate()
        state = cg.jload(self.bundle/'state.json')
        result = state['evaluations'][0]['result']
        self.assertEqual((result['metrics']['latency_cycles'],result['metrics']['latency_worst_cycles']),(1000,1200))
        self.assertEqual(result['schedule_results'][0]['achieved_ii'],4)
        self.assertEqual(result['diagnostics'][0]['scheduled_loop'],'L2')
        self.assertEqual(result['diagnostics'][0]['access_owning_loop'],'L2')
        self.assertEqual(result['action_constraint_graph']['schema'],'cg-bdr-action-constraint-graph-v2')
        before = {str(p.relative_to(self.bundle)):cg.digest(p) for p in self.bundle.rglob('*') if p.is_file()}
        proposals = cg.preview(self.bundle)
        self.assertEqual(proposals[0]['kind'],'banking_style')
        self.assertEqual(before,{str(p.relative_to(self.bundle)):cg.digest(p) for p in self.bundle.rglob('*') if p.is_file()})
        self.evaluate()
        summary = cg.jload(self.bundle/'summary.json')
        self.assertEqual((summary['best']['best_latency_cycles'],summary['best']['best_latency_worst_cycles']),(800,1100))
        self.assertEqual(summary['charged_evaluations'],2)
        self.evaluate()
        state = cg.jload(self.bundle/'state.json')
        self.assertEqual(len(state['evaluations']),2)
        self.assertIn('No eligible untried v2 repair',state['stop'])

    def test_pending_reservation_blocks_retry(self):
        cg.jdump(self.bundle/'pending.json',dict(index=1))
        with self.assertRaisesRegex(RuntimeError,'Pending reserved synthesis'):
            self.evaluate()
        self.assertEqual(cg.jload(self.bundle/'state.json')['evaluations'],[])


class V2NativeTests(unittest.TestCase):
    def test_all_nine_evaluations_verify_without_hls(self):
        with mock.patch.object(cg,'synthesize',side_effect=AssertionError('No HLS allowed in audit')):
            report = audit.audit()
        self.assertEqual(report['verified_evaluations'],9)
        self.assertFalse(report['gan_v2_run_present'])
        self.assertTrue(all(row['speedup'] == 1 for row in report['kernels']))
        self.assertEqual(report,cg.jload(V2/'recorded_results.json'))

    def test_native_candidates_replay_v2_policy(self):
        for bundle in (V2/'evidence').glob('*_4eval'):
            m = cg.jload(bundle/'manifest.json')
            for action in m['actions'].values():
                if action['kind']=='array':action['dims']={int(k):v for k,v in action['dims'].items()}
            m['_source_template_lines'] = (bundle/'inputs/template.cpp').read_text().splitlines()
            history = cg.jload(bundle/'state.json')['evaluations']
            for index,candidate in enumerate(history[1:],1):
                earlier=history[:index]
                winner=min([r for r in earlier if cg.eligible(r['result']['metrics'])],key=cg.objective_value)
                proposals=cg.generate_repair(m,winner['directives'],winner['result'],{r['signature'] for r in earlier},history=earlier)
                self.assertTrue(proposals,bundle.name)
                self.assertEqual(proposals[0]['signature'],candidate['signature'],bundle.name)

    def test_native_best_and_worst_bounds_are_distinct(self):
        bundle=V2/'evidence/covariance_LATENCY_10ns_4eval'
        m=cg.jload(bundle/'manifest.json')
        report=bundle/'projects/eval-01/hls_project/solution1/syn/report/kernel_covariance_csynth.xml'
        metrics=cg.read_report(report,m)
        self.assertEqual((cg.objective_value(metrics),metrics['latency_worst_cycles']),(59643,375643))
        self.assertFalse(cg.eligible(dict(metrics,hls_estimated_clock_within_target=None)))
        with self.assertRaisesRegex(RuntimeError,'context mismatch'):
            cg.read_report(report,dict(m,clock_ns=6.67))


if __name__ == '__main__':
    unittest.main()
