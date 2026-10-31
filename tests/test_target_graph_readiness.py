"""CPU-only real-release and adversarial tests of the experimental tensor view."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

try:
    import torch
    from torch_geometric.data import Batch
    from GNN_branch import target_graph_tensor_v22 as enc
    MISSING = None
except ImportError as exc:
    MISSING = str(exc)

from GNN_branch import target_graph_v22 as overlay, mlir_graph_v2 as program

ROOT=Path(__file__).resolve().parents[1]
FILES=('spcl_example_03-fdea84258953.json','rodinia_lud_1_tiling_0-e35f0c4cd5d4.json',
       'serrano-kalman-filter-96b4d608969f.json')


@unittest.skipIf(MISSING is not None, 'Optional experimental PyG environment required: '+str(MISSING))
class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory(prefix='target-tensor-tests-')
        cls.root=Path(cls.temp.name)
        release=json.loads(overlay.RELEASE.read_text())
        archive=ROOT/release['archive']
        assert overlay.file_hash(archive)==release['archive_sha256']
        with tarfile.open(archive) as stream:
            for folder in ('canonical','compact'):
                for filename in FILES:
                    member=folder+'/'+filename
                    data=stream.extractfile(member).read()
                    assert hashlib.sha256(data).hexdigest()==release['files_sha256'][member]
                    target=cls.root/member;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        cls.graphs=[program.read_graph(cls.root/'canonical'/f) for f in FILES]
        cls.joint=[overlay.enrich(g,'zcu104',10) for g in cls.graphs]
        cls.data=[enc.tensorize(g) for g in cls.joint]

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_closed_vocabularies(self):
        self.assertEqual(len(enc.NODE_TYPES),8)
        self.assertEqual(len(enc.FORWARD_RELATIONS),28)
        self.assertEqual(len(enc.RELATIONS),56)
        self.assertEqual(set(enc.FORWARD_RELATIONS),program.EDGE_KINDS|overlay.RELATIONS)

    def test_real_tensor_shapes_and_finiteness(self):
        present=set()
        for data in self.data:
            self.assertTrue(enc.validate_tensors(data))
            self.assertEqual(data.x.shape[1],36)
            self.assertEqual(data.edge_attr.shape[1],enc.EDGE_DIM)
            present.update(data.node_type.tolist())
        self.assertEqual(present,set(range(8)))

    def test_reverse_relations_keep_types_and_original_keys(self):
        for d in self.data:
            self.assertTrue(torch.equal(d.edge_index[:,0::2].flip(0),d.edge_index[:,1::2]))
            self.assertTrue(torch.equal(d.edge_type[0::2]+28,d.edge_type[1::2]))
            self.assertTrue(torch.equal(d.edge_attr[0::2],d.edge_attr[1::2]))
            self.assertEqual(d.audit['source_edge_keys'][0::2],d.audit['source_edge_keys'][1::2])

    def test_negative_alias_evidence_never_becomes_positive_connectivity(self):
        total=0
        for joint,d in zip(self.joint,self.data):
            keys=[str(k) for _,_,k,e in joint.edges(keys=True,data=True) if e.get('alias_kind')=='no_alias']
            self.assertEqual(set(keys),set(d.audit['negative_edge_keys']))
            self.assertTrue(set(keys).isdisjoint(d.audit['source_edge_keys']))
            self.assertEqual(len(keys),d.negative_edge_index.shape[1]);total+=len(keys)
        self.assertGreater(total,0)

    def test_compact_and_canonical_features_identical(self):
        for filename,d in zip(FILES,self.data):
            compact=program.read_graph(self.root/'compact'/filename)
            other=enc.tensorize(overlay.enrich(compact,'zcu104',10))
            for key,value in d:
                if torch.is_tensor(value):self.assertTrue(torch.equal(value,other[key]),key)

    def test_targets_do_not_change_compiler_inputs(self):
        original=self.data[0];mask=original.node_type<6
        for target,clock in [('u200',10),('u50',10),('zcu104',6.67)]:
            data=enc.tensorize(overlay.enrich(self.graphs[0],target,clock))
            for key in ['x','node_type','op_type','value_type','root_type']:
                self.assertTrue(torch.equal(original[key][mask],data[key][mask]))
            self.assertFalse(torch.equal(original.x,data.x))

    def test_kalman_absolute_masks_and_missing_targets(self):
        data=self.data[2]
        self.assertEqual(int(data.registered_mask.sum()),13)
        self.assertEqual(int(data.matched_mask.sum()),11)
        self.assertEqual(data.unresolved_mask.nonzero()[:,1].tolist(),[6,7])
        self.assertNotIn(6,data.action_slot.tolist());self.assertNotIn(7,data.action_slot.tolist())
        self.assertIn(12,data.action_slot.tolist())
        with self.assertRaisesRegex(ValueError,'truncation'):enc.tensorize(self.joint[2],max_slots=12)

    def test_missing_count_is_not_known_zero(self):
        joint=deepcopy(self.joint[0]);node=self.data[0].audit['node_ids'].index('action:L5')
        joint.graph['action_annotations']['L5']['loop_semantics']['compiler_proven_trip_count']=None
        unknown=enc.tensorize(joint)
        at=2*enc.NUMERICAL.index('trip_count')
        self.assertEqual(unknown.x[node,at:at+2].tolist(),[0.,0.])
        joint.graph['action_annotations']['L5']['loop_semantics']['compiler_proven_trip_count']=0
        known=enc.tensorize(joint)
        self.assertEqual(known.x[node,at:at+2].tolist(),[0.,1.])

    def test_labels_ids_and_paths_are_not_numerical_features(self):
        joint=deepcopy(self.joint[0])
        semantics={label:a.get('loop_semantics',{}) for label,a in joint.graph['action_annotations'].items()}
        for _,attrs in joint.nodes(data=True):
            original=enc._features(attrs,semantics)
            attrs['measured_latency']=999999;attrs['qor']=123;attrs['source_file']='unseen/path'
            self.assertEqual(original,enc._features(attrs,semantics))
        # Exclusion from learning features does not waive release authentication.
        with self.assertRaisesRegex(ValueError,'Original program payload changed'):enc.tensorize(joint)

    def test_root_completeness_and_certainty_stay_separate(self):
        attrs={'evidence_status':'possible','root_candidates_complete':False}
        row=enc._edge_features(attrs)
        self.assertEqual(row[enc.EVIDENCE.index('possible')],1.)
        at=len(enc.EVIDENCE)+enc.EDGE_NUMERICAL.index('root_complete')
        self.assertEqual(row[at:at+2],[0.,1.])
        del attrs['root_candidates_complete']
        self.assertEqual(enc._edge_features(attrs)[at:at+2],[0.,0.])
        attrs['root_candidates_complete']=None
        self.assertEqual(enc._edge_features(attrs)[at:at+2],[0.,0.])
        attrs['loop_carried']=None
        at=len(enc.EVIDENCE)+enc.EDGE_NUMERICAL.index('loop_carried')
        self.assertEqual(enc._edge_features(attrs)[at:at+2],[0.,0.])

    def test_batch_offsets_preserve_action_slots_and_embeddings(self):
        torch.manual_seed(123);model=enc.SmallGraphEncoder().eval()
        batch=Batch.from_data_list([self.data[0],self.data[2]])
        with torch.no_grad():
            memory,_=model.memories(batch)
            self.assertEqual(list(memory.shape),[2,64,128])
            for i,d in enumerate([self.data[0],self.data[2]]):
                standalone,_=model.memories(d)
                self.assertTrue(torch.allclose(memory[i],standalone[0],atol=2e-6))
            self.assertEqual(float(memory[1,6:8].abs().sum()),0.)

    def test_hardware_gradient_path_and_ablated_zero_path(self):
        torch.manual_seed(123);model=enc.SmallGraphEncoder().eval()
        for connected in (True,False):
            data=self.data[0].clone() if connected else enc.without_hardware_messages(self.data[0])
            data.x=data.x.detach().clone().requires_grad_(True)
            model.zero_grad();model(data)[data.action_node_index].square().sum().backward()
            grad=data.x.grad[data.node_type>=6]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()),0.) if connected else self.assertEqual(float(grad.abs().sum()),0.)

    def test_no_mutation_of_canonical_program(self):
        for graph,joint in zip(self.graphs,self.joint):
            before=program.canonical_json(program.graph_payload(graph))
            enc.tensorize(joint)
            self.assertEqual(before,program.canonical_json(program.graph_payload(graph)))

    def test_nonfinite_numerical_inputs_rejected(self):
        joint=deepcopy(self.joint[0]);joint.nodes['hardware:fpga']['clock_period_ns']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):enc.tensorize(joint)

    def test_unknown_vocabulary_cannot_change_dimensions(self):
        attrs={'category':'operation','text':'future.unknown_operation'}
        self.assertEqual(enc._op(attrs),'other')
        self.assertEqual(enc._type({'value_type':'unseen_type'})[0],'aggregate')

    def test_known_zero_budget_and_unknown_platform_are_distinct(self):
        data=enc.tensorize(overlay.enrich(self.graphs[0],'zcu104',10,budgets={'DSP':0}))
        node=data.audit['node_ids'].index('hardware:resource:DSP')
        for name in ('available_capacity','available_fraction','requested_budget'):
            at=2*enc.NUMERICAL.index(name)
            self.assertEqual(data.x[node,at:at+2].tolist(),[0.,1.])
        at=2*enc.NUMERICAL.index('platform_capacity')
        self.assertEqual(data.x[node,at:at+2].tolist(),[0.,0.])

    def test_every_relation_has_an_executable_typed_message(self):
        # Hand-authored tensor-only coverage, not native proof of a new CFG edge.
        torch.manual_seed(123);model=enc.SmallGraphEncoder().eval()
        data=self.data[0].clone()
        pairs=len(enc.FORWARD_RELATIONS)
        data.edge_index=torch.tensor([[0,1]]*pairs+[[1,0]]*pairs).t()
        data.edge_type=torch.arange(len(enc.RELATIONS))
        data.edge_attr=torch.zeros(len(enc.RELATIONS),enc.EDGE_DIM)
        model(data).sum().backward()
        self.assertTrue(torch.isfinite(model.relation.weight.grad).all())
        self.assertTrue((model.relation.weight.grad.abs().sum(1)>0).all())

    def test_candidate_head_is_separate_and_receives_gradients(self):
        from experiments.target_graph_readiness.experiment import CandidatePredictor, CANDIDATE_DIM
        torch.manual_seed(123);model=CandidatePredictor().eval()
        data=Batch.from_data_list([self.data[0]])
        p=torch.zeros(1,64,CANDIDATE_DIM,requires_grad=True)
        initial=model(data,p,torch.tensor([0]),'enriched')
        other=p.detach().clone();other[0,4,0]=1  # Legal pipeline toggle at L5.
        changed=model(data,other,torch.tensor([0]),'enriched')
        self.assertGreater(float((initial-changed).abs().sum()),0.)
        initial.sum().backward()
        self.assertGreater(float(p.grad.abs().sum()),0.)

    def test_candidate_partition_factor_depends_on_selected_dimension(self):
        from types import SimpleNamespace
        from experiments.target_graph_readiness.experiment import _candidate
        from LLM_branch.common.directive_domains import SourceAction
        actions={'L1':SourceAction('array',dimensions=((1,30),(2,50)))}
        mapping={'Array_1':SimpleNamespace(action_id='L1')}
        values,p=_candidate({'Array_1':'cyclic_25_2'},mapping,actions)
        self.assertEqual(values['AUTO{_ARRAY_D_L1}'],'2')
        self.assertEqual(float(p[0,5]),1.)
        with self.assertRaisesRegex(ValueError,'TCAD tuple'):
            _candidate({'Array_1':'cyclic_25_1'},mapping,actions)

    def test_incomplete_candidate_cannot_silently_default_unmapped_actions(self):
        from types import SimpleNamespace
        from experiments.target_graph_readiness.experiment import _candidate
        from LLM_branch.common.directive_domains import SourceAction
        actions={'L1':SourceAction('loop',trip_count=32),'L3':SourceAction('loop',trip_count=32)}
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            _candidate({'OuterLoop_1':'unroll_2'},{'OuterLoop_1':SimpleNamespace(action_id='L1')},actions)


if __name__=='__main__':unittest.main()
