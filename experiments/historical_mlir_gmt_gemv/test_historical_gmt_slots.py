import pathlib
import sys
import unittest
import networkx as nx
import torch

sys.path.insert(0,str(pathlib.Path(__file__).parent))
from historical_gmt_slots import make_evidence, HistoricalRoleGMT, loss_fn, metrics, loop_body

class TestHistoricalGMTPipeline(unittest.TestCase):
    def test_graph_evidence_and_routing(self):
        g=nx.MultiDiGraph()
        for i in range(63):g.add_node(str(i),text='arith.addf',type=0,block=0)
        arr=[23,43,18]
        for i,n in enumerate(arr):g.nodes[str(n)].update(type=104, text='array_scope',action_id=f'L{i+1}',action_resolution='source_location')
        g.nodes['28'].update(text='affine.for',is_loop=1,trip_count=32,action_id='L4',action_resolution='source_location')
        g.nodes['3'].update(text='affine.for',is_loop=1,trip_count=32,loop_depth=1,action_id='L5',action_resolution='source_location')
        for i,(root,mem,read) in enumerate(((23,47,58),(43,61,8),(18,24,1))):
            g.nodes[str(mem)].update(type=1,is_memory=1,memory_rank=2-i,text='memref_foo')
            g.nodes[str(read)].update(text='affine.load')
            g.add_edge(str(root),str(mem),flow=7,role='array_root')
            g.add_edge(str(root),str(read),flow=7,role='array_read')
            g.add_edge(str(mem),str(read),flow=10,role='read')
        g.nodes['32'].update(text='affine.store')
        g.add_edge('28','3',flow=8,role='region_entry')
        g.add_edge('28','1',flow=0,role='next_in_block')
        g.add_edge('3','58',flow=8,role='region_entry')
        g.add_edge('58','8',flow=0,role='next_in_block')
        g.add_edge('8','32',flow=0,role='next_in_block')
        g.add_edge('32','1',flow=12,role='RAW',certainty='proven')
        self.assertNotIn('1',loop_body(g,'28')) # next_in_block after header must not leak out of region
        self.assertIn('58',loop_body(g,'28'))
        pack=dict(node_states=torch.randn(63,128),node_embs=torch.zeros(64,128),node_embs_mask=torch.zeros(64,dtype=torch.bool),action_relation_mask=torch.eye(64,dtype=torch.bool))
        pack['node_embs'][:5]=torch.randn(5,128);pack['node_embs_mask'][:5]=True
        pack['action_relation_mask'][0,3]=True
        mapping=[dict(absolute_memory_slot=i,action=f'L{i+1}',compiler_node_id=[23,43,18,28,3][i],action_type='array' if i<3 else 'loop') for i in range(5)]
        # tempfile only to provide a graph provenance file for checksum
        with __import__('tempfile').TemporaryDirectory() as tmp:
            f=pathlib.Path(tmp)/'graph.gexf';nx.write_gexf(g,f)
            data,meta=make_evidence(g,pack,mapping,f)
        self.assertEqual(data['route'].shape[:2],(5,3))
        self.assertTrue(data['base_relation'][0,3])
        self.assertTrue((data['route'].sum(-1)>0).any())
        self.assertGreater(sum('RAW'==r['identity'].get('role') for r in meta['records'] if r['kind']=='edge'),0)
        model=HistoricalRoleGMT()
        summary,active,pred,attn=model(data,return_attention=True)
        self.assertEqual(summary.shape,(5,3,128))
        self.assertTrue(torch.isfinite(summary).all())
        # Any permutation of evidence records must leave memory unchanged.
        permutation=torch.randperm(len(data['src']))
        reorder={k:v.clone() for k,v in data.items()}
        for key in ('src','dst','edge','group','facts','role_idx','flow_idx','certainty_idx'):
            reorder[key]=data[key][permutation]
        reorder['route']=data['route'][:,:,permutation]
        s_reordered,_,_,_=model(reorder,return_attention=True)
        torch.testing.assert_close(summary,s_reordered,rtol=1e-5,atol=1e-5)
        loss,_=loss_fn(model,data);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.action_query.weight.grad.norm()),0)
        self.assertGreater(len(meta['records']),5)
        self.assertTrue((data['role_idx']>0).any())
        self.assertTrue((data['flow_idx']>0).any())
        for i in range(3):
            self.assertGreater(meta['counts'][f'L{i+1}']['memory'],1)

if __name__=='__main__':unittest.main()
