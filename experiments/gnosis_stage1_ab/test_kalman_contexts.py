import unittest
import torch
from experiments.gnosis_stage1_ab import kalman_contexts as k
from experiments.gnosis_stage1_ab.analyze_kalman_contexts import compare_scores, shared_generation_prefix


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        value = {'input_ids': [ord(c) for c in text]}
        if kwargs.get('return_offsets_mapping'):
            value['offset_mapping'] = [(i, i+1) for i in range(len(text))]
        return value

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


class KalmanTests(unittest.TestCase):
    source = '''/*L1:*/ float a[8];
#pragma HLS array_partition variable=a type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}
/*L2:*/ for(int i=0;i<16;i++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
}'''

    def setUp(self):
        self.tok = CharacterTokenizer()
        self.registry = {k.prod.normalize_kname(k.KERNEL): {
            'AUTO{_ARRAY_T_L1}': ['none','cyclic'],
            'AUTO{_ARRAY_F_L1}': ['0','2','4'],
            'AUTO{_ARRAY_D_L1}': ['0','1'],
            'AUTO{_PIPE_L2}': ['0','1'],
            'AUTO{_UNROLL_L2}': ['0','2','16'],
        }}

    def mask(self, target, padding=0):
        pack = k.prod.build_deterministic_rhs_pack(self.source, target, self.tok,
            directive_domain_registry=self.registry, kernel_name=k.KERNEL)
        ids = torch.tensor([[7,8,9]+pack.input_ids+[0]*padding])
        attn = torch.ones_like(ids)
        if padding:
            attn[:,-padding:] = 0
        return pack, k.causal_kind_mask(self.tok,self.source,self.registry,ids,attn,3)

    def test_partial_multitoken_candidate_preserves_value_causal_positions(self):
        target = 'auto{_ARRAY_T_L1} = cyclic\nauto{_ARRAY_F_L1} = 2\nauto{_ARRAY_D_L1} = 1\nauto{_PIPE_L2} = 1\nauto{_UNROLL_L2} = 16'
        # Production scorer always supplies the required newline continuation.
        pack, mask = self.mask(target+'\n')
        k.apply_decision_policy(pack,self.tok,k.prod.build_rhs_map_from_target(target),'decision')
        expected = torch.zeros_like(mask)
        for i,key in enumerate(pack.site_keys):
            if key and pack.token_weights[i]>0:
                expected[0,3+i-1] = k.trained.kind_id(key)
        torch.testing.assert_close(mask,expected,rtol=0,atol=0)
        self.assertEqual(int((mask==2).sum()),2)
        self.assertTrue(torch.equal(mask[0,:3],torch.zeros(3,dtype=torch.long)))

    def test_forced_values_and_newlines_do_not_receive_film(self):
        _, mask = self.mask('auto{_ARRAY_T_L1} = none\nauto{_ARRAY_F_L1} = 0\nauto{_ARRAY_D_L1} = 0\n')
        self.assertEqual(int((mask==3).sum()),len('none'))
        self.assertEqual(int((mask==4).sum()),0)
        self.assertEqual(int((mask==5).sum()),0)

    def test_padding_is_not_modulated(self):
        _,mask=self.mask('auto{_ARRAY_T_L1} = cyclic\n',padding=7)
        self.assertEqual(int(mask[0,-7:].sum()),0)

    def test_unknown_or_corrupted_target_serialization_fails(self):
        ids=torch.tensor([[7,8,9]+[ord(c) for c in 'unrecognized suffix\n']])
        with self.assertRaises(ValueError):
            k.causal_kind_mask(self.tok,self.source,self.registry,ids,torch.ones_like(ids),3)

    def test_requested_contexts_are_exact_uniform_resource_budgets(self):
        rows=k.contexts(self.source)
        self.assertEqual(len(rows),4)
        self.assertEqual([r['clock_period'] for r in rows],[10.,10.,3.33,10.])
        for r in rows:
            f=.2 if r['context_id'].endswith('20pct') else 1.
            self.assertEqual([r[s] for s in k.trained.FEATURE_KEYS],[f]*4)
            caps=k.mc.DEVICE_RESOURCES[r['device']]
            self.assertEqual(r['avail_dsp'],round(caps['DSP']*f))

    def test_fixed_prefix_reference_margin_and_rank_change(self):
        def site(candidates):
            return [dict(lhs='auto{_UNROLL_L2}',candidates=[
                dict(rhs=rhs,mean_logprob=score) for rhs,score in candidates])]
        a=site([('2',-.1),('0',-.4)])
        b=site([('0',-.2),('2',-.6)])
        result=compare_scores(a,b)[0]
        self.assertTrue(result['winner_changed'])
        self.assertEqual(result['reference_rank_after'],2)
        self.assertAlmostEqual(result['delta_reference_mean_logprob'],-.5)
        self.assertAlmostEqual(result['delta_reference_signed_margin'],-.7)

    def test_different_fixed_prefix_candidate_sets_are_rejected(self):
        a=[dict(lhs='x',candidates=[dict(rhs='0',mean_logprob=0.)])]
        b=[dict(lhs='x',candidates=[dict(rhs='1',mean_logprob=0.)])]
        with self.assertRaises(ValueError):compare_scores(a,b)

    def test_forced_decoder_sentinels_are_not_treated_as_scores(self):
        a=[dict(lhs='x',candidates=[dict(rhs='0',mean_logprob=0.)])]
        self.assertEqual(compare_scores(a,a),[])

    def test_generated_prefix_comparison_stops_after_first_divergence(self):
        def site(lhs,order):
            return dict(lhs=lhs,candidates=[dict(rhs=x,mean_logprob=-float(i+1)) for i,x in enumerate(order)])
        a=dict(prompt_sha256='same',candidate_scores=[site('x',['0','1']),site('y',['0','1']),site('z',['0','1'])])
        b=dict(prompt_sha256='same',candidate_scores=[site('x',['0','1']),site('y',['1','0']),site('z',['1','0'])])
        rows=shared_generation_prefix(a,b)
        self.assertEqual([r['lhs'] for r in rows],['x','y'])
        self.assertEqual(rows[-1]['shared_preceding_directives'],{'X':'0'})


if __name__=='__main__':
    unittest.main()
