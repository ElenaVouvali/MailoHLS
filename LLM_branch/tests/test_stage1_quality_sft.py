"""CPU-only regression tests for Stage-1 quality-aware measured supervision."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.stage1_quality_sft.core import (
    assignment, build_banks, build_compact_request, build_request, candidate_pool_record,
    parse, request_id, sample_target, single_reference, validate_complete_assignment,
)
from experiments.stage1_quality_sft.evaluate import evaluate
from experiments.stage1_quality_sft.evaluate import select_checkpoint
from experiments.stage1_quality_sft.support import constrained_rhs_candidates, normalize_kname


def row(target, latency, util=10, idx=0, kernel='toy', device='xcu200-fsgd2104-2-e',
        clock=10, source='toy.cpp', budget=1.0):
    return {'kernel_name': kernel, 'source_key': kernel, 'source_file': source,
        'device': device, 'clock_period': clock, 'resource_budget_id': f'{budget}',
        'budget_frac_bram_18k': budget, 'budget_frac_dsp': budget,
        'budget_frac_ff': budget, 'budget_frac_lut': budget, 'objective': 'PARETO_LATENCY',
        'target': target, 'latency': latency, 'bram_util_%': util,
        'dsp_util_%': util, 'ff_util_%': util, 'lut_util_%': util,
        '_jsonl_idx': idx, '_family': 'toy-family'}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row('auto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 0', 100, idx=1),
            row('auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 2', 100.5, idx=2),
            row('auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 4', 102, idx=3),
        ]

    def test_identity_complete_assignment_and_latency_tolerance(self):
        banks, _ = build_banks(self.rows)
        request = build_request(self.rows[0], banks, .01)
        self.assertEqual(request['accepted_count'], 2)
        self.assertEqual(request['feasible_count'], 3)
        self.assertEqual(request['accepted'][0]['row_id'], 1)
        self.assertEqual(len(request['request_identity'][-2]), 4)
        self.assertEqual(set(request['accepted'][1]['assignment']), {'AUTO{_PIPE_L1}', 'AUTO{_UNROLL_L1}'})
        self.assertEqual(assignment(self.rows[0]['target']), tuple(sorted(parse(self.rows[0]['target']).items())))

    def test_resource_budget_exact_fraction_and_zero_support(self):
        row_tight = row(self.rows[1]['target'], 100.5, util=50, idx=2, budget=.5)
        bank, _ = build_banks([row_tight])
        self.assertEqual(build_request(row_tight, bank, .01)['feasible_count'], 1)
        impossible = dict(row_tight, budget_frac_ff=0)
        with self.assertRaisesRegex(ValueError, 'no feasible'):
            build_request(impossible, bank, .01)

    def test_predeclared_epsilon_sensitivity_boundaries(self):
        bank, _ = build_banks(self.rows)
        self.assertEqual(build_request(self.rows[0], bank, 0)['accepted_count'], 1)
        self.assertEqual(build_request(self.rows[0], bank, .005)['accepted_count'], 2)
        self.assertEqual(build_request(self.rows[0], bank, .01)['accepted_count'], 2)

    def test_duplicate_conflicting_measurements_rejected(self):
        conflict = dict(self.rows[0], _jsonl_idx=99, latency=101)
        with self.assertRaisesRegex(ValueError, 'Conflicting measured QoR'):
            build_banks([self.rows[0], conflict])

    def test_tie_breaking_and_sampling_are_complete_and_resume_stable(self):
        tied = [row('auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 2', 100, idx=4),
                row('auto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 4', 100, idx=5)]
        bank, _ = build_banks(tied)
        request = build_request(tied[0], bank, .01)
        self.assertEqual(single_reference(request)['assignment_id'],
                         min(c['assignment_id'] for c in request['accepted']))
        schedule = [sample_target(request, 123, i) for i in range(40)]
        self.assertEqual(schedule, [sample_target(request, 123, i) for i in range(40)])
        self.assertEqual({tuple(sorted(c['assignment'].items())) for c in schedule},
                         {tuple(sorted(c['assignment'].items())) for c in request['accepted']})
        for choice in schedule:
            self.assertIn(choice['assignment'], [c['assignment'] for c in request['accepted']])

    def test_context_identity_includes_target_clock_budget_objective(self):
        base = self.rows[0]
        banks, _ = build_banks([base])
        self.assertNotEqual(request_id(base), request_id(dict(base, clock_period=5)))
        self.assertNotEqual(build_request(base, banks)['request_id'],
                            build_request(dict(base, budget_frac_ff=.5), banks)['request_id'])
        self.assertNotEqual(build_request(base, banks)['request_id'],
                            build_request(dict(base, objective='PARETO_ADP'), banks)['request_id'])

    def test_compact_pool_deduplicates_physical_candidate_information(self):
        banks, _ = build_banks(self.rows)
        request = build_compact_request(self.rows[0], banks)
        pool = candidate_pool_record(next(iter(banks)), next(iter(banks.values())))
        self.assertEqual(request['candidate_pool_id'], pool['candidate_pool_id'])
        self.assertEqual(request['accepted_count'], 2)
        self.assertEqual(len(pool['candidates']), 3)
        self.assertTrue(set(request['sensitivity']['0.01']['accepted_candidate_ids']).issubset(
            {c['candidate_id'] for c in pool['candidates']}))

    def test_complete_domain_and_array_tuple_contract(self):
        registry = {'toy': {
            'AUTO{_PIPE_L1}': ['0', '1'], 'AUTO{_UNROLL_L1}': ['0', '2'],
            'AUTO{_ARRAY_T_L2}': ['none', 'block'], 'AUTO{_ARRAY_F_L2}': ['0', '2'],
            'AUTO{_ARRAY_D_L2}': ['0', '1']}}
        good = row('auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 2\nauto{_ARRAY_T_L2} = block\nauto{_ARRAY_F_L2} = 2\nauto{_ARRAY_D_L2} = 1', 100)
        parsed = validate_complete_assignment(good, registry, constrained_rhs_candidates, normalize_kname, parse)
        self.assertEqual(len(parsed), 5)
        bad = dict(good, target=good['target'].replace('ARRAY_D_L2} = 1', 'ARRAY_D_L2} = 0'))
        with self.assertRaises(ValueError):
            validate_complete_assignment(bad, registry, constrained_rhs_candidates, normalize_kname, parse)
        incomplete = dict(good, target='auto{_PIPE_L1} = 1')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate_complete_assignment(incomplete, registry, constrained_rhs_candidates, normalize_kname, parse)


class EvaluationTests(unittest.TestCase):
    def test_unmatched_outputs_have_unknown_qor_and_exact_match_scoring(self):
        source = row('auto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 0', 100, idx=1)
        bank, _ = build_banks([source])
        request = build_compact_request(source, bank)
        pool = candidate_pool_record(next(iter(bank)), next(iter(bank.values())))
        exact = source['target']
        pred = [{'request_id': request['request_id'], 'prediction': exact}]
        report = evaluate([request], pred, {pool['candidate_pool_id']: pool['candidates']})
        self.assertEqual(report['exact_measured_matches'], 1)
        self.assertEqual(report['unknown_qor'], 0)
        unmatched = evaluate([request], [{'request_id': request['request_id'], 'prediction': 'auto{_PIPE_L1} = 1'}],
                             {pool['candidate_pool_id']: pool['candidates']})
        self.assertEqual(unmatched['unknown_qor'], 1)
        self.assertIsNone(unmatched['per_request'][0]['regret'])

    def test_prediction_outside_frozen_validation_requests_fails(self):
        source = row('auto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 0', 100, idx=1)
        bank, _ = build_banks([source])
        request = build_compact_request(source, bank)
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            evaluate([request], [{'request_id': 'future-hidden-result', 'prediction': ''}], {})

    def test_checkpoint_choice_fails_closed_and_is_deterministic(self):
        report = dict(input_sha256=dict(requests='r', candidate_pools='p'), requests=10,
            predictions=10, missing_predictions=0, exact_measured_matches=2,
            measured_feasible=2, exact_match_coverage=.2,
            family_macro=dict(feasible_regret=.1), source_legal_count=9,
            complete_assignment_count=10)
        result = select_checkpoint([
            dict(checkpoint='step-20', step=20, report=report),
            dict(checkpoint='step-10', step=10, report=report)])
        self.assertEqual(result['selected'], 'step-10')
        unknown = dict(report, exact_measured_matches=0)
        self.assertEqual(select_checkpoint([dict(checkpoint='x', step=1, report=unknown)])['status'],
                         'no_hardware_supported_checkpoint')
        with self.assertRaisesRegex(ValueError, 'identical complete'):
            select_checkpoint([dict(checkpoint='a', step=1, report=report),
                               dict(checkpoint='b', step=2, report=dict(report, input_sha256=dict(requests='x', candidate_pools='p')))])


class SplitTests(unittest.TestCase):
    def test_test_rows_are_not_decoded(self):
        from experiments.stage1_quality_sft.build import _read_split_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.jsonl'
            valid = json.dumps(dict(kernel_name='toy', source_key='toy', source_file='toy.cpp',
                                    target='auto{_PIPE_L1} = 0', device='xcu200', clock_period=10))
            path.write_text(valid + '\nNOT JSON TEST LABEL\n')
            selected = _read_split_rows(path, [0], {'toy': '// template auto{_PIPE_L1}'})
            self.assertEqual(len(selected), 1)
            with self.assertRaises(ValueError):
                _read_split_rows(path, [1], {'toy': '// template'})

    def test_observability_requires_raw_and_processed_apl_action(self):
        from experiments.stage1_quality_sft.observability import audit
        from experiments.stage1_quality_sft.support import sha
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Data/CSVS').mkdir(parents=True)
            (root / 'Data/ApplicationAPLMapping').mkdir(parents=True)
            (root / 'Data/ApplicationDataset/toy').mkdir(parents=True)
            processed = root / 'processed'; processed.mkdir()
            raw = root / 'Data/CSVS/toy.csv'
            source = root / 'Data/ApplicationDataset/toy/kernel.cpp'
            info = root / 'Data/ApplicationDataset/toy/kernel_info.txt'
            apl = root / 'Data/ApplicationAPLMapping/toy.txt'
            table = processed / 'preprocessed-toy.csv'
            raw.write_text('Loop_1\npipeline\n')
            source.write_text('L1: for (;;) {}\n')
            info.write_text('toy\nL1,loop,4\n')
            apl.write_text('Loop_1,L1\n')
            table.write_text('Loop_1\npipeline_1\n')
            prep = root / 'prep.json'
            prep.write_text(json.dumps({'kernels': [dict(kernel='toy', input_sha256=sha(raw),
                output_sha256=sha(table))]}))
            ds = dict(kernel_statistics=[dict(kernel='toy', source='Data/ApplicationDataset/toy/kernel.cpp',
                                               source_sha256=sha(source))])
            template = '// auto{_PIPE_L1} auto{_UNROLL_L1}'
            observed = [dict(kernel_name='toy', source_key='toy', preprocessed_row=0,
                target='auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 0')]
            results, eligible = audit(observed, ds, prep, {'toy': template}, root, processed)
            self.assertEqual(eligible, {'toy'})
            self.assertEqual(results['toy']['actions'][0]['status'], 'directly_measured')
            apl.write_text('')
            table.write_text('Version\nrow\n')
            observed[0]['target'] = 'auto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 0'
            prep.write_text(json.dumps({'kernels': [dict(kernel='toy', input_sha256=sha(raw),
                output_sha256=sha(table))]}))
            results, eligible = audit(observed, ds, prep, {'toy': template}, root, processed)
            self.assertEqual(eligible, set())
            self.assertEqual(results['toy']['unknown_action_ids'], ['L1'])


try:
    import importlib.util
    _HAS_MODEL_RUNTIME = bool(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'))
except (ImportError, ValueError):
    _HAS_MODEL_RUNTIME = False


@unittest.skipUnless(_HAS_MODEL_RUNTIME, 'Stage-1 loss integration needs the repository PyTorch/Transformers runtime')
class LossIntegrationTests(unittest.TestCase):
    def test_action_normalized_loss_masks_forced_fields(self):
        from types import SimpleNamespace
        from LLM_branch.train.stage1_decision_loss import apply_decision_policy

        class CharacterTokenizer:
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                payload = dict(input_ids=[ord(c) for c in text])
                if return_offsets_mapping:
                    payload['offset_mapping'] = [(i, i+1) for i in range(len(text))]
                return payload

        # One free PIPE site and one semantically forced UNROLL site in L1.
        site = 'AUTO{_PIPE_L1}'
        ids = [ord('1'), ord('\n'), ord('0'), ord('\n')]
        pack = SimpleNamespace(input_ids=ids, labels=[ids[0], ids[1], -100, -100], token_weights=[1.]*4,
            xattn_target_mask=[1]*4, site_keys=[site, site, 'AUTO{_UNROLL_L1}', 'AUTO{_UNROLL_L1}'])
        tok = CharacterTokenizer()
        result = apply_decision_policy(pack, tok,
            {'AUTO{_PIPE_L1}':'1', 'AUTO{_UNROLL_L1}':'0'}, policy='decision')
        self.assertEqual(result.labels, [ord('1'), -100, -100, -100])
        self.assertEqual(result.xattn_target_mask, [1, 0, 0, 0])
        self.assertAlmostEqual(sum(result.token_weights), 1.)


if __name__ == '__main__':
    unittest.main()
