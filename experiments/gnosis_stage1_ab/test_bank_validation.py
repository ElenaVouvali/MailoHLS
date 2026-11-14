"""Regression tests: restored compact JSONL lowercase completion keys.

Uses the same source-key semantics as production, without loading GPU packages.
"""
import ast
import functools
import math
import unittest
from pathlib import Path


class FakeProduction:
    RESOURCE_KEYS=('BRAM_18K','DSP','FF','LUT')
    UTIL_FIELD_BY_RESOURCE={k:k.lower()+'_util_%' for k in RESOURCE_KEYS}
    DEVICE_RESOURCES={'xczu7ev-ffvc1156-2-e':{}}

    @staticmethod
    def extract_ordered_lhs_plan(source):
        assert 'auto{_PIPE_L1}' in source
        return [('L1','auto{_PIPE_L1}')]

    @staticmethod
    def build_rhs_map_from_target(target):
        result={}
        for line in target.splitlines():
            if '=' in line:
                left,right=line.split('=',1)
                result[left.strip()]=right.strip()
        return result

    @staticmethod
    def _clock_of(row):return row['clock_period']

    @staticmethod
    def _norm_device(x):return x

    @staticmethod
    def constrained_rhs_candidates(kernel,lhs,registry,prior):
        values=registry[kernel][lhs.upper()]
        return values,values


def load_validation():
    src=Path(__file__).with_name('prepare_gnosis_ab.py').read_text()
    tree=ast.parse(src)
    names={'cached_ordered_lhs_plan','validate_row','valid'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    g={'prod':FakeProduction,'RKEYS':FakeProduction.RESOURCE_KEYS,
       'functools':functools,'math':math}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'bank_validation_functions','exec'),g)
    return g


class TestBankValidation(unittest.TestCase):
    def setUp(self):
        self.fn=load_validation()['validate_row']
        self.row=dict(kernel_name='example',input='void k(){ auto{_PIPE_L1}; }',
            target='auto{_PIPE_L1} = 1',device='xczu7ev-ffvc1156-2-e',
            clock_period=5.0,latency=1.0,
            **{k.lower()+'_util_%':4. for k in FakeProduction.RESOURCE_KEYS})
        self.registry={'example':{'AUTO{_PIPE_L1}':['0','1']}}

    def test_lowercase_completion_keys_are_accepted(self):
        self.assertIsNone(self.fn(self.row,{'example'},self.registry))

    def test_uppercase_completion_keys_are_accepted(self):
        self.row['target']='AUTO{_PIPE_L1} = 1'
        self.assertIsNone(self.fn(self.row,{'example'},self.registry))

    def test_illegal_value_must_still_fail(self):
        self.row['target']='auto{_PIPE_L1} = 99'
        self.assertEqual(self.fn(self.row,{'example'},self.registry),
                         'out_of_parent_legal_domain')

    def test_unknown_source_label_must_still_fail(self):
        self.row['target']='auto{_PIPE_L2} = 1'
        self.assertEqual(self.fn(self.row,{'example'},self.registry),
                         'source_completion_lhs_mismatch')

    def test_missing_registry_must_fail_not_invent_domain(self):
        self.assertEqual(self.fn(self.row,{'example'},{}),
                         'missing_parent_registry_or_input_field')


if __name__=='__main__':unittest.main()
