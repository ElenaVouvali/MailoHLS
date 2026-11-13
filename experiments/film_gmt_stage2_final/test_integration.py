"""Focused tests for complete-state joining, cohort gates and exposure coverage."""
import collections
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from experiments.film_gmt_stage2_final import bank_contract as bc,coverage,train


def test_recovered_state_is_joined_before_effective_configuration_grouping():
    from Preprocessing.tests.test_gnosis_recovery import RecoveryTests
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);RecoveryTests().fixture(root)
        path=root/'Data/manifest.json';path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({'kernels':[{'kernel':'k','split':'train',
            'complete_source_configuration_eligible':True,'action_state_recovery':True}]}))
        rows=list(bc.joined_configurations('k',root))
        assert len({row['measurement']['OuterLoop_1'] for row in rows})==1
        assert len({tuple(sorted(row['source_action_states'].items())) for row in rows})==2
        assert all(set(row['source_action_states'])=={'L1','L2'} for row in rows)


def test_locked_test_is_rejected_before_csv_access():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);p=root/'Data/manifest.json';p.parent.mkdir()
        p.write_text(json.dumps({'kernels':[{'kernel':'locked','split':'test'}]}))
        with patch('csv.DictReader',side_effect=AssertionError('locked CSV opened')):
            with pytest.raises(ValueError,match='Locked-test'):
                list(bc.joined_configurations('locked',root))


def test_expanded_source_contract_stops_before_qor_or_partial_bank_creation(tmp_path):
    registry=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run/directive_domain_registry.json')
    if not registry.exists():pytest.skip('Requires the authenticated Liono V4 fixture')
    with patch('csv.DictReader',side_effect=AssertionError('QoR parsed before source-domain qualification')):
        result=bc.qualification(registry,tmp_path/'audit')
    assert result['status']=='NOT_READY'
    assert len(result['blockers'])==9
    assert not result['aggregation_or_selection_started']
    assert not list((tmp_path/'audit').glob('*.jsonl'))


def fixture_data(contexts=180):
    def row(i):
        return {'kernel_name':'k'+str(i%30),'device':'fpga'+str((i//30)%2),
                'clock_period':float(i//60+1),'budget_frac_bram_18k':1.,
                'budget_frac_dsp':1.,'budget_frac_ff':1.,'budget_frac_lut':1.,
                'resource_budget_id':str(i),'_family':'f'+str(i%30)}
    sft=[{'context_id':'c'+str(i),'row':row(i),'source_family':row(i)['_family']} for i in range(contexts)]
    pairs=[{'kind':'clock_switch','group_id':'g','row':row(i)} for i in [0,60]]
    return {'train':{'sft':sft,'pairs':pairs},'val':{'sft':sft[:1],'pairs':pairs}}


def test_old_112_context_cap_cannot_hide_device_clock_or_kernel_coverage():
    data=fixture_data()
    with pytest.raises(ValueError,match='Context cap loses'):
        train.build_schedule(data,450,112,123)
    schedule=train.build_schedule(data,450,0,123)
    assert len(schedule[1])==180
    visits=collections.Counter(schedule[1][i%180] for i in range(450*4))
    assert set(visits)==set(schedule[0])


def test_pair_groups_preserve_both_target_contexts():
    pairs=fixture_data()['train']['pairs']
    assert len(coverage.paired_groups(pairs))==1
    with pytest.raises(ValueError,match='both target contexts'):
        coverage.paired_groups(pairs[:1])


def test_forward_filled_action_routes_cannot_see_a_future_marker():
    import torch
    from LLM_branch.common.structural_xattn import build_placeholder_slot_ids,forward_fill_slot_ids
    ids=torch.tensor([[17,1,1,18,1]])
    slots=forward_fill_slot_ids(build_placeholder_slot_ids(ids,[17,18]))
    assert slots.tolist()==[[1,1,1,2,2]]


def test_volta_bf16_emulation_does_not_replace_native_fp16():
    import types
    fake=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda _: (7,0),
        is_bf16_supported=lambda: True),bfloat16='bf16',float16='fp16')
    assert train.native_autocast_dtype(fake)=='fp16'
    fake.cuda.get_device_capability=lambda _: (8,0)
    assert train.native_autocast_dtype(fake)=='bf16'


def test_original_cached_anchors_are_preserved_as_bytes_not_just_allclose():
    import torch
    out=bc.ROOT/'artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors'
    if not (out/'original_cache_anchors.json').exists():pytest.skip('Requires final local memory export')
    receipt=json.loads((out/'original_cache_anchors.json').read_text())
    for k,entry in receipt['kernels'].items():
        cached=torch.load(entry['original_file'],map_location='cpu',weights_only=True)['node_embs']
        exported=torch.load(out/'memory_B'/(k+'.memory.pt'),map_location='cpu',weights_only=True)['node_embs'][:64]
        assert cached.dtype==exported.dtype
        assert cached.contiguous().numpy().tobytes()==exported.contiguous().numpy().tobytes()
