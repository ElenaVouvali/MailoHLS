import json

import pytest

from experiments.stage1_recovery.evaluate import (
    load_validation_measurements,
    measured_candidate_quality,
    target_transition_support,
)
from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES


DEVICE = 'xcu200-fsgd2104-2-e'
CAP = DEVICE_RESOURCES[DEVICE]
SOURCE = 'Data/ApplicationDataset/toy/kernel.cpp'


def assignment(pipe, unroll):
    return f'auto{{_PIPE_L1}} = {pipe}\nauto{{_UNROLL_L1}} = {unroll}'


def row(idx, target, latency, *, source=SOURCE, device=DEVICE, clock=10.0,
        bram=1.0, dsp=1.0, ff=1.0, lut=1.0):
    return dict(_jsonl_idx=idx, kernel_name='toy', source_key='toy', source_file=source,
        device=device, clock_period=clock, target=target, latency=latency,
        **{'bram_util_%': bram, 'dsp_util_%': dsp, 'ff_util_%': ff, 'lut_util_%': lut})


def request(target=assignment(0, 0), *, idx=900, dsp_budget=CAP['DSP'], budget_id='budget'):
    return dict(_jsonl_idx=idx, kernel_name='toy', source_key='toy', source_file=SOURCE,
        device=DEVICE, selected_clock_period=10.0, resource_budget_id=budget_id,
        obj_mode='PARETO_LATENCY', target=target,
        avail_bram=CAP['BRAM_18K'], avail_dsp=dsp_budget, avail_ff=CAP['FF'], avail_lut=CAP['LUT'],
        budget_frac_bram_18k=1.0, budget_frac_dsp=dsp_budget/CAP['DSP'],
        budget_frac_ff=1.0, budget_frac_lut=1.0)


def prediction(value, *, idx=900, reference=assignment(0, 0), budget_id='budget'):
    return dict(kernel_name='toy', source_key='toy', jsonl_idx=idx, device=DEVICE, selected_clock_period=10.0,
        obj_mode='PARETO_LATENCY', reference_target=reference,
        prediction=value, resource_budget_id=budget_id)


def test_exact_match_requires_full_source_device_clock_and_assignment():
    measurements = [
        row(1, assignment(1, 4), 11.0),
        row(2, assignment(1, 4), 1.0, source='other.cpp'),
        row(3, assignment(1, 4), 1.0, device='xczu7ev-ffvc1156-2-e'),
        row(4, assignment(1, 4), 1.0, clock=5.0),
    ]
    summary, records = measured_candidate_quality(
        [prediction(assignment(1, 4))], measurements, [request()])
    assert summary['exact_measured_match_count'] == 1
    assert records[0]['measured_qor']['latency'] == 11.0
    assert records[0]['feasibility'] is True


def test_repeated_measurement_row_under_two_budgets_keeps_requests_distinct():
    cfg = assignment(1, 4)
    requests = [request(idx=900, budget_id='loose'),
                request(idx=900, dsp_budget=CAP['DSP'] * .5, budget_id='tight')]
    predictions = [prediction(cfg, idx=900, budget_id='loose'),
                   prediction(cfg, idx=900, budget_id='tight')]
    summary, records = measured_candidate_quality(predictions,
        [row(1, cfg, 10.0, dsp=80.0)], requests)
    assert [r['resource_budget_id'] for r in records] == ['loose', 'tight']
    assert [r['feasibility'] for r in records] == [True, False]
    assert summary['exact_measured_match_count'] == 2

    invariant_predictions = [prediction(assignment(0, 0), idx=900, budget_id='loose'),
                             prediction(assignment(0, 0), idx=900, budget_id='tight')]
    transitions = target_transition_support(invariant_predictions, requests)['budget']
    assert transitions['comparisons'] == 1
    assert transitions['invariant'] == 1
    assert transitions['correctly_invariant'] == 1
    assert transitions['distinct_participating_contexts'] == 2


def test_feasibility_regret_and_near_optimal_candidate():
    measurements = [row(1, assignment(0, 0), 10.0),
                    row(2, assignment(1, 4), 10.5),
                    row(3, assignment(1, 8), 7.0, dsp=100.0)]
    summary, records = measured_candidate_quality(
        [prediction(assignment(1, 4))], measurements,
        [request(dsp_budget=CAP['DSP'] * 0.5)])
    assert records[0]['best_feasible_latency'] == 10.0
    assert records[0]['measured_regret'] == pytest.approx(0.05)
    assert summary['within_5_percent_count'] == 1
    assert records[0]['feasibility'] is True

    tight = request(dsp_budget=CAP['DSP'] * 0.5)
    summary, records = measured_candidate_quality(
        [prediction(assignment(1, 8))], measurements, [tight])
    assert records[0]['feasibility'] is False
    assert records[0]['measured_qor']['latency'] == 7.0
    assert records[0]['measured_regret'] is None
    assert summary['matched_infeasible_count'] == 1

    no_feasible, no_feasible_rows = measured_candidate_quality(
        [prediction(assignment(1, 8))], [row(3, assignment(1, 8), 7.0, dsp=100.0)],
        [request(dsp_budget=CAP['DSP'] * 0.5)])
    assert no_feasible_rows[0]['best_feasible_latency'] is None
    assert no_feasible_rows[0]['measured_regret'] is None
    assert no_feasible['requests_without_any_feasible_measured_validation_candidate'] == 1

    optimum, optimum_rows = measured_candidate_quality(
        [prediction(assignment(0, 0))], measurements, [request(dsp_budget=CAP['DSP'] * 0.5)])
    assert optimum_rows[0]['measured_regret'] == 0.0
    assert optimum['best_feasible_latency_match_count'] == 1


def test_unmeasured_prediction_has_unknown_qor_and_no_regret():
    summary, records = measured_candidate_quality(
        [prediction(assignment(1, 2))], [row(1, assignment(0, 0), 10.0)], [request()])
    assert records[0]['measured_match'] is False
    assert records[0]['measured_qor'] is None
    assert records[0]['measured_regret'] is None
    assert summary['exact_measured_match_fraction'] == 0.0
    assert summary['no_measured_match_count'] == 1


def test_duplicate_design_rows_must_have_identical_qor():
    with pytest.raises(ValueError, match='Conflicting repeated measurement'):
        measured_candidate_quality([prediction(assignment(0, 0))],
            [row(1, assignment(0, 0), 10.0), row(2, assignment(0, 0), 11.0)], [request()])


def test_identical_duplicate_receipts_collapse_deterministically():
    summary, records = measured_candidate_quality(
        [prediction(assignment(0, 0))],
        [row(2, assignment(0, 0), 10.0), row(1, assignment(0, 0), 10.0)], [request()])
    assert summary['canonical_measurement_designs'] == 1
    assert summary['duplicate_rows_collapsed'] == 1
    assert records[0]['measured_qor']['latency'] == 10.0


def test_validation_reader_never_parses_nonvalidation_rows(tmp_path):
    data = tmp_path / 'dataset.jsonl'
    data.write_text('{"valid": true}\nnot-json-test-row\n')
    import hashlib
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'val_jsonl_idx': [0]}))
    rows = load_validation_measurements(data, split, hashlib.sha256(data.read_bytes()).hexdigest())
    assert len(rows) == 1 and rows[0]['_jsonl_idx'] == 0


def test_overlapping_budget_pairs_report_context_count_and_correct_invariance():
    requests = [request(assignment(0, 0), idx=900, dsp_budget=CAP['DSP'] * .5),
                request(assignment(1, 4), idx=901, dsp_budget=CAP['DSP'] * .7),
                request(assignment(1, 4), idx=902, dsp_budget=CAP['DSP'] * .9)]
    predictions = [prediction(assignment(0, 0), idx=900),
                   prediction(assignment(1, 4), idx=901, reference=assignment(1, 4)),
                   prediction(assignment(1, 4), idx=902, reference=assignment(1, 4))]
    out = target_transition_support(predictions, requests)['budget']
    assert out['comparisons'] == 3
    assert out['distinct_participating_contexts'] == 3
    assert out['controlled_context_groups'] == 1
    assert out['changed'] == 2 and out['invariant'] == 1
    assert out['correctly_invariant'] == 1
    assert out['changed_fields_correct'] == 2
    assert out['macro_group_complete_transition_accuracy'] == 1.0
    assert out['macro_group_changed_field_accuracy'] == 1.0
    assert out['macro_group_correct_invariance_accuracy'] == 1.0

    collapsed = [prediction(assignment(0, 0), idx=i,
                            reference=assignment(0, 0) if i == 900 else assignment(1, 4))
                 for i in (900, 901, 902)]
    failed = target_transition_support(collapsed, requests)['budget']
    assert failed['correctly_invariant'] == 0
    assert failed['incorrect_constant'] == 2
    assert failed['macro_group_complete_transition_accuracy'] == 0.0
