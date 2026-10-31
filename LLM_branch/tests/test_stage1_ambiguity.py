"""Small measured-design checks for the training-only ambiguity audit."""
import pytest
import json

from experiments.stage1_recovery.ambiguity import audit, candidate_banks, load_training_rows, near_optimal
from experiments.stage1_recovery.prepare import sha


def row(index, directive, latency, *, dsp=10, budget=.5):
    return dict(_jsonl_idx=index, _family='example', kernel_name='example',
                source_key='example', source_file='example.cpp', device='zcu104',
                clock_period=10.0, selected_clock_period=10.0,
                target=f'auto{{_UNROLL_L1}} = {directive}', latency=latency,
                **{'bram_util_%': 1, 'dsp_util_%': dsp, 'ff_util_%': 1,
                   'lut_util_%': 1, 'budget_frac_bram_18k': budget,
                   'budget_frac_dsp': budget, 'budget_frac_ff': budget,
                   'budget_frac_lut': budget})


def test_near_optimal_requires_complete_measured_feasible_design():
    winner = row(0, 1, 100)
    close = row(1, 2, 100.4)
    infeasible = row(2, 4, 90, dsp=80)
    banks, repeats = candidate_banks([winner, close, infeasible])
    assert repeats == 0
    feasible, accepted, optimum = near_optimal(winner, banks[('example', 'example', 'example.cpp', 'zcu104', 10.0)], .005)
    assert optimum == 100
    assert len(feasible) == len(accepted) == 2
    assert audit([winner], [winner, close, infeasible])['tolerances']['0.0']['multiple_accepted'] == 0
    assert audit([winner], [winner, close, infeasible])['tolerances']['0.005']['multiple_accepted'] == 1


def test_repeated_measurement_requires_identical_qor():
    first = row(0, 1, 100)
    repeated = row(1, 1, 100)
    assert candidate_banks([first, repeated])[1] == 1
    repeated['latency'] = 101
    with pytest.raises(ValueError, match='Conflicting measured QoR'):
        candidate_banks([first, repeated])


def test_missing_selected_design_and_infeasibility_fail_closed():
    winner = row(0, 1, 100)
    with pytest.raises(ValueError, match='lack exact measured assignment'):
        audit([row(1, 2, 101)], [winner])
    tight = row(0, 1, 100, budget=.05)
    with pytest.raises(ValueError, match='no feasible measured design'):
        audit([tight], [winner])


def test_training_loader_never_parses_heldout_labels(tmp_path):
    dataset = tmp_path / 'rows.jsonl'
    dataset.write_text(json.dumps(row(0, 1, 100)) + '\n' + 'not valid heldout JSON\n')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train_jsonl_idx=[0], val_jsonl_idx=[1], test_jsonl_idx=[])))
    assert [r['_jsonl_idx'] for r in load_training_rows(dataset, split, sha(dataset))] == [0]
    with pytest.raises(ValueError, match='SHA256 changed'):
        load_training_rows(dataset, split, 'bad')
    split.write_text(json.dumps(dict(train_jsonl_idx=[0], val_jsonl_idx=[0], test_jsonl_idx=[])))
    with pytest.raises(ValueError, match='overlaps'):
        load_training_rows(dataset, split, sha(dataset))
