"""Semantic safeguards for the separately versioned recovery experiments."""
from types import SimpleNamespace

import pytest
import torch

from experiments.stage1_recovery import decisions
from experiments.stage1_recovery.run import remove_delimiter_supervision
from experiments.stage1_recovery.evaluate import transition_metrics
from experiments.stage1_recovery.prepare import prepare, sha
import json
from LLM_branch.train import train_SFT_xattn_new as production


class CharTokenizer:
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(c) for c in text]}


def registry():
    return {'toy': {'AUTO{_PIPE_L3}': ['0', '1'], 'AUTO{_UNROLL_L3}': ['0', '2'],
                    'AUTO{_ARRAY_T_L9}': ['none', 'complete', 'cyclic', 'block'],
                    'AUTO{_ARRAY_F_L9}': ['0', '2', '4'], 'AUTO{_ARRAY_D_L9}': ['0', '1']}}


def source():
    return ('L3:\nauto{_PIPE_L3}\nauto{_UNROLL_L3}\n'
            'L9:\nauto{_ARRAY_T_L9}\nauto{_ARRAY_F_L9}\nauto{_ARRAY_D_L9}\n')


def test_joint_choices_keep_positive_pipe_and_unroll_legal():
    choices = decisions.legal_actions('toy', ['auto{_PIPE_L3}', 'auto{_UNROLL_L3}'], registry())
    assert len(choices) == 4
    assert {'AUTO{_PIPE_L3}': '1', 'AUTO{_UNROLL_L3}': '2'} in choices


def test_joint_arrays_use_existing_tuple_semantics():
    fields = ['auto{_ARRAY_T_L9}', 'auto{_ARRAY_F_L9}', 'auto{_ARRAY_D_L9}']
    choices = decisions.legal_actions('toy', fields, registry())
    assert len(choices) == 6
    for choice in choices:
        decisions.mailohls_contract.validate_directive_assignments(choice)
    assert not any(c['AUTO{_ARRAY_T_L9}'] == 'complete' and c['AUTO{_ARRAY_F_L9}'] != '0' for c in choices)


def test_sparse_action_labels_are_not_renumbered():
    assert [label for label, fields in decisions.action_plan(source())] == ['L3', 'L9']


def test_action_score_has_gradients_only_at_value_prediction_positions():
    tok = CharTokenizer()
    ids, positions = decisions.encode_action(tok, ['auto{_PIPE_L3}', 'auto{_UNROLL_L3}'],
                        {'AUTO{_PIPE_L3}': '1', 'AUTO{_UNROLL_L3}': '2'})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(256, 128))

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.logits[:input_ids.shape[1]][None])

    model = Model()
    score = decisions.action_score(model, [1, 2], ids, positions)
    score.backward()
    nonzero = model.logits.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert nonzero == [2 + i - 1 for i in positions]
    assert float(score.detach()) == pytest.approx(-len(positions) * torch.log(torch.tensor(128.)).item())
    assert all(ids[i] != ord('\n') for i in positions)


def test_value_only_loss_preserves_context_and_site_mass():
    tok = CharTokenizer()
    row = dict(input=source(), kernel_name='toy', target=(
        'auto{_PIPE_L3} = 1\nauto{_UNROLL_L3} = 0\nauto{_ARRAY_T_L9} = cyclic\n'
        'auto{_ARRAY_F_L9} = 4\nauto{_ARRAY_D_L9} = 1'))
    pack = production.build_deterministic_rhs_pack(row['input'], row['target'], tok,
                directive_domain_registry=registry(), kernel_name='toy')
    sample = dict(input_ids=torch.tensor([5, 6] + pack.input_ids),
                  labels=torch.tensor([-100, -100] + pack.labels),
                  token_weights=torch.tensor([0., 0.] + pack.token_weights))
    data = SimpleNamespace(samples=[sample])
    before = sample['input_ids'].clone()
    remove_delimiter_supervision(data, [row], tok, registry())
    assert torch.equal(before, sample['input_ids'])
    assert torch.all(sample['labels'][sample['input_ids'] == ord('\n')] == -100)
    assert sample['token_weights'].sum().item() == pytest.approx(4.)
    assert torch.all(sample['token_weights'][sample['labels'] == -100] == 0.)


def test_balanced_metrics_expose_constant_mode_collapse():
    rows = [dict(reference_target=f'auto{{_PIPE_L3}} = {value}\nauto{{_UNROLL_L3}} = 0',
                 prediction='auto{_PIPE_L3} = 0\nauto{_UNROLL_L3} = 0') for value in ['0', '0', '0', '1']]
    result = decisions.metrics(rows)
    assert result['by_kind']['PIPE']['accuracy'] == .75
    assert result['by_kind']['PIPE']['balanced_accuracy'] == .5
    assert result['joint_action_accuracy'] == .75


def test_metrics_reject_missing_action_slots():
    with pytest.raises(ValueError, match='keys differ'):
        decisions.metrics([dict(reference_target='auto{_PIPE_L9} = 1', prediction='auto{_PIPE_L1} = 1')])


def test_batched_action_scores_equal_scalar_scores():
    torch.manual_seed(5)
    tok = CharTokenizer()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(128, 4)
            self.head = torch.nn.Linear(4, 128)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.head(self.emb(input_ids)))

    model = Model().eval()
    fields = ['auto{_PIPE_L3}', 'auto{_UNROLL_L3}']
    choices = decisions.legal_actions('toy', fields, registry())
    batched = decisions.score_actions_batch(model, tok, [1, 2], fields, choices)
    for score, candidate, _ in batched:
        ids, positions = decisions.encode_action(tok, fields, candidate)
        expected = decisions.action_score(model, [1, 2], ids, positions)
        assert score == pytest.approx(float(expected.detach()), abs=1e-6)


def test_decision_metrics_exclude_forced_array_dimensions():
    row = dict(kernel_name='toy', reference_target='auto{_ARRAY_T_L9} = cyclic\nauto{_ARRAY_F_L9} = 2\nauto{_ARRAY_D_L9} = 1',
               prediction='auto{_ARRAY_T_L9} = cyclic\nauto{_ARRAY_F_L9} = 2\nauto{_ARRAY_D_L9} = 1')
    result = decisions.metrics([row], registry())
    assert 'ARRAY_D' in result['by_kind']
    assert 'ARRAY_D' not in result['decision_by_kind']
    assert result['forced_sites'] == 1


def test_budget_sensitivity_is_not_correctness():
    rows = [dict(reference_target=f'auto{{_UNROLL_L3}} = {g}',
                 prediction=f'auto{{_UNROLL_L3}} = {p}') for g, p in [('2', '4'), ('4', '2')]]
    result = transition_metrics(rows)
    assert result['sensitive_groups'] == 1
    assert result['correct_groups'] == 0


def test_constant_baseline_roundtrips_canonical_uppercase_keys():
    row = dict(input=source(), kernel_name='toy', target=(
        'auto{_PIPE_L3} = 0\nauto{_UNROLL_L3} = 0\nauto{_ARRAY_T_L9} = cyclic\n'
        'auto{_ARRAY_F_L9} = 2\nauto{_ARRAY_D_L9} = 1'))
    modes = decisions.modes_from_train([row])
    pred = decisions.constant_baseline(row, registry(), modes)
    result = decisions.metrics([dict(kernel_name='toy', reference_target=row['target'], prediction=pred)], registry())
    assert result['exact_design_accuracy'] == 1.


def test_prepared_cache_authenticates_parent_and_export_bytes(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    (parent / 'training_contract.json').write_text('{}')
    out = tmp_path / 'selection'
    out.mkdir()
    (out / 'train.jsonl').write_text('sample')
    manifest = dict(parent_contract_sha256=sha(parent / 'training_contract.json'),
                    files={'train.jsonl': sha(out / 'train.jsonl')})
    (out / 'manifest.json').write_text(json.dumps(manifest))
    assert prepare(parent, out) == manifest
    (out / 'train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Selected data changed'):
        prepare(parent, out)
