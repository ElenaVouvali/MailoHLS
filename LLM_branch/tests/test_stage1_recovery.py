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
    domains = registry()
    domains['toy']['AUTO{_UNROLL_L3}'] = ['0', '2', '16']
    choices = decisions.legal_actions('toy', fields, domains)
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


def test_trivial_dimension_cannot_inflate_core_score():
    kinds = {k: {'balanced_accuracy': .25} for k in ['PIPE', 'UNROLL', 'ARRAY_T', 'ARRAY_F']}
    kinds['ARRAY_D'] = {'balanced_accuracy': 1.}
    assert decisions.core_balanced({'decision_by_kind': kinds}) == .25


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


from LLM_branch.train import stage1_decision_loss as decision_loss


def _decision_row():
    return dict(input=source(), kernel_name='toy', target=(
        'auto{_PIPE_L3} = 1\nauto{_UNROLL_L3} = 2\nauto{_ARRAY_T_L9} = cyclic\n'
        'auto{_ARRAY_F_L9} = 4\nauto{_ARRAY_D_L9} = 1'), _family='toy', _jsonl_idx=0)


def test_legacy_pack_is_identical_and_decisions_keep_context():
    row, tok = _decision_row(), CharTokenizer()
    common = dict(directive_domain_registry=registry(), kernel_name='toy')
    legacy = production.build_deterministic_rhs_pack(row['input'], row['target'], tok, **common)
    explicit = production.build_deterministic_rhs_pack(row['input'], row['target'], tok, supervision_policy='legacy', **common)
    new = production.build_deterministic_rhs_pack(row['input'], row['target'], tok, supervision_policy='decision', **common)
    assert legacy == explicit
    assert legacy.input_ids == new.input_ids
    assert sum(new.token_weights) == pytest.approx(1.)
    assert all(y == -100 and w == 0 for x,y,w in zip(new.input_ids,new.labels,new.token_weights) if x==ord('\n'))
    mass = {}
    for key, weight in zip(new.site_keys,new.token_weights):
        mass[key] = mass.get(key,0)+weight
    assert mass['AUTO{_PIPE_L3}'] == pytest.approx(.25)
    assert mass['AUTO{_UNROLL_L3}'] == pytest.approx(.25)
    assert mass['AUTO{_ARRAY_T_L9}'] == pytest.approx(.25)  # six value characters, still equal field mass
    assert mass['AUTO{_ARRAY_F_L9}'] == pytest.approx(.25)
    assert mass['AUTO{_ARRAY_D_L9}'] == 0  # forced dimension


def test_actions_with_different_nonforced_field_counts_have_equal_mass():
    row = _decision_row()
    row['target'] = row['target'].replace('= cyclic','= complete').replace('_ARRAY_F_L9} = 4','_ARRAY_F_L9} = 0')
    pack = production.build_deterministic_rhs_pack(row['input'],row['target'],CharTokenizer(),
        directive_domain_registry=registry(),kernel_name='toy',supervision_policy='decision')
    masses = {label: sum(w for key,w in zip(pack.site_keys,pack.token_weights) if key and label in key) for label in ['L3}', 'L9}']}
    assert masses == pytest.approx({'L3}':.5,'L9}':.5})


def test_alignment_fails_on_a_merged_value_delimiter_token():
    class Merged:
        def __call__(self,text,add_special_tokens=False,**kwargs):
            return dict(input_ids=[1],offset_mapping=[(0,len(text))]) if kwargs else dict(input_ids=[1])
    with pytest.raises(ValueError,match='merges'):
        decision_loss._value_positions(Merged(),'16',[1])


def test_balancing_counts_measurements_once_and_has_bounded_mean_one():
    a = _decision_row()
    b = dict(a,_jsonl_idx=1,target=a['target'].replace('_UNROLL_L3} = 2','_UNROLL_L3} = 0'))
    rows = [a,dict(a,resource_budget_id='repeat'),b]
    fit = decision_loss.fit_value_weights(rows,registry(),production.constrained_rhs_candidates,decisions.parse)
    assert fit['distinct_measured_rows']==2
    assert fit['counts']['UNROLL']=={'2':1,'0':1}
    assert 'ARRAY_D' not in fit['counts']
    for kind,values in fit['counts'].items():
        weights=fit['weights'][kind]
        assert all(.5<=w<=2 for w in weights.values())
        assert sum(values[v]*weights[v] for v in values)/sum(values.values()) == pytest.approx(1.)
    pack=production.build_deterministic_rhs_pack(a['input'],a['target'],CharTokenizer(),kernel_name='toy',
         directive_domain_registry=registry(),supervision_policy='decision_balanced',decision_value_weights=fit['weights'])
    assert sum(pack.token_weights)==pytest.approx(1.)
    with pytest.raises(ValueError,match='training-only'):
        production.build_deterministic_rhs_pack(a['input'],a['target'],CharTokenizer(),kernel_name='toy',
             directive_domain_registry=registry(),supervision_policy='decision_balanced')


def _context(row, budget, idx=0, clock=10., device='zcu'):
    return dict(row,device=device,selected_clock_period=clock,obj_mode='PARETO_LATENCY',source_key='toy',
        _jsonl_idx=idx,avail_bram=budget,avail_dsp=budget,avail_ff=budget,avail_lut=budget,
        budget_frac_bram_18k=budget/100,budget_frac_dsp=budget/100,budget_frac_ff=budget/100,budget_frac_lut=budget/100)


def test_contrasts_distinguish_changed_invariant_and_exact_contexts():
    from experiments.stage1_recovery.prepare import target_contrasts
    row=_decision_row()
    changed=dict(row,target=row['target'].replace('_UNROLL_L3} = 2','_UNROLL_L3} = 0'))
    rows=[_context(row,100),_context(changed,50,1),_context(changed,25,1)]
    pairs=target_contrasts(rows)
    assert sum(p['classification']=='changed' for p in pairs)==2
    assert sum(p['classification']=='invariant' for p in pairs)==1
    assert not next(p for p in pairs if p['classification']=='invariant')['distinct_measurements']
    assert target_contrasts([rows[0],dict(rows[0])])==[]
    before=[r['target'] for r in rows]
    info=decision_loss.emphasize_budget_contrasts(rows,1.25)
    assert info['distinct_measured_rows']==2
    assert sum(r['_sample_weight'] for r in rows)==pytest.approx(len(rows))
    assert min(r['_sample_weight'] for r in rows)>0
    assert before==[r['target'] for r in rows]


def test_clock_contrasts_require_fixed_absolute_budget_and_device_full_context():
    from experiments.stage1_recovery.prepare import target_contrasts
    row=_decision_row()
    a,b=_context(row,100),_context(row,100,1,clock=5.)
    assert [p['kind'] for p in target_contrasts([a,b])]==['clock']
    b['avail_lut']=99
    assert not target_contrasts([a,b])
    b=_context(row,100,1,device='u200')
    assert [p['kind'] for p in target_contrasts([a,b])]==['device']


def test_decision_report_does_not_reward_wrong_invariance_or_swapped_changes():
    a=_context(_decision_row(),100)
    b=_context(dict(a,target=a['target'].replace('_UNROLL_L3} = 2','_UNROLL_L3} = 0')),50,1)
    rows=[dict(a,reference_target=a['target'],prediction=b['target']),dict(b,reference_target=b['target'],prediction=a['target'])]
    result=decisions.decision_report(rows,registry())
    transition=result['target_transitions']['budget']
    assert transition['changed_pairs']==1 and transition['changed_complete_correct']==0
    assert transition['changed_fields_both_endpoints_correct']==0
    assert result['forced_accuracy']==1.
    assert result['complete_loop_action_accuracy']==0.


def test_decision_loss_has_finite_gradients_without_new_prediction_head():
    row=_decision_row()
    pack=production.build_deterministic_rhs_pack(row['input'],row['target'],CharTokenizer(),kernel_name='toy',
         directive_domain_registry=registry(),supervision_policy='decision')
    logits=torch.zeros(len(pack.input_ids),128,requires_grad=True)
    labels=torch.tensor(pack.labels)
    loss=(torch.nn.functional.cross_entropy(logits[:-1],labels[1:],ignore_index=-100,reduction='none')*torch.tensor(pack.token_weights[1:])).sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    expected=[i-1 for i,w in enumerate(pack.token_weights) if w>0]
    assert logits.grad.abs().sum(-1).nonzero().flatten().tolist()==expected


def test_serialization_cache_changes_no_values():
    row=_decision_row()
    expected=production.canonical_completion_key(row['input'],row['target'])
    production.cache_stage1_serialization()
    assert production.canonical_completion_key(row['input'],row['target'])==expected
    assert production.canonical_completion_key(row['input'],row['target'])==expected
    assert production.canonical_completion_key.cache_info().hits>=1


def test_training_update_audit_observes_skips_and_scheduler(tmp_path):
    audit=decision_loss.UpdateAudit(tmp_path)
    audit.trainer=SimpleNamespace(accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
        lr_scheduler=SimpleNamespace(last_epoch=1),optimizer=SimpleNamespace(param_groups=[{'lr':1e-4}]))
    args=SimpleNamespace(num_train_epochs=2,gradient_accumulation_steps=8,warmup_steps=27)
    state=SimpleNamespace(global_step=0,max_steps=546)
    audit.on_train_begin(args,state,None,train_dataloader=list(range(2187)))
    state.global_step=1
    audit.on_step_end(args,state,None)
    audit.trainer.accelerator.optimizer_step_was_skipped=True
    state.global_step=2
    audit.on_step_end(args,state,None)
    saved=json.loads((tmp_path/'optimizer_audit.json').read_text())
    assert saved[-1]['actual_updates']==1
    assert saved[-1]['scheduler_last_epoch']==1
    assert saved[-1]['optimizer_step_skipped']
    schedule=json.loads((tmp_path/'schedule_audit.json').read_text())
    assert schedule['trainer_max_steps']==546 and schedule['dataloader_batches']==2187


def test_promotion_refuses_same_length_different_contexts():
    row=_context(_decision_row(),100)
    left=decisions.decision_report([dict(row,reference_target=row['target'],prediction=row['target'])],registry())
    changed=dict(row,device='other')
    right=decisions.decision_report([dict(changed,reference_target=row['target'],prediction=row['target'])],registry())
    with pytest.raises(ValueError,match='identical validation'):
        decisions.promotion_assessment(left,right,left)


def test_stage1_launcher_explicit_objective_and_boolean_overrides(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    executable=tmp_path/'fake_python'
    executable.write_text('#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    executable.chmod(0o755)
    capture=tmp_path/'arguments.json'
    root=Path(__file__).resolve().parents[2]
    env=dict(os.environ,PYTHON=str(executable),CAPTURE=str(capture),OUT=str(tmp_path/'new'),
             OBJECTIVE='PARETO_LATENCY',STAGE1_EVAL_ON_START='0',STAGE1_SUPERVISION_POLICY='decision',
             STAGE1_BUDGET_INVARIANT_MAX_DUPLICATES='-1')
    subprocess.run(['bash','scripts/training/train_stage1.sh'],cwd=root,env=env,check=True)
    argv=json.loads(capture.read_text())
    assert '--no-eval_on_start' in argv and '--eval_on_start' not in argv
    assert argv[argv.index('--stage1_supervision_policy')+1]=='decision'
    assert argv[argv.index('--budget_invariant_max_duplicates')+1]=='-1'
    assert argv[argv.index('--lr_lora')+1]=='1.5e-5'
    assert '--init_adapter_dir' not in argv


def test_contrast_emphasis_requires_a_measured_feasibility_boundary():
    a=_context(_decision_row(),100)
    b=_context(dict(a,target=a['target'].replace('_UNROLL_L3} = 2','_UNROLL_L3} = 0')),50,1)
    a.update({'bram_util_%':0,'dsp_util_%':10,'ff_util_%':20,'lut_util_%':20})
    b.update({'bram_util_%':0,'dsp_util_%':10,'ff_util_%':10,'lut_util_%':10})
    rows=[a,b]
    info=decision_loss.emphasize_budget_contrasts(rows,1.25)
    assert info['changed_budget_participants']==0  # changed winner, both fit both budgets
    assert info['both_winners_cross_feasible_pairs']==1
    a['lut_util_%']=80
    info=decision_loss.emphasize_budget_contrasts(rows,1.25)
    assert info['changed_budget_participants']==2
    assert info['feasibility_boundary_pairs']==1


def test_full_run_finalizer_does_not_mistake_intermediate_best_for_completion(tmp_path,monkeypatch):
    from experiments.stage1_recovery import finalize
    parent=tmp_path/'parent';parent.mkdir();(parent/'weights').write_text('unchanged')
    prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'manifest.json').write_text('{}')
    (prepared/'train_reconstructed_selected.jsonl').write_text('{"_jsonl_idx": 1}\n')
    run=tmp_path/'run';run.mkdir();(run/'best_custom_stage1').mkdir()
    protocol=dict(input_manifest_sha256=sha(prepared/'manifest.json'),reference_hashes={'weights':sha(parent/'weights')},implementation_sha256={})
    (run/'recovery_protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr(finalize,'prepare',lambda *args:dict(counts={'train':1,'val':1}))
    monkeypatch.setattr(finalize,'DEFAULT_PARENT',str(parent))
    result=finalize.finalize_full_run(run,prepared)
    assert result['status']=='training_not_completed'
    assert result['classification']=='CONDITIONAL'
    assert not result['automatic_promotion']


def test_optimizer_audit_resume_preserves_abandoned_receipts_without_double_count(tmp_path):
    prior=[dict(step=1,optimizer_step_skipped=False),dict(step=2,optimizer_step_skipped=False)]
    (tmp_path/'optimizer_audit.json').write_text(json.dumps(prior))
    audit=decision_loss.UpdateAudit(tmp_path)
    args=SimpleNamespace(num_train_epochs=2,gradient_accumulation_steps=8,warmup_steps=1)
    state=SimpleNamespace(global_step=1,max_steps=3)
    audit.on_train_begin(args,state,None)
    assert audit.completed==1
    assert [r['step'] for r in audit.records]==[1]
    assert len(list(tmp_path.glob('optimizer_audit_before_resume_*.json')))==1


def test_recovery_freeze_rejects_changed_selection_before_loading_model(tmp_path,monkeypatch):
    from experiments.stage1_recovery import prepare as preparation
    prepared=tmp_path/'prepared';prepared.mkdir()
    row=dict(_context(_decision_row(),100),resource_budget_id='full')
    for split in ['train','val']:
        (prepared/f'{split}_reconstructed_selected.jsonl').write_text(json.dumps(row)+'\n')
    monkeypatch.setattr(preparation,'prepare',lambda *args:{})
    args=SimpleNamespace(disable_structural_memory=True,init_adapter_dir='',resume_from_checkpoint='')
    changed=dict(row,target=row['target'].replace('_UNROLL_L3} = 2','_UNROLL_L3} = 0'))
    with pytest.raises(ValueError,match='train selection differs'):
        preparation.freeze_full_run(tmp_path/'run',prepared,[changed],[row],args)
    assert not (tmp_path/'run/recovery_protocol.json').exists()
