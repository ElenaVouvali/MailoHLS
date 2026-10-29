"""Summarize completed controls and label their metadata as experimental."""
import argparse
import json
from pathlib import Path

from experiments.stage1_recovery.run import DEFAULT_PARENT, dump, sha
from experiments.stage1_recovery import decisions
from experiments.stage1_recovery.prepare import DEFAULT_OUTPUT, prepare
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference


def full_saved_baselines():
    """Re-score all 62 stored validation outputs, not fresh model inference."""
    prepare(DEFAULT_PARENT)
    sources = json.loads(Path('artifacts/llm/mailohls_sft.sources.json').read_text())['templates']
    selected = {name: [json.loads(line) for line in (DEFAULT_OUTPUT / (name + '_reconstructed_selected.jsonl')).read_text().splitlines()]
                for name in ['train', 'val']}
    registry = inference.load_directive_domain_registry(str(Path(DEFAULT_PARENT).parent / 'directive_domain_registry.json'))
    modes = decisions.modes_from_train(selected['train'])
    lookup = {(r['_jsonl_idx'], r['resource_budget_id'], r['clock_period']): r for r in selected['val']}
    saved = json.loads((Path(DEFAULT_PARENT) / 'best_selection_metrics.json').read_text())['rows']
    original, baseline = [], []
    for old in saved:
        row = dict(lookup[(old['jsonl_idx'], old['resource_budget_id'], old['selected_clock_period'])])
        row['input'] = sources[row['source_key']]
        original.append(dict(kernel_name=old['kernel_name'], reference_target=old['reference_target'], prediction=old['prediction']))
        baseline.append(dict(kernel_name=old['kernel_name'], reference_target=old['reference_target'],
                             prediction=decisions.constant_baseline(row, registry, modes)))
    return dict(schema='mailohls-stage1-full-saved-validation-baselines-v1', cases=len(saved),
                saved_checkpoint=decisions.metrics(original, registry), constant_baseline=decisions.metrics(baseline, registry),
                training_modes=modes, new_gpu_inference=False)


def normalize_metadata(run):
    path = run / 'adapter/training_contract.json'
    contract = json.loads(path.read_text())
    if contract['schema'] != 'mailohls-training-contract-v1':
        return
    assert contract['experimental_recovery']['parent'] == DEFAULT_PARENT
    backup = run / 'initial_inherited_training_contract.json'
    assert not backup.exists()
    backup.write_bytes(path.read_bytes())
    weights_before = sha(run / 'adapter/adapter_model.safetensors')
    keys = ['model', 'model_revision', 'tokenizer', 'tokenizer_size', 'special_tokens',
            'special_token_ids', 'special_token_to_id', 'embedding_weights_tied', 'objective',
            'prompt_schema_version', 'device_mode', 'experimental_recovery']
    normalized = {key: contract[key] for key in keys}
    normalized.update(schema='mailohls-stage1-recovery-adapter-v1', stage='stage1_experimental')
    dump(path, normalized)
    assert weights_before == sha(run / 'adapter/adapter_model.safetensors')
    report = json.loads((run / 'report.json').read_text())
    report['metadata_normalization'] = dict(original_contract_sha256=sha(backup),
        normalized_contract_sha256=sha(path), weights_unchanged=True)
    dump(run / 'report.json', report)


def finalize_full_run(run, prepared, validation_report=None):
    """Read-only model assessment; never rewrites contracts or promotes weights."""
    run, prepared = Path(run), Path(prepared)
    manifest = prepare(DEFAULT_PARENT, prepared)
    protocol = json.loads((run/'recovery_protocol.json').read_text())
    if (run/'training_contract.json').is_file():
        contract=json.loads((run/'training_contract.json').read_text())
        sealed=(contract.get('stage1_supervision') or {}).get('recovery_protocol_sha256')
        if sealed and sealed != sha(run/'recovery_protocol.json'):
            raise ValueError('Recovery protocol differs from the training contract')
    if protocol['input_manifest_sha256'] != sha(prepared/'manifest.json'):
        raise ValueError('Recovery selection manifest changed')
    for name,digest in protocol['reference_hashes'].items():
        if sha(Path(DEFAULT_PARENT)/name)!=digest:
            raise ValueError(f'Historical checkpoint changed: {name}')
    for name,digest in protocol['implementation_sha256'].items():
        if sha(run/'source_snapshot'/name)!=digest:
            raise ValueError(f'Frozen experiment implementation changed: {name}')
    rows = [json.loads(line) for line in (prepared/'train_reconstructed_selected.jsonl').read_text().splitlines()]
    result = dict(schema='stage1-full-recovery-assessment-v1',test_labels_used=False,
        examples=manifest['counts'],distinct_training_measured_rows=len({r['_jsonl_idx'] for r in rows}),
        protocol_sha256=sha(run/'recovery_protocol.json'),historical_checkpoint_unchanged=True,
        automatic_promotion=False)
    if not (run/'adapter_model.safetensors').is_file():
        result.update(status='training_not_completed',classification='CONDITIONAL',
                      reason='Final root adapter not saved; best/checkpoint subdirectories alone do not establish completion')
        return result
    records = json.loads((run/'optimizer_audit.json').read_text())
    updates = [r for r in records if 'optimizer_step_skipped' in r]
    result.update(status='training_completed_validation_pending',classification='CONDITIONAL',
        actual_updates=sum(not r['optimizer_step_skipped'] for r in updates),
        skipped_updates=sum(r['optimizer_step_skipped'] for r in updates),
        attempted_updates=max(r['step'] for r in updates),
        schedule=json.loads((run/'schedule_audit.json').read_text()),
        final_adapter_sha256=sha(run/'adapter_model.safetensors'))
    best = json.loads((run/'best_custom_stage1/best_selection_metrics.json').read_text())
    result['selected_best'] = dict(step=best['step'],decision_key=best['decision_key'],
        report=best.get('decision_report'),constant_baseline=best.get('constant_baseline_report'))
    result['validation_curve'] = [dict(step=r['step'],ce=r['eval_loss'],
        core_balanced=r.get('decision_report',{}).get('core_decision_balanced_accuracy'),
        complete_actions=r.get('decision_report',{}).get('complete_action_accuracy'))
        for r in (json.loads(path.read_text()) for path in sorted(run.glob('val_selection_step_*.json')))]
    result['validation_curve'].sort(key=lambda r:r['step'])
    # Re-score the original and selected checkpoint on the exact same saved bank.
    from experiments.stage1_recovery.evaluate import evaluate_saved
    val = [json.loads(line) for line in (prepared/'val_reconstructed_selected.jsonl').read_text().splitlines()]
    sources=json.loads(Path('artifacts/llm/mailohls_sft.sources.json').read_text())['templates']
    review=run/'saved_decision_review'
    review.mkdir(exist_ok=True)
    comparison=evaluate_saved([str(Path(DEFAULT_PARENT)/'best_selection_metrics.json'),
        str(run/'best_custom_stage1/best_selection_metrics.json')],rows,val,sources,review)
    reports=list(comparison['checkpoints'].values())
    result['saved_bank_promotion']=decisions.promotion_assessment(reports[1]['report'],reports[0]['report'],reports[1]['constant_baseline'])
    if validation_report:
        full=json.loads(Path(validation_report).read_text())
        if full['cases'] != len(val):
            raise ValueError('Full validation report does not cover the selected validation split')
        result['full_validation_report_sha256']=sha(validation_report)
        result['full_validation']=full
        comparisons=list(full['adapters'].values())
        if len(comparisons)==2:
            expected=[Path(DEFAULT_PARENT),run/'best_custom_stage1']
            for record,path in zip(comparisons,expected):
                if Path(record['path']).resolve()!=path.resolve() or record['adapter_sha256']!=sha(path/'adapter_model.safetensors'):
                    raise ValueError('Validation report adapter lineage does not match the reference/recovery pair')
            gate=decisions.promotion_assessment(comparisons[1]['field_greedy'],comparisons[0]['field_greedy'],full['baseline'])
            result['full_validation_promotion']=gate
            result['status']='evaluation_completed'
            checks=gate['checks']
            action_gain=(checks['complete_action_better_than_reference'] and checks['complete_action_better_than_constant'])
            result['classification']=('VALIDATED RECOVERY' if gate['eligible_on_measured_decisions'] else
                                      'CONDITIONAL' if action_gain else 'NO IMPROVEMENT')
    dump(run/'full_run_report.json',result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/diagnostics/stage1_recovery_20261028')
    ap.add_argument('--full-run', help='Assess a fresh full-data run without normalizing historical control metadata')
    ap.add_argument('--prepared', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--full-validation-report')
    args = ap.parse_args()
    if args.full_run:
        print(json.dumps(finalize_full_run(args.full_run,args.prepared,args.full_validation_report),indent=2))
        return
    root = Path(args.root)
    result = dict(schema='mailohls-stage1-recovery-summary-v1', training={}, heldout_test_used=False)
    initial, selections = set(), set()
    for name in ['ce_control', 'ce_lr_control', 'decision_ce', 'joint_choice']:
        run = root / name
        normalize_metadata(run)
        report = json.loads((run / 'report.json').read_text())
        assert report['parent_unchanged'] and report['trainables_changed'] and report['elapsed_seconds'] > 0
        selections.add(sha(run / 'selected_train.json'))
        initial.add(report['initial_trainable_hashes']['combined_sha256'])
        generated = json.loads((run / 'train_greedy.json').read_text())
        result['training'][name] = dict(
            settings=report['settings'], actual_updates=report['actual_updates'],
            skipped_updates=report['settings']['steps'] - report['actual_updates'],
            teacher_forced_before=report['initial_train_teacher'], teacher_forced_after=report['final_train_teacher'],
            four_training_case_field_greedy=decisions.metrics([
                dict(row, prediction=row['candidates'][0]['prediction']) for row in generated]),
            adapter_sha256=sha(run / 'adapter/adapter_model.safetensors'), elapsed_seconds=report['elapsed_seconds'])
    assert len(initial) == len(selections) == 1
    result['identical_initial_trainables_and_selected_examples'] = True
    result['selected_examples_sha256'] = next(iter(selections))
    parent = Path(DEFAULT_PARENT)
    first = json.loads((root / 'ce_control/report.json').read_text())['parent_hashes']
    assert first == {name: sha(parent / name) for name in first}
    result['parent_hashes_unchanged'] = first
    result['validation'] = json.loads((root / 'validation/report.json').read_text())
    assert len(result['validation']['adapters']) == 4
    validation = result['validation']
    validation['baseline']['core_decision_balanced_accuracy'] = decisions.core_balanced(validation['baseline'])
    result['full_training_followup_candidates'] = []
    for index, record in validation['adapters'].items():
        for method in ['field_greedy', 'joint_greedy']:
            record[method]['core_decision_balanced_accuracy'] = decisions.core_balanced(record[method])
        record['core_balanced_gain'] = (record['joint_greedy']['core_decision_balanced_accuracy']
                                       - validation['baseline']['core_decision_balanced_accuracy'])
        if record['pilot_gate']['passed'] and record['core_balanced_gain'] > 0:
            result['full_training_followup_candidates'].append(record['path'])
    result['production_checkpoint_promoted'] = False
    result['full_saved_validation_baselines'] = full_saved_baselines()
    dump(root / 'full_saved_validation_baselines.json', result['full_saved_validation_baselines'])
    dump(root / 'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
