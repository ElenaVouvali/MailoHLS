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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/diagnostics/stage1_recovery_20261028')
    args = ap.parse_args()
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
