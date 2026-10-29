"""Validation-only pilot with legal constant baselines and budget pairs."""
from __future__ import annotations

import argparse
import collections
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from experiments.stage1_recovery import decisions
from experiments.stage1_recovery.run import AUDIT, DEFAULT_PARENT, dump, evaluate_greedy, sha
from experiments.stage1_recovery.prepare import prepare


def validation_pairs(rows, sources):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['kernel_name'], row['device'], row['clock_period'])].append(row)
    chosen = []
    for index, kernel in enumerate(sorted({r['kernel_name'] for r in rows})):
        options = []
        for (name, device, clock), group in sorted(groups.items()):
            if name != kernel:
                continue
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if a['target'] != b['target'] and a['resource_budget_id'] != b['resource_budget_id']:
                        # Alternate preferred devices and retain a shared clock.
                        preferred = sorted({r['device'] for r in rows})[index % 2]
                        options.append(((device == preferred, clock == 10.,
                                         sum(a.get('budget_frac_' + k, 1.) + b.get('budget_frac_' + k, 1.)
                                             for k in ['bram_18k', 'dsp', 'ff', 'lut'])), a, b))
        if not options:
            raise ValueError(f'No distinct-target budget pair available: {kernel}')
        _, a, b = max(options, key=lambda x: x[0])
        chosen.extend(dict(r, input=sources[r['source_key']]) for r in [a, b])
    return chosen


def transition_metrics(rows):
    groups = 0
    correct = sensitive = 0
    for a, b in zip(rows[::2], rows[1::2]):
        ga, gb = decisions.parse(a['reference_target']), decisions.parse(b['reference_target'])
        pa, pb = decisions.parse(a['prediction']), decisions.parse(b['prediction'])
        changed = [k for k in ga if ga[k] != gb[k]]
        assert changed
        groups += 1
        sensitive += any(pa[k] != pb[k] for k in changed)
        correct += all(pa[k] == ga[k] and pb[k] == gb[k] for k in changed)
    return dict(groups=groups, sensitive_groups=sensitive, correct_groups=correct,
                exact_changed_field_transition_accuracy=correct / groups)


def summarize(rows, registry):
    return dict(**decisions.metrics(rows, registry), budget_transitions=transition_metrics(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adapters', nargs='+', default=[DEFAULT_PARENT])
    ap.add_argument('--output', required=True)
    ap.add_argument('--joint', action='store_true')
    ap.add_argument('--prepared', default=str(AUDIT))
    ap.add_argument('--full-validation', action='store_true', help='All selected validation contexts via production callback')
    ap.add_argument('--saved-metrics', nargs='+', help='CPU-only reanalysis of saved selection predictions on identical cases')
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.perf_counter()
    prepared = Path(args.prepared)
    prepare(DEFAULT_PARENT, prepared)
    sources = json.loads(Path('artifacts/llm/mailohls_sft.sources.json').read_text())['templates']
    split = json.loads(Path('mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    all_train = [json.loads(l) for l in (prepared / 'train_reconstructed_selected.jsonl').read_text().splitlines()]
    all_val = [json.loads(l) for l in (prepared / 'val_reconstructed_selected.jsonl').read_text().splitlines()]
    if args.saved_metrics:
        evaluate_saved(args.saved_metrics, all_train, all_val, sources, out)
        return
    rows = ([dict(r,input=sources[r['source_key']]) for r in all_val]
            if args.full_validation else validation_pairs(all_val, sources))
    assert {r['_jsonl_idx'] for r in rows} <= set(split['val_jsonl_idx'])
    modes = decisions.modes_from_train(all_train)
    registry = inference.load_directive_domain_registry(str(Path(DEFAULT_PARENT).parent / 'directive_domain_registry.json'))
    baseline = [dict(kernel_name=r['kernel_name'], reference_target=r['target'],
                     prediction=decisions.constant_baseline(r, registry, modes)) for r in rows]
    result = dict(schema='mailohls-stage1-recovery-validation-pilot-v1', heldout_test_used=False,
                  cases=len(rows), kernels=sorted({r['kernel_name'] for r in rows}),
                  evaluation_seed=123 if args.full_validation else None,
                  dataset_sha256=sha('artifacts/llm/mailohls_sft.jsonl'),
                  prepared_manifest_sha256=sha(prepared/'manifest.json'),
                  baseline=(decisions.decision_report([
                      dict(r,reference_target=r['target'],prediction=b['prediction']) for r,b in zip(rows,baseline)], registry)
                      if args.full_validation else summarize(baseline, registry)), adapters={}, train_modes=modes)
    dump(out / 'cases.json', [{k: v for k, v in r.items() if k != 'input'} for r in rows])
    for index, path in enumerate(args.adapters):
        root = Path(path)
        contract = json.loads((root / 'training_contract.json').read_text())
        if args.full_validation:
            from transformers import set_seed
            set_seed(123)  # identical initialization of any frozen resized rows
        tok = inference.load_checkpoint_tokenizer(str(root), contract)
        model = inference.load_stage_model(SimpleNamespace(model=contract['model'],
            model_revision=contract['model_revision'], no_4bit=False, device_map='auto', stage='stage1',
            embedding_weights_tied=False, lora_adapter_dir='', adapter_dir=str(root)), tok)
        model.eval()
        if args.full_validation:
            from LLM_branch.train import train_SFT_xattn_new as production
            cases = [production.SelectionCase(r['kernel_name'],r['obj_mode'],r['input'],r['target'],r) for r in rows]
            callback = production.StageValSelectionCallback(tok,cases,registry,str(out/f'callback_{index}'),
                max_prompt_tokens=contract['max_length'],candidate_batch_size=4)
            generated = [callback._run_case(model,case) for case in cases]
            ordinary_report = decisions.decision_report(generated,registry)
            record = dict(path=str(root),adapter_sha256=sha(root/'adapter_model.safetensors'),
                          field_greedy=ordinary_report)
            dump(out/f'adapter_{index}_field_greedy.json',generated)
            result['adapters'][str(index)] = record
            if index>0:
                reference = result['adapters']['0']['field_greedy']
                record['promotion'] = decisions.promotion_assessment(ordinary_report,reference,result['baseline'])
            dump(out/'report.json',result)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        generated = evaluate_greedy(model, tok, rows, registry)
        ordinary = [dict(r, prediction=r['candidates'][0]['prediction']) for r in generated]
        record = dict(path=str(root), adapter_sha256=sha(root / 'adapter_model.safetensors'),
                      field_greedy=summarize(ordinary, registry))
        dump(out / f'adapter_{index}_field_greedy.json', generated)
        if args.joint:
            joint = [decisions.joint_decode(model, tok, row, registry) for row in rows]
            dump(out / f'adapter_{index}_joint_greedy.json', joint)
            record['joint_greedy'] = summarize(joint, registry)
        score = record.get('joint_greedy', record['field_greedy'])
        base = result['baseline']
        # This gate defines evidence needed before promoting a pilot. It is not
        # a synthesis/QoR acceptance criterion or a claim of statistical power.
        record['pilot_gate'] = dict(
            exceeds_constant_balanced_accuracy=score['decision_macro_kind_balanced_accuracy'] > base['decision_macro_kind_balanced_accuracy'],
            exceeds_constant_joint_accuracy=score['joint_action_accuracy'] > base['joint_action_accuracy'],
            has_correct_budget_transition=score['budget_transitions']['correct_groups'] > 0)
        record['pilot_gate']['passed'] = all(record['pilot_gate'].values())
        result['adapters'][str(index)] = record
        result['elapsed_seconds'] = time.perf_counter() - start
        dump(out / 'report.json', result)
        print('EVALUATED', json.dumps(record), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    dump(out / 'report.json', result)


def evaluate_saved(paths, train, val, sources, output):
    """Reanalyze saved predictions without loading any model or test labels."""
    import hashlib
    registry = inference.load_directive_domain_registry(str(Path(DEFAULT_PARENT).parent/'directive_domain_registry.json'))
    modes = decisions.modes_from_train(train)
    bank = {(r['kernel_name'],r['device'],r['selected_clock_period'],r['resource_budget_id'],r['_jsonl_idx']):r for r in val}
    result = dict(schema='stage1-saved-decision-review-v1',test_labels_used=False,training_modes=modes,checkpoints={})
    identity = None
    for path in paths:
        saved = json.loads(Path(path).read_text())['rows']
        enriched = []
        for row in saved:
            key=(row['kernel_name'],row['device'],row['selected_clock_period'],row['resource_budget_id'],row['jsonl_idx'])
            original=bank[key]
            if decisions.parse(row['reference_target']) != decisions.parse(original['target']):
                raise ValueError('Saved reference differs from authenticated validation selection')
            enriched.append(dict(original,**row))
        case_hash=hashlib.sha256(json.dumps([(r['kernel_name'],r['device'],r['selected_clock_period'],r['resource_budget_id'],r['jsonl_idx'],r['reference_target']) for r in enriched],sort_keys=True).encode()).hexdigest()
        if identity is not None and case_hash!=identity:
            raise ValueError('Checkpoint comparison requires identical ordered validation cases')
        identity=case_hash
        baseline=[dict(r,prediction=decisions.constant_baseline(dict(r,input=sources[r['source_key']]),registry,modes)) for r in enriched]
        record=dict(prediction_file_sha256=sha(path),report=decisions.decision_report(enriched,registry),
                    constant_baseline=decisions.decision_report(baseline,registry))
        result['checkpoints'][str(path)]=record
    result['case_identity_sha256']=identity
    dump(Path(output)/'report.json',result)
    return result


if __name__ == '__main__':
    main()
