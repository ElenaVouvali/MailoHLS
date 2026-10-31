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


def _assignment_key(text):
    """Canonical complete directive assignment; formatting/order do not matter."""
    return tuple(sorted(decisions.parse(text).items()))


def _context_identity(row):
    """Identity for a generated request, including repeated-row budget contexts."""
    index = row.get('jsonl_idx', row.get('_jsonl_idx'))
    clock = row.get('selected_clock_period', row.get('clock_period'))
    return (row['kernel_name'], row.get('source_key'), int(index), str(row['device']).strip().lower(),
            round(float(clock), 2), str(row.get('resource_budget_id')),
            str(row.get('obj_mode', '')))


def measured_candidate_quality(predictions, validation_measurements, requests):
    """Match constrained outputs to exact validation-family measured designs.

    The best-feasible denominator is the complete raw measured validation pool
    for the same source, device and clock, filtered by the request's resource
    limits. Duplicate physical design rows are accepted only when all recorded
    QoR fields agree; the lowest source-row index is then the canonical receipt.
    Unmatched generated configurations retain unknown QoR.
    """
    from collections import defaultdict
    from statistics import median
    from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES, RESOURCE_KEYS, UTIL_FIELD_BY_RESOURCE, AVAIL_FIELD_BY_RESOURCE

    request_by_id = {_context_identity(r): r for r in requests}
    if len(request_by_id) != len(requests):
        raise ValueError('Selected validation request contexts are not uniquely identified')
    measured = defaultdict(list)
    for row in validation_measurements:
        key = (row['kernel_name'], row.get('source_key'), row.get('source_file'),
               str(row['device']).strip().lower(), round(float(row['clock_period']), 2),
               _assignment_key(row['target']))
        measured[key].append(row)

    canonical, duplicate_rows = {}, 0
    qos = ('latency',) + tuple(UTIL_FIELD_BY_RESOURCE[k] for k in RESOURCE_KEYS)
    for key, rows in measured.items():
        rows = sorted(rows, key=lambda r: int(r.get('_jsonl_idx', -1)))
        signatures = {tuple(float(r[field]) for field in qos) for r in rows}
        if len(signatures) != 1:
            raise ValueError(f'Conflicting repeated measurement for exact design: {key[:5]}')
        duplicate_rows += len(rows) - 1
        canonical[key] = rows[0]

    pools = defaultdict(list)
    for key, row in canonical.items():
        pool_key = key[:5]
        device = pool_key[3]
        capacities = DEVICE_RESOURCES.get(device)
        if capacities is None:
            continue
        used = {resource: float(row[UTIL_FIELD_BY_RESOURCE[resource]]) / 100.0 * capacities[resource]
                for resource in RESOURCE_KEYS}
        pools[pool_key].append((key[5], row, used))

    by_group = defaultdict(list)
    records = []
    winner_matches = measured_best_matches = 0
    measured_count = feasible_count = infeasible_count = 0
    regrets = []
    within1 = within5 = 0
    best_pool_missing = 0
    for prediction in predictions:
        identity = _context_identity(prediction)
        request = request_by_id.get(identity)
        if request is None:
            raise ValueError(f'Generated prediction has no exact selected validation request: {identity}')
        if _assignment_key(prediction['reference_target']) != _assignment_key(request['target']):
            raise ValueError(f'Prediction reference differs from authenticated selected label: {identity}')
        device = str(request['device']).strip().lower()
        clock = round(float(request['selected_clock_period']), 2)
        pool_key = (request['kernel_name'], request.get('source_key'), request.get('source_file'), device, clock)
        capacities = DEVICE_RESOURCES.get(device)
        if capacities is None:
            raise ValueError(f'Unknown validation device capacity: {device}')
        limits = {}
        fraction_limits = {}
        for resource in RESOURCE_KEYS:
            available_field = AVAIL_FIELD_BY_RESOURCE[resource]
            frac_field = {'BRAM_18K':'budget_frac_bram_18k','DSP':'budget_frac_dsp',
                          'FF':'budget_frac_ff','LUT':'budget_frac_lut'}[resource]
            if request.get(frac_field) is not None:
                # This is the same fractional feasibility contract used by
                # production design_fits_shared_budget. Integer prompt counts
                # are rounded for display and can differ by <1 resource unit.
                fraction_limits[resource] = float(request[frac_field])
            elif request.get(available_field) not in (None, ''):
                limits[resource] = float(request[available_field])
            else:
                limits[resource] = float(capacities[resource])
        candidates = pools.get(pool_key, [])
        feasible = [(key, row, used) for key, row, used in candidates
                    if all((float(row[UTIL_FIELD_BY_RESOURCE[k]]) / 100.0 <= fraction_limits[k] + 1e-9)
                           if k in fraction_limits else used[k] <= limits[k] + 1e-8
                           for k in RESOURCE_KEYS)]
        best_latency = min((float(row['latency']) for _, row, _ in feasible), default=None)
        if best_latency is None:
            best_pool_missing += 1

        pred_key = (request['kernel_name'], request.get('source_key'), request.get('source_file'),
                    device, clock, _assignment_key(prediction['prediction']))
        measured_row = canonical.get(pred_key)
        is_feasible = None
        regret = None
        if measured_row is not None:
            measured_count += 1
            used = {resource: float(measured_row[UTIL_FIELD_BY_RESOURCE[resource]]) / 100.0 * capacities[resource]
                    for resource in RESOURCE_KEYS}
            is_feasible = all((float(measured_row[UTIL_FIELD_BY_RESOURCE[k]]) / 100.0 <= fraction_limits[k] + 1e-9)
                               if k in fraction_limits else used[k] <= limits[k] + 1e-8
                               for k in RESOURCE_KEYS)
            if is_feasible:
                feasible_count += 1
                if best_latency is not None:
                    regret = (float(measured_row['latency']) - best_latency) / best_latency
                    # The predicted design belongs to this exact pool, so a
                    # tiny negative value can only be floating-point roundoff.
                    if regret < -1e-10:
                        raise ValueError('Matched candidate beats the computed feasible-pool optimum')
                    regret = max(0.0, regret)
                    regrets.append(regret)
                    within1 += regret <= 0.01 + 1e-12
                    within5 += regret <= 0.05 + 1e-12
            else:
                infeasible_count += 1
            qos_record = dict(latency=float(measured_row['latency']),
                resources_used={k: used[k] for k in RESOURCE_KEYS},
                resource_utilization_percent={k: float(measured_row[UTIL_FIELD_BY_RESOURCE[k]]) for k in RESOURCE_KEYS})
        else:
            qos_record = None
        selected_match = _assignment_key(prediction['prediction']) == _assignment_key(request['target'])
        winner_matches += selected_match
        if measured_row is not None and is_feasible and best_latency is not None:
            measured_best_matches += abs(float(measured_row['latency']) - best_latency) <= max(1e-12, best_latency * 1e-12)
        rec = dict(kernel=request['kernel_name'], source_key=request.get('source_key'),
            jsonl_idx=identity[2], device=device, clock_period=clock,
            resource_budget_id=request.get('resource_budget_id'), prediction=prediction['prediction'],
            budget_limit_source='fractional_shared_budget' if fraction_limits else 'absolute_available_resources',
            selected_reference_match=bool(selected_match), measured_match=measured_row is not None,
            feasibility=is_feasible, measured_qor=qos_record, best_feasible_latency=best_latency,
            measured_regret=regret, measured_candidate_count=len(candidates),
            feasible_measured_candidate_count=len(feasible))
        records.append(rec)
        group_key = (request['kernel_name'], device, clock, request.get('resource_budget_id'))
        by_group[group_key].append(rec)

    n = len(records)
    summary = dict(schema='stage1-validation-measured-candidate-quality-v1',
        evaluation_pool='all raw authenticated measurement rows whose original JSONL indices belong to the family-disjoint validation split',
        best_feasible_definition='minimum measured latency among exact-source/device/clock validation measurements feasible under the request resource limits; fractional budgets use the production design_fits_shared_budget percentage contract, otherwise absolute available resources',
        latency_field='dataset latency; source preprocessing column Latency_msec (milliseconds)',
        duplicate_measurement_policy='exact duplicate designs collapsed to lowest source row only when latency and all four utilization fields agree',
        requests=n, exact_measured_match_count=measured_count,
        exact_measured_match_fraction=measured_count/n if n else None,
        no_measured_match_count=n-measured_count,
        matched_feasible_count=feasible_count, matched_infeasible_count=infeasible_count,
        matched_feasible_fraction=feasible_count/measured_count if measured_count else None,
        feasible_selected_candidate_rate=feasible_count/n if n else None,
        selected_reference_winner_match_count=winner_matches,
        selected_reference_winner_match_fraction=winner_matches/n if n else None,
        best_feasible_latency_match_count=measured_best_matches,
        measured_regret_denominator='matched, measured, feasible outputs only; unmatched QoR remains unknown',
        measured_regret_count=len(regrets), mean_measured_regret=sum(regrets)/len(regrets) if regrets else None,
        median_measured_regret=median(regrets) if regrets else None,
        within_1_percent_count=within1, within_1_percent_fraction=within1/len(regrets) if regrets else None,
        within_5_percent_count=within5, within_5_percent_fraction=within5/len(regrets) if regrets else None,
        requests_without_any_feasible_measured_validation_candidate=best_pool_missing,
        canonical_measurement_designs=len(canonical), duplicate_rows_collapsed=duplicate_rows,
        groups=[dict(kernel=k[0],device=k[1],clock_period=k[2],resource_budget_id=k[3],
            requests=len(v),measured_match_count=sum(x['measured_match'] for x in v),
            matched_feasible_count=sum(x['feasibility'] is True for x in v),
            matched_infeasible_count=sum(x['feasibility'] is False for x in v),
            measured_regret_count=sum(x['measured_regret'] is not None for x in v),
            mean_measured_regret=(sum(x['measured_regret'] for x in v if x['measured_regret'] is not None)/
                sum(x['measured_regret'] is not None for x in v)) if any(x['measured_regret'] is not None for x in v) else None)
            for k,v in sorted(by_group.items())])
    return summary, records


def load_validation_measurements(dataset_path, split_path, expected_dataset_sha256):
    """Stream only authenticated validation rows; test records are never parsed."""
    split = json.loads(Path(split_path).read_text())
    validation_ids = set(int(i) for i in split['val_jsonl_idx'])
    if sha(dataset_path) != expected_dataset_sha256:
        raise ValueError('Measured-candidate dataset hash differs from the authenticated audit')
    selected = []
    with Path(dataset_path).open() as handle:
        for index, line in enumerate(handle):
            if index not in validation_ids:
                continue
            row = json.loads(line)
            row['_jsonl_idx'] = index
            selected.append(row)
    if {r['_jsonl_idx'] for r in selected} != validation_ids:
        raise ValueError('Validation measurement rows do not exactly cover authenticated validation split IDs')
    return selected


def target_transition_support(predictions, requests):
    """Describe paired responses without treating overlapping pairs as IID."""
    from collections import defaultdict
    from experiments.stage1_recovery.prepare import target_contrasts

    request_index = {_context_identity(r): i for i, r in enumerate(requests)}
    pred_index = {_context_identity(r): r for r in predictions}
    ordered_predictions = [pred_index[_context_identity(r)] for r in requests]
    if len(pred_index) != len(predictions) or set(pred_index) != set(request_index):
        raise ValueError('Transition analysis requires exactly one prediction for every selected validation context')
    pairs = target_contrasts(requests)
    support = defaultdict(lambda: dict(comparisons=0, changed=0, invariant=0, unsupported=0,
        complete_correct=0, correctly_invariant=0, changed_fields_correct=0,
        invariant_prediction_same=0, prediction_responsive=0, incorrect_responsive=0, incorrect_constant=0,
        participants=set(), kernels=set(), device_clock_groups=set(),
        group_complete_results=defaultdict(list), group_changed_results=defaultdict(list),
        group_invariant_results=defaultdict(list)))
    for pair in pairs:
        kind, i, j = pair['kind'], pair['left'], pair['right']
        a, b = requests[i], requests[j]
        pa, pb = decisions.parse(ordered_predictions[i]['prediction']), decisions.parse(ordered_predictions[j]['prediction'])
        ga, gb = decisions.parse(a['target']), decisions.parse(b['target'])
        item = support[kind]
        item['comparisons'] += 1
        item['participants'].update((_context_identity(a), _context_identity(b)))
        item['kernels'].update((a['kernel_name'], b['kernel_name']))
        item['device_clock_groups'].update(((a['device'], float(a['selected_clock_period'])),
                                           (b['device'], float(b['selected_clock_period']))))
        if not pair['comparable']:
            item['unsupported'] += 1
            continue
        fixed = (a['kernel_name'], a.get('source_key'), a.get('obj_mode'))
        if kind == 'budget':
            group = fixed + (a['device'], float(a['selected_clock_period']))
        elif kind == 'clock':
            group = fixed + (a['device'], tuple(a.get(k) for k in ('avail_bram','avail_dsp','avail_ff','avail_lut')))
        else:
            group = fixed + (float(a['selected_clock_period']), 'full_device')
        is_complete = ga == pa and gb == pb
        item['complete_correct'] += is_complete
        item['group_complete_results'][group].append(bool(is_complete))
        changed = pair['changed_fields']
        if not changed:
            item['invariant'] += 1
            correct = is_complete
            item['correctly_invariant'] += correct
            item['invariant_prediction_same'] += pa == pb
            item['group_invariant_results'][group].append(bool(correct))
            continue
        item['changed'] += 1
        fields_correct = all(pa.get(k) == ga.get(k) and pb.get(k) == gb.get(k) for k in changed)
        responsive = any(pa.get(k) != pb.get(k) for k in changed)
        item['changed_fields_correct'] += fields_correct
        item['prediction_responsive'] += responsive
        item['incorrect_responsive'] += responsive and not fields_correct
        item['incorrect_constant'] += not responsive
        item['group_changed_results'][group].append(bool(fields_correct))
    output = {}
    for kind, values in support.items():
        complete_groups = values.pop('group_complete_results')
        changed_groups = values.pop('group_changed_results')
        invariant_groups = values.pop('group_invariant_results')
        participants = values.pop('participants')
        kernels = values.pop('kernels')
        device_clock = values.pop('device_clock_groups')
        def macro(groups):
            scores = [sum(v)/len(v) for v in groups.values() if v]
            return sum(scores)/len(scores) if scores else None
        output[kind] = dict(values, distinct_participating_contexts=len(participants),
            distinct_kernels=len(kernels), distinct_device_clock_groups=len(device_clock),
            controlled_context_groups=len(complete_groups),
            macro_group_complete_transition_accuracy=macro(complete_groups),
            macro_group_changed_field_accuracy=macro(changed_groups),
            macro_group_correct_invariance_accuracy=macro(invariant_groups),
            group_aggregation_note='Descriptive macro-average over fixed-context groups; pair comparisons overlap and are not independent samples.')
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adapters', nargs='+', default=[DEFAULT_PARENT])
    ap.add_argument('--output', required=True)
    ap.add_argument('--joint', action='store_true')
    ap.add_argument('--prepared', default=str(AUDIT))
    ap.add_argument('--full-validation', action='store_true', help='All selected validation contexts via production callback')
    ap.add_argument('--saved-metrics', nargs='+', help='CPU-only reanalysis of saved selection predictions on identical cases')
    ap.add_argument('--measured-quality-predictions', nargs='+',
                    help='CPU-only measured-QoR matching of saved full-validation constrained-decoder outputs')
    ap.add_argument('--dataset', default='artifacts/llm/mailohls_sft.jsonl')
    ap.add_argument('--split', default='mailohls_runs/mailohls_final_family_split_s123.json')
    ap.add_argument('--domain-registry', default='mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json')
    ap.add_argument('--training-contract', default='mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1/training_contract.json')
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.perf_counter()
    prepared = Path(args.prepared)
    if args.measured_quality_predictions:
        evaluate_measured_quality_files(args.measured_quality_predictions, prepared,
                                        args.dataset, args.split, args.domain_registry,
                                        args.training_contract, out)
        return
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


def evaluate_measured_quality_files(prediction_paths, prepared, dataset_path, split_path,
                                    domain_registry_path, training_contract_path, output):
    """Read-only QoR matching for completed constrained-decoder validation files."""
    prepared = Path(prepared)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    input_hashes = manifest.get('inputs', {})
    expected_data = manifest['dataset_sha256']
    expected_split = input_hashes.get(str(Path(split_path)))
    for input_path, expected_hash in input_hashes.items():
        if not Path(input_path).is_file() or sha(input_path) != expected_hash:
            raise ValueError(f'Authenticated preparation input changed or is missing: {input_path}')
    if expected_split is None or sha(split_path) != expected_split:
        raise ValueError('Family split does not match the authenticated prepared-data manifest')
    if input_hashes.get(str(Path(dataset_path))) != expected_data or sha(dataset_path) != expected_data:
        raise ValueError('Dataset does not match the authenticated prepared-data manifest')
    requests = [json.loads(line) for line in (prepared / 'val_reconstructed_selected.jsonl').read_text().splitlines()]
    train = [json.loads(line) for line in (prepared / 'train_reconstructed_selected.jsonl').read_text().splitlines()]
    source_payload = json.loads(Path('artifacts/llm/mailohls_sft.sources.json').read_text())
    templates = source_payload['templates']
    parent_contract = json.loads(Path(training_contract_path).read_text())
    if sha(domain_registry_path) != parent_contract.get('directive_domain_registry_sha256'):
        raise ValueError('Directive-domain registry does not match the historical training contract')
    from LLM_branch.inference.eval_stage1_stage2_stage3 import load_directive_domain_registry
    registry = load_directive_domain_registry(str(domain_registry_path))
    modes = decisions.modes_from_train(train)
    baseline = [dict(request, reference_target=request['target'],
        prediction=decisions.constant_baseline(dict(request, input=templates[request['source_key']]), registry, modes))
        for request in requests]
    validation_measurements = load_validation_measurements(dataset_path, split_path, expected_data)
    identity = None
    result = dict(schema='stage1-full-validation-measured-quality-review-v1',
        dataset_sha256=expected_data, split_sha256=expected_split,
        prepared_manifest_sha256=sha(prepared / 'manifest.json'),
        validation_requests=len(requests), validation_measurement_rows=len(validation_measurements),
        heldout_test_labels_parsed=False, checkpoints={}, target_transitions={})
    candidates = [("legal_constant_training_baseline", None, baseline)]
    for path in prediction_paths:
        generated = json.loads(Path(path).read_text())
        if not isinstance(generated, list):
            raise ValueError(f'Expected a JSON list of constrained-decoder predictions: {path}')
        case_ids = [_context_identity(row) for row in generated]
        request_ids = [_context_identity(row) for row in requests]
        if len(case_ids) != len(requests) or len(set(case_ids)) != len(case_ids) or set(case_ids) != set(request_ids):
            raise ValueError(f'Prediction file does not cover each selected validation context exactly once: {path}')
        case_hash = __import__('hashlib').sha256(json.dumps(case_ids, separators=(',', ':')).encode()).hexdigest()
        if identity is not None and case_hash != identity:
            raise ValueError('Measured QoR comparison requires identical ordered validation contexts')
        identity = case_hash
        candidates.append((str(path), path, generated))
    if len(candidates) != 3:
        raise ValueError(f'Expected exactly historical and recovery checkpoints, received {len(candidates)-1} prediction files')
    for label, path, generated in candidates:
        summary, records = measured_candidate_quality(generated, validation_measurements, requests)
        summary['decision_metrics'] = decisions.decision_report(generated, registry)
        if path is not None:
            summary['prediction_file_sha256'] = sha(path)
        else:
            summary['prediction_source'] = 'training-derived legal constant baseline; complete assignment generated from source domains'
        result['checkpoints'][str(label)] = summary
        result['target_transitions'][str(label)] = target_transition_support(generated, requests)
        dump(output / (Path(path).stem + '_measured_quality.json' if path is not None
                       else 'legal_constant_training_baseline_measured_quality.json'), records)
    result['case_identity_sha256'] = identity
    dump(Path(output) / 'report.json', result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


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
