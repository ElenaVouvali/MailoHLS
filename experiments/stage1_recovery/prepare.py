"""Reconstruct the parent's train/validation selection without using test targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from LLM_branch.train import train_SFT_xattn_new as production

DEFAULT_OUTPUT = Path('artifacts/diagnostics/stage1_recovery_data')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parent, output=DEFAULT_OUTPUT):
    production.cache_stage1_serialization()
    output = Path(output)
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['parent_contract_sha256'] != sha(Path(parent) / 'training_contract.json'):
            raise ValueError('Selected data belongs to a different parent contract')
        for filename, digest in manifest['files'].items():
            if sha(output / filename) != digest:
                raise ValueError(f'Selected data changed: {filename}')
        for filename, digest in manifest.get('inputs', {}).items():
            if sha(filename) != digest:
                raise ValueError(f'Selection input changed: {filename}')
        return manifest
    contract = json.loads((Path(parent) / 'training_contract.json').read_text())
    if contract['auto_frequency_fraction'] != 0 or contract['objective'] != 'PARETO_LATENCY':
        raise ValueError('This reconstruction supports the audited specified-clock latency parent')
    assert sha('artifacts/llm/mailohls_sft.jsonl') == contract['dataset_sha256']
    split = json.loads(Path('mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    membership = {k: sorted(split[k + '_jsonl_idx']) for k in ['train', 'val', 'test']}
    digest = hashlib.sha256(json.dumps(membership, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == contract['split_sha256']
    production.TARGET_CFG = production.TargetAwareConfig(device_mode=contract['device_mode'],
        budget_mode='random', random_budgets_per_case=contract['random_budgets_per_case'],
        min_budget_frac=contract['random_budget_min_frac'], min_feasible_candidates=contract['min_feasible_candidates_per_budget'],
        candidate_pool_per_objective=contract['candidate_pool_per_objective'], auto_frequency_fraction=0,
        min_auto_clock_count=contract['min_auto_clock_count'], effective_area_floor=contract['effective_area_floor'], seed=contract['seed'])
    # load_rows authenticates the source sidecar. Test targets are not selected,
    # summarized, exported, or used in any training/validation decision.
    rows = production.load_rows('artifacts/llm/mailohls_sft.jsonl')
    train_idx, val_idx = set(membership['train']), set(membership['val'])
    train = [r for r in rows if r['_jsonl_idx'] in train_idx]
    val = [r for r in rows if r['_jsonl_idx'] in val_idx]
    del rows
    registry = production.load_directive_domain_registry(str(Path(parent).parent / 'directive_domain_registry.json'))
    chosen = {}
    for name, source, seed in [('train', train, contract['seed']), ('val', val, contract['seed'] + 10000)]:
        augmented = production.augment_rows_with_random_resource_budgets(source,
            contract['random_budgets_per_case'], seed, contract['min_feasible_candidates_per_budget'])
        selected, _ = production.select_goal_rows(augmented, contract['objective'], contract['top_k'],
            contract['goal_domination_penalty'], contract['goal_max_dominated_gap'],
            contract['score_weight_min'], contract['score_weight_power'], False)
        if name == 'train':
            selected = production.compact_duplicate_budget_targets(selected, contract['budget_target_max_duplicates'], invariant_max_duplicates=None)
        chosen[name] = selected
        del augmented
    assert production.compute_directive_loss_weights(chosen['train'], contract['directive_loss_weighting'], registry) == contract['directive_loss_weights']
    cases = production.build_selection_cases(chosen['val'], contract['objective'], 0, 4, .85, 2)
    saved = json.loads((Path(parent) / 'best_selection_metrics.json').read_text())['rows']
    expected = {(r['kernel_name'], r['device'], r['selected_clock_period'], r['resource_budget_id']):
                (r['jsonl_idx'], r['reference_target']) for r in saved}
    assert len(cases) == len(expected)
    for c in cases:
        key = (c.kernel_name, c.row['device'], c.row['selected_clock_period'], c.row['resource_budget_id'])
        assert expected[key] == (c.row['_jsonl_idx'], c.reference_target)
    output.mkdir(parents=True)
    files = {}
    for name, selected in chosen.items():
        filename = name + '_reconstructed_selected.jsonl'
        with (output / filename).open('w') as handle:
            for row in selected:
                handle.write(json.dumps({k: v for k, v in row.items() if k != 'input'}, sort_keys=True) + '\n')
        files[filename] = sha(output / filename)
    manifest = dict(schema='mailohls-stage1-recovery-selected-data-v1', parent_contract_sha256=sha(Path(parent) / 'training_contract.json'),
                    dataset_sha256=contract['dataset_sha256'], split_sha256=digest, files=files,
                    counts={k: len(v) for k, v in chosen.items()}, saved_validation_cases_matched=len(cases),
                    heldout_targets_used=False)
    manifest['inputs'] = {str(p): sha(p) for p in [
        Path('artifacts/llm/mailohls_sft.jsonl'), Path('artifacts/llm/mailohls_sft.sources.json'),
        Path('artifacts/llm/mailohls_sft.manifest.json'), Path('mailohls_runs/mailohls_final_family_split_s123.json')]}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest




def target_contrasts(rows):
    """Comparable selected requests, never synthetic preference labels.

    Budget: same program/device/clock. Clock: identical absolute resource limits.
    Device: same source/clock and FULL device budget (different capacities are
    the intended device change). Other device-budget comparisons are excluded.
    A changed measured winner is diagnostic, not a statistically proven boundary.
    """
    from collections import defaultdict
    from itertools import combinations
    from experiments.stage1_recovery.decisions import parse
    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        base = (row['kernel_name'], row.get('source_key'), row.get('obj_mode'))
        clock = row.get('selected_clock_period', row.get('clock_period'))
        budget = tuple(row.get(k) for k in ('avail_bram', 'avail_dsp', 'avail_ff', 'avail_lut'))
        fractions = tuple(row.get(k) for k in ('budget_frac_bram_18k', 'budget_frac_dsp', 'budget_frac_ff', 'budget_frac_lut'))
        groups[('budget', base + (row['device'], clock))][budget] = i
        groups[('clock', base + (row['device'], budget))][clock] = i
        if fractions == (1., 1., 1., 1.):
            groups[('device', base + (clock, 'full_device'))][row['device']] = i
    result = []
    for (kind, key), contexts in sorted(groups.items(), key=lambda item: repr(item[0])):
        for left, right in combinations(sorted(contexts.values()), 2):
            a, b = rows[left], rows[right]
            ga, gb = parse(a['target']), parse(b['target'])
            comparable = set(ga) == set(gb)
            changed = [k for k in ga if comparable and ga[k] != gb[k]]
            result.append(dict(kind=kind, left=left, right=right, kernel=a['kernel_name'],
                               comparable=comparable, changed_fields=changed,
                               classification=('changed' if changed else 'invariant') if comparable else 'unsupported',
                               distinct_measurements=a.get('_jsonl_idx') != b.get('_jsonl_idx')))
    return result


def supervision_audit(rows, registry, fit_balance=True):
    """Counts complete legal actions and nonforced decisions on selected rows."""
    from collections import Counter, defaultdict
    from experiments.stage1_recovery.decisions import parse
    from LLM_branch.train.stage1_decision_loss import decision_fields, fit_value_weights
    fields, forced, loops, arrays, context = defaultdict(Counter), defaultdict(Counter), Counter(), Counter(), defaultdict(list)
    configs = set()
    for row in rows:
        parsed = parse(row['target'])
        configs.add((row['kernel_name'], tuple(sorted(parsed.items()))))
        actions = defaultdict(dict)
        for key, kind, rhs, is_forced in decision_fields(row, registry, production.constrained_rhs_candidates, parse):
            (forced if is_forced else fields)[kind][rhs] += 1
            actions[key.rsplit('_L', 1)[-1]][kind] = rhs
        for action in actions.values():
            if 'PIPE' in action:
                loops[(action['PIPE'], action['UNROLL'])] += 1
            else:
                arrays[(action['ARRAY_T'], action['ARRAY_F'], action['ARRAY_D'])] += 1
        clock = row.get('selected_clock_period', row.get('clock_period'))
        context[(row['device'], clock)].append(row)
    contrasts = target_contrasts(rows)
    summaries = {}
    for kind in ('budget', 'clock', 'device'):
        subset = [c for c in contrasts if c['kind'] == kind]
        summaries[kind] = dict(pairs=len(subset), classifications=dict(Counter(c['classification'] for c in subset)),
                               changed_kernels=sorted({c['kernel'] for c in subset if c['classification'] == 'changed'}))
    return dict(examples=len(rows), distinct_measured_rows=len({r['_jsonl_idx'] for r in rows}),
                kernels=len({r['kernel_name'] for r in rows}), families=dict(Counter(r['_family'] for r in rows)),
                distinct_complete_assignments=len(configs), repeated_winner_contexts=len(rows)-len({r['_jsonl_idx'] for r in rows}),
                complete_loop_actions={','.join(k): v for k,v in sorted(loops.items())},
                complete_array_actions={','.join(k): v for k,v in sorted(arrays.items())},
                nonforced_values={k: dict(v) for k,v in fields.items()}, forced_values={k: dict(v) for k,v in forced.items()},
                context_coverage=[dict(device=d, clock_ns=c, examples=len(v),
                     distinct_measured_rows=len({r['_jsonl_idx'] for r in v}),
                     distinct_complete_assignments=len({r['target'] for r in v}),
                     budget_ranges={k: [min(r[k] for r in v),max(r[k] for r in v)]
                                    for k in ('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut')})
                                  for (d,c),v in sorted(context.items())],
                target_contrasts=summaries, informative_target_contrasts=sum(c['classification']=='changed' for c in contrasts),
                nearly_equivalent_alternatives='requires feasible-measurement gap audit; changes are not proven preference boundaries',
                balancing=(fit_value_weights(rows, registry, production.constrained_rhs_candidates, parse) if fit_balance else None))


def audit_prepared(parent, output):
    """Authenticate reconstruction first; inspect only train raw labels."""
    from collections import Counter, defaultdict
    from experiments.stage1_recovery.decisions import parse
    manifest = prepare(parent, output)
    output = Path(output)
    registry = production.load_directive_domain_registry(str(Path(parent).parent / 'directive_domain_registry.json'))
    selected = {s: [json.loads(line) for line in (output / f'{s}_reconstructed_selected.jsonl').read_text().splitlines()]
                for s in ('train', 'val')}
    audit = dict(schema='mailohls-full-supervision-audit-v1', selection_manifest=manifest,
                 compaction=dict(max_duplicates=4,invariant_max_duplicates=None), test_labels_used=False,
                 implementation_sha256={p:sha(p) for p in ['experiments/stage1_recovery/prepare.py','LLM_branch/train/stage1_decision_loss.py']},
                 train=supervision_audit(selected['train'], registry),
                 validation_reporting_only=supervision_audit(selected['val'], registry, fit_balance=False))
    # Only training membership is read; test labels never enter audit statistics.
    membership = set(json.loads(Path('mailohls_runs/mailohls_final_family_split_s123.json').read_text())['train_jsonl_idx'])
    raw, configs, feasible_bank = 0, set(), defaultdict(list)
    with Path('artifacts/llm/mailohls_sft.jsonl').open() as handle:
        for i, line in enumerate(handle):
            if i not in membership:
                continue
            row = json.loads(line)
            raw += 1
            configs.add((row['kernel_name'],tuple(sorted(parse(row['target']).items()))))
            feasible_bank[(row['kernel_name'],row['device'],row['clock_period'])].append(row)
    audit['train']['raw_measurements'] = raw
    audit['train']['distinct_raw_directive_assignments'] = len(configs)
    gaps = []
    for row in selected['train']:
        bank = feasible_bank[(row['kernel_name'],row['device'],row['clock_period'])]
        feasible = [r for r in bank if all(float(r[k]) <= 100*float(row[f])+1e-8 for k,f in
            [('bram_util_%','budget_frac_bram_18k'),('dsp_util_%','budget_frac_dsp'),('ff_util_%','budget_frac_ff'),('lut_util_%','budget_frac_lut')])]
        others = [float(r['latency']) for r in feasible if parse(r['target']) != parse(row['target'])]
        gap = (min(others)-float(row['latency']))/float(row['latency']) if others and float(row['latency'])>0 else None
        gaps.append(dict(measured_row=row['_jsonl_idx'], budget=row['resource_budget_id'],feasible_measurements=len(feasible),
                         relative_latency_gap_to_best_different_assignment=gap))
    conflicts = defaultdict(set)
    for r in selected['train']:
        conflicts[(r['kernel_name'],r['device'],r['selected_clock_period'],r['resource_budget_id'],r['obj_mode'])].add(r['target'])
    audit['train']['conflicting_identical_requests'] = sum(len(v)>1 for v in conflicts.values())
    audit['train']['gap_audit'] = dict(contexts=len(gaps), unavailable=sum(g['relative_latency_gap_to_best_different_assignment'] is None for g in gaps),
        within_one_percent=sum(abs(g['relative_latency_gap_to_best_different_assignment'])<=.01 for g in gaps if g['relative_latency_gap_to_best_different_assignment'] is not None),
        warning='Latency gap only; no repeated-measurement uncertainty or statistical significance established')
    for name,value in [('supervision_audit.json',audit),('training_contrasts.json',target_contrasts(selected['train'])),('training_winner_gaps.json',gaps)]:
        (output / name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return audit


def validate_prepared_tokens(parent, output):
    """CPU preflight using the pinned checkpoint tokenizer, never model weights."""
    from LLM_branch.inference.eval_stage1_stage2_stage3 import load_checkpoint_tokenizer
    from experiments.stage1_recovery import decisions
    from LLM_branch.train import stage1_decision_loss
    manifest = prepare(parent, output)
    contract = json.loads((Path(parent)/'training_contract.json').read_text())
    tok = load_checkpoint_tokenizer(str(parent),contract)
    registry = production.load_directive_domain_registry(str(Path(parent).parent/'directive_domain_registry.json'))
    rows = [json.loads(line) for line in (Path(output)/'train_reconstructed_selected.jsonl').read_text().splitlines()]
    sources = json.loads(Path('artifacts/llm/mailohls_sft.sources.json').read_text())['templates']
    rows = [dict(r,input=sources[r['source_key']]) for r in rows]
    balance = stage1_decision_loss.fit_value_weights(rows,registry,production.constrained_rhs_candidates,decisions.parse)
    data = production.SFTDataset(rows,tok,contract['max_length'],kind_loss_weights=contract['directive_loss_weights'],
        directive_domain_registry=registry,input_only_special_ids=contract['special_token_ids'])
    counts = dict.fromkeys(stage1_decision_loss.POLICIES,0)
    for row in rows:
        legacy = production.build_deterministic_rhs_pack(row['input'],row['target'],tok,
            directive_domain_registry=registry,kernel_name=row['kernel_name'])
        for policy in stage1_decision_loss.POLICIES:
            pack = production.build_deterministic_rhs_pack(row['input'],row['target'],tok,
                directive_domain_registry=registry,kernel_name=row['kernel_name'],supervision_policy=policy,
                decision_value_weights=balance['weights'])
            if pack.input_ids != legacy.input_ids:
                raise ValueError('Loss mode changed the target serialization')
            if policy!='legacy' and abs(sum(pack.token_weights)-1)>1e-6:
                raise ValueError('Incorrect action-normalized mass')
            counts[policy]+=1
    result=dict(schema='mailohls-stage1-token-preflight-v1',selection_manifest_sha256=sha(Path(output)/'manifest.json'),
        tokenizer_sha256=sha(Path(parent)/'tokenizer.json'),examples_by_policy=counts,
        max_sample_tokens=max(data.lengths),min_sample_tokens=min(data.lengths),
        tokenizer_size=len(tok),domain_registry_sha256=sha(Path(parent).parent/'directive_domain_registry.json'),
        unchanged_target_ids=True,supervised_special_tokens=False,model_loaded=False,
        effective_updates_per_epoch=(len(rows)+7)//8,epochs=2,coverage_update_budget=2*((len(rows)+7)//8),trainer_451_epoch_based_max_steps=2*(len(rows)//8),warmup_updates=int(.05*2*((len(rows)+7)//8)))
    (Path(output)/'token_preflight.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', default='mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1')
    ap.add_argument('--output', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--audit', action='store_true', help='Training supervision/contrast audit; validation reported separately')
    ap.add_argument('--validate-tokens', action='store_true', help='CPU tokenizer and all-policy mask preflight')
    args = ap.parse_args()
    result = (validate_prepared_tokens(args.parent, args.output) if args.validate_tokens else
              audit_prepared(args.parent, args.output) if args.audit else prepare(args.parent, args.output))
    print(json.dumps(result, indent=2))
