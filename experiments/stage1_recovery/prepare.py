"""Reconstruct the parent's train/validation selection without using test targets."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path

from LLM_branch.train import train_SFT_xattn_new as production

DEFAULT_OUTPUT = Path('artifacts/diagnostics/stage1_recovery_data')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parent, output=DEFAULT_OUTPUT):
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
    production.extract_ordered_lhs_plan = functools.lru_cache(maxsize=256)(production.extract_ordered_lhs_plan)
    production.reorder_target_by_source_order = functools.lru_cache(maxsize=100000)(production.reorder_target_by_source_order)
    production.canonical_completion_key = functools.lru_cache(maxsize=100000)(production.canonical_completion_key)
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', default='mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1')
    ap.add_argument('--output', default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    print(json.dumps(prepare(args.parent, args.output), indent=2))
