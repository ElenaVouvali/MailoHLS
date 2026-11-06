"""Select feasible measured preferences from the existing authenticated V4 banks."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import math
import numpy as np

from LLM_branch.common import directive_domains, mailohls_contract, source_tool_policy
from experiments.stage1_quality_sft import core
from experiments.stage1_quality_sft.build import _read_split_rows
from experiments.stage1_quality_sft.training import _serialize_target_against_source
from experiments.stage1_quality_sft.support import load_registry, normalize_kname, constrained_rhs_candidates
from .common import *


def eligible_candidates(pool, request, registry, actions, policy, exclusions):
    retained = []
    for c in pool['candidates']:
        q = [c.get('latency'), *[c.get('utilization', {}).get(k) for k in core.UTILIZATION]]
        if any(v is None or not math.isfinite(float(v)) or float(v) < 0 for v in q) or float(q[0]) <= 0:
            exclusions['missing_or_invalid_qor'] += 1; continue
        if any(float(c['utilization'][u]) > 100 * request['budget'][b] + 1e-9
               for u, b in zip(core.UTILIZATION, core.BUDGET)):
            exclusions['outside_requested_budget'] += 1; continue
        kernel = pool['physical_context'][0]
        assignment = c['assignment']
        row = dict(kernel_name=kernel, target='\n'.join(f'{k} = {v}' for k,v in assignment.items()))
        try:
            core.validate_complete_assignment(row, registry, constrained_rhs_candidates, normalize_kname, core.parse)
            source_tool_policy.validate_source_assignment(assignment, actions)
            source_tool_policy.validate_complete_assignment(assignment, actions, policy)
        except ValueError as e:
            exclusions['source_or_tool:' + str(e).split(':')[0]] += 1; continue
        assert c['candidate_id'] == __import__('hashlib').sha256(repr(tuple(sorted(assignment.items()))).encode()).hexdigest()
        retained.append(c)
    return retained


def one_pair(request, feasible, seed=123):
    if not feasible: return None, 'no_admissible_feasible_configuration', 0
    best = min(float(c['latency']) for c in feasible)
    accepted = [c for c in feasible if float(c['latency']) <= 1.01 * best + 1e-12]
    choices = []
    combinations = 0
    for preferred in accepted:
        rejected = [c for c in feasible if float(c['latency']) >= 1.10 * float(preferred['latency']) - 1e-12]
        combinations += len(rejected)
        if rejected: choices.append((preferred, rejected))
    if not choices: return None, 'no_feasible_rejection_with_10pct_gap', 0
    preferred, rejected = min(choices, key=lambda p: digest([seed, request['request_id'], p[0]['candidate_id']]))
    rejected = min(rejected, key=lambda c: digest([seed, request['request_id'], preferred['candidate_id'], c['candidate_id']]))
    return dict(request=request, preferred=preferred, rejected=rejected, best_feasible_latency=best,
                latency_gap=float(rejected['latency']) / float(preferred['latency']) - 1.,
                pair_id=digest([request['request_id'], preferred['candidate_id'], rejected['candidate_id']])), None, combinations


def balance(pairs, maximum=8, seed=123):
    families = defaultdict(list)
    for p in pairs:
        families[p['request']['family']].append(p)
    family_counts=defaultdict(Counter);global_counts=Counter();chosen=[]
    def keys(p):
        r=p['request'];d,c=r['physical_context'][3:5];b=tuple(r['budget'][k] for k in core.BUDGET)
        return [('device',d),('clock',c),('budget',b),('stratum',d,c,b),('kernel',r['physical_context'][0])]
    # Round robin families, selecting the least represented hardware categories.
    # Hash tie-breaks prevent sorted device names from consuming a small family cap.
    for exposure in range(maximum):
        for family in sorted(families):
            if not families[family]:continue
            def priority(p):
                ks=keys(p)
                return tuple(family_counts[family][k] for k in ks)+tuple(global_counts[k] for k in ks[:4])+(digest([seed,p['pair_id']]),)
            p=min(families[family],key=priority);families[family].remove(p);chosen.append(p)
            for k in keys(p):family_counts[family][k]+=1;global_counts[k]+=1
    return chosen


def prepare(out=OUT):
    if (out / 'manifest.json').exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    contract, _ = authenticate()
    parent = read(V4.parent / 'train.manifest.json')
    assert sha(PREPARED / 'eligibility_manifest.json') == parent['protocol_manifest_sha256']
    protocol = read(PREPARED / 'eligibility_manifest.json')
    assert protocol['authenticated'] and protocol['test_rows_decoded'] == 0
    for name, h in protocol['files'].items(): assert sha(PREPARED / name) == h, name
    raw_split_path = PARENT_WORKTREE / 'mailohls_runs/mailohls_final_family_split_s123.json'
    assert sha(raw_split_path) == protocol['split_sha256']
    raw_split = read(raw_split_path)
    dataset = PARENT_WORKTREE / 'artifacts/llm/mailohls_sft.jsonl'
    assert sha(dataset) == protocol['dataset_sha256'] == parent['parent_dataset_sha256']
    sources = read(V4.parent / 'train.sources.json')['templates']
    assert sha(V4.parent / 'train.sources.json') == parent['sources_sha256']
    assert sha(V4.parent / 'train.jsonl') == parent['output_sha256'] == contract['dataset_sha256']
    arm_split = read(V4.parent / 'split.json')
    assert set(arm_split['train_families']).isdisjoint(arm_split['val_families'])
    assert not arm_split['test_jsonl_idx']
    registry = load_registry(V4 / 'directive_domain_registry.json')
    policy = source_tool_policy.load_policy(POLICY)
    actions = {}
    records, audit = {}, {}
    for split, prefix in [('train','train'), ('validation','validation')]:
        requests = rows(PREPARED / f'{prefix}_requests.jsonl.gz')
        pools = {p['candidate_pool_id']: p for p in rows(PREPARED / f'{prefix}_candidate_pools.jsonl.gz')}
        exclusions, unsupported = Counter(), Counter()
        all_pairs, combination_count = [], 0
        allowed_ids = set(raw_split['train_jsonl_idx' if split == 'train' else 'val_jsonl_idx'])
        allowed_families = set(arm_split['train_families' if split == 'train' else 'val_families'])
        for request_index,request in enumerate(requests):
            if request_index%100==0:print(f'[AUDIT] {split} request {request_index}/{len(requests)}',flush=True)
            assert request['family'] in allowed_families
            assert request['objective'] == 'PARETO_LATENCY'
            pool = pools[request['candidate_pool_id']]
            assert pool['physical_context'] == request['physical_context']
            assert all(c['row_id'] in allowed_ids for c in pool['candidates'])
            kernel = pool['physical_context'][0]
            if kernel not in actions: actions[kernel] = directive_domains.load_source_actions(kernel, ROOT / 'Data/ApplicationDataset')
            feasible = eligible_candidates(pool, request, registry, actions[kernel], policy, exclusions)
            pair, reason, combinations = one_pair(request, feasible)
            combination_count += combinations
            if pair: all_pairs.append(pair)
            else: unsupported[reason] += 1
        selected = balance(all_pairs) if split == 'train' else all_pairs
        records[split] = selected
        gaps = [p['latency_gap'] for p in selected]
        audit[split] = dict(requests=len(requests), eligible_requests=len(all_pairs), valid_pair_combinations=combination_count,
            sealed_pairs=len(selected), request_pair_cap=1, family_pair_cap=8 if split == 'train' else None,
            pairs_per_family=dict(Counter(p['request']['family'] for p in selected)),
            device_clock_budget_counts=dict(Counter(f"{p['request']['physical_context'][3]}@{p['request']['physical_context'][4]}ns/{p['request']['budget'][core.BUDGET[0]]}" for p in selected)),
            gap_quantiles=dict(zip(['min','p25','median','p75','max'], np.quantile(gaps,[0,.25,.5,.75,1]).tolist())) if gaps else {},
            excluded_candidate_request_occurrences=dict(exclusions), excluded_requests=dict(unsupported))
    ids = {p[side]['row_id'] for ps in records.values() for p in ps for side in ['preferred','rejected']}
    measured = {r['_jsonl_idx']:r for r in _read_split_rows(dataset, ids, sources)}
    def materialize(pair):
        request = pair['request']; context = request['physical_context']
        src = sources[context[1]]
        fields = dict(context_id=request['request_id'], request_id=request['request_id'], kernel_name=context[0],
            source_key=context[1], source_file=context[2], input=src, objective='PARETO_LATENCY', obj_mode='PARETO_LATENCY',
            device=context[3], clock_period=context[4], frequency_mode='specified', selected_clock_period=context[4],
            resource_budget_id=f"uniform-{int(100*request['budget'][core.BUDGET[0]]):03d}", **request['budget'])
        for res, avail in mailohls_contract.AVAIL_FIELD_BY_RESOURCE.items():
            b = core.BUDGET[list(mailohls_contract.RESOURCE_KEYS).index(res)]
            fields[avail] = int(round(mailohls_contract.DEVICE_RESOURCES[context[3]][res] * request['budget'][b]))
        for side in ['preferred','rejected']:
            c = pair[side]; r = measured[c['row_id']]
            assert core.physical_context(r) == tuple(context)
            assert core.parse(r['target']) == c['assignment']
            assert float(r['latency']) == float(c['latency'])
            assert all(float(r[k]) == float(c['utilization'][k]) for k in core.UTILIZATION)
            pair[side + '_target'] = _serialize_target_against_source(c['assignment'], src)
        return dict(pair, row=fields)
    for split, ps in records.items(): jsonl(out / f'{split}_pairs.jsonl', [materialize(p) for p in ps])
    supported = (len(records['train']) >= 64 and len(audit['train']['pairs_per_family']) >= 8
                 and len(records['validation']) >= 20 and len(audit['validation']['pairs_per_family']) >= 2)
    manifest = dict(schema='stage1-qor-preference-v1', seed=123, beta=.1, lambda_dpo=.2,
        near_optimal_epsilon=.01, minimum_rejection_gap=.10, training_justified_for_bounded_pilot=supported,
        coverage_gate='at least 64 train pairs/8 train families and 20 validation pairs/2 validation families; no outcome tuning',
        preferred_rule='within 1% of minimum measured latency among source/tool-admissible feasible complete configurations',
        reference_adapter_sha256=ADAPTER_SHA, reference_contract_sha256=CONTRACT_SHA,
        prepared_protocol_sha256=sha(PREPARED/'eligibility_manifest.json'), parent_dataset_sha256=sha(dataset),
        raw_family_split_sha256=sha(raw_split_path), tokenizer_sha256=sha(V4/'tokenizer.json'),
        registry_sha256=sha(V4/'directive_domain_registry.json'), tool_policy_sha256=sha(POLICY),
        pair_files={s:sha(out/f'{s}_pairs.jsonl') for s in records}, audit=audit,
        test_measurements_decoded=0, complete_measured_assignments_preserved=True,
        exclusion_counting='candidate-request occurrences; a design can appear under several budgets',
        measurement_units='authenticated dataset latency units; ratios only; no inferred cycle conversion')
    write(out/'manifest.json',manifest)
    write(out/'reference_contract.json',contract)
    print(json.dumps(dict(training_justified=supported, audit=audit),indent=2),flush=True)
    return manifest


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args();prepare(a.out)
