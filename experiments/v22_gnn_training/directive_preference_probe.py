"""Bounded preference-learning probe for candidate-independent V4 memory.

This experiment compares compiler-only memory, zero-gated target FiLM memory,
and the existing V4 hardware-message memory. It is separate from the V4 QoR
regression experiments and does not write production structural memory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch import target_graph_v22 as overlay
from experiments.v22_gnn_training import fast_probe, train, checkpointing


TRAIN_KERNELS = (
    'machsuite-gemm-blocked', 'machsuite-gemm-ncubed', 'machsuite-md-knn',
    'machsuite-spmv-ellpack', 'machsuite-stencil2d',
    'rodinia-backprop-0-baseline-back', 'rodinia-backprop-0-baseline-forward',
    'rodinia-hotspot-0-baseline', 'rodinia-kmeans-0-baseline',
    'rodinia_cfd_step_factor_1_tiling_0', 'rodinia_dilate_0_baseline_0',
    'rodinia_lavaMD_0_baseline', 'rodinia_lc_gicov_0_baseline_0',
    'rodinia_streamcluster_1_tiling_0',
)
VALIDATION_KERNELS = (
    'machsuite-viterbi', 'rodinia_pathfinder_0_baseline_0',
    'rodinia_pathfinder_4_doublebuffer_0',
)
PROFILES = ('zcu104', 'u200')
CLOCKS = (3.33, 5.0, 10.0)
BUDGETS = (1.0, 0.75, 0.5, 0.25, 0.1)
OBJECTIVES = ('PARETO_ADP', 'PARETO_LATENCY', 'PARETO_AREA')
N_CANDIDATES = 16
SEED = 123
PREFERENCE_MARGIN = 0.05
SCHEMA = 'salt-hls-v4-directive-preference-memory-probe-v1'
RESOURCE_ORDER = train.RESOURCE_ORDER


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _deduplicate(rows):
    result = {}
    for row in rows:
        old = result.get(row['config_hash'])
        if old is not None:
            if not np.allclose(old['physical'], row['physical'], rtol=0, atol=1e-12):
                raise ValueError('Duplicate complete assignment has conflicting physical labels')
            continue
        result[row['config_hash']] = row
    return result


def _select_assignments(data, kernel, split):
    contexts = [(kernel, p, c) for p in PROFILES for c in CLOCKS]
    maps = [_deduplicate(data.groups.get(key, [])) for key in contexts]
    shared = set.intersection(*(set(item) for item in maps))
    support=defaultdict(int); exemplar={}
    for mapping in maps:
        for signature,row in mapping.items():
            support[signature]+=1
            exemplar.setdefault(signature,row)
    candidates = list(exemplar.values())
    if split == 'train':
        remaining=sorted(candidates,key=lambda row:(-support[row['config_hash']],row['config_hash']))
        chosen=[remaining.pop(0)] if remaining else []
        while remaining and len(chosen)<N_CANDIDATES:
            next_row=min(remaining,key=lambda row:(
                -min(fast_probe._assignment_distance(row['assignment'],item['assignment']) for item in chosen),
                -support[row['config_hash']],row['config_hash']))
            chosen.append(next_row); remaining.remove(next_row)
    else:
        # Prefer candidates measured in more target contexts; hash breaks ties.
        chosen = sorted(candidates,key=lambda row:(-support[row['config_hash']],row['config_hash']))[:N_CANDIDATES]
    core_hashes=sorted(row['config_hash'] for row in chosen)
    per_context={}; selected_union=set()
    for (_,profile,clock),mapping in zip(contexts,maps):
        current=[h for h in core_hashes if h in mapping]
        candidates=list(mapping.values())
        while len(current)<min(N_CANDIDATES,len(candidates)):
            remaining=[row for row in candidates if row['config_hash'] not in current]
            if not remaining: break
            if current:
                next_row=min(remaining,key=lambda row:(
                    -min(fast_probe._assignment_distance(row['assignment'],mapping[h]['assignment'])
                          for h in current),-support[row['config_hash']],row['config_hash']))
            else:
                next_row=min(remaining,key=lambda row:(-support[row['config_hash']],row['config_hash']))
            current.append(next_row['config_hash'])
        current=sorted(current); per_context[f'{profile}|{clock:g}ns']=current
        selected_union.update(current)
    hashes=sorted(selected_union)
    support_counts={key:support[key] for key in hashes}
    by_context={key:len(values) for key,values in per_context.items()}
    return hashes, {'all_six_contexts':len(shared),'selected_support':support_counts,
                    'selected_core_hashes':core_hashes,'selected_core_count':len(core_hashes),
                    'selected_assignment_hashes_by_context':per_context,
                    'selected_candidates_by_context':by_context}


def _area(record):
    return sum(v / c for v, c in zip(record['physical'][1:], record['capacities'])) / 4.0


def _score(record, objective):
    return train.measured_objective(record, objective)


def _is_zero_area_ambiguous(record):
    return (record['kernel'] == 'machsuite-viterbi' and record['profile'] == 'u200'
            and float(record['clock']) == 5.0 and _area(record) == 0.0)


def _preference_pairs(records, fraction, objective, margin=PREFERENCE_MARGIN):
    """Pairs are (preferred index, worse index, pair class); ties/2x infeasible skip."""
    if not records:
        return [], {'feasible_vs_infeasible': 0, 'objective': 0,
                    'near_tie': 0, 'both_infeasible': 0, 'zero_area_ambiguous': 0}
    capacities = records[0]['capacities']
    limits = [capacity * fraction for capacity in capacities]
    feasible = [all(v <= cap for v, cap in zip(row['physical'][1:], limits)) for row in records]
    pairs = []
    counts = defaultdict(int)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if feasible[i] != feasible[j]:
                pairs.append((i, j, 'feasibility') if feasible[i] else (j, i, 'feasibility'))
                counts['feasible_vs_infeasible'] += 1
                continue
            if not feasible[i]:
                counts['both_infeasible'] += 1
                continue
            if objective in ('PARETO_AREA', 'PARETO_ADP') and (
                    _is_zero_area_ambiguous(records[i]) or _is_zero_area_ambiguous(records[j])):
                counts['zero_area_ambiguous'] += 1
                continue
            left, right = _score(records[i], objective), _score(records[j], objective)
            scale = max(abs(left), abs(right), 1e-12)
            if abs(left - right) / scale <= margin:
                counts['near_tie'] += 1
                continue
            pairs.append((i, j, 'objective') if left < right else (j, i, 'objective'))
            counts['objective'] += 1
    return pairs, dict(counts)


def build_cohort(data):
    roster = TRAIN_KERNELS + VALIDATION_KERNELS
    split = data.kernel_split
    split_manifest = data.split
    family_map = {}
    for family in split_manifest['train_families'] + split_manifest['val_families']:
        family_norm = ''.join(ch for ch in family.lower() if ch.isalnum())
        for kernel in split_manifest['train_kernels'] + split_manifest['val_kernels']:
            kernel_norm = ''.join(ch for ch in kernel.lower() if ch.isalnum())
            if kernel_norm.startswith(family_norm):
                family_map[kernel] = family
    if any(split.get(kernel) != 'train' for kernel in TRAIN_KERNELS):
        raise ValueError('Frozen training cohort crosses the authenticated family split')
    if any(split.get(kernel) != 'val' for kernel in VALIDATION_KERNELS):
        raise ValueError('Frozen validation cohort crosses the authenticated family split')
    selection = {}
    for kernel in roster:
        selected, coverage = _select_assignments(data, kernel, split[kernel])
        selection[kernel] = {'split': split[kernel], **coverage,
                             'selected_assignment_hashes': selected,
                             'selected_count': len(selected)}

    chosen = []
    for row in data.records:
        if row['kernel'] not in selection:
            continue
        target_key=f"{row['profile']}|{row['clock']:g}ns"
        if row['config_hash'] in selection[row['kernel']]['selected_assignment_hashes_by_context'][target_key]:
            chosen.append(row)
    by_identity = {}
    for row in chosen:
        key = (row['kernel'], row['profile'], row['clock'], row['config_hash'])
        if key in by_identity:
            if not np.allclose(by_identity[key]['physical'], row['physical'], rtol=0, atol=1e-12):
                raise ValueError('Repeated measured assignment has inconsistent QoR')
        else:
            by_identity[key] = row
    records = list(by_identity.values())
    by_context = defaultdict(list)
    for row in records:
        by_context[(row['kernel'], row['profile'], row['clock'])].append(row)
        row['kernel_family'] = family_map.get(row['kernel'], row['kernel'])
    for rows in by_context.values():
        rows.sort(key=lambda row: row['config_hash'])

    pair_audit = {'schema': SCHEMA + '-pair-audit', 'margin': PREFERENCE_MARGIN,
                  'budget_fractions': list(BUDGETS), 'objectives': list(OBJECTIVES),
                  'contexts': {}, 'totals': defaultdict(int)}
    preference_groups = {'train': [], 'val': []}
    for (kernel, profile, clock), rows in sorted(by_context.items()):
        split_name = split[kernel]
        for fraction in BUDGETS:
            for objective in OBJECTIVES:
                pairs, counts = _preference_pairs(rows, fraction, objective)
                key = f'{kernel}|{profile}|{clock:g}ns|{fraction:g}|{objective}'
                pair_audit['contexts'][key] = {
                    'split': split_name, 'kernel': kernel, 'profile': profile,
                    'clock_ns': clock, 'budget_fraction': fraction,
                    'objective': objective, 'candidate_count': len(rows),
                    'candidate_pool_sha256': _sha([row['config_hash'] for row in rows]),
                    'measured_feasible': sum(all(v <= c*fraction for v,c in zip(r['physical'][1:],r['capacities'])) for r in rows),
                    'measured_infeasible': sum(not all(v <= c*fraction for v,c in zip(r['physical'][1:],r['capacities'])) for r in rows),
                    'preference_pairs': len(pairs),
                    'feasible_vs_infeasible_pairs': counts.get('feasible_vs_infeasible',0),
                    'feasible_objective_pairs': counts.get('objective',0),
                    'near_ties_excluded': counts.get('near_tie',0),
                    'both_infeasible_pairs_excluded': counts.get('both_infeasible',0),
                    'zero_area_ambiguous_pairs_excluded': counts.get('zero_area_ambiguous',0),
                }
                for name, value in counts.items():
                    pair_audit['totals'][name] += value
                pair_audit['totals']['preference_pairs'] += len(pairs)
                preference_groups[split_name].append({
                    'key': key, 'kernel': kernel, 'profile': profile,
                    'clock': clock, 'fraction': fraction, 'objective': objective,
                    'rows': rows, 'pairs': pairs,
                })
    pair_audit['totals'] = dict(pair_audit['totals'])
    graph_audit = {}
    kernel_stats = {item['kernel']: item for item in data.kernel_manifest}
    for kernel in roster:
        graph_audit[kernel] = next(item for item in data.audit['kernels']
            if Path(item['graph_member']).name.startswith(kernel + '-'))
    manifest = {
        'schema': SCHEMA + '-cohort', 'seed': SEED, 'candidate_selection': {
            'training': 'greedy complete-directive Hamming diversity with hash tie-breaks; QoR-blind',
            'validation': 'highest target-support assignments, then hash tie-break; QoR-blind',
            'candidate_count_cap': N_CANDIDATES, 'six_target_intersection_required': False,
            'target_coverage_priority': 'prefer assignments measured in more target contexts; retain per-context count variation',
        },
        'selection_rationale': ('Label-independent coverage screen using authenticated training-side graph/action inventory, '
            'per-target valid measurement counts, exact assignment support across six device/clock contexts, and '
            'structural diversity including loop-only and multiple array-action kernels. Validation QoR was not used.'),
        'training_kernels': list(TRAIN_KERNELS), 'validation_kernels': list(VALIDATION_KERNELS),
        'family_split': {k: split[k] for k in roster},
        'family_identity': {k: family_map.get(k, k) for k in roster}, 'kernels': selection,
        'dataset_sha256': data.audit['dataset']['dataset_sha256'],
        'family_split_sha256': data.audit['dataset']['split_sha256'],
        'graph_release_sha256': data.audit['graph_release']['archive_sha256'],
        'graph_provenance': {k: {name:graph_audit[k].get(name) for name in
            ('graph_identity','graph_member','graph_member_sha256','source_sha256',
             'metadata_sha256','apl_sha256','preprocessed_csv_sha256')}
            for k in roster},
        'source_action_mapping_sha256': {k:_sha([
            {'action_id':label,'kind':a.kind,'trip_count':a.trip_count,
             'dimensions':a.dimensions} for label,a in sorted(data.actions[k].items())])
            for k in roster},
        'measurement_record_count': len(records),
        'selected_measurements': [{
            'kernel':row['kernel'],'family':row['kernel_family'],
            'device':row['device'],'profile':row['profile'],'clock_ns':row['clock'],
            'preprocessed_csv_sha256':kernel_stats[row['kernel']]['table_sha256'],
            'preprocessed_row':row['preprocessed_row'],
            'complete_assignment_sha256':row['config_hash'],
        } for row in sorted(records,key=lambda item:(item['kernel'],item['profile'],
                                                   item['clock'],item['config_hash']))],
        'physical_measurement_key': ['kernel', 'complete_assignment_hash', 'device', 'clock_ns'],
        'budget_views_are_correlated': True,
        'test_kernels_loaded_for_probe': [],
    }
    manifest['manifest_sha256'] = _sha(manifest)
    pair_audit['cohort_manifest_sha256'] = manifest['manifest_sha256']
    coverage = target_preference_coverage(by_context, split, selection, family_map)
    coverage['cohort_manifest_sha256'] = manifest['manifest_sha256']
    return manifest, pair_audit, coverage, preference_groups, records, by_context


def target_preference_coverage(by_context, split, selection, family_map=None):
    family_map=family_map or {}
    report = {'schema': SCHEMA + '-target-preference-coverage', 'axes': {},
              'family_macro': {}, 'budget_views_correlated': True,
              'assignment_pools':{}}
    def pool_reference(hashes):
        values=sorted(hashes); digest=_sha(values)
        report['assignment_pools'].setdefault(digest,values)
        return digest
    comparisons = defaultdict(list)
    for kernel in selection:
        targets = {}
        for profile in PROFILES:
            for clock in CLOCKS:
                rows = by_context.get((kernel, profile, clock), [])
                targets[(profile, clock)] = {r['config_hash']: r for r in rows}
        for clock in CLOCKS:
            left, right = targets[('zcu104', clock)], targets[('u200', clock)]
            shared = sorted(set(left) & set(right))
            if shared:
                comparisons['device'].append({'kernel':kernel,'family':family_map.get(kernel,kernel),
                    'context':f'{clock:g}ns','matched_assignments':len(shared),
                    'physical_qor_pair_count':len(shared),
                    'matched_assignment_pool_sha256':pool_reference(shared)})
        for profile in PROFILES:
            for i,left_clock in enumerate(CLOCKS):
                for right_clock in CLOCKS[i+1:]:
                    left,right=targets[(profile,left_clock)],targets[(profile,right_clock)]
                    shared=sorted(set(left)&set(right))
                    if shared:
                        comparisons['clock'].append({'kernel':kernel,'family':family_map.get(kernel,kernel),
                            'context':f'{profile}|{left_clock:g}->{right_clock:g}ns',
                            'matched_assignments':len(shared),'physical_qor_pair_count':len(shared),
                            'matched_assignment_pool_sha256':pool_reference(shared)})
        # Preference reversal analysis uses the common measured assignment pool.
        for objective in OBJECTIVES:
            for axis, pairs in [('device', [(('zcu104',c),('u200',c),f'{c:g}ns') for c in CLOCKS]),
                                ('clock', [( (p,c1),(p,c2),f'{p}|{c1:g}->{c2:g}ns') for p in PROFILES for i,c1 in enumerate(CLOCKS) for c2 in CLOCKS[i+1:]])]:
                for t1,t2,label in pairs:
                    left,right=targets[t1],targets[t2]
                    shared=sorted(set(left)&set(right))
                    if len(shared)<2: continue
                    for fraction in (1.0,0.5,0.25,0.1):
                        a=[left[h] for h in shared]; b=[right[h] for h in shared]
                        def best(rows):
                            feas=[r for r in rows if all(v<=c*fraction for v,c in zip(r['physical'][1:],r['capacities']))]
                            if not feas:return None
                            if objective in ('PARETO_AREA','PARETO_ADP'):
                                feas=[r for r in feas if not _is_zero_area_ambiguous(r)]
                            if not feas:return None
                            ordered=sorted(feas,key=lambda r:(_score(r,objective),r['config_hash']))
                            first=_score(ordered[0],objective)
                            second=_score(ordered[1],objective) if len(ordered)>1 else None
                            margin=(abs(second-first)/max(abs(first),abs(second),1e-12) if second is not None else None)
                            meaningful=(second is None or margin>PREFERENCE_MARGIN)
                            return {'winner':ordered[0]['config_hash'],'margin_to_runner_up':margin,
                                    'meaningful':meaningful,'feasible_count':len(ordered)}
                        x,y=best(a),best(b)
                        entry={'kernel':kernel,'family':family_map.get(kernel,kernel),'pair':label,
                               'objective':objective,'budget_fraction':fraction,
                               'left_request':{'profile':t1[0],'clock_ns':t1[1],
                                               'budget_fraction':fraction,'objective':objective},
                               'right_request':{'profile':t2[0],'clock_ns':t2[1],
                                                'budget_fraction':fraction,'objective':objective},
                               'matched_candidate_count':len(shared),
                               'matched_assignment_pool_sha256':pool_reference(shared),
                               'left':x,'right':y,
                               'classification':_preference_change_class(x,y)}
                        comparisons[axis+'_preference'].append(entry)
        # Budget-only and objective-only preference changes use the same
        # measured assignment pool; physical QoR is never regenerated per budget.
        for profile in PROFILES:
            for clock in CLOCKS:
                rows_by_hash=targets[(profile,clock)]
                if not rows_by_hash: continue
                rows=[rows_by_hash[h] for h in sorted(rows_by_hash)]
                for objective in OBJECTIVES:
                    for i,low in enumerate(BUDGETS):
                        for high in BUDGETS[i+1:]:
                            x=_best_preference(rows,low,objective)
                            y=_best_preference(rows,high,objective)
                            comparisons['budget_preference'].append({
                                'kernel':kernel,'family':family_map.get(kernel,kernel),
                                'pair':f'{profile}|{clock:g}ns|{low:g}->{high:g}',
                                'left_request':{'profile':profile,'clock_ns':clock,
                                    'budget_fraction':low,'objective':objective},
                                'right_request':{'profile':profile,'clock_ns':clock,
                                    'budget_fraction':high,'objective':objective},
                                'objective':objective,'matched_candidate_count':len(rows),
                                'matched_assignment_pool_sha256':pool_reference(rows_by_hash),
                                'left':x,'right':y,
                                'classification':_preference_change_class(x,y)})
                for fraction in BUDGETS:
                    for i,left_objective in enumerate(OBJECTIVES):
                        for right_objective in OBJECTIVES[i+1:]:
                            x=_best_preference(rows,fraction,left_objective)
                            y=_best_preference(rows,fraction,right_objective)
                            comparisons['objective_preference'].append({
                                'kernel':kernel,'family':family_map.get(kernel,kernel),
                                'pair':f'{profile}|{clock:g}ns|{fraction:g}|{left_objective}->{right_objective}',
                                'left_request':{'profile':profile,'clock_ns':clock,
                                    'budget_fraction':fraction,'objective':left_objective},
                                'right_request':{'profile':profile,'clock_ns':clock,
                                    'budget_fraction':fraction,'objective':right_objective},
                                'objective':f'{left_objective}->{right_objective}',
                                'matched_candidate_count':len(rows),
                                'matched_assignment_pool_sha256':pool_reference(rows_by_hash),
                                'left':x,'right':y,
                                'classification':_preference_change_class(x,y)})
    for axis, rows in comparisons.items():
        report['axes'][axis]={'comparison_count':len(rows),'distinct_kernels':len({r['kernel'] for r in rows}),
                              'rows':rows}
    return report


def _best_preference(rows, fraction, objective):
    feasible=[r for r in rows if all(v<=c*fraction for v,c in zip(r['physical'][1:],r['capacities']))]
    if objective in ('PARETO_AREA','PARETO_ADP'):
        feasible=[r for r in feasible if not _is_zero_area_ambiguous(r)]
    if not feasible: return None
    ordered=sorted(feasible,key=lambda r:(_score(r,objective),r['config_hash']))
    first=_score(ordered[0],objective)
    second=_score(ordered[1],objective) if len(ordered)>1 else None
    gap=abs(second-first)/max(abs(first),abs(second),1e-12) if second is not None else None
    return {'winner':ordered[0]['config_hash'],'gap_to_runner_up':gap,
            'meaningful':(second is None or gap>PREFERENCE_MARGIN),
            'feasible_count':len(ordered)}


def _preference_change_class(left,right):
    if left is None or right is None: return 'unsupported_empty_feasible_pool'
    if left['meaningful'] and right['meaningful'] and left['winner']!=right['winner']:
        return 'meaningful_change'
    if left['meaningful'] and right['meaningful'] and left['winner']==right['winner']:
        return 'meaningful_invariant'
    if left['winner']==right['winner']: return 'invariant_near_tie'
    return 'winner_change_near_tie'


def capability_vector(registry, profile_name, clock):
    profile=registry['profiles'][profile_name]
    caps=[]; known=[]
    for resource in ('BRAM_18K','DSP','FF','LUT'):
        item=profile['resources'][resource]
        caps.append(math.log1p(float(item['value']))/24.0 if item.get('known') else 0.0)
        known.append(float(bool(item.get('known'))))
    cap_values=[]; cap_masks=[]
    for res in ('BRAM_18K','DSP','FF','LUT'):
        props=registry['capabilities'][res].get('properties',{})
        for name in sorted(props):
            prop=props[name]; value=prop.get('value'); is_known=bool(prop.get('known')) and value is not None
            cap_values.append(math.log1p(float(value))/16.0 if is_known else 0.0)
            cap_masks.append(float(is_known))
    mem=profile.get('external_memory',{})
    mem_kind=mem.get('architecture','unknown').lower()
    memory_kind=[float('ddr' in mem_kind),float('hbm' in mem_kind),float(mem_kind=='unknown')]
    mem_bw=mem.get('bandwidth_bytes_per_second')
    return torch.tensor(caps+known+[1.0/float(clock)]+cap_values+cap_masks+memory_kind+
                        [math.log1p(float(mem_bw))/40 if mem_bw is not None else 0.0,
                         float(bool(mem.get('bandwidth_known')))],dtype=torch.float32)


def decision_context(profile, clock, fraction, objective, registry):
    capacities=[float(registry['profiles'][profile]['resources'][r]['value']) for r in RESOURCE_ORDER]
    target=[math.log1p(v)/24.0 for v in capacities]+[1.0/float(clock)]
    budget=[float(fraction)]*4
    objective_hot=[float(objective==name) for name in train.OBJECTIVES]
    return torch.tensor(target+budget+objective_hot,dtype=torch.float32)


class PreferenceMemoryModel(nn.Module):
    """Shared V4 action encoder and preference head with one of three memory paths."""
    def __init__(self, mode, profile_feature_dim=None):
        super().__init__()
        if mode not in ('compiler', 'target_film', 'hardware_messages'):
            raise ValueError('Unknown memory mode')
        self.mode=mode
        self.encoder=tensors.SmallGraphEncoder(hidden=64)
        self.action_kind=nn.Embedding(3,4)
        self.local=nn.Sequential(nn.Linear(128+train.CANDIDATE_DIM+
            train.CANDIDATE_DEMAND_DIM+5,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.scorer=nn.Sequential(nn.Linear(32+12,64),nn.ReLU(),nn.Linear(64,1))
        self.modulation=None
        self.gate=None
        if mode=='target_film':
            if profile_feature_dim is None:
                raise ValueError('Target-FiLM needs a fixed authenticated profile width')
            self.modulation=nn.Sequential(nn.Linear(128+profile_feature_dim+tensors.ACTION_DEMAND_DIM,128),
                                          nn.ReLU(),nn.Linear(128,256))
            self.gate=nn.Parameter(torch.zeros(()))

    def encode_memory(self, graph, profile_features=None):
        memory,_=self.encoder.memories(graph)
        if memory.shape[0]!=1:
            raise ValueError('One graph is required per target-context memory')
        if self.mode=='target_film':
            if profile_features is None: raise ValueError('Target-FiLM requires authenticated profile features')
            active=(graph.registered_mask & graph.matched_mask).squeeze(0)
            relevance=graph.action_demand[0]
            z=profile_features.to(memory.device).reshape(1,-1).expand(memory.shape[1],-1)
            raw=self.modulation(torch.cat([memory[0],z,relevance],dim=-1))
            gamma,beta=raw.chunk(2,dim=-1)
            mod=memory[0]+torch.tanh(self.gate)*(torch.tanh(gamma)*memory[0]+beta)
            result=memory.clone()
            result[0,active]=mod[active]
            return result
        return memory

    def score(self, graph, memory, candidates, demands, decision):
        n=candidates.shape[0]
        kind=self.action_kind(graph.action_kind_slot.expand(n,-1))
        matched=graph.matched_mask.expand(n,-1).float().unsqueeze(-1)
        local=self.local(torch.cat([memory.expand(n,-1,-1),candidates,demands,kind,matched],dim=-1))
        registered=graph.registered_mask.expand(n,-1).float().unsqueeze(-1)
        pooled=(local*registered).sum(1)/registered.sum(1).clamp_min(1)
        return self.scorer(torch.cat([pooled,decision.to(local.device).expand(n,-1)],dim=-1)).squeeze(-1)


def _target_tensor(data, row):
    base,auth=data.graphs[row['kernel']]
    # Primary structural memory depends on physical target/clock, never on the
    # externally requested feasibility budget or optimization objective.
    capacities={r:int(v) for r,v in zip(RESOURCE_ORDER,row['capacities'])}
    enriched=overlay.enrich(base,row['profile'],row['clock'],budgets=capacities,authentication=auth)
    return tensors.tensorize(enriched)


def prepare_features(data, records, timeout_seconds=1800):
    started=time.perf_counter()
    graph_cache={}; cand_cache={}; demand_cache={}; report={}
    for i,row in enumerate(records):
        key=(row['kernel'],row['profile'],float(row['clock']))
        if key not in graph_cache:
            graph=_target_tensor(data,row)
            graph_cache[key]={'hardware':graph.cpu(), 'compiler':tensors.without_hardware_messages(graph).cpu()}
        identity=(row['kernel'],row['profile'],float(row['clock']),row['config_hash'])
        if identity not in cand_cache:
            cand,demand=data.candidate_inputs(row)
            cand_cache[identity]=cand.cpu(); demand_cache[identity]=demand.cpu()
        if time.perf_counter()-started>timeout_seconds:
            raise TimeoutError('Feature preparation exceeded the 30-minute limit')
        if (i+1)%100==0: print(f'prepared {i+1}/{len(records)} candidate contexts',flush=True)
    report={'graph_tensor_count':len(graph_cache),'candidate_context_count':len(cand_cache),
            'elapsed_seconds':time.perf_counter()-started}
    return graph_cache,cand_cache,demand_cache,report


def _group_records_for_preference(groups, kernel, profile, clock, fraction, objective):
    context_key=f'{kernel}|{profile}|{clock:g}ns|{fraction:g}|{objective}'
    return next((g for g in groups if g['key']==context_key),None)


def _memory_ablation(memory, graph, variant):
    if variant=='correct': return memory
    active=(graph.registered_mask & graph.matched_mask).squeeze(0)
    result=memory.clone()
    if variant=='zero':
        result.zero_()
    elif variant=='global_mean':
        if active.any(): result[0,active]=memory[0,active].mean(0)
    elif variant=='shuffled':
        slots=torch.where(active)[0]
        if len(slots)>1: result[0,slots]=memory[0,slots.roll(1)]
    else:
        raise ValueError('Unknown memory ablation: '+variant)
    return result


def evaluate_model(model, groups, graph_cache, cand_cache, demand_cache, registry,
                   split_name, device, profile_cache=None, memory_variant='correct',
                   mismatched_memories=None, deadline=None):
    model.eval(); pair_acc=[]; family_acc=defaultdict(list); top1=[]; top3=[]; contexts=[]
    unsupported_memory_contexts=[]
    if profile_cache is None: profile_cache={}
    memory_cache={}
    with torch.no_grad():
        for group in groups:
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError('Evaluation exceeded the frozen GPU wall-clock budget')
            records=group['rows']; kernel=group['kernel']; profile=group['profile']; clock=group['clock']
            if not records:
                top1.append({'context':group['key'],'kernel':kernel,
                             'family':None,'accuracy':None,'pair_count':0,
                             'selected_hash':None,'selected_feasible':None,
                             'profile':group['profile'],'clock_ns':group['clock'],
                             'budget_fraction':group['fraction'],'objective':group['objective'],
                             'measured_feasible_count':0,'no_candidate_pool':True,
                             'top1_regret':None,'top1_absolute_regret':None,
                             'top3_regret':None,'top3_absolute_regret':None})
                continue
            gkey=(kernel,profile,clock); graph=graph_cache[gkey][
                'hardware' if model.mode=='hardware_messages' else 'compiler'].to(device)
            target_key=(profile,clock)
            if target_key not in profile_cache:
                profile_cache[target_key]=capability_vector(registry,profile,clock)
            mkey=(model.mode,kernel,profile,clock)
            if mkey not in memory_cache:
                memory_cache[mkey]=model.encode_memory(graph,profile_cache[target_key].to(device))
            memory=memory_cache[mkey]
            if memory_variant=='target_mismatched':
                if mismatched_memories is None or (kernel,profile,clock) not in mismatched_memories:
                    unsupported_memory_contexts.append(group['key'])
                    continue
                memory=mismatched_memories[(kernel,profile,clock)]
            else:
                memory=_memory_ablation(memory,graph,memory_variant)
            identities=[(r['kernel'],r['profile'],r['clock'],r['config_hash']) for r in records]
            candidates=torch.stack([cand_cache[i] for i in identities]).to(device)
            demands=torch.stack([demand_cache[i] for i in identities]).to(device)
            decision=decision_context(profile,clock,group['fraction'],group['objective'],registry)
            scores=model.score(graph,memory,candidates,demands,decision).cpu().numpy()
            pairs=group['pairs']
            correct=sum(scores[a]<scores[b] for a,b,_ in pairs)
            if pairs:
                acc=correct/len(pairs); pair_acc.append(acc); family_acc[records[0].get('kernel_family',records[0]['kernel'])].append(acc)
            feasible=[all(v<=c*group['fraction'] for v,c in zip(r['physical'][1:],r['capacities'])) for r in records]
            eligible=np.asarray([not (_is_zero_area_ambiguous(r) and group['objective'] in
                                     ('PARETO_AREA','PARETO_ADP')) for r in records],dtype=bool)
            order=np.asarray([int(i) for i in np.argsort(scores,kind='stable') if eligible[int(i)]],dtype=int)
            if not len(order):
                top1.append({'context':group['key'],'kernel':kernel,
                             'profile':profile,'clock_ns':clock,
                             'budget_fraction':group['fraction'],'objective':group['objective'],
                             'accuracy':None,'pair_count':len(pairs),
                             'selected_hash':None,'selected_feasible':None,
                             'measured_feasible_count':sum(feasible),'no_candidate_pool':True,
                             'top1_regret':None,'top1_absolute_regret':None,
                             'top3_regret':None,'top3_absolute_regret':None})
                continue
            selected=int(order[0])
            top1_row={'context':group['key'],'kernel':kernel,
                      'family':records[0].get('kernel_family',kernel),
                      'profile':profile,'clock_ns':clock,
                      'budget_fraction':group['fraction'],'objective':group['objective'],
                      'accuracy':(correct/len(pairs) if pairs else None),
                      'pair_count':len(pairs),'selected_hash':records[selected]['config_hash'],
                      'candidate_scores':[{'config_hash':r['config_hash'],'score':float(scores[i]),
                          'measured_feasible':bool(feasible[i])} for i,r in enumerate(records)],
                      'selected_feasible':bool(feasible[selected]),
                      'measured_feasible_count':sum(bool(feasible[i] and eligible[i]) for i in range(len(feasible))),
                      'top1_regret':None,'top1_absolute_regret':None,'top3_regret':None,
                      'top3_absolute_regret':None}
            valid_feasible=[i for i,yes in enumerate(feasible) if yes and eligible[i]]
            if valid_feasible:
                objective_values=[_score(r,group['objective']) for r in records]
                best=min(objective_values[i] for i in valid_feasible)
                if feasible[selected]:
                    value=objective_values[selected]
                    top1_row['top1_regret']=(value-best)/max(abs(best),1e-12)
                    top1_row['top1_absolute_regret']=value-best
                shortlist=[i for i in order[:3] if feasible[int(i)] and eligible[int(i)]]
                if shortlist:
                    value=min(objective_values[int(i)] for i in shortlist)
                    top1_row['top3_regret']=(value-best)/max(abs(best),1e-12)
                    top1_row['top3_absolute_regret']=value-best
            top1.append(top1_row)
            contexts.append({'key':group['key'],'scores':scores.tolist(),'pair_accuracy':top1_row['accuracy'],
                             'selected_hash':top1_row['selected_hash'],'selected_feasible':top1_row['selected_feasible']})
    feasible_contexts=[r for r in top1 if r['measured_feasible_count']>0 and r.get('selected_feasible') is not None]
    mean_top1=np.mean([r['top1_regret'] for r in top1 if r['top1_regret'] is not None]) if any(r['top1_regret'] is not None for r in top1) else None
    mean_top3=np.mean([r['top3_regret'] for r in top1 if r['top3_regret'] is not None]) if any(r['top3_regret'] is not None for r in top1) else None
    family_summary={}
    for family in sorted({r['family'] for r in top1 if r.get('family')}):
        rows=[r for r in top1 if r.get('family')==family]
        rows_with_pairs=[r for r in rows if r['accuracy'] is not None]
        family_summary[family]={
            'contexts':len(rows),'pair_contexts':len(rows_with_pairs),
            'preference_accuracy_macro':float(np.mean([r['accuracy'] for r in rows_with_pairs])) if rows_with_pairs else None,
            'selected_feasible_rate':float(np.mean([r['selected_feasible'] for r in rows if r.get('selected_feasible') is not None])) if any(r.get('selected_feasible') is not None for r in rows) else None,
            'top1_regret_macro':float(np.mean([r['top1_regret'] for r in rows if r['top1_regret'] is not None])) if any(r['top1_regret'] is not None for r in rows) else None,
        }
    by_objective={}
    for objective in OBJECTIVES:
        objective_rows=[r for r in top1 if r.get('objective')==objective]
        pair_rows=[r for r in objective_rows if r['accuracy'] is not None]
        family_pairs=defaultdict(list)
        for row in pair_rows:
            family_pairs[row['family']].append(row['accuracy'])
        objective_feasible=[r for r in objective_rows
            if r['measured_feasible_count']>0 and r.get('selected_feasible') is not None]
        objective_regrets=[r['top1_regret'] for r in objective_rows if r['top1_regret'] is not None]
        objective_top3=[r['top3_regret'] for r in objective_rows if r['top3_regret'] is not None]
        by_objective[objective]={
            'context_count':len(objective_rows),'pair_context_count':len(pair_rows),
            'preference_accuracy_family_macro':float(np.mean(
                [np.mean(values) for values in family_pairs.values()])) if family_pairs else None,
            'top1_actual_feasible_rate':float(np.mean(
                [r['selected_feasible'] for r in objective_feasible])) if objective_feasible else None,
            'top1_actual_feasible_denominator':len(objective_feasible),
            'selected_infeasible_count':sum(r.get('selected_feasible') is False for r in objective_rows),
            'top1_relative_regret_macro':float(np.mean(objective_regrets)) if objective_regrets else None,
            'top3_relative_regret_macro':float(np.mean(objective_top3)) if objective_top3 else None,
        }
    pair_population=[r for r in top1 if r['accuracy'] is not None]
    result={'split':split_name,'preference_accuracy_pair_micro':sum(
        r['accuracy']*r['pair_count'] for r in pair_population)/max(1,sum(r['pair_count'] for r in pair_population)),
        'preference_accuracy_context_macro':float(np.mean(pair_acc)) if pair_acc else None,
        'preference_accuracy_family_macro':float(np.mean([np.mean(x) for x in family_acc.values()])) if family_acc else None,
        'top1_actual_feasible_rate':float(np.mean([r['selected_feasible'] for r in feasible_contexts])) if feasible_contexts else None,
        'top1_actual_feasible_denominator':len(feasible_contexts),
        'top1_relative_regret_macro':float(mean_top1) if mean_top1 is not None else None,
        'top3_relative_regret_macro':float(mean_top3) if mean_top3 is not None else None,
        'selected_infeasible_count':sum(r.get('selected_feasible') is False for r in top1),
        'context_count':len(top1),'no_candidate_pool_count':sum(bool(r.get('no_candidate_pool')) for r in top1),
        'unsupported_memory_context_count':len(unsupported_memory_contexts),
        'unsupported_memory_contexts':unsupported_memory_contexts,
        'family_summary':family_summary,'memory_variant':memory_variant,'contexts':top1}
    result['by_objective']=by_objective
    return result


def build_mismatched_memories(model, groups, graph_cache, registry, device):
    """Encode a different physical target while the scorer target stays fixed."""
    model.eval(); cache={}; available={}
    for group in groups:
        if group['rows']:
            available[(group['kernel'],group['profile'],group['clock'])]=None
    with torch.no_grad():
        for kernel,profile,clock in sorted(available):
            alternatives=[key for key in available if key[0]==kernel and key!=(kernel,profile,clock)]
            # Prefer a device change at the same clock, then a clock change.
            alternatives.sort(key=lambda key:(key[2]!=clock,key[1]==profile,key[1],key[2]))
            if not alternatives: continue
            _,other_profile,other_clock=alternatives[0]
            source_graph=graph_cache[(kernel,other_profile,other_clock)][
                'hardware' if model.mode=='hardware_messages' else 'compiler'].to(device)
            pfeat=capability_vector(registry,other_profile,other_clock).to(device)
            cache[(kernel,profile,clock)]=model.encode_memory(source_graph,pfeat)
    return cache


def target_response_report(coverage, validation_result):
    """Compare target preference on the exact common measured candidate pool.

    Each gold target comparison is constructed over a referenced assignment pool.
    Ranking the full per-context pool here would compare different candidate sets
    and could credit a prediction for selecting an assignment unavailable on the
    paired target. Re-score only the authenticated intersection.
    """
    predictions={row['context']:{candidate['config_hash']:candidate['score']
                    for candidate in row.get('candidate_scores', [])}
                 for row in validation_result['contexts']}
    report={}
    validation=set(VALIDATION_KERNELS)
    for axis in ('device_preference','clock_preference','budget_preference','objective_preference'):
        axis_rows=coverage.get('axes',{}).get(axis,{}).get('rows',[])
        totals=defaultdict(int); family=defaultdict(lambda:defaultdict(int)); detail=[]
        for item in axis_rows:
            if item['kernel'] not in validation: continue
            gold_class=item['classification']; left_gold=item.get('left'); right_gold=item.get('right')
            if gold_class in ('meaningful_change','meaningful_invariant') and left_gold and right_gold:
                totals['meaningful_pairs']+=1; family[item['family']]['meaningful_pairs']+=1
            if gold_class not in ('meaningful_change','meaningful_invariant'):
                totals[gold_class]+=1; family[item['family']][gold_class]+=1
                continue
            lreq=item['left_request']; rreq=item['right_request']
            lkey=f"{item['kernel']}|{lreq['profile']}|{lreq['clock_ns']:g}ns|{lreq['budget_fraction']:g}|{lreq['objective']}"
            rkey=f"{item['kernel']}|{rreq['profile']}|{rreq['clock_ns']:g}ns|{rreq['budget_fraction']:g}|{rreq['objective']}"
            matched=item.get('matched_assignment_hashes')
            if matched is None:
                matched=coverage.get('assignment_pools',{}).get(
                    item.get('matched_assignment_pool_sha256'),[])
            matched=set(matched)
            left_scores=predictions.get(lkey, {})
            right_scores=predictions.get(rkey, {})
            common=matched & set(left_scores) & set(right_scores)
            pred_l=(min(common,key=lambda h:(left_scores[h],h)) if common else None)
            pred_r=(min(common,key=lambda h:(right_scores[h],h)) if common else None)
            if pred_l is None or pred_r is None:
                cls='prediction_context_unavailable'
            elif gold_class=='meaningful_change':
                if pred_l==left_gold['winner'] and pred_r==right_gold['winner']:
                    cls='correct_changed_complete_winners'
                elif pred_l!=pred_r:
                    cls='changed_but_wrong_winner'
                else:
                    cls='incorrect_invariance'
            elif pred_l==left_gold['winner'] and pred_r==right_gold['winner']:
                cls='correct_invariant_complete_winner'
            elif pred_l==pred_r:
                cls='invariant_but_wrong_winner'
            else:
                cls='unnecessary_change'
            totals[cls]+=1; family[item['family']][cls]+=1
            detail.append({'kernel':item['kernel'],'family':item['family'],'pair':item['pair'],
                'expected':gold_class,'prediction_class':cls,'left_selected':pred_l,
                'right_selected':pred_r,'left_measured_winner':left_gold['winner'],
                'right_measured_winner':right_gold['winner'],
                'matched_candidate_count':len(common)})
        report[axis]={'totals':dict(totals),'family_counts':{k:dict(v) for k,v in family.items()},
                      'comparisons':detail}
    return report


def _sample_pairs(group, count, rng):
    # Balance feasible-vs-infeasible and within-feasible objective comparisons.
    by_kind=defaultdict(list)
    for pair in group['pairs']: by_kind[pair[2]].append(pair)
    kinds=[k for k,v in by_kind.items() if v]
    result=[]
    for _ in range(count):
        kind=rng.choice(kinds); result.append(rng.choice(by_kind[kind]))
    return result


def _gradient_norm(parameters):
    values=[p.grad.detach().float().norm().square() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def _gradient_breakdown(model):
    hardware_relations=set(overlay.RELATIONS)
    row_ids=[i for i,name in enumerate(tensors.RELATIONS)
             if name.removeprefix('reverse:') in hardware_relations]
    relation_grad=model.encoder.relation.weight.grad
    hardware_relation_norm=(float(relation_grad[row_ids].detach().float().norm().cpu())
                            if relation_grad is not None and row_ids else 0.0)
    head=list(model.action_kind.parameters())+list(model.local.parameters())+list(model.scorer.parameters())
    conditioning=(list(model.modulation.parameters())+[model.gate]
                  if model.modulation is not None else [])
    result={'encoder_l2':_gradient_norm(model.encoder.parameters()),
            'hardware_relation_embedding_l2':hardware_relation_norm,
            'target_conditioner_l2':_gradient_norm(conditioning),
            'preference_head_l2':_gradient_norm(head)}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite parameter-group gradient statistic')
    return result


def _train_arm(mode, train_groups, val_groups, graph_cache, cand_cache, demand_cache,
               registry, output_dir, device, protocol_sha256,
               updates=300, eval_every=100, time_limit=1200):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    profile_dim=capability_vector(registry,'zcu104',10.0).numel()
    model=PreferenceMemoryModel(mode,profile_dim).to(device)
    common_parameters=(list(model.encoder.parameters())+list(model.action_kind.parameters())+
                       list(model.local.parameters())+list(model.scorer.parameters()))
    initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in common_parameters)).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng=random.Random(SEED)
    profile_cache={(p,c):capability_vector(registry,p,c) for p in PROFILES for c in CLOCKS}
    started=time.perf_counter(); history=[]; sampling_trace=[]; best=None; best_score=-float('inf'); best_step=None
    groups=[g for g in train_groups if g['pairs']]
    for step in range(1,updates+1):
        model.train(); group=rng.choice(groups)
        rows=group['rows']; graph=graph_cache[(group['kernel'],group['profile'],group['clock'])][
            'hardware' if mode=='hardware_messages' else 'compiler'].to(device)
        pc=profile_cache[(group['profile'],group['clock'])].to(device)
        memory=model.encode_memory(graph,pc)
        ids=[(r['kernel'],r['profile'],r['clock'],r['config_hash']) for r in rows]
        candidates=torch.stack([cand_cache[i] for i in ids]).to(device)
        demands=torch.stack([demand_cache[i] for i in ids]).to(device)
        dctx=decision_context(group['profile'],group['clock'],group['fraction'],group['objective'],registry).to(device)
        scores=model.score(graph,memory,candidates,demands,dctx)
        sampled=_sample_pairs(group,16,rng)
        sampling_trace.append({'step':step,'group':group['key'],
            'pairs':[[rows[a]['config_hash'],rows[b]['config_hash'],kind] for a,b,kind in sampled]})
        loss=F.softplus(torch.stack([scores[a]-scores[b] for a,b,_ in sampled])).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite preference loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        sampled_classes={kind:sum(pair[2]==kind for pair in sampled)
                         for kind in ('feasibility','objective')}
        gradient_groups=_gradient_breakdown(model)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        if not math.isfinite(grad_norm): raise ValueError('Nonfinite preference gradient')
        optimizer.step()
        if step%25==0 or step==1:
            history.append({'step':step,'loss':float(loss.detach()),'grad_norm':grad_norm,
                            'clip_applied':bool(grad_norm>1.0),
                            'sampled_pair_class_counts':sampled_classes,
                            'gradient_groups':gradient_groups,
                            'gate':float(torch.tanh(model.gate).detach()) if model.gate is not None else None})
        if step%eval_every==0 or step==updates:
            report=evaluate_model(model,val_groups,graph_cache,cand_cache,demand_cache,
                                  registry,'val',device,profile_cache,
                                  deadline=started+time_limit)
            score=report['by_objective']['PARETO_ADP']['preference_accuracy_family_macro']
            if score is not None and score>best_score:
                best_score=score; best_step=step; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append({'step':step,'validation':report,'selection_score':score})
            checkpointing.atomic_torch_save(output_dir/f'{mode}_step_{step}.pt',{
                'schema':SCHEMA,'mode':mode,'state_dict':model.state_dict(),'step':step,
                'validation_family_macro':score,'protocol_seed':SEED,
                'protocol_sha256':protocol_sha256})
            print(f'{mode} step {step}/{updates}: loss={float(loss.detach()):.5f} val_pref={score}',flush=True)
        if step%25==0 or step==1:
            checkpointing.atomic_json(output_dir/f'{mode}_progress.json',{
                'mode':mode,'last_completed_update':step,'planned_updates':updates,
                'elapsed_seconds':time.perf_counter()-started,
                'history':history,'sampling_trace_sha256':_sha(sampling_trace),
                'best_validation_step':best_step,'best_validation_score':best_score,
                'protocol_sha256':protocol_sha256})
        if time.perf_counter()-started>time_limit:
            raise TimeoutError(f'{mode} exceeded the 20-minute GPU budget')
    if best is None: raise RuntimeError('No validation-selected preference checkpoint')
    model.load_state_dict(best)
    checkpoint={
        'schema':SCHEMA,'mode':mode,'state_dict':best,'seed':SEED,'updates':updates,
        'best_validation_step':best_step,'best_validation_family_macro':best_score,
        'protocol_sha256':protocol_sha256,
        'initial_parameter_sha256':initial_hash,
        'parameter_count':sum(p.numel() for p in model.parameters()),
    }
    checkpoint_path=output_dir/f'{mode}_selected.pt'
    checkpointing.atomic_torch_save(checkpoint_path,checkpoint)
    final=evaluate_model(model,val_groups,graph_cache,cand_cache,demand_cache,registry,'val',device,profile_cache)
    return {'mode':mode,'checkpoint':str(checkpoint_path),'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'parameters':checkpoint['parameter_count'],'initial_parameter_sha256':initial_hash,
            'best_validation_step':best_step,'training_seconds':time.perf_counter()-started,
            'sampling_trace':sampling_trace,'sampling_trace_sha256':_sha(sampling_trace),
            'history':history,'selected_validation':final,'selected_state':best,'model':model}


def fixed_context_diagnostic(train_groups, graph_cache, cand_cache, demand_cache,
                             registry, device, updates=100):
    """Training-only check that one measured preference pool is learnable."""
    group=next((g for g in train_groups if g['kernel']=='machsuite-gemm-ncubed'
        and g['profile']=='u200' and g['clock']==10.0 and g['fraction']==1.0
        and g['objective']=='PARETO_ADP' and len(g['rows'])>=8 and len(g['pairs'])>=20),None)
    if group is None: return {'status':'no_supported_fixed_context'}
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model=PreferenceMemoryModel('hardware_messages',capability_vector(registry,'zcu104',10).numel()).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    graph=graph_cache[(group['kernel'],group['profile'],group['clock'])]['hardware'].to(device)
    pfeat=capability_vector(registry,group['profile'],group['clock']).to(device)
    ids=[(r['kernel'],r['profile'],r['clock'],r['config_hash']) for r in group['rows']]
    candidates=torch.stack([cand_cache[i] for i in ids]).to(device)
    demands=torch.stack([demand_cache[i] for i in ids]).to(device)
    decision=decision_context(group['profile'],group['clock'],group['fraction'],group['objective'],registry).to(device)
    pairs=group['pairs']; rng=random.Random(SEED); history=[]
    def diagnostics():
        model.eval()
        with torch.no_grad():
            memory=model.encode_memory(graph,pfeat)
            scores=model.score(graph,memory,candidates,demands,decision)
            correct=sum(scores[a]<scores[b] for a,b,_ in pairs)
            pair_loss=float(F.softplus(torch.stack([scores[a]-scores[b] for a,b,_ in pairs])).mean())
        model.train(); return pair_loss,correct/len(pairs)
    initial=diagnostics()
    for step in range(1,updates+1):
        sampled=_sample_pairs(group,min(16,len(pairs)),rng)
        memory=model.encode_memory(graph,pfeat)
        scores=model.score(graph,memory,candidates,demands,decision)
        loss=F.softplus(torch.stack([scores[a]-scores[b] for a,b,_ in sampled])).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if step%20==0 or step==updates:
            full=diagnostics(); history.append({'step':step,'sampled_loss':float(loss.detach()),
                                                'fixed_pair_loss':full[0],'pair_accuracy':full[1]})
    final=diagnostics()
    return {'status':'completed_training_only','context':group['key'],'candidate_count':len(group['rows']),
            'meaningful_pairs':len(pairs),'initial_pair_loss':initial[0],
            'initial_pair_accuracy':initial[1],'final_pair_loss':final[0],
            'final_pair_accuracy':final[1],'updates':updates,'history':history}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,required=True)
    parser.add_argument('--preprocessed-dir',type=Path,default=None,
                        help='Authenticated preprocessed CSV directory; default is the repository path')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--authenticated-audit',type=Path,default=None,
                        help='Reuse a byte-verified full dataset audit instead of rerunning it')
    parser.add_argument('--updates',type=int,default=300)
    parser.add_argument('--preparation-limit-seconds',type=int,default=1800)
    parser.add_argument('--gpu-limit-seconds',type=int,default=1200)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--gpu-name',default='A2')
    parser.add_argument('--gpu-uuid',default='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty output: '+str(args.output))
    if not args.audit_only and subprocess.check_output(
            ['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Training requires a clean, committed source snapshot')
    if not args.audit_only and args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('The requested GPU is unavailable; this experiment does not silently use CPU')
    if not args.audit_only and args.device.startswith('cuda') and torch.cuda.device_count()!=1:
        raise RuntimeError('Expose only the isolated A2 GPU to this process')
    args.output.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    audit_path=args.output/'authenticated_audit.json'
    data=train.PreparedData(args.release_dir,audit_path,selected_kernels=TRAIN_KERNELS+VALIDATION_KERNELS,
                            authenticated_audit_path=args.authenticated_audit,
                            preprocessed_dir=args.preprocessed_dir)
    checkpointing.atomic_json(args.output/'authenticated_audit.json',data.audit)
    manifest,pair_audit,coverage,groups,records,by_context=build_cohort(data)
    if time.perf_counter()-t0>args.preparation_limit_seconds:
        raise TimeoutError('Authenticated data/cohort preparation exceeded 30 minutes')
    checkpointing.atomic_json(args.output/'preference_cohort_manifest.json',manifest)
    checkpointing.atomic_json(args.output/'preference_pair_audit.json',pair_audit)
    checkpointing.atomic_json(args.output/'preference_target_coverage.json',coverage)
    if args.audit_only:
        checkpointing.atomic_json(args.output/'preference_audit_receipt.json',{
            'schema':SCHEMA+'-audit-receipt','manifest_sha256':manifest['manifest_sha256'],
            'pair_audit_sha256':_sha(pair_audit),'target_coverage_sha256':_sha(coverage),
            'selected_measurement_count':len(records),'training_preference_contexts':len(groups['train']),
            'validation_preference_contexts':len(groups['val']),
            'audit_seconds':time.perf_counter()-t0,'test_measurements_loaded':False})
        print(json.dumps({'output':str(args.output),'manifest_sha256':manifest['manifest_sha256'],
            'records':len(records),'train_groups':len(groups['train']),'validation_groups':len(groups['val']),
            'pair_totals':pair_audit['totals']},indent=2))
        return
    train_groups=groups['train']; val_groups=groups['val']
    if not train_groups or not val_groups: raise ValueError('No train/validation preference groups')
    prep_started=time.perf_counter()
    graph_cache,cand_cache,demand_cache,feat_report=prepare_features(data,records,args.preparation_limit_seconds)
    feat_report['elapsed_including_authenticated_load_and_cohort']=time.perf_counter()-t0
    checkpointing.atomic_json(args.output/'feature_preparation.json',feat_report)
    if time.perf_counter()-t0>args.preparation_limit_seconds:
        checkpointing.atomic_json(args.output/'preparation_limit.status.json',{
            'status':'STOPPED_PREPARATION_LIMIT','elapsed_seconds':time.perf_counter()-t0,
            'limit_seconds':args.preparation_limit_seconds,'features_prepared':feat_report})
        raise TimeoutError('Authenticated data and feature preparation exceeded its 30-minute limit')
    gpu_identity=None
    if args.device.startswith('cuda'):
        # Reuse the repository's UUID/name/process-isolation gate immediately
        # before GPU work; never silently select ordinal zero or share a GPU.
        gpu_identity=fast_probe._verify_idle_gpu(args.gpu_name,args.gpu_uuid)
        if args.gpu_name.casefold() not in torch.cuda.get_device_name(0).casefold():
            raise RuntimeError('CUDA device identity differs from the requested physical GPU')
    # Reuse deterministic cached features for all three matched arms.
    registry=data.registry
    run_protocol={'schema':SCHEMA,'seed':SEED,'updates_per_arm':args.updates,
        'evaluation_steps':list(range(100,args.updates+1,100)) if args.updates>=100 else [args.updates],
        'train_kernels':list(TRAIN_KERNELS),'validation_kernels':list(VALIDATION_KERNELS),
        'candidate_count_per_kernel':N_CANDIDATES,'budgets':list(BUDGETS),'objectives':list(OBJECTIVES),
        'preference_margin':PREFERENCE_MARGIN,'loss':'softplus(score_preferred-score_worse), balanced pair classes',
        'checkpoint_selection':'validation ADP preference accuracy macro-averaged over represented families',
        'optimizer':'AdamW','learning_rate':3e-4,'weight_decay':1e-4,
        'graph_release_sha256':manifest['graph_release_sha256'],'dataset_sha256':manifest['dataset_sha256'],
        'preprocessing_manifest_sha256':data.audit['dataset']['preprocessing_manifest_sha256'],
        'cohort_manifest_sha256':manifest['manifest_sha256'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_hashes':{name:overlay.file_hash(Path(path)) for name,path in {
            'preference_probe':'experiments/v22_gnn_training/directive_preference_probe.py',
            'training_model':'experiments/v22_gnn_training/train.py',
            'tensorizer':'GNN_branch/target_graph_tensor_v22.py',
            'target_overlay':'GNN_branch/target_graph_v22.py',
            'action_demand':'GNN_branch/action_demand_v22.py',
            'semantic_features':'GNN_branch/semantic_tensor_features_v22.py',
            'hardware_profiles':'GNN_branch/targets/fpga_profiles.json',
            'dataset_reconstruction':'Preprocessing/create_jsonl.py',
            'source_domains':'LLM_branch/common/directive_domains.py'}.items()},
        'test_data_loaded':False,'vitis_synthesis':False,'stage2_training':False,
        'gpu_identity':gpu_identity}
    run_protocol['protocol_sha256']=_sha(run_protocol)
    checkpointing.atomic_json(args.output/'protocol.json',run_protocol)
    cache_path=args.output/'feature_cache.pt'
    checkpointing.atomic_torch_save(cache_path,{'graph':graph_cache,'candidates':cand_cache,'demands':demand_cache,
       'manifest_sha256':manifest['manifest_sha256']})
    cache_hash=hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    wall=time.perf_counter(); deadline=wall+args.gpu_limit_seconds
    fixed=fixed_context_diagnostic(train_groups,graph_cache,cand_cache,demand_cache,
                                   registry,torch.device(args.device),updates=100)
    checkpointing.atomic_json(args.output/'fixed_context_diagnostic.json',fixed)
    if time.perf_counter()>deadline:
        raise TimeoutError('Fixed-context diagnostic exceeded the GPU time budget')
    arm_results=[]; models={}
    for mode in ('compiler','target_film','hardware_messages'):
        remaining=deadline-time.perf_counter()
        if remaining<=0: raise TimeoutError('GPU runtime budget exhausted before all matched arms completed')
        try:
            result=_train_arm(mode,train_groups,val_groups,graph_cache,cand_cache,demand_cache,
                              registry,args.output,torch.device(args.device),run_protocol['protocol_sha256'],
                              args.updates,time_limit=remaining)
        except TimeoutError as error:
            checkpointing.atomic_json(args.output/'gpu_phase.status.json',{
                'status':'STOPPED_GPU_TIME_LIMIT','message':str(error),
                'elapsed_seconds':time.perf_counter()-wall,
                'limit_seconds':args.gpu_limit_seconds,
                'active_arm':mode,'completed_arms':[item['mode'] for item in arm_results],
                'active_arm_progress':str(args.output/f'{mode}_progress.json'),
                'step_checkpoints':[str(path) for path in sorted(args.output.glob(f'{mode}_step_*.pt'))],
                'protocol_sha256':run_protocol['protocol_sha256']})
            return
        models[mode]=result.pop('model'); result.pop('selected_state',None)
        arm_results.append(result)
        if time.perf_counter()>deadline:
            raise TimeoutError('Three-arm GPU phase exceeded 20 minutes')
    initial_hashes={r['initial_parameter_sha256'] for r in arm_results}
    trace_hashes={r['sampling_trace_sha256'] for r in arm_results}
    if len(initial_hashes)!=1: raise ValueError('Matched arms have different common initial parameter states')
    if len(trace_hashes)!=1: raise ValueError('Matched arms consumed different preference pair traces')
    memory_ablation={}
    for mode,model in models.items():
        mismatched=build_mismatched_memories(model,val_groups,graph_cache,registry,torch.device(args.device))
        variants={}
        for variant in ('correct','zero','shuffled','global_mean','target_mismatched'):
            try:
                evaluation=evaluate_model(model,val_groups,graph_cache,cand_cache,demand_cache,
                    registry,'val',torch.device(args.device),memory_variant=variant,
                    mismatched_memories=mismatched,deadline=deadline)
            except TimeoutError as error:
                checkpointing.atomic_json(args.output/'gpu_phase.status.json',{
                    'status':'STOPPED_GPU_TIME_LIMIT','message':str(error),
                    'elapsed_seconds':time.perf_counter()-wall,
                    'limit_seconds':args.gpu_limit_seconds,
                    'completed_arms':[item['mode'] for item in arm_results],
                    'completed_memory_ablations':{key:sorted(value) for key,value in memory_ablation.items()},
                    'selected_checkpoints':[item['checkpoint'] for item in arm_results],
                    'protocol_sha256':run_protocol['protocol_sha256']})
                return
            variants[variant]={'selection':evaluation,
                               'target_response':target_response_report(coverage,evaluation)}
            memory_ablation[mode]=variants
            checkpointing.atomic_json(args.output/'memory_ablation_progress.json',memory_ablation)
        memory_ablation[mode]=variants
        if time.perf_counter()>deadline:
            checkpointing.atomic_json(args.output/'gpu_phase.status.json',{
                'status':'STOPPED_GPU_TIME_LIMIT','elapsed_seconds':time.perf_counter()-wall,
                'limit_seconds':args.gpu_limit_seconds,
                'completed_arms':[item['mode'] for item in arm_results],
                'completed_memory_ablations':{key:sorted(value) for key,value in memory_ablation.items()},
                'selected_checkpoints':[item['checkpoint'] for item in arm_results],
                'protocol_sha256':run_protocol['protocol_sha256']})
            return
    gpu_seconds=time.perf_counter()-wall
    result={'schema':SCHEMA,'protocol_sha256':run_protocol['protocol_sha256'],
       'feature_cache_sha256':cache_hash,'feature_cache_path':str(cache_path),
       'feature_preparation':feat_report,'cohort_manifest_sha256':manifest['manifest_sha256'],
       'training_group_count':len(train_groups),'validation_group_count':len(val_groups),
       'fixed_context_diagnostic':fixed,'matched_common_initial_parameter_sha256':next(iter(initial_hashes)),
       'matched_sampling_trace_sha256':next(iter(trace_hashes)),
       'arms':[{k:v for k,v in arm.items() if k not in ('history','selected_validation')} for arm in arm_results],
       'validation':{arm['mode']:arm['selected_validation'] for arm in arm_results},
       'target_response':{arm['mode']:target_response_report(coverage,arm['selected_validation'])
                          for arm in arm_results},
       'memory_ablation':memory_ablation,
       'training_histories':{arm['mode']:arm['history'] for arm in arm_results},
       'gpu_seconds':gpu_seconds,
       'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated()) if args.device.startswith('cuda') else None}
    checkpointing.atomic_json(args.output/'results.json',result)
    print(json.dumps({'output':str(args.output),'protocol_sha256':run_protocol['protocol_sha256'],
      'manifest_sha256':manifest['manifest_sha256'],'gpu_seconds':result['gpu_seconds'],
      'arms':[{k:a[k] for k in ('mode','checkpoint','parameters','best_validation_step','training_seconds')} for a in arm_results]},indent=2))


if __name__=='__main__': main()
