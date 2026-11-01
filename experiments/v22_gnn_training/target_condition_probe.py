"""Small cached three-arm target-conditioning probe on authenticated GNΩSIS data.

This uses the existing V4 CandidateQoRModel, loss and compiler graph APIs. It
is a diagnostic only; no test split or synthesis results are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch import nn
from torch.nn import functional as F

from experiments.v22_gnn_training import checkpointing, fast_probe, train

TRAIN_KERNELS = fast_probe.TRAIN_KERNELS
VALIDATION_KERNELS = fast_probe.VALIDATION_KERNELS
PROFILES = ('zcu104', 'u200')
CLOCKS = (3.33, 5.0, 10.0)
N_CANDIDATES = 16
SEED = 123
SCHEMA = 'salt-hls-v4-target-condition-small-probe-v1'


def _assignment_intersection(groups, kernel):
    maps = [
        {row['config_hash']: row for row in groups.get((kernel, profile, clock), [])}
        for profile in PROFILES for clock in CLOCKS
    ]
    return set.intersection(*(set(item) for item in maps)) if maps and all(maps) else set()


class CandidateOnlyModel(nn.Module):
    """Small non-graph control with candidate/action and numeric context only."""
    def __init__(self):
        super().__init__()
        width = 64 * (train.CANDIDATE_DIM + train.CANDIDATE_DEMAND_DIM) + train.tensors.CONTEXT_DIM
        self.net = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, context, candidates, demands):
        n = candidates.shape[0]
        return self.net(torch.cat([candidates.flatten(1), demands.flatten(1),
                                   context.expand(n, -1)], dim=-1))


def _metric_rank(fn, actual, predicted):
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return None
    value = float(fn(actual, predicted)[0])
    return value if math.isfinite(value) else None


def _relative_regret(scores, selected_score):
    """Return relative regret only when the measured optimum is well-scaled."""
    best = float(np.min(scores))
    scale = max(float(np.max(np.abs(scores))), 1e-12)
    if abs(best) <= 1e-6 * scale:
        return None, 'near_zero_measured_optimum'
    return float((float(selected_score)-best)/abs(best)), None


def _features(data, records):
    graphs = {}
    candidates = {}
    demands = {}
    groups = defaultdict(list)
    for row in records:
        key = (row['kernel'], row['profile'], float(row['clock']))
        if key not in graphs:
            graphs[key] = {}
            for fraction in (1.0, 0.5):
                hardware = data.graph_tensor(row, fraction, 'hardware').cpu()
                graphs[key][fraction] = {
                    'hardware': hardware,
                    'numerical': train.tensors.without_hardware_messages(hardware),
                }
        identity = (row['kernel'], row['profile'], float(row['clock']), row['config_hash'])
        if identity not in candidates:
            candidates[identity], demands[identity] = (x.cpu() for x in data.candidate_inputs(row))
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: row['config_hash'])
    return graphs, candidates, demands, groups


def _select_cohort(data):
    groups = data.groups
    selected = {}
    selection = {}
    for kernel in TRAIN_KERNELS:
        common = _assignment_intersection(groups, kernel)
        if len(common) < N_CANDIDATES:
            raise ValueError(f'{kernel}: only {len(common)} assignments common across all six training contexts')
        per = [groups[(kernel, p, c)] for p in PROFILES for c in CLOCKS]
        exemplar = {r['config_hash']: r for r in per[0]}
        seed_rows = [exemplar[h] for h in sorted(common)]
        chosen = fast_probe._select_training_candidates(seed_rows, N_CANDIDATES)
        hashes = [r['config_hash'] for r in chosen]
        for profile in PROFILES:
            for clock in CLOCKS:
                lookup = {r['config_hash']: r for r in groups[(kernel, profile, clock)]}
                selected[(kernel, profile, clock)] = [lookup[h] for h in hashes]
        selection[kernel] = {'split': 'train', 'common_all_six_available': len(common),
                             'selected_assignments': hashes,
                             'selection': 'max-Hamming assignment diversity with hash tie break; no QoR labels'}

    for kernel in VALIDATION_KERNELS:
        common = _assignment_intersection(groups, kernel)
        if len(common) >= 8:
            hashes = sorted(common)[:min(N_CANDIDATES, len(common))]
            method = 'hash-first exact assignment intersection across six contexts; QoR-blind'
            for profile in PROFILES:
                for clock in CLOCKS:
                    lookup = {r['config_hash']: r for r in groups[(kernel, profile, clock)]}
                    selected[(kernel, profile, clock)] = [lookup[h] for h in hashes]
        else:
            # No six-way assignment intersection: choose hash-first independently
            # in each target context, and retain exact pair intersections later.
            per_context = {}
            for profile in PROFILES:
                for clock in CLOCKS:
                    rows = groups[(kernel, profile, clock)]
                    chosen = sorted(rows, key=lambda r: r['config_hash'])[:N_CANDIDATES]
                    selected[(kernel, profile, clock)] = chosen
                    per_context[f'{profile}|{clock:g}'] = [r['config_hash'] for r in chosen]
            method = 'hash-first independently per context; QoR-blind; paired analyses intersect exact hashes'
            hashes = []
        selection[kernel] = {'split': 'validation', 'common_all_six_available': len(common),
                             'selection': method,
                             'selected_assignments': hashes if hashes else per_context}
    return selected, selection


def _collate(group, graph, candidates, demands, device):
    c, d = [], []
    for row in group:
        key = (row['kernel'], row['profile'], float(row['clock']), row['config_hash'])
        c.append(candidates[key]); d.append(demands[key])
    return torch.stack(c).to(device), torch.stack(d).to(device)


def _predict(model, arm, graph, records, candidates, demands, device):
    c, d = _collate(records, graph, candidates, demands, device)
    if arm == 'candidate_target':
        context = graph.hardware_context
        z = model(context, c, d)
    else:
        z = model(graph, c, d, arm=arm)
    return z


def _evaluate(model, arm, groups, graph_cache, candidates, demands, mean, std, device):
    model.eval()
    per_context = []
    physical_by_hash = {}
    with torch.no_grad():
        for key in sorted(groups):
            rows = groups[key]
            budget_predictions = {}
            for fraction in (1.0, 0.5):
                graph = graph_cache[key][fraction][arm if arm in ('hardware','numerical') else 'hardware'].to(device)
                z = _predict(model, arm, graph, rows, candidates, demands, device)
                budget_predictions[fraction] = torch.expm1(z * std + mean).clamp_min(0).cpu().numpy()
            prediction = budget_predictions[1.0]
            actual = np.asarray([row['physical'] for row in rows], dtype=np.float64)
            capacities = np.asarray(rows[0]['capacities'], dtype=np.float64)
            area = np.mean(actual[:, 1:] / capacities, axis=1)
            pred_area = np.mean(prediction[:, 1:] / capacities, axis=1)
            scores = {
                'latency': (actual[:, 0], prediction[:, 0]),
                'area': (area, pred_area),
                'adp': (actual[:, 0] * area, prediction[:, 0] * pred_area),
            }
            ranking = {}
            for name, (y, p) in scores.items():
                selected = int(np.argmin(p))
                order = np.argsort(p, kind='stable')
                top3 = order[:min(3, len(order))]
                top1_regret, regret_reason = _relative_regret(y, y[selected])
                top3_regret, _ = _relative_regret(y, float(np.min(y[top3])))
                ranking[name] = {
                    'spearman': _metric_rank(spearmanr, y, p),
                    'kendall': _metric_rank(kendalltau, y, p),
                    'top1_regret': top1_regret,
                    'top3_regret': top3_regret,
                    'regret_undefined_reason': regret_reason,
                    'pool_size': len(rows),
                }
            feasibility = {}
            full_selected_hash = None
            for fraction in (1.0, 0.5):
                pred_for_budget = budget_predictions[fraction]
                measured_feasible = np.all(actual[:, 1:] <= (capacities*fraction)[None,:],axis=1)
                predicted_feasible = np.all(pred_for_budget[:,1:] <= (capacities*fraction)[None,:],axis=1)
                selected = int(np.argmin(np.where(predicted_feasible, pred_for_budget[:,0], np.inf))) if predicted_feasible.any() else None
                if fraction == 1.0 and selected is not None:
                    full_selected_hash = rows[selected]['config_hash']
                feasibility[str(fraction)] = {
                    'accuracy': float(np.mean(measured_feasible==predicted_feasible)),
                    'false_feasible': int(np.sum(~measured_feasible & predicted_feasible)),
                    'false_infeasible': int(np.sum(measured_feasible & ~predicted_feasible)),
                    'measured_feasible_count': int(measured_feasible.sum()),
                    'predicted_feasible_count': int(predicted_feasible.sum()),
                    'always_feasible_baseline_accuracy': float(np.mean(measured_feasible)),
                    'selected_is_measured_feasible': bool(measured_feasible[selected]) if selected is not None else None,
                }
                feasibility[str(fraction)]['top1_regret'] = (
                    float((actual[selected,0]-np.min(actual[measured_feasible,0])) /
                          max(abs(float(np.min(actual[measured_feasible,0]))),1e-12))
                    if selected is not None and measured_feasible.any() and measured_feasible[selected] else None)
            per_context.append({
                'kernel': key[0], 'profile': key[1], 'clock_ns': key[2],
                'latency_mae_ms': float(np.mean(np.abs(prediction[:, 0]-actual[:, 0]))),
                'latency_mape': float(np.mean(np.abs(prediction[:, 0]-actual[:, 0])/
                                              np.maximum(actual[:, 0], 1e-12))),
                'resource_mae_fraction': np.mean(np.abs(prediction[:,1:]-actual[:,1:])/
                                                 capacities[None,:], axis=0).tolist(),
                'ranking': ranking,
                'feasibility_by_budget_fraction': feasibility,
                'mean_abs_budget_induced_physical_prediction_delta': np.mean(
                    np.abs(budget_predictions[1.0]-budget_predictions[0.5]),axis=0).tolist(),
                'selected_config_hash_full_budget': full_selected_hash,
            })
            for row, pred in zip(rows, prediction):
                physical_by_hash[(key, row['config_hash'])] = pred
    return {'contexts': per_context, 'predictions': physical_by_hash}


def _target_contrasts(groups, predictions):
    records_by_key = {key: {row['config_hash']: row for row in rows}
                      for key, rows in groups.items()}
    reports = {'clock': [], 'device': []}
    for kernel in VALIDATION_KERNELS:
        for profile in PROFILES:
            for i, c1 in enumerate(CLOCKS):
                for c2 in CLOCKS[i+1:]:
                    k1, k2 = (kernel, profile, c1), (kernel, profile, c2)
                    left, right = records_by_key.get(k1, {}), records_by_key.get(k2, {})
                    common = sorted(set(left) & set(right))
                    if not common: continue
                    reports['clock'].append(_contrast_row(
                        'clock', kernel, f'{profile}:{c1:g}->{c2:g}ns', common,
                        k1, k2, left, right, predictions))
        for clock in CLOCKS:
            k1, k2 = (kernel, 'zcu104', clock), (kernel, 'u200', clock)
            left, right = records_by_key.get(k1, {}), records_by_key.get(k2, {})
            common = sorted(set(left) & set(right))
            if common:
                reports['device'].append(_contrast_row(
                    'device', kernel, f'{clock:g}ns:zcu104->u200', common,
                    k1, k2, left, right, predictions))
    return reports


def _contrast_row(kind, kernel, target_pair, common, k1, k2, left, right, predictions):
    actual, predicted = [], []
    for h in common:
        actual.append(np.asarray(right[h]['physical']) - np.asarray(left[h]['physical']))
        predicted.append(predictions[(k2, h)] - predictions[(k1, h)])
    a, p = np.asarray(actual), np.asarray(predicted)
    threshold = np.maximum(np.abs(a[:,0]), 1e-12) * .01
    nontrivial = np.abs(a[:,0]) > threshold
    sign = np.sign(a[nontrivial,0]) == np.sign(p[nontrivial,0])
    return {'kernel': kernel, 'target_pair': target_pair,
            'matched_assignments': len(common),
            'nontrivial_measured_latency_contrasts': int(nontrivial.sum()),
            'mean_absolute_paired_error_by_target': np.mean(np.abs(p-a),axis=0).tolist(),
            'latency_sign_accuracy': float(np.mean(sign)) if len(sign) else None,
            'near_zero_measured_latency_contrast_count': int((~nontrivial).sum())}


def run(args):
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = train.PreparedData(args.release_dir, out/'data_audit.json',
                              selected_kernels=TRAIN_KERNELS + VALIDATION_KERNELS)
    selected, selection = _select_cohort(data)
    records = [row for rows in selected.values() for row in rows]
    if any(row['split'] == 'test' for row in records): raise ValueError('Test records selected')
    prep = time.monotonic()-start
    if prep > args.preparation_limit_seconds: raise TimeoutError('Data audit/cohort selection exceeded prep cap')
    feature_start = time.monotonic()
    graph_cache, candidate_cache, demand_cache, all_groups = _features(data, records)
    features_seconds = time.monotonic()-feature_start
    if prep + features_seconds > args.preparation_limit_seconds:
        raise TimeoutError('Authenticated data plus feature preparation exceeded the preparation cap')
    selected_groups = {key: sorted(rows,key=lambda r:r['config_hash']) for key,rows in selected.items()}
    train_rows = [r for key,rows in selected_groups.items() if rows[0]['split']=='train' for r in rows]
    mean, std = train._fit_normalization(train_rows)
    rank_scales = train.fit_ranking_scales(train_rows)
    manifest = {
        'schema': SCHEMA, 'seed': SEED, 'dataset_sha256': data.audit['dataset']['dataset_sha256'],
        'split_sha256': data.audit['dataset']['split_sha256'],
        'graph_archive_sha256': data.audit['graph_release']['archive_sha256'],
        'training_kernels': list(TRAIN_KERNELS), 'validation_kernels': list(VALIDATION_KERNELS),
        'target_contexts': [f'{p}|{c:g}ns' for p in PROFILES for c in CLOCKS],
        'selection': selection,
        'training_rows': len(train_rows), 'validation_rows': sum(len(r) for k,r in selected_groups.items() if r[0]['split']=='val'),
        'test_records_used_for_selection_or_fit': False,
        'normalization': 'log1p target mean/std fitted on selected training rows only',
        'physical_target_budget_invariance': 'same measured row reused under 100% and 50% request budgets; budget affects feasibility only',
        'feature_preparation_seconds': features_seconds,
        'source_hashes': fast_probe._source_hashes(),
        'target_condition_probe_sha256': fast_probe.overlay.file_hash(Path(__file__)),
    }
    manifest['sha256'] = fast_probe._canonical_hash(manifest)
    checkpointing.atomic_json(out/'cohort_manifest.json',manifest)

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    gpu=fast_probe._verify_idle_gpu(args.gpu_name,args.gpu_uuid)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Requires one isolated CUDA GPU')
    device=torch.device('cuda:0')
    if args.gpu_name.casefold() not in torch.cuda.get_device_name(device).casefold(): raise RuntimeError('GPU identity mismatch')
    gpu_start=time.monotonic(); deadline=gpu_start+args.gpu_limit_seconds
    mean,std=mean.to(device),std.to(device)
    # Freeze identical update sampling before training any arm.
    rng=random.Random(SEED)
    train_keys=sorted(k for k,r in selected_groups.items() if r[0]['split']=='train')
    trace=[]
    for _ in range(args.updates):
        k=rng.choice(train_keys); rows=selected_groups[k]
        ix=sorted(rng.sample(range(len(rows)),min(args.candidates_per_update,len(rows))))
        trace.append({'group':list(k),'indices':ix,'objective':rng.choice(train.OBJECTIVES),
                      'budget':rng.choice(train.BUDGET_FRACTIONS)})
    init_seed=SEED
    random.seed(init_seed); np.random.seed(init_seed); torch.manual_seed(init_seed); torch.cuda.manual_seed_all(init_seed)
    base=train.CandidateQoRModel()
    initial_state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    init_hash=train.model_state_sha256(base)
    arms={}
    for arm in ('numerical','hardware','candidate_target'):
        if time.monotonic()>deadline: raise TimeoutError('GPU time cap reached')
        random.seed(init_seed); np.random.seed(init_seed); torch.manual_seed(init_seed); torch.cuda.manual_seed_all(init_seed)
        if arm=='candidate_target': model=CandidateOnlyModel().to(device)
        else:
            model=train.CandidateQoRModel().to(device); model.load_state_dict(initial_state)
        model_hash=train.model_state_sha256(model) if arm!='candidate_target' else None
        if arm!='candidate_target' and model_hash!=init_hash: raise ValueError('Graph arms initial states differ')
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        losses=[]; gradients=[]; relation_gradients=[]; arm_start=time.monotonic()
        for step,item in enumerate(trace,1):
            if time.monotonic()>deadline: raise TimeoutError('GPU time cap reached during training')
            key=tuple(item['group']); rows=selected_groups[key]
            batch=[rows[i] for i in item['indices']]
            graph=graph_cache[key][item['budget']][arm if arm in ('hardware','numerical') else 'hardware'].to(device)
            c,d=_collate(batch,graph,candidate_cache,demand_cache,device)
            target=(torch.tensor([r['transformed'] for r in batch],dtype=torch.float32,device=device)-mean)/std
            if arm=='candidate_target': z=model(graph.hardware_context,c,d)
            else: z=model(graph,c,d,arm=arm)
            reg=F.smooth_l1_loss(z,target)
            budget=[x*item['budget'] for x in batch[0]['capacities']]
            rank,pairs=train.ranking_loss(z,batch,item['objective'],budget,mean,std,
                                           return_pair_count=True,objective_scales=rank_scales)
            loss=reg+.1*rank
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            opt.zero_grad(set_to_none=True); loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
            relation_grad = (fast_probe._hardware_relation_gradient_norm(model)
                             if arm != 'candidate_target' else 0.0)
            relation_gradients.append(relation_grad)
            gradients.append({'norm':float(norm),'ranking_pairs':pairs,
                              'weighted_rank_loss':float((.1*rank).detach()),'regression_loss':float(reg.detach())})
            opt.step(); losses.append(float(loss.detach()))
        val_groups={k:r for k,r in selected_groups.items() if r[0]['split']=='val'}
        # Evaluate small selected pools at both requested budgets and record
        # budget-induced physical QoR prediction changes separately.
        report=_evaluate(model,arm,val_groups,graph_cache,candidate_cache,demand_cache,mean,std,device)
        ablation=None
        if arm=='hardware':
            original={key:{frac:dict(arms) for frac,arms in by_frac.items()}
                      for key,by_frac in graph_cache.items()}
            for key in graph_cache:
                for frac in graph_cache[key]:
                    graph_cache[key][frac]['hardware']=graph_cache[key][frac]['numerical']
            disabled=_evaluate(model,arm,val_groups,graph_cache,candidate_cache,demand_cache,mean,std,device)
            graph_cache.update(original)
            embedding_shifts=[]
            with torch.no_grad():
                for key, rows in val_groups.items():
                    full_graph=graph_cache[key][1.0]['hardware'].to(device)
                    no_edge_graph=graph_cache[key][1.0]['numerical'].to(device)
                    full_memory,full_pool=model.graph.memories(full_graph)
                    no_memory,no_pool=model.graph.memories(no_edge_graph)
                    active=torch.nonzero(full_graph.matched_mask[0],as_tuple=False).flatten().tolist()
                    if active:
                        a,b=full_memory[:,active],no_memory[:,active]
                        embedding_shifts.append({
                            'kernel':key[0],'profile':key[1],'clock_ns':key[2],
                            'relative_action_memory_l2':float((a-b).norm()/a.norm().clamp_min(1e-12)),
                            'pooled_memory_l2':float((full_pool-no_pool).norm())})
            ablation={'disabled_metrics':{'contexts':disabled['contexts']},
                      'enabled_metrics':{'contexts':report['contexts'],
                      'embedding_shifts':embedding_shifts,
                      'mean_relative_action_memory_l2':float(np.mean(
                          [x['relative_action_memory_l2'] for x in embedding_shifts])) if embedding_shifts else None}}
        ckpt=out/f'{arm}_probe.pt'
        checkpointing.atomic_torch_save(ckpt,{'schema':SCHEMA,'cohort_sha256':manifest['sha256'],
            'arm':arm,'model':{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
            'normalization_mean':mean.cpu(),'normalization_std':std.cpu()})
        arms[arm]={'initial_model_sha256':model_hash,'final_model_sha256':train.model_state_sha256(model),
            'updates':len(losses),'loss_first':losses[0],'loss_last':losses[-1],
            'loss_min':min(losses),'training_and_validation_seconds':time.monotonic()-arm_start,
            'ranking_gradient_audit':gradients,
            'hardware_relation_gradient_norm_mean':float(np.mean(relation_gradients)),
            'hardware_relation_gradient_norm_max':float(np.max(relation_gradients)),
            'validation':{'contexts':report['contexts']},'hardware_edge_ablation':ablation,
            'checkpoint_sha256':fast_probe.overlay.file_hash(ckpt)}
        del model,opt; torch.cuda.empty_cache()
    elapsed=time.monotonic()-gpu_start
    if elapsed>args.gpu_limit_seconds: raise TimeoutError('GPU time cap exceeded')
    # Paired exact-assignment clock/device contrasts for each arm.
    contrasts={}
    for arm in ('numerical','hardware','candidate_target'):
        model_class=CandidateOnlyModel if arm=='candidate_target' else train.CandidateQoRModel
        ckpt=torch.load(out/f'{arm}_probe.pt',map_location=device,weights_only=False)
        model=model_class().to(device); model.load_state_dict(ckpt['model']); model.eval()
        pred={}
        with torch.no_grad():
            for key,rows in selected_groups.items():
                if rows[0]['split']!='val': continue
                for frac in (1.0,.5):
                    graph=graph_cache[key][frac][arm if arm in ('hardware','numerical') else 'hardware'].to(device)
                    c,d=_collate(rows,graph,candidate_cache,demand_cache,device)
                    z=(model(graph.hardware_context,c,d) if arm=='candidate_target'
                       else model(graph,c,d,arm=arm))
                    physical=torch.expm1(z*std+mean).clamp_min(0).cpu().numpy()
                    pred[(key,frac)]={r['config_hash']:y for r,y in zip(rows,physical)}
        by_budget={}
        for frac in (1.0,.5):
            by_budget[frac]=_target_contrasts(selected_groups,{(key,h):y
                for (key,b),m in pred.items() if b==frac for h,y in m.items()})
        budget_delta=[]
        for key in val_groups:
            common=set(pred[(key,1.0)])&set(pred[(key,.5)])
            budget_delta += [np.abs(pred[(key,1.0)][h]-pred[(key,.5)][h]).tolist() for h in common]
        contrasts[arm]={'clock_device':by_budget,
                        'mean_abs_budget_induced_physical_prediction_delta':
                            np.mean(budget_delta,axis=0).tolist() if budget_delta else None}
    output={'status':'completed','manifest_sha256':manifest['sha256'],
        'prep_seconds':prep,'feature_seconds':features_seconds,'gpu_seconds':elapsed,
        'gpu_name':torch.cuda.get_device_name(device),'gpu_uuid':args.gpu_uuid,
        'gpu_inventory_before_launch':gpu,'updates_per_arm':args.updates,
        'candidates_per_update':args.candidates_per_update,
        'sampling_trace_sha256':fast_probe._canonical_hash(trace),
        'initial_graph_arm_state_sha256':init_hash,'arms':arms,'target_contrasts':contrasts}
    checkpointing.atomic_json(out/'results.json',output)
    checkpointing.atomic_json(out/'status.json',{'status':'completed','manifest_sha256':manifest['sha256'],
                               'gpu_seconds':elapsed})
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu-name',default='A2'); ap.add_argument('--gpu-uuid',default='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35')
    ap.add_argument('--updates',type=int,default=100); ap.add_argument('--candidates-per-update',type=int,default=8)
    ap.add_argument('--preparation-limit-seconds',type=int,default=600); ap.add_argument('--gpu-limit-seconds',type=int,default=1200)
    args=ap.parse_args()
    if not 1<=args.updates<=600: ap.error('updates must be 1..600')
    result=run(args); print(json.dumps({'status':result['status'],'prep_seconds':result['prep_seconds'],
        'feature_seconds':result['feature_seconds'],'gpu_seconds':result['gpu_seconds'],
        'cohort':result['manifest_sha256']},indent=2))

if __name__=='__main__': main()
