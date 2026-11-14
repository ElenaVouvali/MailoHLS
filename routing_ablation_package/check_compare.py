#!/usr/bin/env python3
"""Check scientific comparability of checkpoint-300 routing variants.

Also produces a descriptive Kalman prediction comparison, not a QoR claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def data(p):return json.loads(Path(p).read_text())

def protocol(run):
    run=Path(run)
    p=data(run/'protocol.json')
    rec=data(run/'checkpoint-0300/receipt.json')
    if int(rec['step']) != 300:raise ValueError(f'{run}: not checkpoint 300')
    if rec['structural_state_sha256']!=sha(run/'checkpoint-0300/structural_xattn.pt'):
        raise ValueError(f'{run}: corrupted structural checkpoint')
    if rec['protocol_sha256']!=hashlib.sha256(json.dumps(p,sort_keys=True,default=str).encode()).hexdigest():
        raise ValueError(f'{run}: protocol/checkpoint mismatch')
    return p,rec


def ensure_compatible(runs):
    versions={mode:protocol(path) for mode,path in runs.items()}
    ref,rr=versions['compiler_relational']
    if not ref['routing'].startswith('compiler_relational;'):
        raise ValueError('Baseline routing protocol is not compiler relational')
    comparable=(
        'dataset_sha256','memory_sha256','parent','model','loss',
        'causal_apply_mask','selection'
    )
    fixed_params=(
        'seed','steps','grad_accum','contexts','max_length','eval_interval',
        'val_sft','val_pair_groups','lr_xattn','lr_gate','gate_open_steps',
        'gate_open_lr','probe_sft','rank_budget','rank_clock','weight_decay',
        'max_grad_norm','min_train_families','min_val_families','cohort'
    )
    for mode,(p,rec) in versions.items():
        for k in comparable:
            if p.get(k)!=ref.get(k):
                raise ValueError(f'CONFUND: {mode} has different {k}')
        for k in fixed_params:
            if p.get('train_config',{}).get(k) != ref.get('train_config',{}).get(k):
                raise ValueError(f'CONFUND: {mode} changed hyperparameter {k}: '
                                 f"{p.get('train_config',{}).get(k)} != {ref.get('train_config',{}).get(k)}")
        if rec['frozen_composite_sha256'] != rr['frozen_composite_sha256']:
            raise ValueError(f'CONFUND: {mode} has different frozen Stage-1 weights/FiLM')
        if not str(p['routing']).startswith(mode + ';'):
            raise ValueError(f'Routing mismatch in {mode} protocol')
        if p['train_config']['steps'] != 450:
            raise ValueError(f'{mode} must use same 450-step learning-rate schedule')
        if mode!='compiler_relational':
            status=data(Path(runs[mode])/'TARGET_REACHED.json')
            if status.get('trained_updates')!=300 or status.get('schedule_total_updates')!=450:
                raise ValueError(f'{mode}: no clean 300-update stopping receipt')
    return versions


def load_record(file):
    rows=[json.loads(l) for l in Path(file).read_text().splitlines() if l.strip()]
    matches=[r for r in rows if r.get('kernel','').lower()=='kalman']
    if len(matches)!=1:raise ValueError(f'{file}: expected exactly one Kalman record, got {len(matches)}')
    r=matches[0]
    if r['status']!='inference_only_not_synthesized':raise ValueError('Inappropriate result status')
    return r


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--baseline-run',type=Path,required=True)
    a.add_argument('--self-run',type=Path)
    a.add_argument('--full-run',type=Path)
    a.add_argument('--predictions',type=Path)
    args=a.parse_args()
    runs={'compiler_relational':args.baseline_run}
    if args.self_run:runs['self_only']=args.self_run
    if args.full_run:runs['full_memory']=args.full_run
    if len(runs)==3:versions=ensure_compatible(runs)
    else:
        versions={mode:protocol(path) for mode,path in runs.items()}
        if versions['compiler_relational'][0]['train_config']['steps']!=450:
            raise ValueError('Baseline was not trained under 450-step schedule')
    print('[CHECKPOINTS VERIFIED]')
    for mode,(p,r) in versions.items():
        print(mode, 'checkpoint-0300', r['structural_state_sha256'], 'routing:',p['routing'])
    if not args.predictions:
        return
    if len(runs)!=3:raise ValueError('Need all 3 run folders to compare predictions')
    preds={m:load_record(args.predictions/(m+'.jsonl')) for m in runs}
    for mode,row in preds.items():
        if row.get('routing_mode')!=mode:raise ValueError('Output row routing mode incorrect')
        if row['checkpoint_sha256']!=versions[mode][1]['structural_state_sha256']:
            raise ValueError(f'{mode}: inference checkpoint weight SHA mismatch')
    reference=preds['compiler_relational']
    conditions=('device','clock_period_ns','budget_percent','source_sha256','memory_manifest_sha256','objective')
    for mode,row in preds.items():
        for key in conditions:
            if row.get(key)!=reference.get(key):
                raise ValueError(f'Inference condition mismatch {mode}: {key}')
    def keyed(row):return {d['action_id']:d for d in row['directives']}
    tables={m:keyed(r) for m,r in preds.items()}
    ids=set(tables['compiler_relational'])
    if any(set(t)!=ids for t in tables.values()):
        raise ValueError('Kalman variants do not have the same action IDs')
    keys=['pipeline_ii','unroll_factor','partition_type','partition_factor','partition_dim']
    lines=[
        '# Kalman checkpoint-300 routing ablation — inference-only',
        '',
        '**Warning:** These are directive predictions, NOT Vitis HLS QoR results.',
        '',
        f"- Device: {reference['device']}; clock period: {reference['clock_period_ns']} ns; resource budget: {reference['budget_percent']}%",
        '- Frozen Stage-1/FiLM, training supervision, structural memory and training hyperparameters: verified identical.',
        '- All three Stage-2 structural checkpoints evaluated at step 300.',
        '',
        '| Action / field | Compiler-guided | Self-only | Full-memory |',
        '|---|---|---|---|',
    ]
    diffs=0
    for aid in sorted(ids,key=lambda x:int(x[1:])):
        for key in keys:
            present=[key in tables[m][aid] for m in runs]
            if not any(present):continue
            v=[str(tables[m][aid].get(key,'—')) for m in ('compiler_relational','self_only','full_memory')]
            diffs+=len(set(v))>1
            lines.append(f'| {aid} {key} | '+' | '.join(v)+' |')
    lines.extend(['',f'**Directive-field differences:** {diffs}.', '',
      '## Same-fold validation at the matched 300-update checkpoint',
      '',
      '| Routing | Two-context group accuracy | Pair accuracy | SFT NLL | XAttn ON/OFF score delta |',
      '|---|---:|---:|---:|---:|'])
    for mode in ('compiler_relational','self_only','full_memory'):
        val=data(Path(runs[mode])/'val_0300.json')
        effect=val.get('structural_effect',{})
        fields=(val.get('macro_two_context_group_accuracy'), val.get('macro_pair_accuracy'),
                val.get('sft_nll'),effect.get('mean_abs_score_delta'))
        if any(x is None for x in fields):raise ValueError(f'{mode} lacks required Stage-2 validation metrics')
        lines.append(f'| {mode} | '+' | '.join(f'{float(x):.6g}' for x in fields)+' |')
    lines.extend(['',
      'The held-out validation fold is the original two-family Stage-2 validation split; it is not a substitute for an unseen-kernel QoR study.',
      '',
      'Candidate log-probabilities are checkpoint-dependent and are not HLS latency estimates.',
      'Synthesize all three generated directive sets with the same Vitis HLS toolchain, device, clock and resource budget to establish measured performance/feasibility.', ''])
    args.predictions.joinpath('kalman_routing_comparison.md').write_text('\n'.join(lines))
    print('[COMPARE]',args.predictions/'kalman_routing_comparison.md')
    print('[WARNING] Inference-only. No synthesis latency or resource numbers exist yet.')

if __name__=='__main__':main()
