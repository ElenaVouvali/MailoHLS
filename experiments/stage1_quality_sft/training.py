"""Matched exposure construction and a scoped bridge to the production Stage-1 trainer."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import math
from pathlib import Path

from LLM_branch.common import mailohls_contract
from experiments.stage1_quality_sft.build import (
    DATASET, SPLIT, SOURCES, REGISTRY, _read_split_rows,
)
from experiments.stage1_quality_sft.core import parse, sample_target
from experiments.stage1_quality_sft.support import sha

BACKBONE = 'deepseek-ai/deepseek-coder-6.7b-base'
BACKBONE_REVISION = 'ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912'


def _jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in rows))


def _read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]



# Preserve literal source-marker spelling: production Stage-1 matches source LHS
# against target LHS using a case-sensitive lookup before constructing labels.
_DIRECTIVE_SOURCE_RE = re.compile(
    r"auto\{_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*\}", re.IGNORECASE
)


def _serialize_target_against_source(assignment, source_text):
    if not isinstance(assignment, dict) or not assignment:
        raise ValueError('Missing measured full directive assignment')
    normalized = {}
    for lhs, rhs in assignment.items():
        key = str(lhs).strip().upper()
        if key in normalized:
            raise ValueError(f'Duplicate measured directive after case normalization: {key}')
        normalized[key] = str(rhs).strip()
    observed = {}
    for match in _DIRECTIVE_SOURCE_RE.finditer(source_text):
        literal = match.group(0)
        key = literal.upper()
        if key in observed and observed[key] != literal:
            raise ValueError(f'Source uses conflicting marker casing: {key}')
        observed[key] = literal
    if not observed or set(observed) != set(normalized):
        raise ValueError(
            f'Full measured assignment differs from source action markers; '
            f'missing={sorted(set(observed)-set(normalized))[:8]}, '
            f'extra={sorted(set(normalized)-set(observed))[:8]}'
        )
    return '\n'.join(f'{literal} = {normalized[key]}' for key, literal in observed.items())

def _export_arm(arm, prepared: Path, out: Path, exposures: int, seed: int):
    requests = _read_jsonl(prepared / 'train_requests.jsonl.gz')
    val_requests = _read_jsonl(prepared / 'validation_requests.jsonl.gz')
    train_pools = {p['candidate_pool_id']: {c['candidate_id']: dict(c, assignment_id=c['candidate_id'])
        for c in p['candidates']} for p in _read_jsonl(prepared / 'train_candidate_pools.jsonl.gz')}
    val_pools = {p['candidate_pool_id']: {c['candidate_id']: dict(c, assignment_id=c['candidate_id'])
        for c in p['candidates']} for p in _read_jsonl(prepared / 'validation_candidate_pools.jsonl.gz')}
    split = json.loads(SPLIT.read_text())
    templates = json.loads(SOURCES.read_text())['templates']
    source_rows = _read_split_rows(DATASET, split['train_jsonl_idx'], templates)
    val_source_rows = _read_split_rows(DATASET, split['val_jsonl_idx'], templates)
    by_idx = {int(r['_jsonl_idx']): r for r in source_rows + val_source_rows}
    train_rows, validation_rows, target_schedule = [], [], []

    def materialize(request, candidate, split_name, exposure):
        source = by_idx[int(candidate['row_id'])]
        device = str(source.get('device', '')).lower()
        caps = mailohls_contract.DEVICE_RESOURCES.get(device)
        if caps is None:
            raise ValueError(f'No existing physical capacity definition for {device}')
        fraction = float(request['budget']['budget_frac_bram_18k'])
        used = {res: float(source[mailohls_contract.UTIL_FIELD_BY_RESOURCE[res]]) * caps[res] / 100.0
                for res in mailohls_contract.RESOURCE_KEYS}
        available = {res: float(caps[res]) * fraction for res in mailohls_contract.RESOURCE_KEYS}
        if any(used[res] > available[res] + 1e-9 for res in mailohls_contract.RESOURCE_KEYS):
            raise ValueError('Selected measured candidate does not fit its request budget')
        row = dict(source)
        for res in mailohls_contract.RESOURCE_KEYS:
            field = mailohls_contract.AVAIL_FIELD_BY_RESOURCE[res]
            row[field] = int(round(available[res]))
        row['resource_budget_frac'] = fraction
        row.update({k: float(v) for k, v in request['budget'].items()})
        row.update(objective='PARETO_LATENCY', obj_mode='PARETO_LATENCY',
                   selected_clock_period=float(source['clock_period']),
                   frequency_mode='specified',
                   resource_budget_id=f'uniform-{int(float(request["budget"]["budget_frac_bram_18k"])*100):03d}',
                   target=_serialize_target_against_source(candidate['assignment'], templates[source['source_key']]),
                   _source_measurement_jsonl_idx=int(candidate['row_id']),
                   _quality_request_id=request['request_id'],
                   _quality_exposure=exposure, _quality_arm=arm, _quality_split=split_name)
        # Retain authenticated per-candidate QoR for the production loader.
        # These measurements are NOT used as prompt fields or supervised targets.
        # Forbid invented/dummy QoR and verify candidate/source-row identity.
        source_assignment = {
            str(k).strip().upper(): str(v).strip()
            for k, v in parse(source['target']).items()
        }
        selected_assignment = {
            str(k).strip().upper(): str(v).strip()
            for k, v in candidate['assignment'].items()
        }
        if source_assignment != selected_assignment:
            raise ValueError(
                f"Candidate/source assignment mismatch: measured row {candidate['row_id']}"
            )
        measured_fields = (
            'latency', 'area', 'bram_util_%', 'dsp_util_%',
            'ff_util_%', 'lut_util_%',
        )
        for key in measured_fields:
            if key not in source:
                raise ValueError(
                    f"Missing authenticated {key} in measured row {candidate['row_id']}"
                )
            value = float(source[key])
            if not math.isfinite(value) or value < 0.0 or (key == 'latency' and value <= 0.0):
                raise ValueError(
                    f"Invalid authenticated {key}={source[key]!r} "
                    f"in measured row {candidate['row_id']}"
                )
            row[key] = value
        # Remove derived preferences, weights and extraneous payloads; keep
        # objective/device/clock/budget conditioning unchanged for both arms.
        for key in (
            'weight', 'is_pareto', 'resource_pressure_max_pct',
            'preprocessed_row', 'resource_pressure', 'used_bram_18k',
            'used_dsp', 'used_ff', 'used_lut',
        ):
            row.pop(key, None)
        row['_sample_weight'] = 1.0
        return row

    for request in requests:
        pool = train_pools[request['candidate_pool_id']]
        choices = [pool[cid] for cid in request['sensitivity']['0.01']['accepted_candidate_ids']]
        sample_request = dict(request, accepted=choices)
        for exposure in range(exposures):
            chosen = pool[request['best_candidate_id']] if arm == 'single_reference' else sample_target(sample_request, seed, exposure)
            train_rows.append(materialize(request, chosen, 'train', exposure))
            target_schedule.append(dict(request_id=request['request_id'], exposure=exposure,
                candidate_id=chosen['candidate_id'], measured_jsonl_idx=chosen['row_id'],
                assignment_id=chosen['assignment_id'], assignment=chosen['assignment']))
    for request in val_requests:
        # Validation examples are fixed by the predeclared request manifest; the
        # requested target is a legal representative, never a model-selection oracle.
        validation_rows.append(materialize(request, val_pools[request['candidate_pool_id']][request['best_candidate_id']], 'validation', 0))

    # Dataset indices are arm-specific; every exposure is retained exactly once.
    combined = train_rows + validation_rows
    for index, row in enumerate(combined):
        row['_jsonl_idx'] = index
    out.mkdir(parents=True, exist_ok=False)
    dataset_path = out / 'train.jsonl'
    _jsonl(dataset_path, combined)
    source_path = out / 'train.sources.json'
    source_path.write_text(json.dumps(dict(schema='mailohls-sft-sources-v1', templates=templates),
                                     ensure_ascii=False, sort_keys=True) + '\n')
    manifest = dict(schema='mailohls-sft-jsonl-manifest-v2-compact-source',
        examples=len(combined), output=str(dataset_path), output_sha256=sha(dataset_path),
        sources=str(source_path), sources_sha256=sha(source_path),
        utilization_policy='reported_percentages_unchanged',
        area_metric='arithmetic_mean_of_bram_dsp_ff_lut_percentages',
        parent_dataset_sha256=sha(DATASET), protocol_manifest_sha256=sha(prepared / 'eligibility_manifest.json'))
    (out / 'train.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    split_payload = dict(train_jsonl_idx=list(range(len(train_rows))),
                         val_jsonl_idx=list(range(len(train_rows), len(combined))),
                         test_jsonl_idx=[], dataset_sha256=sha(dataset_path),
                         train_families=sorted({r['_family'] for r in train_rows}),
                         val_families=sorted({r['_family'] for r in validation_rows}),
                         test_families=[], effective_area_floor=1e-12,
                         quality_protocol_sha256=json.loads((prepared / 'eligibility_manifest.json').read_text())['protocol_sha256'])
    split_path = out / 'split.json'
    split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + '\n')
    report = dict(arm=arm, seed=int(seed), train_requests=len(requests), exposures_per_request=exposures,
                  training_examples=len(train_rows), validation_requests=len(validation_rows),
                  optimizer_exposure_target=len(train_rows), sampling='uniform across complete epsilon=.01 assignments' if arm == 'multi_reference' else 'best feasible latency, assignment-hash tie break',
                  distinct_train_assignments=len({(r['source_key'], r['target']) for r in train_rows}),
                  target_coverage=len({r['_source_measurement_jsonl_idx'] for r in train_rows}),
                  dataset_sha256=sha(dataset_path), split_sha256=sha(split_path),
                  validation_families=split_payload['val_families'], training_families=split_payload['train_families'])
    (out / 'arm_manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    _jsonl(out / 'target_schedule.jsonl', target_schedule)
    return report


def export(prepared: Path, output: Path, exposures=2, seed=123):
    if exposures <= 0:
        raise ValueError('exposures must be positive')
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for arm in ('single_reference', 'multi_reference'):
        reports.append(_export_arm(arm, prepared, output / arm, exposures, seed))
    if reports[0]['training_examples'] != reports[1]['training_examples']:
        raise AssertionError('SFT arms do not have matched exposure counts')
    # Validate the intended treatment isolation at serialized model-input level.
    left, right = (list(_read_jsonl(output / name / 'train.jsonl')) for name in ('single_reference', 'multi_reference'))
    if len(left) != len(right):
        raise AssertionError('SFT arm dataset lengths differ')
    ignored = {'target', '_source_measurement_jsonl_idx', '_quality_arm',
               'latency', 'area', 'bram_util_%', 'dsp_util_%',
               'ff_util_%', 'lut_util_%'}
    for i, (a, b) in enumerate(zip(left, right)):
        if {k: v for k, v in a.items() if k not in ignored} != {k: v for k, v in b.items() if k not in ignored}:
            raise AssertionError(f'SFT arms differ outside target selection at serialized row {i}')
    manifest = dict(schema='stage1-quality-sft-matched-arms-v1', arms=reports,
        shared_backbone=BACKBONE, backbone_revision=BACKBONE_REVISION,
        independent_fresh_adapters=True, historical_adapter_initialization=False,
        matched_seed=int(seed), matched_exposures=True, exposure_count_per_request=int(exposures),
        scheduler='existing production cosine schedule; one pass over the sealed exposure rows',
        loss='stage1-action-normalized-values-v1',
        training_protocol_sha256=sha(prepared / 'eligibility_manifest.json'),
        validation_request_sha256=sha(prepared / 'validation_requests.jsonl.gz'),
        code_sha256={name: sha(Path(name)) for name in (
            'experiments/stage1_quality_sft/core.py', 'experiments/stage1_quality_sft/build.py',
            'experiments/stage1_quality_sft/training.py', 'LLM_branch/train/train_SFT_xattn_new.py',
            'LLM_branch/train/stage1_decision_loss.py', 'LLM_branch/common/directive_domains.py',
            'LLM_branch/common/mailohls_contract.py')})
    (output / 'matched_arms.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def _passthrough_select(rows, *args, **kwargs):
    # Each input row already represents one measured, budget-conditioned target.
    # Returning it unchanged preserves multi-reference exposures through Trainer.
    return [dict(row) for row in rows], dict(policy='preselected_quality_sft_targets', rows=len(rows))


def run_production(arm_dir: Path, argv=None):
    """Run the current production Stage-1 stack with only the selector scoped.

    Call from a process that already has the documented CUDA/model environment.
    The patched selector is restored even if training raises.
    """
    import sys
    args = list(argv if argv is not None else sys.argv[1:])
    owned = {'--run_mode', '--split_mode', '--batch_size', '--grad_accum', '--num_workers', '--family_sampling_power', '--gradient_checkpointing', '--dataset', '--split_json', '--init_adapter_dir', '--output_dir',
             '--directive_domain_registry_json', '--seed', '--model', '--model_revision',
             '--epochs', '--lora_r', '--lora_alpha', '--lora_dropout', '--lora_target_modules',
             '--objective', '--resource_budget_mode', '--resource_budget_fracs',
             '--random_budgets_per_case', '--directive_loss_weighting',
             '--stage1_supervision_policy', '--stage1_contrast_weight',
             '--stage1_quality_skip_selection',
             '--budget_target_max_duplicates'}
    if any(arg.split('=', 1)[0] in owned for arg in args):
        raise ValueError('Runner owns dataset/split/registry/output and forbids historical adapter initialization')
    from experiments.stage1_quality_sft.gpu_preflight import check as gpu_check
    gpu_receipt = gpu_check()
    arm_manifest = json.loads((arm_dir / 'arm_manifest.json').read_text())
    run_dir = (arm_dir / 'run').resolve()
    resume_args = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '--resume_from_checkpoint']
    if '--resume_from_checkpoint' in args:
        if len(resume_args) != 1 or not Path(resume_args[0]).resolve().is_relative_to(run_dir):
            raise ValueError('Resume is permitted only from a checkpoint in this arm\'s own run directory')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run output already exists; resume from an authenticated checkpoint: {run_dir}')
    args += ['--dataset', str((arm_dir / 'train.jsonl').resolve()),
             '--split_json', str((arm_dir / 'split.json').resolve()),
             '--directive_domain_registry_json', str(REGISTRY.resolve()),
             '--run_mode', 'single', '--split_mode', 'random_design', '--resource_budget_mode', 'none',
             '--objective', 'PARETO_LATENCY', '--top_k', '1',
             '--budget_target_max_duplicates', '0',
             '--stage1_supervision_policy', 'decision', '--stage1_contrast_weight', '1.0',
             '--directive_loss_weighting', 'uniform',
             '--seed', str(arm_manifest['seed']),
             '--model', BACKBONE, '--model_revision', BACKBONE_REVISION,
             '--epochs', '1', '--batch_size', '1', '--grad_accum', '8', '--num_workers', '0', '--family_sampling_power', '0', '--gradient_checkpointing', '--lora_r', '8', '--lora_alpha', '16', '--lora_dropout', '0.05',
             '--lora_target_modules', 'attention',
             '--stage1_decision_diagnostics', '--stage1_quality_skip_selection',
             '--no-eval_on_start',
             '--eval_steps', '1000',
             '--selection_eval_steps', '1000',
             '--save_steps', '100',
             '--early_stopping_patience', '0',
             '--output_dir', str(run_dir), '--disable_structural_memory']
    (arm_dir / 'gpu_preflight.json').write_text(json.dumps(gpu_receipt, indent=2, sort_keys=True) + '\n')
    from LLM_branch.train import train_SFT_xattn_new as production
    original = production.select_goal_rows
    original_argv = sys.argv
    production.select_goal_rows = _passthrough_select
    try:
        sys.argv = [sys.argv[0], *args]
        production.main()
    finally:
        production.select_goal_rows = original
        sys.argv = original_argv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    ex = sub.add_parser('export')
    ex.add_argument('--prepared', required=True); ex.add_argument('--output', required=True)
    ex.add_argument('--exposures', type=int, default=4); ex.add_argument('--seed', type=int, default=123)
    run = sub.add_parser('run')
    run.add_argument('--arm-dir', required=True)
    run.add_argument('trainer_args', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.command == 'export':
        print(json.dumps(export(Path(args.prepared), Path(args.output), args.exposures, args.seed), indent=2))
    else:
        run_production(Path(args.arm_dir), args.trainer_args)


if __name__ == '__main__':
    main()
