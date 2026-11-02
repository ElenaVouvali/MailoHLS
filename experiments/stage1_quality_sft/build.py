"""Authenticate raw split rows and freeze train targets / QoR-blind val requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import gzip
from collections import defaultdict
from pathlib import Path

from experiments.stage1_quality_sft.core import BUDGET, physical_context
from experiments.stage1_quality_sft.core import (
    EPSILONS, build_banks, build_compact_request, candidate_pool_record,
    candidate_pool_id, observability_record, parse,
    validate_complete_assignment,
)
from experiments.stage1_quality_sft.support import (
    constrained_rhs_candidates, family_id_from_kernel_name, load_registry,
    normalize_kname, source_fields as source_action_fields,
    sha,
)
from experiments.stage1_quality_sft.observability import audit as audit_observability

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'artifacts/llm/mailohls_sft.jsonl'
SPLIT = ROOT / 'mailohls_runs/mailohls_final_family_split_s123.json'
REGISTRY = ROOT / 'mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json'
PREPROCESSED = Path.home() / 'MailoHLS/.local-archive/reorganization-20261023/retired-worktree-source/LLM_branch/Data/preprocessed_CSVS'
MANIFEST = DATASET.with_suffix('.manifest.json')
SOURCES = DATASET.with_suffix('.sources.json')
BUDGET_LEVELS = (1.0, 0.5, 0.25)


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _read_split_rows(path, allowed_indices, templates):
    allowed = set(map(int, allowed_indices))
    result = []
    with path.open(encoding='utf-8') as handle:
        for idx, line in enumerate(handle):
            if idx not in allowed:
                continue  # test JSON is not decoded or inspected
            row = json.loads(line)
            row['_jsonl_idx'] = idx
            row['_family'] = family_id_from_kernel_name(row['kernel_name'])
            if not row.get('input'):
                row['input'] = templates[row['source_key']]
            result.append(row)
    if {r['_jsonl_idx'] for r in result} != allowed:
        raise ValueError('Split contains missing/out-of-range dataset row identities')
    return result


def _budget_row(row, fraction):
    out = dict(row)
    out.update({key: float(fraction) for key in BUDGET})
    out.update(resource_budget_id=f'uniform-{int(fraction * 100):03d}',
               objective='PARETO_LATENCY', obj_mode='PARETO_LATENCY',
               selected_clock_period=float(row['clock_period']),
               frequency_mode='specified')
    return out


def _request_rows(rows, banks, registry, output, split_name):
    grouped = {}
    for row in rows:
        grouped.setdefault(physical_context(row), row)
    requests, unsupported, epsilon_sensitivity = [], [], defaultdict(int)
    for context, row in sorted(grouped.items(), key=lambda x: repr(x[0])):
        for fraction in BUDGET_LEVELS:
            request_row = _budget_row(row, fraction)
            try:
                primary = build_compact_request(request_row, banks)
            except ValueError as exc:
                if 'no feasible' in str(exc).lower() or 'no measured feasible' in str(exc).lower():
                    unsupported.append(dict(context=context, budget_fraction=fraction,
                                            reason='no_feasible_measured_configuration'))
                    continue
                raise
            # All rows entering these banks have already passed the independent
            # source/action legality audit above; the pool contains only those
            # authenticated complete assignments.
            for epsilon in EPSILONS:
                epsilon_sensitivity[str(epsilon)] += primary['sensitivity'][str(epsilon)]['accepted_count'] > 1
            primary['split'] = split_name
            requests.append(primary)
    return requests, unsupported, dict(epsilon_sensitivity)


def prepare(output: Path, dataset=DATASET, split_path=SPLIT, registry_path=REGISTRY,
            preprocessed_dir=PREPROCESSED):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite prepared protocol: {output}')
    dataset, split_path, registry_path = map(Path, (dataset, split_path, registry_path))
    dataset_manifest = json.loads(MANIFEST.read_text())
    split = json.loads(split_path.read_text())
    sources = json.loads(SOURCES.read_text())
    if dataset_manifest.get('schema') != 'mailohls-sft-jsonl-manifest-v2-compact-source':
        raise ValueError('Unsupported authenticated GNΩSIS dataset manifest')
    actual_dataset_sha = sha(dataset)
    if actual_dataset_sha != dataset_manifest.get('output_sha256') or actual_dataset_sha != split.get('dataset_sha256'):
        raise ValueError('Dataset bytes do not match both preprocessing and family-split identities')
    if sha(SOURCES) != dataset_manifest.get('sources_sha256'):
        raise ValueError('Source sidecar hash does not match dataset manifest')
    if sha(Path(dataset_manifest['preprocessing_manifest'])) != dataset_manifest['preprocessing_manifest_sha256']:
        raise ValueError('Preprocessing manifest changed')
    if sources.get('schema') != 'mailohls-sft-sources-v1':
        raise ValueError('Unsupported source-template sidecar')
    templates = sources['templates']
    membership = {s: set(map(int, split[f'{s}_jsonl_idx'])) for s in ('train', 'val', 'test')}
    if membership['train'] & membership['val'] or membership['train'] & membership['test'] or membership['val'] & membership['test']:
        raise ValueError('Authenticated family split overlaps')
    if not set(split.get('train_families', ())).isdisjoint(split.get('val_families', ())) or not set(split.get('train_families', ())).isdisjoint(split.get('test_families', ())):
        raise ValueError('Family-disjoint split contract is broken')

    train = _read_split_rows(dataset, membership['train'], templates)
    val = _read_split_rows(dataset, membership['val'], templates)
    print(f'[quality-sft] loaded train={len(train)} val={len(val)}; test rows were not decoded', flush=True)
    registry = load_registry(registry_path)
    action_audit, eligible_kernels = audit_observability(
        train + val, dataset_manifest, Path(dataset_manifest['preprocessing_manifest']),
        templates, ROOT, preprocessed_dir)
    print(f'[quality-sft] observability audit: eligible={len(eligible_kernels)} kernels', flush=True)
    raw_train_count, raw_val_count = len(train), len(val)
    excluded_train_kernels = sorted({r['kernel_name'] for r in train} - eligible_kernels)
    excluded_val_kernels = sorted({r['kernel_name'] for r in val} - eligible_kernels)
    excluded_train_families = sorted({r['_family'] for r in train if r['kernel_name'] in excluded_train_kernels})
    excluded_validation_families = sorted({r['_family'] for r in val if r['kernel_name'] in excluded_val_kernels})
    train = [r for r in train if r['kernel_name'] in eligible_kernels]
    val = [r for r in val if r['kernel_name'] in eligible_kernels]
    # Source markers are observed from the authenticated source-template sidecar.
    fields_by_source = {}
    for key, source in templates.items():
        fields_by_source[key] = source_action_fields(source)
    observability, seen = [], set()
    checked = set()
    for row in train:
        key = (row['source_key'], tuple(sorted(parse(row['target']).items())))
        # validate all rows: this is source/action legality, not graph mapping.
        if key not in checked:
            validate_complete_assignment(row, registry, constrained_rhs_candidates,
                                         normalize_kname, parse)
            checked.add(key)
        if row['source_key'] not in seen:
            observability.append(observability_record(row, fields_by_source[row['source_key']], registry,
                                                       normalize_kname, parse))
            seen.add(row['source_key'])
    checked = set()
    for row in val:
        key = (row['source_key'], tuple(sorted(parse(row['target']).items())))
        if key not in checked:
            validate_complete_assignment(row, registry, constrained_rhs_candidates,
                                         normalize_kname, parse)
            checked.add(key)

    train_banks, duplicate_train = build_banks(train)
    val_banks, duplicate_val = build_banks(val)
    print(f'[quality-sft] candidate banks built: train={sum(map(len, train_banks.values()))} val={sum(map(len, val_banks.values()))}', flush=True)
    train_requests, train_unsupported, train_sensitivity = _request_rows(
        train, train_banks, registry, output, 'train')
    # Validation request identities are generated from source/target/clock/budget
    # strata before model inference; their QoR is used only to mark support/score.
    val_requests, val_unsupported, val_sensitivity = _request_rows(
        val, val_banks, registry, output, 'validation')
    train_pools = (candidate_pool_record(context, bank) for context, bank in sorted(train_banks.items(), key=lambda item: repr(item[0])))
    val_pools = (candidate_pool_record(context, bank) for context, bank in sorted(val_banks.items(), key=lambda item: repr(item[0])))
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        'train_requests.jsonl.gz': train_requests,
        'validation_requests.jsonl.gz': val_requests,
        'train_candidate_pools.jsonl.gz': train_pools,
        'validation_candidate_pools.jsonl.gz': val_pools,
        'source_action_eligibility.json': observability,
        'raw_observability_audit.json': action_audit,
    }
    file_hashes = {}
    for filename, value in payloads.items():
        path = output / filename
        if filename.endswith('.jsonl.gz'):
            with gzip.open(path, 'wt', encoding='utf-8', newline='') as handle:
                for item in value:
                    handle.write(json.dumps(item, sort_keys=True, separators=(',', ':')) + '\n')
        else:
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        file_hashes[filename] = sha(path)
    audit = dict(schema='stage1-quality-sft-eligibility-v1', authenticated=True,
        dataset_sha256=actual_dataset_sha, dataset_manifest_sha256=sha(MANIFEST),
        source_sidecar_sha256=sha(SOURCES), preprocessing_manifest_sha256=sha(Path(dataset_manifest['preprocessing_manifest'])),
        split_sha256=sha(split_path), registry_sha256=sha(registry_path),
        raw_train_rows_before_observability=raw_train_count, raw_validation_rows_before_observability=raw_val_count,
        eligible_train_rows=len(train), eligible_validation_rows=len(val), test_rows_decoded=0,
        train_kernels=len({r['kernel_name'] for r in train}), validation_kernels=len({r['kernel_name'] for r in val}),
        train_families=len({r['_family'] for r in train}), validation_families=len({r['_family'] for r in val}),
        directly_measured_action_assignments=sum(len(parse(r['target'])) for r in train),
        source_registry_actions=sum(len(v) for v in fields_by_source.values()),
        observable_source_templates=len(observability), source_action_eligibility='complete rows equal source markers and registry; targets taken only from authenticated measured records',
        train_unique_physical_candidates=sum(len(v) for v in train_banks.values()),
        validation_unique_physical_candidates=sum(len(v) for v in val_banks.values()),
        identical_duplicate_train_rows=duplicate_train, identical_duplicate_validation_rows=duplicate_val,
        train_requests=len(train_requests), validation_requests=len(val_requests),
        eligible_kernels=sorted(eligible_kernels),
        excluded_train_kernels=excluded_train_kernels, excluded_validation_kernels=excluded_val_kernels,
        excluded_train_families=excluded_train_families,
        excluded_validation_families=excluded_validation_families,
        train_multi_reference_requests=sum(r['accepted_count'] > 1 for r in train_requests),
        train_near_optimal_alternatives_epsilon_0_005=train_sensitivity.get('0.005', 0),
        train_near_optimal_alternatives_epsilon_0_01=train_sensitivity.get('0.01', 0),
        validation_unsupported_contexts=val_unsupported, train_unsupported_contexts=train_unsupported,
        budget_levels=list(BUDGET_LEVELS), primary_epsilon=.01,
        validation_protocol='fixed Cartesian observed source/device/clock contexts x uniform budgets; no model outputs or winner-reversal selection',
        validation_request_coverage={
            'device_clock_budget_counts': {
                f'{r["physical_context"][3]}@{r["physical_context"][4]}ns/{r["budget"][BUDGET[0]]:.2f}':
                    sum(1 for q in val_requests if q['physical_context'][3] == r['physical_context'][3]
                        and q['physical_context'][4] == r['physical_context'][4]
                        and q['budget'][BUDGET[0]] == r['budget'][BUDGET[0]])
                for r in val_requests
            },
            'kernels': sorted({r['physical_context'][0] for r in val_requests}),
            'families': sorted({r['family'] for r in val_requests}),
            'budgets': list(BUDGET_LEVELS),
        },
        preprocessed_measurement_archive=str(preprocessed_dir),
        raw_observability_summary={k: dict(eligible=v['eligible'], unknown_action_ids=v.get('unknown_action_ids', []), block_reason=v.get('block_reason')) for k, v in action_audit.items()},
        code_sha256={str(p.relative_to(ROOT)): sha(ROOT / p) for p in (
            Path('experiments/stage1_quality_sft/core.py'), Path('experiments/stage1_quality_sft/build.py'),
            Path('experiments/stage1_quality_sft/support.py'), Path('experiments/stage1_quality_sft/observability.py'),
            Path('LLM_branch/common/mailohls_contract.py'), Path('LLM_branch/common/directive_domains.py'),
            Path('LLM_branch/train/train_SFT_xattn_new.py'), Path('LLM_branch/train/stage1_decision_loss.py'))},
        files=file_hashes)
    audit['protocol_sha256'] = _digest_json(audit)
    (output / 'eligibility_manifest.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dataset', default=str(DATASET))
    ap.add_argument('--split', default=str(SPLIT))
    ap.add_argument('--registry', default=str(REGISTRY))
    ap.add_argument('--preprocessed-dir', default=str(PREPROCESSED))
    args = ap.parse_args()
    print(json.dumps(prepare(Path(args.output), args.dataset, args.split, args.registry,
                             args.preprocessed_dir), indent=2))


if __name__ == '__main__':
    main()
