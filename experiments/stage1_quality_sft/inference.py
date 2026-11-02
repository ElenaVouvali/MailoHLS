"""Create QoR-free production-decoder inputs and normalize its prediction rows."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from LLM_branch.common import mailohls_contract
from experiments.stage1_quality_sft.support import sha


def read_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def export_validation_inputs(prepared: Path, source_sidecar: Path, output: Path):
    requests = read_jsonl(prepared / 'validation_requests.jsonl.gz')
    templates = json.loads(source_sidecar.read_text())['templates']
    rows = []
    for req in requests:
        kernel, source_key, source_file, device, clock = req['physical_context']
        capacity = mailohls_contract.DEVICE_RESOURCES[str(device).lower()]
        fractions = req['budget']
        # Stage-1 prompts accept exact available physical resource counts.
        fields = ('budget_frac_bram_18k', 'budget_frac_dsp', 'budget_frac_ff', 'budget_frac_lut')
        resources = dict(zip(mailohls_contract.RESOURCE_KEYS, fields))
        row = dict(context_id=req['request_id'], request_id=req['request_id'],
            kernel_name=kernel, source_key=source_key, source_file=source_file,
            input=templates[source_key], objective='PARETO_LATENCY', obj_mode='PARETO_LATENCY',
            device=device, clock_period=float(clock), selected_clock_period=float(clock),
            frequency_mode='specified', available_clock_periods=[float(clock)],
            resource_budget_id='uniform-' + str(int(fractions[fields[0]] * 100)).zfill(3),
            quality_budget_fractions=fractions)
        for resource, key in resources.items():
            row[mailohls_contract.AVAIL_FIELD_BY_RESOURCE[resource]] = int(round(
                capacity[resource] * float(fractions[key])))
        # No target, measured latency, or reference winner is included.
        rows.append(row)
    output = Path(output)
    output.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows))
    return dict(rows=len(rows), output=str(output), sha256=sha(output),
                requests_sha256=sha(prepared / 'validation_requests.jsonl.gz'),
                labels_exposed_to_decoder=False)


def normalize_production_predictions(prediction_jsonl: Path, output: Path):
    raw = read_jsonl(prediction_jsonl)
    result = []
    for row in raw:
        request_id = row.get('context_id') or row.get('request_id')
        candidates = row.get('candidates') or []
        if not request_id:
            raise ValueError('Production inference row has no context_id/request_id')
        if not candidates:
            result.append(dict(request_id=request_id, prediction=''))
            continue
        selected_id = row.get('best_unique_sample_id_by_model_score')
        choice = next((c for c in candidates if c.get('sample_id') == selected_id), None)
        if choice is None:
            raise ValueError(f'No selected candidate exists for inference context {request_id}')
        result.append(dict(request_id=request_id, prediction=choice.get('canonical_prediction', choice.get('prediction', '')),
                           selected_sample_id=selected_id,
                           sequence_score=choice.get('sequence_score')))
    output = Path(output)
    output.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in result))
    return dict(rows=len(result), output=str(output), sha256=sha(output),
                production_inference_sha256=sha(prediction_jsonl))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export-validation')
    export.add_argument('--prepared', required=True); export.add_argument('--sources', required=True); export.add_argument('--output', required=True)
    normalize = sub.add_parser('normalize')
    normalize.add_argument('--predictions', required=True); normalize.add_argument('--output', required=True)
    args = ap.parse_args()
    result = export_validation_inputs(Path(args.prepared), Path(args.sources), Path(args.output)) if args.command == 'export-validation' else normalize_production_predictions(Path(args.predictions), Path(args.output))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
