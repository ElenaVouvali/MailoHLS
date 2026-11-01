"""Trace the Viterbi/U200/5 ns zero-area records to source and preprocessing."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import tempfile

from GNN_branch import target_graph_v22 as overlay
from Preprocessing.create_jsonl import build_examples, file_sha256
from experiments.v22_gnn_training import checkpointing, train
from experiments.v22_gnn_training.audit import _candidate_values


ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED = ROOT / 'LLM_branch/Data/preprocessed_CSVS'
RAW = ROOT / 'Data/CSVS/machsuite-viterbi.csv'
RESOURCE_COLUMNS = ('bram_util_%', 'dsp_util_%', 'ff_util_%', 'lut_util_%')


def audit(output: Path):
    # Reconstruct this kernel's rows alone. The full family split and held-out
    # test tables are intentionally not opened by this source-anomaly audit.
    with tempfile.TemporaryDirectory(prefix='v22-viterbi-zero-audit-') as temp:
        isolated = Path(temp)
        shutil.copyfile(PREPROCESSED / 'preprocessed-machsuite-viterbi.csv',
                        isolated / 'preprocessed-machsuite-viterbi.csv')
        examples, _, _ = build_examples(isolated)
    profile = overlay.load_registry()['profiles']['u200']
    source_rows = {}
    with RAW.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key = (row['Version'], row['Device'].lower(), float(row['Clock_Period_nsec']))
            if key in source_rows:
                raise ValueError(f'Duplicate original measurement key: {key}')
            source_rows[key] = row
    with (PREPROCESSED / 'preprocessed-machsuite-viterbi.csv').open(
            newline='', encoding='utf-8') as stream:
        preprocessed_rows = list(csv.DictReader(stream))
    affected = []
    for row in examples:
        if (row['kernel_name'] != 'machsuite-viterbi'
                or row['device'].lower() != 'xcu200-fsgd2104-2-e'
                or float(row['clock_period']) != 5.0
                or float(row['latency']) <= 0
                or any(float(row[column]) != 0 for column in RESOURCE_COLUMNS)):
            continue
        # `build_examples` retains the version in its preprocessing row only
        # through the row index; recover it from the authenticated table.
        pre = preprocessed_rows[int(row['preprocessed_row'])]
        version = pre['Version']
        raw = source_rows.get((version, row['device'].lower(), float(row['clock_period'])))
        if raw is None:
            raise ValueError(f'No exact original CSV row for {version}')
        raw_utils = [raw[column] for column in (
            'BRAM_Utilization_percentage', 'DSP_Utilization_percentage',
            'FF_Utilization_percentage', 'LUT_Utilization_percentage')]
        pre_utils = [pre[column] for column in (
            'BRAM_Utilization_percentage', 'DSP_Utilization_percentage',
            'FF_Utilization_percentage', 'LUT_Utilization_percentage')]
        if raw['Latency_msec'] != pre['Latency_msec'] or raw_utils != pre_utils:
            raise ValueError(f'Original/preprocessed measurement differs for {version}')
        transformed, physical = train.physical_targets(row, profile)
        capacity = [float(profile['resources'][name]['value']) for name in train.RESOURCE_ORDER]
        area_proxy = sum(value / limit for value, limit in zip(physical[1:], capacity)) / 4
        affected.append({
            'version': version,
            'device': row['device'],
            'clock_ns': float(row['clock_period']),
            'latency_ms_source': raw['Latency_msec'],
            'synthesis_time_sec_source': raw.get('Synthesis_Time_sec'),
            'raw_resource_utilization_percent_strings': raw_utils,
            'preprocessed_resource_utilization_percent_strings': pre_utils,
            'preprocessed_area_field': pre['Area'],
            'assignment_sha256': train.sha256_json(_candidate_values(row['target'])),
            'converted_physical_targets_order_bram_dsp_ff_lut': physical[1:],
            'device_capacities_order_bram_dsp_ff_lut': capacity,
            'area_proxy': area_proxy,
            'adp_ms_times_area_proxy': physical[0] * area_proxy,
            'source_fields_present_and_parseable': True,
        })
    result = {
        'schema': 'salt-hls-v4-zero-area-source-audit-v1',
        'audit_script_sha256': overlay.file_hash(Path(__file__)),
        'raw_csv': str(RAW.relative_to(ROOT)),
        'raw_csv_sha256': file_sha256(RAW),
        'preprocessed_csv': str((PREPROCESSED / 'preprocessed-machsuite-viterbi.csv').relative_to(ROOT)),
        'preprocessed_csv_sha256': file_sha256(PREPROCESSED / 'preprocessed-machsuite-viterbi.csv'),
        'preprocessing_manifest_sha256': file_sha256(PREPROCESSED / 'preprocessing_manifest.json'),
        'fpga_profile_registry_sha256': overlay.file_hash(overlay.PROFILES),
        'context': {'kernel': 'machsuite-viterbi', 'profile': 'u200', 'clock_ns': 5.0},
        'affected_record_count': len(affected),
        'resource_order': list(train.RESOURCE_ORDER),
        'percentage_conversion': 'physical count = recorded percentage / 100 * authenticated profile capacity',
        'records': affected,
        'conclusion': (
            'All four source CSV utilization fields are present and recorded as literal integer-percent zero; '
            'preprocessing preserves those values and conversion therefore produces zero physical counts, '
            'zero area proxy and zero ADP. The source provides no sub-percent raw resource counts or precision '
            'metadata, so it is unresolved whether the zeros mean exact zero or values rounded/truncated below '
            'the reported percentage resolution. No conversion defect is evidenced; retain the original labels, '
            'mark relative regret undefined for zero optimum, and report absolute measured-pool regret.'),
    }
    checkpointing.atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print(f"wrote {args.output}: {result['affected_record_count']} exact source records")


if __name__ == '__main__':
    main()
