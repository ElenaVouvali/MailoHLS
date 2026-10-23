#!/usr/bin/env python3
"""Validate downloaded reports against inputs, then stage enriched JSONL."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
inputs = [json.loads(s) for s in (ROOT / 'TA_experiments.input.jsonl').read_text().splitlines()]
cases = sorted((ROOT / 'reports').glob('[0-9][0-9]_*'))
assert len(cases) == len(inputs) == 18
rows = []
for original, case in zip(inputs, cases):
    row = json.loads((case / 'result.json').read_text())
    for key, value in original.items():
        if key not in ('latency', 'resources'):
            assert row[key] == value, (case.name, key)
    meta = row['synthesis']
    assert meta['status'] == 'success' and meta['returncode'] == 0
    report = case / 'hls_project/solution1/syn/report/forward_propagation_csynth.xml'
    assert hashlib.sha256(report.read_bytes()).hexdigest() == meta['report_sha256']
    xml = ET.parse(report).getroot()
    assert row['latency'] == int(xml.findtext('PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency'))
    for key, tag in [('bram', 'BRAM_18K'), ('dsp', 'DSP'), ('ff', 'FF'), ('lut', 'LUT'), ('uram', 'URAM')]:
        assert row['resources'][key] == int(xml.findtext('AreaEstimates/Resources/' + tag))
    for name, digest in meta['source_sha256'].items():
        assert hashlib.sha256((case / name).read_bytes()).hexdigest() == digest
    log = (case / 'synthesis.log').read_text()
    assert not re.search(r'^ERROR:', log, re.M)
    meta['loop_constraints_satisfied'] = 'All loop constraints were satisfied.' in log
    meta['ignored_partial_unroll_loops'] = sorted(set(re.findall(r"Ignored partial unroll directive for loop '(L\d+)'", log)), key=lambda s: int(s[1:]))
    meta['local_report_path'] = str(report)
    meta['local_log_path'] = str(case / 'synthesis.log')
    rows.append(row)
out = ROOT / 'TA_experiments.enriched.jsonl'
out.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
print('Validated all 18 reports, source hashes, resources, latency and unchanged input metadata.')
print('device clock_ns budget latency_cycles bram dsp ff lut loop_constraints_satisfied')
for row in rows:
    print(row['device'], row['clock_period_ns'], row['resource_budget'], row['latency'], *[row['resources'][k] for k in ['bram','dsp','ff','lut']], row['synthesis']['loop_constraints_satisfied'])
