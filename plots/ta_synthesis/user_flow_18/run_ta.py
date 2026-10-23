#!/usr/bin/env python3
"""Prepare and synthesize one independent GAN project per input JSONL row."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/bonus24/Application/GAN')
PARTS = {'ZCU104': 'xczu7ev-ffvc1156-2-e', 'U200': 'xcu200-fsgd2104-2-e'}

def run_one(item):
    index, row = item
    case = ROOT / ('%02d_%s_%sns_%s' % (index, row['device'], str(row['clock_period_ns']).replace('.', 'p'), row['resource_budget']))
    case.mkdir(exist_ok=False)
    (case / 'input.json').write_text(json.dumps(row, indent=2) + '\n')
    source = re.sub(r'^\s*#pragma\s+HLS[^\n]*\n', '', (SOURCE / 'network.cpp').read_text(), flags=re.M)
    for d in row['directives']:
        if d['kind'] == 'array':
            pragma = '#pragma HLS array_partition variable=%s type=%s factor=%d dim=%d' % (d['variable'], d['partition_type'], d['partition_factor'], d['partition_dim'])
        else:
            pragma = '#pragma HLS pipeline II=%d\n#pragma HLS unroll factor=%d' % (d['pipeline_ii'], d['unroll_factor'])
        pattern = r'(^\s*' + d['action_id'] + r':[^\n]*\n)'
        source, count = re.subn(pattern, lambda m: m.group(1) + pragma + '\n', source, flags=re.M)
        assert count == 1, d
    (case / 'network.cpp').write_text(source)
    for name in ['network.h', 'weight_definitions.h', 'tanh.h', 'ap_fixed.h']:
        shutil.copy2(str(SOURCE / name), str(case / name))
    # Use the application-local fixed-point header, as in the user-validated baseline.
    tcl = ['open_project -reset hls_project', 'set_top forward_propagation',
           'add_files network.cpp',
           'open_solution -reset solution1 -flow_target vivado',
           'set_part {%s}' % PARTS[row['device']],
           'create_clock -period %s -name default' % row['clock_period_ns'],
           'config_array_partition -complete_threshold 0 -throughput_driven off',
           'config_compile -pipeline_loops 0', 'csynth_design', 'exit']
    (case / 'run_hls.tcl').write_text('\n'.join(tcl) + '\n')
    start = time.time()
    print('START', case.name, flush=True)
    with (case / 'synthesis.log').open('w') as log:
        proc = subprocess.run(['timeout', '--kill-after=30s', '45m', '/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls', '-f', 'run_hls.tcl'], cwd=str(case), stdout=log, stderr=subprocess.STDOUT)
    report = case / 'hls_project/solution1/syn/report/forward_propagation_csynth.xml'
    result = dict(row)
    meta = {'status': 'failed', 'returncode': proc.returncode, 'elapsed_seconds': round(time.time()-start, 3),
            'host': 'coroni.microlab.ntua.gr', 'run_directory': str(case), 'part': PARTS[row['device']],
            'fixed_point_library': '/home/bonus24/Application/GAN/ap_fixed.h (application-local header)',
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir() if p.suffix in ['.cpp', '.h', '.tcl']}}
    if proc.returncode == 0 and report.exists():
        xml = ET.parse(str(report)).getroot()
        def val(path, kind=float):
            return kind(xml.findtext(path))
        assert xml.findtext('UserAssignments/Part') == PARTS[row['device']]
        assert abs(val('UserAssignments/TargetClockPeriod')-row['clock_period_ns']) < 0.001
        result['latency'] = val('PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency', int)
        result['latency_unit'] = 'cycles'
        result['latency_ns'] = round(result['latency'] * row['clock_period_ns'], 6)
        result['resources'] = {key: val('AreaEstimates/Resources/' + tag, int) for key, tag in [('bram','BRAM_18K'),('dsp','DSP'),('ff','FF'),('lut','LUT'),('uram','URAM')]}
        meta.update(status='success', report_path=str(report), report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(), tool_version=xml.findtext('ReportVersion/Version'),
                    estimated_clock_period_ns=val('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod'),
                    clock_uncertainty_ns=val('UserAssignments/ClockUncertainty'), bram_unit='BRAM_18K',
                    latency_best_cycles=val('PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency',int),
                    interval_min_cycles=val('PerformanceEstimates/SummaryOfOverallLatency/Interval-min',int),
                    interval_max_cycles=val('PerformanceEstimates/SummaryOfOverallLatency/Interval-max',int))
    result['synthesis'] = meta
    (case / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('END', case.name, meta['status'], result['latency'], flush=True)
    return index, result

def main():
    rows = [json.loads(line) for line in (ROOT / 'TA_experiments.input.jsonl').read_text().splitlines()]
    assert len(rows) == 18
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, item) for item in enumerate(rows, 1)]
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results[index] = result
            temp = ROOT / 'TA_experiments.results.jsonl.tmp'
            temp.write_text(''.join(json.dumps(results[i], separators=(',', ':'))+'\n' for i in sorted(results)))
            os.replace(str(temp), str(ROOT / 'TA_experiments.results.jsonl'))
    (ROOT / 'COMPLETE').write_text('%d/%d successful\n' % (sum(r['synthesis']['status']=='success' for r in results.values()), len(rows)))

if __name__ == '__main__':
    main()
