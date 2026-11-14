#!/usr/bin/env python3
"""Parse one identified Vitis 2021.1 top report; Python 3.6 standard library only.

This does not run HLS or certify that its process completed successfully.
Undefined HLS latency remains None; co-simulation is a separate measurement.
"""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def numeric(text):
    if text is None:
        return None
    try:
        value = Decimal(text.strip().replace(',', ''))
    except InvalidOperation:
        return None
    if not value.is_finite() or value < 0:
        return None
    return value


def count(text):
    value = numeric(text)
    if value is None or value != value.to_integral_value():
        return None
    return int(value)


def parse_csynth(path, expected_top, expected_part, expected_clock_ns):
    path = Path(path)
    root = ET.parse(str(path)).getroot()
    expected = {'ReportVersion/Version': '2021.1',
                'UserAssignments/TopModelName': expected_top,
                'UserAssignments/Part': expected_part,
                'UserAssignments/FlowTarget': 'vivado',
                'UserAssignments/unit': 'ns',
                'PerformanceEstimates/SummaryOfOverallLatency/unit': 'clock cycles'}
    for key, value in expected.items():
        if root.findtext(key) != value:
            raise ValueError('Wrong report identity {}: {!r} != {!r}'.format(key, root.findtext(key), value))
    requested = numeric(str(expected_clock_ns))
    reported = numeric(root.findtext('UserAssignments/TargetClockPeriod'))
    if requested is None or requested == 0 or reported is None or reported == 0:
        raise ValueError('Missing/invalid positive clock')
    # These XML files print requested clocks to two decimal places. This is
    # report-format rounding tolerance, not permission to change the Tcl clock.
    if abs(requested - reported) > Decimal('0.005000001'):
        raise ValueError('Wrong report clock: {} vs {}'.format(reported, requested))
    prefix = 'PerformanceEstimates/SummaryOfOverallLatency/'
    raw = {k: root.findtext(prefix + name) for k, name in
           [('best', 'Best-caseLatency'), ('average', 'Average-caseLatency'), ('worst', 'Worst-caseLatency')]}
    latencies = {key: count(value) for key, value in raw.items()}
    used, available, percentages = {}, {}, {}
    warnings = []
    for resource, names in [('BRAM_18K', ['BRAM_18K']),
                            ('DSP', ['DSP', 'DSP48E', 'DSP48E1', 'DSP48E2']),
                            ('FF', ['FF']), ('LUT', ['LUT'])]:
        for target, location in [(used, 'Resources'), (available, 'AvailableResources')]:
            values = [count(root.findtext('AreaEstimates/' + location + '/' + name)) for name in names]
            target[resource] = next((value for value in values if value is not None), None)
        u, a = used[resource], available[resource]
        percentages[resource] = None if u is None or a in (None, 0) else float(Decimal(100) * u / a)
        if percentages[resource] is None:
            warnings.append('Missing usage or positive availability for ' + resource)
    if latencies['worst'] is None:
        warnings.append('Top-level HLS worst-case latency is undefined; do not substitute a loop/subfunction or historical value')
    estimate = numeric(root.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod'))
    return {'top': expected_top, 'part': expected_part, 'requested_clock_period_ns': float(requested),
            'requested_frequency_mhz': float(Decimal(1000) / requested),
            'report_clock_period_ns': float(reported),
            'latency_cycles': latencies['worst'], 'latency_basis': 'csynth_worst',
            'latency_cycles_best': latencies['best'], 'latency_cycles_average': latencies['average'],
            'latency_raw': raw, 'resources_used': used, 'resources_available': available,
            'resources_pct': percentages, 'resource_metrics_complete': all(x is not None for x in percentages.values()),
            'estimated_clock_period_ns': None if estimate is None else float(estimate),
            'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'warnings': warnings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml')
    parser.add_argument('--top', required=True)
    parser.add_argument('--part', required=True)
    parser.add_argument('--clock-ns', required=True)
    args = parser.parse_args()
    print(json.dumps(parse_csynth(args.xml, args.top, args.part, args.clock_ns), indent=2, allow_nan=False))
