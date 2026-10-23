import json
from pathlib import Path

target_path = Path('Target_general_experiments.jsonl')
salt_path = Path('SALT_experiments.jsonl')

target = [json.loads(line) for line in target_path.read_text().splitlines() if line.strip()]
salt = [json.loads(line) for line in salt_path.read_text().splitlines() if line.strip()]

ref_kernels = ['GAN', 'chstone-jpeg', 'Kalman', 'covariance']
salt_refs = {
    r['kernel']: r for r in salt
    if r.get('objective') == 'LATENCY' and r.get('stage') == 'stage-3' and r['kernel'] in ref_kernels
}
assert set(salt_refs) == set(ref_kernels)

def ref_row(kernel):
    s = salt_refs[kernel]
    jpeg = kernel == 'chstone-jpeg'
    return {
        'kernel': kernel,
        'device': 'ZCU104',
        'clock_period_ns': 10.0,
        'resource_budget': 'full',
        'stage': 'reference-salt-stage-3',
        'objective': 'LATENCY',
        'directives': s['directives'],
        'status': 'success',
        'latency': s['latency'],
        'resources': s['resources'],
        'latency_source': 'SALT_experiments.jsonl',
        'latency_measurement': 'rtl_cosim_verilog_cycles' if jpeg else 'csynth_worst_case_cycles',
        'synthesis_status': 'completed',
        'implementation_status': 'not_run',
        'cosimulation_status': 'passed' if jpeg else 'not_run',
        'result_source': 'SALT_experiments.jsonl',
        'flow_script': 'jpeg_csynth_cosim.tcl' if jpeg else 'kernel_csynth.tcl',
        'reference_source': 'SALT_experiments.jsonl',
    }

rows = [ref_row(k) for k in ref_kernels]
for old in target:
    # The first two target rows are replaced by the exact SALT references above.
    if old.get('device') == 'ZCU104' and old.get('clock_period_ns') == 10.0:
        continue
    row = dict(old)
    row.setdefault('stage', 'target-evaluation')
    row.setdefault('objective', 'LATENCY')
    row.setdefault('resource_budget', 'full')
    row.setdefault('implementation_status', 'not_run')
    row.setdefault('synthesis_status', 'completed')
    row.setdefault('flow_script', 'jpeg_csynth_cosim.tcl' if row['kernel'] == 'chstone-jpeg' else 'kernel_csynth.tcl')
    row.setdefault('reference_source', None)
    rows.append(row)

assert len(rows) == 12
assert [(r['kernel'], r['device'], r['clock_period_ns']) for r in rows[:4]] == [
    ('GAN', 'ZCU104', 10.0), ('chstone-jpeg', 'ZCU104', 10.0),
    ('Kalman', 'ZCU104', 10.0), ('covariance', 'ZCU104', 10.0)]
for row in rows[:4]:
    s = salt_refs[row['kernel']]
    assert row['latency'] == s['latency']
    assert row['resources'] == s['resources']
    assert row['directives'] == s['directives']

target_path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
print('wrote', target_path, 'rows=', len(rows))
