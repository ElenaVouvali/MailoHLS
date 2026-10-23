import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'paper_plots' / 'Target_general_experiments.jsonl'
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

array_vars = {
    'GAN': {'L1': 'xbuf', 'L2': 'layer_1_out', 'L3': 'layer_2_out'},
    'Kalman': {'L1': 'in_local', 'L2': 'out_local', 'L5': 'u_hat_arr', 'L6': 'p_arr_1', 'L7': 'k_arr', 'L8': 'calc_temp_arr'},
    'chstone-jpeg': {'L5': 'QuantBuff', 'L6': 'HuffBuff', 'L7': 'IDCTBuff', 'L15': 'huffsize', 'L16': 'huffcode'},
}

def expand(kernel, shorthand):
    result = []
    arrays = array_vars.get(kernel, {})
    for action_id, value in shorthand.items():
        if action_id in arrays:
            if value == 'NONE' or value == 'N':
                result.append({'action_id': action_id, 'kind': 'array', 'variable': arrays[action_id], 'partition_type': 'none', 'partition_factor': 0, 'partition_dim': 0})
            elif value == 'COMP':
                result.append({'action_id': action_id, 'kind': 'array', 'variable': arrays[action_id], 'partition_type': 'complete', 'partition_factor': 0, 'partition_dim': 1})
            else:
                kind = {'C': 'cyclic', 'B': 'block'}[value[0]]
                result.append({'action_id': action_id, 'kind': 'array', 'variable': arrays[action_id], 'partition_type': kind, 'partition_factor': int(value[1:]), 'partition_dim': 1})
        else:
            result.append({'action_id': action_id, 'kind': 'loop', 'pipeline_ii': 1 if value == 'P' else 0, 'unroll_factor': int(value[1:]) if value.startswith('U') else 0})
    return result

formatted = []
for row in rows:
    directives = row['directives']
    if isinstance(directives, dict):
        directives = expand(row['kernel'], directives)
    formatted.append({
        'kernel': row['kernel'],
        'stage': 'stage-3',
        'objective': row.get('objective') or 'LATENCY',
        'latency': row.get('latency'),
        'resources': row['resources'],
        'directives': directives,
    })

assert len(formatted) == 12
assert all(list(row) == ['kernel', 'stage', 'objective', 'latency', 'resources', 'directives'] for row in formatted)
assert all(isinstance(row['directives'], list) for row in formatted)
path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in formatted))
print('formatted', len(formatted), 'rows in SALT schema')
