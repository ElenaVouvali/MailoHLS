import json
from pathlib import Path

path = Path('Target_general_experiments.jsonl')
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
contexts = [
    ('ZCU104', 10.0), ('ZCU104', 10.0), ('ZCU104', 10.0), ('ZCU104', 10.0),
    ('ZCU104', 6.67), ('ZCU104', 6.67), ('ZCU104', 6.67), ('ZCU104', 6.67),
    ('U50', 10.0), ('U50', 10.0), ('U50', 10.0), ('U50', 10.0),
]
assert len(rows) == len(contexts) == 12
formatted = []
for row, (device, clock) in zip(rows, contexts):
    formatted.append({
        'kernel': row['kernel'],
        'device': device,
        'clock_period_ns': clock,
        'resource_budget': 'full',
        'stage': row['stage'],
        'objective': row['objective'],
        'latency': row['latency'],
        'resources': row['resources'],
        'directives': row['directives'],
    })
path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in formatted))
print('restored target context for', len(formatted), 'rows')
