"""Summarize completed GAN FiLM predictions without inferring physical QoR."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    m = json.loads((a.run/'manifest.json').read_text())
    rows = json.loads((a.run/'predicted_directives.json').read_text())
    assert m['status'] == 'INFERENCE_COMPLETED' and len(rows) == 18
    assert len({r['context_id'] for r in rows}) == 18
    assert all(r['film_enabled'] and len(r['directives']) == 25 for r in rows)
    configurations = {}
    table = []
    for r in rows:
        key = json.dumps(r['directives'], sort_keys=True)
        config = configurations.setdefault(key, f'C{len(configurations)+1}')
        line = dict(request=r['context_id'], device='ZCU104' if r['device'].startswith('xczu') else 'U200',
                    clock_ns=r['clock_ns'], budget_pct=r['budget_pct'], configuration=config)
        for label in range(1, 12):
            kinds = ('ARRAY_T', 'ARRAY_F', 'ARRAY_D') if label <= 3 else ('PIPE', 'UNROLL')
            line[f'L{label}'] = '/'.join(r['directives'][f'AUTO{{_{kind}_L{label}}}'] for kind in kinds)
        table.append(line)
    reference = next(r for r in rows if r['context_id'] == 'GAN:ZCU104:10ns:100pct')
    for line, r in zip(table, rows):
        line['changed_fields_vs_reference'] = sum(v != reference['directives'][k] for k, v in r['directives'].items())
    keys = list(table[0])
    with (a.run/'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(table)
    text = ['# GAN: selected Stage-1 FiLM checkpoint, 18 TCAD requests', '',
        f'All 18 requests completed on Liono V100S with arm B step {m["checkpoint_step"]}, FiLM ON.', '',
        'Objective: `PARETO_LATENCY`. No optimizer update or synthesis was performed.', '',
        'Arrays L1–L3 show `type/factor/dimension`; loops L4–L11 show `pipeline/unroll`.',
        'Array names: L1=`xbuf`, L2=`layer_1_out`, L3=`layer_2_out`.',
        'Values use the checkpoint encoding: unroll 0 means disabled; complete partition uses factor 0.', '',
        '|Device|Clock ns|Budget %|Config|L1|L2|L3|L4|L5|L6|L7|L8|L9|L10|L11|',
        '|'+'---|'*15]
    for line in table:
        values = [line[k] for k in ['device', 'clock_ns', 'budget_pct', 'configuration', *[f'L{i}' for i in range(1,12)]]]
        text.append('|'+'|'.join(map(str, values))+'|')
    text.extend(['', f'{len(configurations)} distinct complete assignments across 18 requests.', '',
        'These are raw constrained-decoder predictions, checked against the sequential checkpoint proposal domains.',
        'Resource feasibility, achieved clock, and hardware latency remain unmeasured.', '',
        'Full directive text for every request: [DIRECTIVES.md](DIRECTIVES.md).',
        'Every individual field: [directives.csv](directives.csv).',
        'Candidate scores: `request_01.json` through `request_18.json`.', '',
        '## Checkpoint identity', '', f'- Parent SHA256: `{m["parent_sha256"]}`',
        f'- Residual LoRA SHA256: `{m["lora_sha256"]}`', f'- FiLM SHA256: `{m["film_sha256"]}`', ''])
    (a.run/'SUMMARY.md').write_text('\n'.join(text))
    (a.run/'summary.json').write_text(json.dumps(dict(completed=18, unique_assignments=len(configurations),
        checkpoint_step=m['checkpoint_step'], film_enabled=True, requests=table), indent=2)+'\n')
    print(json.dumps(dict(completed=18, unique_assignments=len(configurations))))


if __name__ == '__main__':
    main()
