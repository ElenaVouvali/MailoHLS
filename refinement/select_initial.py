#!/usr/bin/env python3
"""Select one existing prediction by its complete deployment context; no inference."""
import argparse
import json
from pathlib import Path


def select(path, kernel, objective, part, clock, budget):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    matches = [r for r in rows if r.get('kernel') == kernel and
               r.get('objective') == objective and r.get('device') == part and
               r.get('clock_period_ns') == clock and r.get('budget_percent') == budget]
    if len(matches) != 1 or not isinstance(matches[0].get('directives'), list):
        raise ValueError('Expected exactly one matching prediction with directives; got {}'.format(len(matches)))
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--objective', default='LATENCY', choices=['LATENCY'])
    parser.add_argument('--part', default='xczu7ev-ffvc1156-2-e')
    parser.add_argument('--clock', type=float, default=10)
    parser.add_argument('--budget-percent', type=float, default=100)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    row = select(args.jsonl, args.kernel, args.objective, args.part, args.clock, args.budget_percent)
    # Exclusive creation prevents accidentally replacing a frozen prediction.
    with Path(args.out).open('x') as output:
        json.dump(row, output, indent=2, allow_nan=False)
        output.write('\n')
    print('Saved existing prediction; its stored QoR is annotation, not a fresh HLS evaluation')


if __name__ == '__main__':
    main()
