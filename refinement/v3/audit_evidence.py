#!/usr/bin/env python3
"""Read-only cross-version audit and optional v3 native bundle verification."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cg_bdr as cg
import runner_v3 as runner


def audit(bundles=()):
    archived = runner.audit()
    rows = []
    for bundle in bundles:
        cg.verify_bundle(Path(bundle))
        rows.append(cg.summary(bundle))
    return dict(legacy_verified=archived['archived']['legacy']['verified_evaluations'],
                v2_verified=archived['archived']['v2']['verified_evaluations'], v3=rows,
                new_hls_calls_by_audit=0, mutation_performed=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles',nargs='*',default=[])
    args = parser.parse_args()
    print(json.dumps(audit(args.bundles),indent=2,sort_keys=True))
