"""Command-line entry point for the isolated quality-aware Stage-1 study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.stage1_quality_sft.build import prepare
from experiments.stage1_quality_sft.training import export, run_production
from experiments.stage1_quality_sft.gpu_preflight import check as gpu_check
from experiments.stage1_quality_sft.inference import export_validation_inputs, normalize_production_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='authenticate raw data and freeze request/candidate manifests; CPU only')
    prep.add_argument('--output', required=True); prep.add_argument('--dataset')
    prep.add_argument('--split'); prep.add_argument('--registry'); prep.add_argument('--preprocessed-dir')
    exp = commands.add_parser('export', help='materialize equal-exposure Arm A and Arm B datasets; CPU only')
    exp.add_argument('--prepared', required=True); exp.add_argument('--output', required=True)
    exp.add_argument('--exposures', type=int, default=4); exp.add_argument('--seed', type=int, default=123)
    pf = commands.add_parser('preflight', help='check the explicitly assigned idle V100S; no training')
    pf.add_argument('--expected-uuid')
    run = commands.add_parser('run', help='launch one matched arm after fail-closed GPU preflight')
    run.add_argument('--arm-dir', required=True); run.add_argument('trainer_args', nargs=argparse.REMAINDER)
    infer = commands.add_parser('export-validation-inputs', help='create label-free production-decoder cases')
    infer.add_argument('--prepared', required=True); infer.add_argument('--sources', required=True); infer.add_argument('--output', required=True)
    norm = commands.add_parser('normalize-predictions', help='convert production inference JSONL for QoR evaluation')
    norm.add_argument('--predictions', required=True); norm.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        from experiments.stage1_quality_sft.build import DATASET, SPLIT, REGISTRY, PREPROCESSED
        result = prepare(Path(args.output), args.dataset or DATASET, args.split or SPLIT,
                         args.registry or REGISTRY, args.preprocessed_dir or PREPROCESSED)
        print(json.dumps(result, indent=2))
    elif args.command == 'export':
        print(json.dumps(export(Path(args.prepared), Path(args.output), args.exposures, args.seed), indent=2))
    elif args.command == 'preflight':
        print(json.dumps(gpu_check(args.expected_uuid), indent=2))
    elif args.command == 'export-validation-inputs':
        print(json.dumps(export_validation_inputs(Path(args.prepared), Path(args.sources), Path(args.output)), indent=2))
    elif args.command == 'normalize-predictions':
        print(json.dumps(normalize_production_predictions(Path(args.predictions), Path(args.output)), indent=2))
    else:
        run_production(Path(args.arm_dir), args.trainer_args)


if __name__ == '__main__':
    main()
