"""Apply the frozen generated-decision plus measured-hardware checkpoint rule."""
import argparse
import json
from pathlib import Path

from experiments.stage1_quality_sft.evaluate import select_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', required=True,
        help='JSON array of {checkpoint,step,report_path} records')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.evaluations).read_text())
    items = [dict(checkpoint=r['checkpoint'], step=int(r['step']),
                  report=json.loads(Path(r['report_path']).read_text())) for r in records]
    result = select_checkpoint(items)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
