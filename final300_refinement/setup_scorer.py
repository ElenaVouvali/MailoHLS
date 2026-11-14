#!/usr/bin/env python3
"""Create isolated V100S checkpoint-300 scorer from campaign-authenticated inference.

Never modifies deployed inference, Stage-2 training, adapter, or memory.
"""
import argparse, ast, hashlib, pathlib

REFERENCE_SHA = '811700e59e443e58d6c7049a33db231afbcc2af1766305a493ca14f41ed8f691'

def one(s, old, new):
    count = s.count(old)
    if count != 1:
        raise RuntimeError(f'Inference contract changed: expected one anchor ({count} found): {old[:120]}')
    return s.replace(old, new)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path(__file__).with_name('score_checkpoint300_liono.py'))
    a=p.parse_args()
    src=pathlib.Path(__file__).with_name('stage2_infer_reference.py')
    assert hashlib.sha256(src.read_bytes()).hexdigest()==REFERENCE_SHA, 'Supplied inference reference changed'
    s=src.read_text()
    s=one(s,"V100 = 'GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f'", "V100 = os.environ.get('SALT_GPU_UUID', 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb')")
    s=one(s,"    a.parent = Path('/home/evouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')", "    a.parent = Path(os.environ.get('SALT_PARENT_STAGE1', '/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run'))")
    s=one(s,"    p.add_argument('--dry-run', action='store_true'", "    p.add_argument('--score-proposals', type=Path, help='JSON slate of complete candidate/reference assignments')\n    p.add_argument('--score-out', type=Path, help='Output JSON with local causal likelihood deltas')\n    p.add_argument('--dry-run', action='store_true'")
    s=one(s,"        assignment, model_score = decode(case, packed, actions, model, film, hooks, tok, registry, dtype)", "        if a.score_proposals:\n            from score_local import score_slate\n            if len(cases)!=1 or not a.score_out:\n                raise ValueError('Scoring needs --kernel one and --score-out')\n            slate=json.loads(a.score_proposals.read_text())\n            result=score_slate(slate, case, packed, actions, model, film, hooks, tok, registry, dtype)\n            a.score_out.parent.mkdir(parents=True,exist_ok=True)\n            a.score_out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\\n')\n            print('[SCORED]', len(result['candidates']), a.score_out, flush=True)\n            continue\n        assignment, model_score = decode(case, packed, actions, model, film, hooks, tok, registry, dtype)")
    ast.parse(s)
    if a.output.exists() and a.output.read_text()!=s:raise FileExistsError('Different scorer already exists: '+str(a.output))
    a.output.write_text(s)
    print('Generated isolated scorer:',a.output)
    print('SHA256:',hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':main()
