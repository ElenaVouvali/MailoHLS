"""Scoped reuse of V4 supervision. Historical trainer defaults are untouched."""
import argparse, contextlib, hashlib, json, sys
from pathlib import Path
from LLM_branch.train.stage1_decision_loss import apply_decision_policy, VERSION

def preserve_references(rows, *args, **kwargs):
    if kwargs.get('goal_mode', 'PARETO_LATENCY') != 'PARETO_LATENCY':
        raise ValueError('This experiment supports latency only')
    selected=[]
    for row in rows:
        if row.get('_quality_arm') != 'multi_reference':
            raise ValueError('Missing authenticated preselected multi-reference identity')
        out=dict(row, obj_mode='PARETO_LATENCY', _sample_weight=1.)
        selected.append(out)
    return selected, {'policy':'preserve_sealed_multireference_exposures_v1','rows':len(selected)}

@contextlib.contextmanager
def corrected(production):
    old_pack,old_select=production.build_deterministic_rhs_pack,production.select_goal_rows
    old_dump=getattr(production,"dump_json",None)
    def pack(source_text,target_text,tok,*args,**kwargs):
        result=old_pack(source_text,target_text,tok,*args,**kwargs)
        return apply_decision_policy(result,tok,production.build_rhs_map_from_target(target_text),'decision')
    def dump(path,payload):
        if Path(path).name=='training_contract.json' and isinstance(payload,dict):
            payload=dict(payload, directive_site_weighting='equal_actions_equal_nonforced_fields_value_tokens_only', research_supervision={'schema':'v3-supervision-v1','policy':VERSION,'reference_selection':'sealed exposures retained without deduplication','historical_top_k_argument_is_not_applied':True})
        return old_dump(path,payload)
    if old_dump is not None:production.dump_json=dump
    production.build_deterministic_rhs_pack=pack
    production.select_goal_rows=preserve_references
    try: yield
    finally:
        production.build_deterministic_rhs_pack=old_pack
        production.select_goal_rows=old_select
        if old_dump is not None:production.dump_json=old_dump

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    opts,argv=parser.parse_known_args()
    if '--' in argv:argv.remove('--')
    from LLM_branch.train import train_SFT_xattn_new as production
    helper=Path(__file__).resolve().parents[2]/'LLM_branch/train/stage1_decision_loss.py'
    receipt={'schema':'v3-supervision-v1','policy':VERSION,'implementation_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'reference_selection':'preserve every sealed complete measured exposure; do not rerank/deduplicate','loss':'action-normalized nonforced RHS values; delimiter/EOS excluded','argv':argv}
    opts.receipt.parent.mkdir(parents=True,exist_ok=True)
    if opts.receipt.exists():
        if json.loads(opts.receipt.read_text())!=receipt:raise ValueError('Supervision receipt mismatch')
    else:opts.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    sys.argv=[sys.argv[0],*argv]
    with corrected(production):production.main()
if __name__=='__main__':main()
