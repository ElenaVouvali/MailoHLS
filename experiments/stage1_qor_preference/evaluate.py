"""Matched native inference and existing GEMV trace reuse; no checkpoint selection."""
from __future__ import annotations
import argparse
import shutil
import sys
from .common import *

PAIRED = PARENT_WORKTREE/'artifacts/stage1_quality_sft/v4_paired_20261105'
GEMV = ROOT/'artifacts/gemv_framework_deep_dive/v4_20261106'


def setup54():
    out=OUT/'validation54';out.mkdir(parents=True,exist_ok=True)
    frozen=read(PAIRED/'manifest.json')
    assert sha(PAIRED/'frozen_validation_requests.jsonl')==frozen['request_file_sha256']
    assert sha(PAIRED/'candidate_pools.jsonl.gz')==frozen['candidate_pool_file_sha256']
    assert sha(PAIRED/'metric_requests.jsonl.gz')==frozen['metric_request_file_sha256']
    assert frozen['adapters']['multi_reference']['adapter_sha256']==ADAPTER_SHA
    for name in ['frozen_validation_requests.jsonl','candidate_pools.jsonl.gz','metric_requests.jsonl.gz','manifest.json','synthesis_subset.json']:
        target=out/name
        if target.exists():assert sha(target)==sha(PAIRED/name)
        else:shutil.copyfile(PAIRED/name,target)
    (out/'predictions').mkdir(exist_ok=True)
    baseline=PAIRED/'predictions/multi_reference_legacy.jsonl'
    shutil.copyfile(baseline,out/'predictions/baseline_v4.jsonl')
    write(out/'baseline_receipt.json',dict(original_path=str(baseline),sha256=sha(baseline),
        adapter_sha256=ADAPTER_SHA,contract_sha256=CONTRACT_SHA,requests_sha256=sha(out/'frozen_validation_requests.jsonl'),
        reuse='authenticated historical V4 legacy greedy outputs; no production or single-reference model used'))
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
    assert len(inference.load_inference_cases_jsonl(str(out/'frozen_validation_requests.jsonl')))==54
    return out


def infer54():
    out=setup54();checkpoint=MODEL_OUT/'run';prediction=out/'predictions/refined_dpo.jsonl'
    if prediction.exists():raise FileExistsError(prediction)
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
    argv=['--stage','stage1','--adapter_dir',str(checkpoint),'--directive_domain_registry_json',str(checkpoint/'directive_domain_registry.json'),
        '--application_dataset_dir',str(ROOT/'Data/ApplicationDataset'),'--input_jsonl',str(out/'frozen_validation_requests.jsonl'),
        '--output_jsonl',str(prediction),'--admissibility_policy','legacy','--score_reduction','mean','--num_samples','1',
        '--candidate_max_prefix_tokens','0','--candidate_keep_head_tokens','256','--debug_topk','10000']
    from experiments.stage1_quality_sft.gpu_preflight import check
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')==GPU
    gpu=check(GPU)
    write(out/'inference_execution_manifest.json',dict(gpu=gpu,generation=argv,
        models={name:dict(directory=str(path),adapter_sha256=sha(path/'adapter_model.safetensors'),
            training_contract_sha256=sha(path/'training_contract.json'),registry_path=str(path/'directive_domain_registry.json'))
            for name,path in [('baseline_v4',V4),('refined_dpo',checkpoint)]}))
    old=sys.argv
    try:sys.argv=[inference.__file__,*argv];inference.main()
    finally:sys.argv=old


def summarize54():
    from experiments.stage1_quality_sft.evaluate_paired_v4 import evaluate_prediction_file
    out=OUT/'validation54'
    reports={name:evaluate_prediction_file(name,out/'predictions'/f'{name}.jsonl',output_dir=out,policy_path=POLICY)
             for name in ['baseline_v4','refined_dpo']}
    a=rows(out/'predictions/baseline_v4.jsonl');b=rows(out/'predictions/refined_dpo.jsonl')
    from experiments.stage1_quality_sft.core import parse
    changes=[]
    for x,y in zip(a,b,strict=True):
        assert x['context_id']==y['context_id']
        px=parse(x['candidates'][0]['canonical_prediction']);py=parse(y['candidates'][0]['canonical_prediction'])
        changes.append(dict(request_id=x['context_id'],changed_fields=[k for k in px if px[k]!=py.get(k)],
                            baseline=px,refined=py))
    write(OUT/'validation54_comparison.json',dict(reports=reports,changes=changes,
        requests_changed=sum(bool(r['changed_fields']) for r in changes),
        interpretation='Unmatched complete assignments have unknown physical QoR, never a hardware failure'))


def gemv():
    from experiments.gemv_framework_deep_dive import run as existing
    from experiments.gemv_framework_deep_dive import analyze as analysis
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
    checkpoint=MODEL_OUT/'run';out=OUT/'gemv';out.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','requests.jsonl','effective_gemv_domains.json']:
        target=out/name
        if target.exists():assert sha(target)==sha(GEMV/name)
        else:shutil.copyfile(GEMV/name,target)
    if not (out/'prompts').exists():shutil.copytree(GEMV/'prompts',out/'prompts')
    original_history=read(GEMV/'historical_reference.json')
    write(out/'baseline_historical_reference.json',original_history)
    write(out/'historical_reference.json',dict(requests={},reason='Refined checkpoint; historical V4 outputs are comparison data, not equality requirements'))
    baseline=rows(GEMV/'official.jsonl')
    common_reference=inference.parse_assignment_dict(baseline[0]['candidates'][0]['canonical_prediction'])
    write(out/'controlled_prefix_reference.json',dict(source='authentic V4 R1, shared by both models',
        source_sha256=sha(GEMV/'official.jsonl'),assignments=common_reference))
    baseline_checkpoint=read(GEMV/'checkpoint.json')
    write(out/'checkpoint.json',dict(adapter_dir=str(checkpoint),files={name:sha(checkpoint/name) for name in baseline_checkpoint['files']},
        controlled_prefix_reference='fixed V4 R1 assignments',training_contract=read(checkpoint/'training_contract.json')))
    old_adapter,old_auth,old_replay=existing.ADAPTER,existing.authenticate,existing.replay
    def auth():
        c=read(checkpoint/'training_contract.json');assert c['qor_preference']['parent_adapter_sha256']==ADAPTER_SHA
        return c,inference.load_checkpoint_tokenizer(str(checkpoint),c)
    def shared_prefix(row,official,reference_assignments,model,tok,registry,trace):
        return old_replay(row,official,common_reference,model,tok,registry,trace)
    try:
        existing.ADAPTER=checkpoint;existing.authenticate=auth;existing.replay=shared_prefix
        existing.run(out)
    finally:existing.ADAPTER=old_adapter;existing.authenticate=old_auth;existing.replay=old_replay
    # Historical V4 outputs are plotting/comparison evidence only, never an
    # equality gate for the newly refined checkpoint.
    write(out/'historical_reference.json',original_history)
    analysis.analyze(out)
    refined=rows(out/'official.jsonl')
    changes=[]
    for a,b in zip(baseline,refined,strict=True):
        va=inference.parse_assignment_dict(a['candidates'][0]['canonical_prediction'])
        vb=inference.parse_assignment_dict(b['candidates'][0]['canonical_prediction'])
        changes.append(dict(request=a['context_id'],baseline=va,refined=vb,
            changed_fields=[k for k in va if va[k]!=vb[k]]))
    write(OUT/'gemv_comparison.json',dict(changes=changes,
        baseline_pairwise_sha256=sha(GEMV/'pairwise.csv'),refined_pairwise_sha256=sha(out/'pairwise.csv'),
        comparison='Within-model target contrast statistics only; no cross-checkpoint vector cosine or common learned-basis assumption',
        controlled_prefix='identical authentic V4 R1 preceding assignments for both models; own natural predictions retained',
        checkpoint_selection='frozen final checkpoint selected before any GEMV evaluation'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['setup54','infer54','summarize54','gemv']);a=ap.parse_args()
    {'setup54':setup54,'infer54':infer54,'summarize54':summarize54,'gemv':gemv}[a.command]()
