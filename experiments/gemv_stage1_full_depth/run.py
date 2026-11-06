"""Run the frozen target/source matrix through the production constrained Stage-1 path."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from LLM_branch.common import directive_domains, source_tool_policy
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from experiments.stage1_quality_sft.gpu_preflight import check
from .common import ADAPTER, DEFAULT_OUT, GPU, PACKAGE, POLICY, contract, digest_object, read_json, sha, write_json
from .prepare import auth, FOCUS
from .trace import SelectedStates, assert_same_prediction, replay

REF_ID='GRID_ZCU104_10P0NS_100PCT'

def run(out):
    out=Path(out); manifest=read_json(out/'manifest.json')
    assert sha(out/'requests.jsonl')==manifest['requests_file_sha256']
    assert sha(PACKAGE/'inputs/gemv.c')==manifest['source_sha256']
    if (out/'official.jsonl').exists() or (out/'completed.json').exists(): raise FileExistsError('Refusing to overwrite official evidence')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU, 'CUDA_VISIBLE_DEVICES must name the assigned V100S UUID'
    gpu=check(GPU); assert gpu['uuid']==GPU and gpu['name']=='Tesla V100S-PCIE-32GB'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)==gpu['name']
    c,tok=auth(); frozen=read_json(out/'checkpoint.json')
    assert sha(ADAPTER/'adapter_model.safetensors')==frozen['adapter_sha256']
    state={}; original_load,original_predict=inf.load_stage_model,inf.predict_case
    request_rows=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
    by_id={r['request_id']:r for r in manifest['requests']}
    validations=[]; results=[]; controlled_prefixes={}; embedding_written=False
    baseline_actions=directive_domains.load_source_actions('gemv',PACKAGE/'inputs')
    policy=source_tool_policy.load_policy(POLICY)
    def loaded(args,tokenizer):
        model=original_load(args,tokenizer);model.eval()
        assert not getattr(model,'initialized_structural_xattn',False)
        state.update(model=model,tok=tokenizer,trace=None)
        return model
    def predict(**kwargs):
        row=kwargs['case'].platform_row; rid=row['context_id']; expected=by_id[rid]
        assert digest_object(row)==expected['request_sha256']
        previous=state.get('trace')
        if previous is not None: previous.close()
        official=original_predict(**kwargs)
        assignment=inf.parse_assignment_dict(official['candidates'][0]['canonical_prediction'])
        assert len(assignment)==13
        result=dict(request_id=rid,prediction=official,assignment=assignment,source=bool(official['candidates']))
        validations.append(dict(request_id=rid,complete_fields=len(assignment)==13,source_tool=bool(official['candidates'])))
        # Audit the latest source/tool policy using the exact action map for this source.
        source_name=rid.split('__',1)[0] if '__' in rid else None
        source_dir=PACKAGE/'sources'/source_name if source_name else PACKAGE/'inputs'
        actions=directive_domains.load_source_actions('gemv',source_dir)
        try: source_tool_policy.validate_source_assignment(assignment,actions); legacy=True; legacy_error=None
        except ValueError as e: legacy=False;legacy_error=str(e)
        try: source_tool_policy.validate_complete_assignment(assignment,actions,policy);v2=True;v2_error=None
        except ValueError as e: v2=False;v2_error=str(e)
        result['validation']={'source':legacy,'source_error':legacy_error,'source_tool_v2':v2,'source_tool_v2_error':v2_error}
        # Extract special-token deltas from the exact loaded wrapper once.
        nonlocal embedding_written
        if not embedding_written:
            _save_embedding_audit(state['model'],tok,out,c)
            embedding_written=True
        trace=state['trace']=SelectedStates(state['model'])
        if previous is not None:
            trace.index,trace.vectors=previous.index,previous.vectors
        # A representative full re-decode checks that attached read-only hooks do not alter greedy output.
        if not state.get('full_hook_check_done'):
            second=original_predict(**kwargs);assert_same_prediction(official,second)
            state['full_hook_check_done']=True;result['full_decode_hook_equivalence']=True
        if rid==REF_ID: state['reference_assignments']=assignment
        assert 'reference_assignments' in state, 'Reference request must precede all structural subset rows'
        records=replay(row,official,state['reference_assignments'],kwargs['model'],kwargs['tok'],kwargs['directive_domain_registry'],trace,controlled_fields=FOCUS)
        for rec in records:
            if rec['prefix_mode']=='controlled':
                old=controlled_prefixes.setdefault(rec['field'],rec['prior_output_ids'])
                assert old==rec['prior_output_ids'],f'Controlled prefix mismatch: {rec["field"]}'
        result['decisions']=official['candidates'][0]['site_debug']
        result['trace_records']=records
        results.append(result)
        write_json(out/f"{rid}_official.json",result)
        write_json(out/'candidate_scores.json',results)
        write_json(out/'validation.json',validations)
        # Each request is a separate compressed activation chunk with a colocated index.
        current_idx=[x for x in trace.index if x['request_id']==rid]
        indices=[x['vector_index'] for x in current_idx]
        vectors=np.stack([trace.vectors[i] for i in indices]).astype(np.float32)
        outdir=out/'hidden_states';outdir.mkdir(exist_ok=True)
        np.savez_compressed(outdir/f'{rid}.npz',vectors=vectors)
        local=[]
        for n,item in enumerate(current_idx):
            entry=dict(item);entry['vector_index']=n;local.append(entry)
        write_json(outdir/f'{rid}.index.json',local)
        # Release host RAM for this request; hooks stay installed but empty.
        trace.index=[];trace.vectors=[]
        print(f'[FULLDEPTH] {rid} complete; source-valid={legacy} v2-valid={v2}',flush=True)
        return official
    argv=['--stage','stage1','--adapter_dir',str(ADAPTER),'--directive_domain_registry_json',str(ADAPTER/'directive_domain_registry.json'),'--application_dataset_dir',str(PACKAGE/'inputs'),'--input_jsonl',str(out/'requests.jsonl'),'--output_jsonl',str(out/'official.jsonl'),'--admissibility_policy','legacy','--score_reduction','mean','--num_samples','1','--sample_seed','123','--max_prompt_tokens','7168','--debug_topk','10000','--candidate_max_prefix_tokens','0','--candidate_keep_head_tokens','256']
    write_json(out/'launch.json',dict(entrypoint='LLM_branch.inference.eval_stage1_stage2_stage3.main',argv=argv,gpu=gpu,manifest_sha256=sha(out/'manifest.json'),capture_layers=list(range(1,33)),capture_boundaries=['post_self_attention_residual','post_decoder_residual','final_rms_normalized_lm_head_input']))
    saved=sys.argv
    try:
        inf.load_stage_model,inf.predict_case=loaded,predict;sys.argv=[inf.__file__,*argv]
        with torch.inference_mode(): inf.main()
        assert len(results)==manifest['request_count']
        assert state.get('full_hook_check_done') and all(x.get('full_decode_hook_equivalence') for x in results if x['request_id']==REF_ID)
        assert all(x['validation']['source'] for x in results)
        assert sha(ADAPTER/'adapter_model.safetensors')==frozen['adapter_sha256']
        write_json(out/'completed.json',dict(status='COMPLETED',request_count=len(results),grid=18,historical=1,structure=24,full_depth_layers=32,hidden_states_are_selected_positions=True,all_source_valid=True,weights_unchanged=True,gpu=gpu,controlled_prefixes_identical=True,official_prediction_count=len(results)))
    finally:
        inf.load_stage_model,inf.predict_case=original_load,original_predict;sys.argv=saved
        if state.get('trace') is not None:state['trace'].close()

def _save_embedding_audit(model,tok,out,training_contract):
    emb=model.get_input_embeddings()
    adapter=getattr(emb,'token_adapter',None)
    if adapter is None: raise RuntimeError('Expected trained input-only special-token wrapper is absent')
    name=adapter.active_adapters[0] if adapter.active_adapters else next(iter(adapter.token_indices))
    ids=list(adapter.token_indices[name])
    trained=adapter.trainable_tokens_delta[name].detach().float().cpu().numpy()
    initial=adapter.trainable_tokens_original[name].detach().float().cpu().numpy()
    if trained.shape!=initial.shape or trained.shape[0]!=len(ids): raise ValueError(f'token delta/index mismatch {trained.shape}/{initial.shape}/{len(ids)}')
    change=trained-initial; mapping=training_contract['special_token_to_id']; token_by_id={int(v):k for k,v in mapping.items()}
    rows=[]
    for i,(tid,before,after,delta) in enumerate(zip(ids,initial,trained,change)):
        b=before.astype(np.float64);a=after.astype(np.float64);d=delta.astype(np.float64)
        denom=float(np.linalg.norm(b)*np.linalg.norm(a));cos=float(np.dot(b,a)/denom) if denom else None
        rows.append(dict(token_id=int(tid),token=token_by_id.get(int(tid)),row=i,trainable=True,initial_l2=float(np.linalg.norm(b)),trained_l2=float(np.linalg.norm(a)),update_l2=float(np.linalg.norm(d)),relative_update_l2=float(np.linalg.norm(d)/(np.linalg.norm(b)+1e-12)),initial_to_trained_cosine=cos,update_sha256=__import__('hashlib').sha256(delta.tobytes()).hexdigest()))
    np.savez_compressed(out/'special_token_embeddings.npz',token_ids=np.asarray(ids,dtype=np.int64),initial=initial.astype(np.float32),trained=trained.astype(np.float32),updates=change.astype(np.float32))
    write_json(out/'special_token_embeddings.json',dict(tensor_shape=list(trained.shape),tokens=rows,role=training_contract.get('special_token_role'),module=training_contract.get('trainable_token_modules'),initial_rows='PEFT TrainableTokensLayer.trainable_tokens_original, preserved from exact resized frozen base at adapter initialization',pairwise_cosines='The plot/report will distinguish pretrained-row geometry from update geometry; similarity is not FPGA physical similarity.'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT_OUT);a=ap.parse_args();run(a.out)
if __name__=='__main__':main()
