"""Summarize executed runs only; keep raw activations/checkpoints out of Git."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from experiments.gemv_structural_explainability.audit import sha,write_json,GNN_SHA
from experiments.gemv_structural_explainability.memory_alignment import MemoryAlignment


def rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def summarize_training(run_dirs,out):
    records={}
    for mode,directory in run_dirs.items():
        done=json.loads((directory/'alignment_completion.json').read_text())
        if done['updates']<=0:raise ValueError('Untrained run cannot enter matched comparison')
        protocol=json.loads((directory/'alignment_protocol.json').read_text())
        state=max((json.loads(p.read_text()) for p in directory.rglob('trainer_state.json')),
                  key=lambda s:s['global_step'])
        records[mode]=dict(completion=done,protocol=protocol,log_history=state['log_history'],
            checkpoint_path=str(directory/'structural_xattn.pt'),
            checkpoint_bytes=(directory/'structural_xattn.pt').stat().st_size)
    signatures=[(r['protocol']['data_sha256'],r['protocol']['split_sha256'],r['protocol']['stage1_sha256'],
                 r['protocol']['gnn_sha256'],r['protocol']['steps']) for r in records.values()]
    if len(set(signatures))!=1:raise ValueError('Training arms are not matched')
    write_json(out/'training_metrics.json',dict(runs=records,matched_inputs_and_budget=True,
        convergence_established=False,qualification='20-update pilot; independent training budget qualification required'))
    return records


def summarize_tokenizer(run,raw_dir,out):
    raw=torch.load(raw_dir/'gemv.pt',map_location='cpu',weights_only=False)
    weights=torch.load(run/'structural_xattn.pt',map_location='cpu',weights_only=True)
    state={k.split('.attn.alignment.',1)[1]:v for k,v in weights.items() if '.attn.alignment.' in k}
    if not state:raise ValueError('No trained alignment state')
    module=MemoryAlignment(raw['edge_dim'])
    module.load_state_dict(state,strict=True);module.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        initial=MemoryAlignment(raw['edge_dim'])
    displacement={k:float((value-initial.state_dict()[k]).float().norm()) for k,value in state.items()}
    with torch.no_grad():output=module(raw)
    n=raw['a'].shape[1];summary=output['memory'][0,64+n:].reshape(64,2,128)[:5]
    evidence=output['memory'][0,64:64+n]
    attention=output['summary_attention'][0,:5]
    flat=summary.flatten(0,1);unit=torch.nn.functional.normalize(flat,dim=-1)
    diagnostics=[]
    for k in range(5):
        eligible=int(raw['routes'][0,k].sum())
        for h in range(4):
            for s in range(2):
                w=attention[k,h,s]
                entropy=float(-(w*w.clamp_min(1e-30).log()).sum())
                diagnostics.append(dict(action='L'+str(k+1),head=h,summary=s,eligible=eligible,
                    entropy=entropy,normalized_entropy=entropy/np.log(eligible) if eligible>1 else 0.,
                    top1=float(w.max()),top3=float(w.topk(min(3,n)).values.sum()),
                    positive_weight_record_count=int(w.gt(0).sum())))
    gates={k:float(v.tanh()) for k,v in weights.items() if k.endswith('attn_gate')}
    metrics=dict(checkpoint_sha256=sha(run/'structural_xattn.pt'),gnn_sha256=GNN_SHA,
        evidence_norms=evidence.norm(dim=-1).tolist(),summary_norms=summary.norm(dim=-1).tolist(),
        summary_cosine=(unit@unit.T).tolist(),same_action_summary_cosine=
            torch.nn.functional.cosine_similarity(summary[:,0],summary[:,1],dim=-1).tolist(),
        summary_variance=float(summary.var(unbiased=False)),
        nonfinite_count=int((~torch.isfinite(output['memory'])).sum()),
        evidence_records=n,active_summaries=10,evidence_to_summary_ratio=n/10,
        active_memory_tokens=int(output['mask'].sum()),memory_shape=list(output['memory'].shape),
        tokenizer_trainable_parameters=sum(p.numel() for p in module.parameters()),
        parameter_displacement_from_seeded_initialization=displacement,
        diagnostics=diagnostics,gates=gates,
        caveat='Nonzero attention denotes aggregation weight; does not establish useful interpretation or causality')
    write_json(out/'alignment_embedding_metrics.json',metrics)
    from experiments.gemv_structural_explainability.evaluation import write_csv
    attention_rows=[]
    for k in range(5):
        for h in range(4):
            for s in range(2):
                for i,w in enumerate(attention[k,h,s].tolist()):
                    attention_rows.append(dict(action='L'+str(k+1),head=h,summary=s,
                        evidence_id=raw['records'][i]['id'],evidence_type=raw['records'][i]['type'],
                        eligible=bool(raw['routes'][0,k,i]),attention=w))
    write_csv(out/'memory_attention.csv',attention_rows)
    np.savez_compressed(out/'learned_summaries.npz',summaries=summary.numpy(),evidence=evidence.numpy(),
                        summary_attention=attention.numpy())
    return metrics


def compact_attention(evaluation_dir,out):
    candidates=rows(evaluation_dir/'candidate_scores.csv')
    selected={(r['lhs'],r['rhs']) for r in candidates if r['intervention']=='original' and r['selected']=='True'}
    raw=evaluation_dir/'decoder_attention.csv'
    sums=defaultdict(lambda:[0.,0.,0.,0.,0])
    invalid_nonzero=0
    with raw.open() as stream:
        for r in csv.DictReader(stream):
            if r['eligible']!='True':
                invalid_nonzero+=float(r['attention'])!=0.;continue
            if (r['lhs'],r['rhs']) not in selected:continue
            key=tuple(r[k] for k in ('model','action','lhs','layer','head','memory_id'))
            value=sums[key]
            for i,name in enumerate(('attention','normalized_entropy','gate','residual_hidden_ratio')):value[i]+=float(r[name])
            value[4]+=1
    if invalid_nonzero:raise ValueError('Invalid-key attention is nonzero')
    compact=[]
    for key,values in sums.items():
        compact.append(dict(zip(('model','action','lhs','layer','head','memory_id'),key),
            attention=values[0]/values[4],normalized_entropy=values[1]/values[4],
            gate=values[2]/values[4],residual_hidden_ratio=values[3]/values[4],query_token_count=values[4]))
    from experiments.gemv_structural_explainability.evaluation import write_csv
    write_csv(out/'decoder_attention.csv',compact)
    write_json(out/'decoder_attention_receipt.json',dict(raw_path=str(raw),raw_sha256=sha(raw),
        raw_bytes=raw.stat().st_size,invalid_nonzero_count=invalid_nonzero,
        aggregation='selected complete RHS continuation: mean over its causal query positions, preserving layer/head/key',
        compact_rows=len(compact)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--evaluation-root',type=Path,required=True);p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    runs={mode:args.runs_root/mode for mode in ('action','evidence','gmt')}
    training=summarize_training(runs,args.out)
    summarize_tokenizer(runs['gmt'],args.raw_dir,args.out)
    compact_attention(args.evaluation_root/'gmt',args.out)
    import shutil
    for name in ('memory_attention.csv','intervention_results.csv','end_to_end_interventions.json'):
        path=args.evaluation_root/'gmt'/name
        if path.is_file():shutil.copyfile(path,args.out/name)
    evaluations={}
    candidate_rows=[]
    for mode in runs:
        path=args.evaluation_root/mode/'candidate_scores.csv'
        scores=rows(path)
        base=[r for r in scores if r['intervention']=='original']
        candidate_rows.extend(base)
        evaluations[mode]=dict(selected={r['lhs']:r['rhs'] for r in base if r['selected']=='True'},
            candidate_scores_sha256=sha(path),checkpoint_sha256=training[mode]['completion']['checkpoint_sha256'],
            evaluation=json.loads((args.evaluation_root/mode/'evaluation_receipt.json').read_text()))
    from experiments.gemv_structural_explainability.evaluation import write_csv
    write_csv(args.out/'candidate_scores.csv',candidate_rows)
    write_json(args.out/'model_comparison.json',dict(models=evaluations,training_budget='20 updates each',
        physical_qor_improvement='NOT EVALUATED',general_superiority_established=False))


if __name__=='__main__':main()
