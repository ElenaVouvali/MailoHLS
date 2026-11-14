"""Strict paired summary. Does not claim physical QoR or automatically promote B."""
import argparse,collections,json,math
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def aggreg(p):
    out=collections.defaultdict(list)
    for x in p:out[x['family'],x['kind'],x['group_id']].append(x)
    return {'complete_groups':len(out),'correct_both':sum(len(x)==2 and all(z['correct'] for z in x) for x in out.values()),
            'family_count':len(set(k[0] for k in out))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--A',type=Path,required=True);p.add_argument('--B',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    pa,pb=read(a.A/'protocol.json'),read(a.B/'protocol.json')
    matching=['parent_sha256','parent_contract_sha256','parent_registry_sha256','bank_manifest_sha256','initial_lora_tensor_sha256','seed','steps','eval_sft','eval_pairs']
    mismatch=[k for k in matching if pa[k]!=pb[k]]
    if mismatch:raise RuntimeError('Not a matched A/B experiment: '+str(mismatch))
    ba,bb=read(a.A/'best.json'),read(a.B/'best.json')
    offA=ba['selection'];onB=bb['selection'];offB=read(a.B/('eval_%04d.json'%bb['step']))['arms']['off']
    ga,gb,go=[aggreg(x['paired_results']) for x in (offA,onB,offB)]
    if not(ga['complete_groups']==gb['complete_groups']==go['complete_groups']):
        raise ValueError('Different validation group sets')
    paired=[]
    for old,new in zip(offA['paired_results'],onB['paired_results']):
        if any(old[k]!=new[k] for k in ('family','kind','group_id')):raise ValueError('Different validation pair ordering')
        paired.append({'family':old['family'],'kind':old['kind'],'group_id':old['group_id'],
                       'A_margin':old['margin'],'B_margin':new['margin'],'A_correct':old['correct'],'B_correct':new['correct']})
    # Declare FiLM qualified only with independent held-out switch coverage.
    enough=gb['family_count']>=2 and gb['complete_groups']>=8
    diff=onB['macro_two_context_group_accuracy']-offA['macro_two_context_group_accuracy']
    film_is_causal=onB['macro_two_context_group_accuracy']>offB['macro_two_context_group_accuracy']+1e-8
    sft_stable=onB['sft_nll']<=offA['sft_nll']+.05
    if not enough:decision='INCONCLUSIVE_INDEPENDENT_TARGET_SWITCH_COVERAGE';default='A'
    elif diff>=.05 and film_is_causal and sft_stable:decision='B_ADVANTAGE_ON_FIXED_PREFIX_CONTRASTS_REQUIRES_GENERATION_QOR';default='B_candidate'
    else:decision='NO_DEMONSTRATED_FILM_ADVANTAGE';default='A'
    report={'decision':decision,'stage1_default':default,'caveat':'Always perform complete-generation, source-domain and measured feasible-QoR verification before final architecture freeze.',
            'comparison':{'A':{'step':ba['step'],'metrics':{k:offA[k] for k in ('macro_pair_accuracy','macro_two_context_group_accuracy','sft_nll')},'groups':ga},
                          'B_ON':{'step':bb['step'],'metrics':{k:onB[k] for k in ('macro_pair_accuracy','macro_two_context_group_accuracy','sft_nll')},'groups':gb},
                          'B_OFF':{'metrics':{k:offB[k] for k in ('macro_pair_accuracy','macro_two_context_group_accuracy','sft_nll')},'groups':go}},
            'delta_B_minus_A_group_accuracy':diff,'B_on_improves_over_B_off':film_is_causal,
            'sft_nll_noninferior':sft_stable,'minimum_independent_validation_coverage':enough,
            'paired':paired,'limitations':['HLS-estimate percentage constraints','No new synthesis','Non-independent budget views','No independent device generalization','Evaluation uses fixed-prefix likelihoods, not generated QoR']}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='paired'},indent=2))
if __name__=='__main__':main()
