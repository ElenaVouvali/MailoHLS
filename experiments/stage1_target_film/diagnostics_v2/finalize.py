"""Finalize only from completed no-training receipts; never promote partial jobs."""
import json,hashlib,re,statistics,time
from pathlib import Path
ROOT=Path('artifacts/stage1_target_film/diagnostics_v2');OUT=Path('experiments/stage1_target_film/diagnostics_v2')
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def assignments(p):
 result={}
 for row in rows(p):
  d={k.upper():v for k,v in re.findall(r'(?im)^(auto\{[^}]+\})\s*=\s*(\S+)\s*$',row['candidates'][0]['prediction'])};assert len(d)==25;result[row['context_id']]=d
 assert len(result)==18;return result

def main():
 required=[ROOT/'development/summary.json',ROOT/'training_precision_control/summary.json',ROOT/'bank_v2/TARGET_CONTRAST_AUDIT.json',*[ROOT/('gan_'+m)/'execution.json' for m in ('on','off','legacy')]];started=time.monotonic()
 while not all(p.exists() for p in required):
  if time.monotonic()-started>21600:raise RuntimeError('Diagnostics incomplete; no final verdict fabricated')
  time.sleep(30)
 from .fixed_prefix import save
 fixed_prefix=save(ROOT,OUT)
 dev=read(required[0]);prec=read(required[1]);bank=read(required[2]);actual={m:assignments(ROOT/('gan_'+m)/'predictions.jsonl') for m in ('on','off','legacy')};historical=assignments(Path('artifacts/stage1_target_film/liono_inference/B_checkpoint48/predictions.jsonl'));trace=rows(ROOT/'paired_scores.jsonl');comparison={}
 for label,x,y in [('ON_vs_OFF',actual['on'],actual['off']),('corrected_vs_matched_legacy',actual['on'],actual['legacy']),('corrected_vs_original_B',actual['on'],historical)]:
  cases=[]
  for context in x:
   changed=[k for k in sorted(x[context]) if x[context][k]!=y[context][k]];cases.append({'context':context,'changed_fields':changed,'distance':len(changed)})
  comparison[label]={'changed_requests':sum(bool(c['distance']) for c in cases),'mean_assignment_fraction':statistics.mean(c['distance'] for c in cases)/25,'cases':cases}
 pairdev=rows(ROOT/'development/pairs.jsonl');pairprec=rows(ROOT/'training_precision_control/pairs.jsonl');pd={r['pair_id']:r for r in pairdev};precision_effect={'pairs':len(pairprec),'max_preferred_score_change_from_native':max(abs(r['scores']['preferred_value']-pd[r['pair_id']]['scores']['preferred_value']) for r in pairprec),'max_ON_OFF_margin_change_in_prepared_precision':max(abs((r['scores']['preferred_value']-r['scores']['rejected_value'])-(r['scores']['preferred_off']-r['scores']['rejected_off'])) for r in pairprec)}
 result={'fixed_prefix_summary':fixed_prefix,'development':dev,'precision_control':prec,'precision_effect':precision_effect,'GAN_comparison':comparison,'fixed_prefix_candidate_traces':len(trace),'max_fixed_prefix_ON_OFF_mean_score_change':max(abs(r['scores']['value']['mean_logprob']-r['scores']['off']['mean_logprob']) for r in trace),'max_fixed_prefix_value_legacy_mean_score_change':max(abs(r['scores']['value']['mean_logprob']-r['scores']['legacy']['mean_logprob']) for r in trace),'full_assignments':actual,'input_sha256':{str(p):sha(p) for p in required},'training_updates':0,'HLS_calls':0}
 (OUT/'COMPLETED_DIAGNOSTICS.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'TARGET_CONTRAST_AUDIT.json').write_text(json.dumps(bank,indent=2)+'\n')
 num=read(OUT/'FILM_NUMERICS.json');num['GAN_ablation_status']='COMPLETED';num['precision_control_status']='COMPLETED';num['precision_control']=precision_effect;num['fixed_prefix_candidate_traces']=len(trace);num['max_fixed_prefix_ON_OFF_mean_score_change']=result['max_fixed_prefix_ON_OFF_mean_score_change'];(OUT/'FILM_NUMERICS.json').write_text(json.dumps(num,indent=2)+'\n')
 with (OUT/'FILM_ABLATION.md').open('a') as f:f.write('\n## Completed GAN and precision controls\n\n'+json.dumps({'GAN':{k:{z:v for z,v in x.items() if z!='cases'} for k,x in comparison.items()},'precision':precision_effect,'fixed_prefix_mean_score_effect':result['max_fixed_prefix_ON_OFF_mean_score_change']},indent=2)+'\n\nFull assignments and per-request changed fields: `COMPLETED_DIAGNOSTICS.json`. OFF uses the same B48 residual, never A32. Exact-prefix scores are reused only where all inputs match; divergent-prefix cache misses receive real forwards. Matched legacy replay separates masking from hardware/kernel differences in the original report.\n')
 lines=['# Broader non-test contrast audit','', 'Complete source configuration eligibility was checked at canonical commit `'+bank['canonical_registry_commit']+'`. Restored source/action cohorts and unobservable all-zero resources were excluded from the frozen old-bank candidate audit. No historical dataset or checkpoint was replaced.','', '|Split/axis/class|Derived views|Unique candidate pairs|Independent kernels|Independent families|','|---|---:|---:|---:|---:|']
 for name,v in bank.get('independence',{}).items():lines.append(f'|{name}|{v["derived_views"]}|{v["unique_candidate_pairs"]}|{v["kernels"]}|{len(v["families"])}|')
 lines+=['','## Qualification limits','','Resource uncertainty is a sensitivity bound of +/-1 percentage point, not an authenticated report-rounding contract or exact counts. Device contrasts use device-relative capacities and do not isolate device architecture. Latency is the recorded measured field, not cycles times requested clock. Counts cover pair preferences, not necessarily near-optimal winner reversals. Exhaustive source/tool legality and restored complete-action reconstruction are required before an optimization bank is released. Accordingly no enriched bank is falsely labeled authenticated.','', 'Original sampler: one near-optimal preferred design (within1%) and one >=10% slower proxy-feasible design per request, followed by8 exposures/family. This loses matching pair identity across targets;128/54 selected pairs have zero same-pair switch/stay matches. Exact losses attributable separately to legality, near-optimal filtering and family balancing have not been disaggregated: INCONCLUSIVE. Do not substitute total pair counts for those missing stage-wise receipts.']
 (OUT/'RAW_CANDIDATE_SCREEN.md').write_text('\n'.join(lines)+'\n')
 robust=sum(v for fold in bank['statistics'].values() for k,v in fold.items() if 'robust_proxy_switch' in k);verdict='EXPLORATORY_RETRAIN_ONLY' if robust else 'DO_NOT_RETRAIN'
 (OUT/'COMPLETED_RETRAINING_STATUS.md').write_text('# Retraining decision\n\n**'+verdict+' — no retraining launched.**\n\nRaw matched pair contrasts are not an authenticated enriched training bank. The conservative resource bounds are assumptions; restored action states require reconstruction and all candidate pairs require explicit legality and source-identity qualification. Pair reversals below a shared dominant design do not establish a useful optimal-policy switch. These gaps block RETRAIN_RECOMMENDED and its automatic training authorization. GAN diversity never entered this decision.\n\nRetain authenticated V4 as the conservative Stage-1 parent. A32/B48 remain experimental; their original successful-update budgets were unequal (A interrupted after58; B completed64), so no final matched-training superiority is claimed. B ON/OFF isolates its inference FiLM contribution. No feasible-latency improvement, timing benefit, or generalization claim is established without matched synthesis receipts.\n\nBudget adaptation is proxy-qualified only. Clock/device candidate-pair effects require independent near-optimal development switch coverage and source/tool qualification. The registry restoration adds measured action columns, but its old corrupted/missing states cannot be silently attached to frozen bank labels.\n')
 decision=OUT/'RETRAINING_DECISION.md'
 text=decision.read_text().replace('The GAN natural-decoder ON/OFF/matched-legacy evaluation is still running.', 'The GAN natural-decoder ON/OFF/matched-legacy evaluation completed; see COMPLETED_DIAGNOSTICS.json.').replace('No deployment-benefit conclusion is inferred from its partial output.', 'No measured deployment benefit is inferred from directive or score changes.')
 decision.write_text(text)
 print('[FINAL_REPORTS_WRITTEN]',verdict,flush=True)
if __name__=='__main__':main()
