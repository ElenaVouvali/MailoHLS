"""Append-only final evidence snapshot. Requires completed baseline retry.
Use a fresh --out directory; --dry-run validates inputs without writing reports.
"""
import argparse,csv,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();r=ROOT/'experiments/final_architecture_v3/results_v3';art=ROOT/'artifacts/final_architecture_v3';retry=art/'stage3_anchor_confirmation/result.json'
 if not retry.exists():raise RuntimeError('Physical baseline confirmation not complete; no final snapshot fabricated')
 inputs=[r/'architecture_decisions.json',r/'corrected_static_completed.json',r/'stage3_completed.json',r/'decoder_comparison.json',r/'measured_bank_analysis.json',r/'joint_trip_probe.json',r/'resource_ledger.json',retry,art/'latency_refinement_path4/bounded_cancellation.json',art/'latency_refinement_path4/active_query_after_cancel.json']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs};print('Validated',len(hashes),'source receipts; FREEZE=NO, pending independent physical endpoint qualification.')
 if a.dry_run:return
 if a.out.exists():raise FileExistsError('Use a new output directory; historical snapshots are immutable')
 a.out.mkdir(parents=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 experiment=[]
 def add(id,hypothesis,baseline,variant,status,metrics,decision,limitations,updates=0,host='CPU',uuid=None,checkpoint=None,contexts=None):
  experiment.append({'experiment_id':id,'hypothesis':hypothesis,'baseline':baseline,'variant':variant,'code_commit':commit,'dataset_identity':'See hashed source receipts; no locked-test load','split_identity':'seed123 frozen family split;17 train families/2 validation families; preference16 train families after spcl exclusion','checkpoint_identity':checkpoint,'seed':123,'GPU_host':host,'GPU_UUID':uuid,'training_updates':updates,'training_runtime':None,'evaluation_contexts':contexts,'available_metrics':metrics,'confidence_intervals_if_applicable':None,'synthesis_receipts':str(retry) if id=='R8' else 'See measured_bank_analysis.json' if id=='R3' else [],'status':status,'limitations':limitations,'decision':decision})
 static=load(r/'corrected_static_completed.json');stage3=load(r/'stage3_completed.json');decode=load(r/'decoder_comparison.json');bank=load(r/'measured_bank_analysis.json');physical=load(retry)
 add('R1','Correct V4-compatible fusion supervision','V4 frozen','static C fresh fusion','PASS',static,'Select50 as research parent; no physical promotion','150-update bounded pilot; only2 validation families',150,'liono','GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb',static['run'],18)
 add('R2','Compiler/LLM facts complementary and used','Source/LLM','GNN probes + evidence intervention','INCONCLUSIVE',{'probe':load(r/'joint_trip_probe.json')['models'],'controls':decode['evidence_controls']},'Recoverability and weak score dependence only','Explicit input trip-count labels;74/13 actions, high-dimensional probes, two families')
 add('R3','Meaningful same-program target switches','Identical full assignments','two clocks + exact-count budget views','INCONCLUSIVE',{'contrasts':bank['clock_contrasts'],'binding':bank['binding_budget_contrasts']},'No clock reversal; one OOD budget boundary; do not infer conditioned-memory winner','12 physical jobs,2 families,3 candidates/context; HLS estimates only')
 add('R7','Required DPO improves selected Stage2','S2 and matched chosen-only SFT','summed-value DPO beta.1','INCONCLUSIVE',stage3,'DPO_REQUIRED_BUT_NOT_QUALIFIED','34 updates/one seed; same25/54 validation ranking, generated9/9 assignments unchanged; resource-proxy labels',34,'liono','GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb','qualification/preferences/dpo/checkpoint-34',54)
 add('decode','Real generation differs/useful','Stage1','Stage2/SFT/Stage3','PASS',decode['summary'],'No model QoR superiority established','36 complete outputs but2 independent families; no exact physical joins',0,'davinci','GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f',None,9)
 add('R8','Optional measured-feedback refinement improves Stage3','Actual Stage3 Pathfinder4 output','coordinate/RC3-pair + one longer baseline retry','BLOCKED' if physical['measurement']['synthesis_status']!='success' else 'INCONCLUSIVE',{'baseline_confirmation':physical['measurement'],'cancellation':load(art/'latency_refinement_path4/bounded_cancellation.json')},'AVAILABLE_EXPERIMENTAL; no measured refinement gain','Initial per-job timeouts; policy comparison canceled; one longer exact-baseline confirmation. No post-route claim',0,'coroni',None,None,1)
 for id in ('R4_conditioning','R5_gate_layers','R6_contrast_SFT','locked_final_test'):
  add(id,'Conditional qualification gate','Established compatible path','Not escalated','NOT_RUN',{},'NOT_IDENTIFIABLE; not rejected with fair evidence','Informative independent-family target switches/useful physical fusion signal absent; locked test remains unopened')
 for row in experiment:
  if row['experiment_id']=='R7':row['training_runtime']={k:v['runtime_seconds'] for k,v in stage3['arms'].items()}
  if row['experiment_id']=='R1':row['training_runtime']=load(r/'resource_ledger.json')['corrected_static_inclusive_file_timestamp_interval_seconds']
 manifest={'schema':'salt-v3-final-snapshot-v1','generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code_commit':commit,'git_dirty_at_generation':dirty,'input_receipts_sha256':hashes,'large_artifacts':'Ignored persistent run directories; never committed; no secrets copied','weights':{str(p):sha(p) for p in [Path('/home/elvouvali/mailohls_runs/final_architecture_v3/qualification/stage2_selected_export/structural_xattn.pt'),Path('/home/elvouvali/mailohls_runs/final_architecture_v3/qualification/preferences/dpo/checkpoint-34/structural_xattn.pt')]}}
 for name,obj in [('experiment_results.json',experiment),('artifact_manifest.json',manifest),('architecture_decisions.json',load(r/'architecture_decisions.json'))]:(a.out/name).write_text(json.dumps(obj,indent=2)+'\n')
 with (a.out/'experiment_results.csv').open('w') as f:
  fields=['experiment_id','status','training_updates','GPU_host','GPU_UUID','decision','limitations'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in experiment)
 ledger=load(r/'resource_ledger.json');ledger['baseline_retry_reserved']=1;ledger['hls_charged_attempts_including_pending']=11+12+ledger['refinement_reservations']+1;ledger['final_baseline_retry_status']=physical['measurement']['synthesis_status'];(a.out/'resource_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
 print(a.out)
if __name__=='__main__':main()
