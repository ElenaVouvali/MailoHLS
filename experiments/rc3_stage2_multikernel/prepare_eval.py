"""Freeze four held-out Viterbi target requests for official decoder checks."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/rc3_stage2_multikernel/v1/audit/stage2_pilot.jsonl'
CONTRASTS=(
 ('R1','xczu7ev-ffvc1156-2-e',10.0,1.0),
 ('R2','xczu7ev-ffvc1156-2-e',3.33,1.0),
 ('R3','xcu200-fsgd2104-2-e',10.0,1.0),
 ('R4','xcu200-fsgd2104-2-e',3.33,1.0),
)
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=DATA)
 p.add_argument('--out',type=Path,default=ROOT/'artifacts/rc3_stage2_multikernel/v1/eval'); a=p.parse_args()
 rows=[json.loads(line) for line in a.data.read_text().splitlines() if line.strip()]
 selected=[]
 for rid,device,clock,budget in CONTRASTS:
  matches=[r for r in rows if r['_quality_split']=='validation' and r['kernel_name']=='machsuite-viterbi'
    and r['device']==device and abs(float(r['selected_clock_period'])-clock)<0.01
    and abs(float(r['resource_budget_frac'])-budget)<1e-9]
  if len(matches)!=1: raise ValueError(f'{rid}: expected one matched held-out row, found {len(matches)}')
  rec=dict(matches[0]); rec['_eval_request_id']=rid; selected.append(rec)
 source_hash=sha(selected[0]['input'].encode())
 if any(r['input']!=selected[0]['input'] for r in selected): raise ValueError('request sources differ')
 a.out.mkdir(parents=True,exist_ok=True)
 payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected).encode()
 path=a.out/'viterbi_four_targets.jsonl'; path.write_bytes(payload)
 manifest={'schema':'salt-hls-stage2-multikernel-heldout-inference-v1','input_jsonl':str(path),
  'input_sha256':sha(payload),'source_sha256':source_hash,'kernel':'machsuite-viterbi',
  'family':'machsuite_viterbi','split':'validation','objective':'PARETO_LATENCY',
  'requests':[{'id':r['_eval_request_id'],'device':r['device'],'clock_ns':r['selected_clock_period'],
    'budget_frac':r['resource_budget_frac'],'source_measurement_jsonl_idx':r['_source_measurement_jsonl_idx'],
    'request_id':r['_quality_request_id']} for r in selected],
  'note':'Rows are held-out authenticated Stage-1 requests; target assignments are retained only as reference metadata and are not supplied as model input.'}
 mp=a.out/'manifest.json'; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'input_jsonl':str(path),'input_sha256':manifest['input_sha256'],'manifest':str(mp),
  'requests':manifest['requests']},indent=2))
if __name__=='__main__': main()
