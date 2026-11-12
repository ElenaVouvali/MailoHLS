#!/usr/bin/env python3
"""Conservative no-truncation certificate for matched Stage-2 rows."""
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer
ap=argparse.ArgumentParser()
ap.add_argument('--data',type=Path,required=True)
ap.add_argument('--model',required=True)
ap.add_argument('--max-length',type=int,required=True)
a=ap.parse_args()
tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
lengths=[];failed=[]
for i,line in enumerate(a.data.open()):
 if not line.strip():continue
 row=json.loads(line)
 # Conservative upper bound relative to typical single-prompt source + target.
 n=len(tok(row['input']+'\n'+row['target'],add_special_tokens=False)['input_ids'])
 lengths.append(n)
 if n>a.max_length-256: failed.append((i,row['kernel_name'],n))
print(json.dumps({'rows':len(lengths),'max_tokens':max(lengths,default=0),
 'max_length':a.max_length,'conservative_overflow_count':len(failed),
 'overflow_examples':failed[:10]},indent=2))
if failed:raise SystemExit('Refusing truncated-source experiment; increase MAX_LENGTH or revise cohort explicitly')
