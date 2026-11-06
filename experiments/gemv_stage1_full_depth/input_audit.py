"""Tokenize the frozen target fields with the saved checkpoint tokenizer."""
import argparse,re
from pathlib import Path
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from .common import ADAPTER,DEFAULT_OUT,read_json,sha,write_json

def run(out):
 out=Path(out); c=read_json(ADAPTER/'training_contract.json');tok=inf.load_checkpoint_tokenizer(str(ADAPTER),c); rows=[]
 for line in (out/'requests.jsonl').read_text().splitlines():
  r=__import__('json').loads(line); text=(out/'prompts'/f"{r['context_id']}.txt").read_text();enc=tok(text,add_special_tokens=False,return_offsets_mapping=True); offsets=enc['offset_mapping']; ids=enc['input_ids'];targets=[]
  for name,pattern in [('BRAM_18K',r'BRAM_18K=(\d+)'),('BRAM_pct',r'BRAM_18K=\d+ \(([\d.]+)%'),('DSP',r'DSP=(\d+)'),('DSP_pct',r'DSP=\d+ \(([\d.]+)%'),('FF',r'FF=(\d+)'),('FF_pct',r'FF=\d+ \(([\d.]+)%'),('LUT',r'LUT=(\d+)'),('LUT_pct',r'LUT=\d+ \(([\d.]+)%')]:
   m=re.search(pattern,text); assert m
   a,b=m.span(1); loc=[i for i,(x,y) in enumerate(offsets) if x<b and y>a]
   targets.append(dict(field=name,text=m.group(1),char_span=[a,b],token_indices=loc,token_ids=[ids[i] for i in loc],decoded=tok.decode([ids[i] for i in loc])))
  learned=[]
  for t in c['special_tokens']:
   tid=tok.convert_tokens_to_ids(t); positions=[i for i,x in enumerate(ids) if x==tid]
   if positions: learned.append(dict(token=t,token_id=tid,positions=positions))
  rows.append(dict(request_id=r['context_id'],prompt_sha256=sha(out/'prompts'/f"{r['context_id']}.txt"),prompt_token_count=len(ids),resource_numbers=targets,learned_special_tokens=learned,selected_device=r['device'],selected_clock=r['clock_period']))
 write_json(out/'input_tokenization_audit.json',dict(tokenizer_json_sha256=sha(ADAPTER/'tokenizer.json'),request_count=len(rows),records=rows))
 print('tokenization audited',len(rows),'prompts')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT);run(p.parse_args().out)
