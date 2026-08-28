"""Audit whether a DPO pair cache supports the single-edit experiment."""
import argparse, json
from collections import Counter

def main():
 p=argparse.ArgumentParser(); p.add_argument('--pairs',required=True); p.add_argument('--min_train',type=int,default=300); p.add_argument('--min_kernels',type=int,default=20); a=p.parse_args()
 rows=[json.loads(x) for x in open(a.pairs,encoding='utf-8') if x.strip()]
 one=[r for r in rows if int(r.get('directive_diff_count',-1))==1]
 kernels={r.get('kernel_name') for r in one}; families={r.get('family') for r in one}
 report={'pairs':len(one),'kernels':len(kernels),'families':len(families),'sufficient':len(one)>=a.min_train and len(kernels)>=a.min_kernels,'diff_histogram':dict(Counter(int(r.get('directive_diff_count',-1)) for r in rows))}
 print(json.dumps(report,indent=2)); raise SystemExit(0 if report['sufficient'] else 2)
if __name__=='__main__': main()
