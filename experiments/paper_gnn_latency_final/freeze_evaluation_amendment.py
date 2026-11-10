import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
from types import SimpleNamespace
from LLM_branch.common import directive_domains as domains
from experiments.paper_gnn.data import ROOT,sha,fingerprint
from experiments.paper_gnn.train import write_json
from experiments.paper_gnn_latency_final.evaluation_cohort import include_radix_ignoring_l6

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'evaluation_amendment';out.mkdir()
    split=json.loads((ROOT/'mailohls_runs/mailohls_final_family_split_s123.json').read_text());data=SimpleNamespace(split=split,actions={'machsuite-sort-radix':domains.tcad.load_source_actions('machsuite-sort-radix',ROOT/'Data/ApplicationDataset')},groups=defaultdict(list),identity={'area_floor':split['effective_area_floor']})
    amendment=include_radix_ignoring_l6(data,a.root/'cohort/corrected_cohort_v1')
    pool=[json.loads(line) for line in (a.root/'cohort/candidate_pool.jsonl').read_text().splitlines() if json.loads(line)['split']=='val']
    for records in data.groups.values():
        for r in records:pool.append({k:r[k] for k in ('split','kernel','family','device','clock','design_id','row_index','ignored_actions')})
    pool.sort(key=lambda r:(r['kernel'],r['device'],r['clock'],r['design_id']))
    with (out/'candidate_pool.jsonl').open('w') as f:
        for r in pool:f.write(json.dumps(r,sort_keys=True)+'\n')
    amendment.update(candidate_pool_sha256=sha(out/'candidate_pool.jsonl'),candidate_identity_sha256=fingerprint(sorted((r['kernel'],r['device'],r['clock'],r['design_id']) for r in pool)),unique_observed_configurations=len(pool),kernel_counts=dict(Counter(r['kernel'] for r in pool)),families=sorted({r['family'] for r in pool}),B_started_at_manifest_creation=(a.root/'B/last.pt').exists())
    write_json(amendment,out/'manifest.json');print(json.dumps(amendment,indent=2))
