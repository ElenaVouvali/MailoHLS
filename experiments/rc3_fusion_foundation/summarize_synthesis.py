"""Compute budget-view feasibility/regret from authenticated batch receipts."""
import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bundle',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); bundle=Path(a.bundle); results=Path(a.results)
    manifest=json.loads((bundle/'manifest.json').read_text())
    report_path=results/'report.json'
    if not report_path.exists():
        Path(a.out).write_text(json.dumps({'status':'pending','reason':'runner report.json not found','jobs':[]},indent=2)+'\n'); return
    report=json.loads(report_path.read_text()); by_id={r['job_id']:r for r in report['results']}
    rows=[]
    for job in manifest['jobs']:
        result=by_id.get(job['job_id']); metrics=(result or {}).get('metrics') or {}
        resources=metrics.get('resources')
        loop_ii=[]
        report_dir=results/job['job_id']/'hls_project/solution1/syn/report'
        if report_dir.exists():
            for path in sorted(report_dir.glob('*Pipeline*_csynth.xml')):
                try:
                    root=ET.parse(path).getroot()
                    ii=root.findtext('.//PipelineII')
                    imin=root.findtext('.//Interval-min'); imax=root.findtext('.//Interval-max')
                    if ii is not None: loop_ii.append({'report':path.name,'pipeline_ii':int(ii),
                        'interval_min_cycles':int(imin) if imin and imin.isdigit() else None,
                        'interval_max_cycles':int(imax) if imax and imax.isdigit() else None})
                except (ET.ParseError, ValueError):
                    loop_ii.append({'report':path.name,'status':'unparseable'})
        for view in job['budget_views_for_posthoc_feasibility']:
            budget=view['resources']
            feasible=(None if not resources or not result or result.get('status')!='success' else
                      all(resources.get(k) is not None and resources[k] <= budget[k] for k in budget))
            rows.append({'job_id':job['job_id'],'candidate':job['arm'],'request':job['request_id'],
                'part':job['part'],'clock_ns':job['clock_ns'],'budget_fraction':view['fraction'],
                'budget':budget,'status':(result or {}).get('status','missing'),
                'latency_cycles':metrics.get('latency_cycles'),'latency_at_requested_clock_ns':metrics.get('latency_at_requested_clock_ns'),
                'resources':resources,'resource_feasible':feasible,'estimated_clock_ns':metrics.get('estimated_clock_ns'),
                'estimated_clock_meets_request':metrics.get('hls_estimated_clock_meets_request'),
                'estimated_latency_ns':(metrics.get('latency_cycles')*metrics.get('estimated_clock_ns')
                    if metrics.get('latency_cycles') is not None and metrics.get('estimated_clock_ns') is not None else None),
                'timing_evidence':'HLS estimate, not post-route','functional_validation':False,
                'loop_pipeline_reports':loop_ii,'budget_view_is_duplicate_synthesis':False,
                'physical_job_shared_across_budget_views':True})
    for row in rows:
        peers=[x for x in rows if x['request']==row['request'] and x['budget_fraction']==row['budget_fraction']
               and x['resource_feasible'] is True and x['status']=='success' and x['latency_cycles'] is not None]
        best=min((x['latency_cycles'] for x in peers),default=None)
        row['best_feasible_pool_latency_cycles']=best
        row['latency_regret_cycles']=None if best is None or row['resource_feasible'] is not True or row['latency_cycles'] is None else row['latency_cycles']-best
        row['latency_regret_fraction']=None if best in (None,0) or row['latency_regret_cycles'] is None else row['latency_regret_cycles']/best
        timed=[x for x in peers if x['estimated_clock_meets_request'] is True]
        best_clock_cycles=min((x['latency_cycles'] for x in timed),default=None)
        best_estimated_ns=min((x['estimated_latency_ns'] for x in timed if x['estimated_latency_ns'] is not None),default=None)
        row['best_resource_and_clock_feasible_pool_latency_cycles']=best_clock_cycles
        row['latency_regret_cycles_clock_feasible']=None if best_clock_cycles is None or row['resource_feasible'] is not True or row['estimated_clock_meets_request'] is not True or row['latency_cycles'] is None else row['latency_cycles']-best_clock_cycles
        row['best_resource_and_clock_feasible_estimated_latency_ns']=best_estimated_ns
        row['estimated_latency_regret_ns']=None if best_estimated_ns is None or row['resource_feasible'] is not True or row['estimated_clock_meets_request'] is not True or row['estimated_latency_ns'] is None else row['estimated_latency_ns']-best_estimated_ns
    payload={'schema':'salt-hls-gemv-synthesis-summary-v1','status':'measured' if any(x['status']=='success' for x in rows) else 'pending_or_failed',
        'manifest_sha256':__import__('hashlib').sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
        'small_pool_only':True,'global_optimum_claim':False,'rows':rows}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
