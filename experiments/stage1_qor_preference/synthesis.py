"""Seal at most four validation-only jobs using the existing authenticated runner."""
from __future__ import annotations
import shutil
from LLM_branch.common import mailohls_contract
from experiments.refinement.synthesis import materialize
from experiments.stage1_quality_sft.prepare_paired_synthesis import _normalized
from experiments.stage1_quality_sft.core import parse
from .common import *


def prepare():
    protocol=read(OUT/'synthesis_protocol.json');paired=OUT/'validation54';dest=OUT/'synthesis_bundle'
    if dest.exists():raise FileExistsError(dest)
    frozen={r['request_id']:r for r in rows(paired/'frozen_validation_requests.jsonl')}
    predictions={arm:_normalized(paired/'predictions'/f'{arm}.jsonl') for arm in ['baseline_v4','refined_dpo']}
    old=PARENT_WORKTREE/'artifacts/stage1_quality_sft/v4_paired_20261105/synthesis_bundle_v2'
    old_manifest=read(old/'manifest.json')
    assert sha(old/'run_batch.py')==old_manifest['runner_sha256']
    dest.mkdir();(dest/'jobs').mkdir();shutil.copyfile(old/'run_batch.py',dest/'run_batch.py')
    jobs={}
    for selected in protocol['requests']:
        r=frozen[selected['request_id']]
        assert all(r[k]==v for k,v in selected.items())
        source=ROOT/r['source_file'];kernel=source.parent
        top=(kernel/'kernel_info.txt').read_text().splitlines()[0].strip()
        for arm in predictions:
            pred=predictions[arm][r['request_id']];values=parse(pred)
            configured=materialize(r['input'],'\n'.join(f'{k} = {v}' for k,v in values.items()))
            source_hash=__import__('hashlib').sha256(configured.encode()).hexdigest()
            key=(r['source_key'],r['device'],r['clock_period'],source_hash)
            attribution=dict(arm=arm,request_id=r['request_id'],budget_id=r['resource_budget_id'],
                budget_fractions=r['quality_budget_fractions'],prediction=pred)
            if key in jobs:jobs[key]['attributions'].append(attribution);continue
            jid=f'qor_{len(jobs)+1:02d}_{source_hash[:12]}';folder=dest/'jobs'/jid;folder.mkdir()
            (folder/source.name).write_text(configured);headers={}
            for p in sorted(kernel.iterdir()):
                if not p.is_file() or p.name in [source.name,'kernel_info.txt','ap_int.h','ap_fixed.h']:continue
                shutil.copyfile(p,folder/p.name)
                if p.suffix.lower() in ['.h','.hpp']:headers[p.name]=sha(p)
            caps=mailohls_contract.DEVICE_RESOURCES[r['device']]
            budgets={res:r[field] for res,field in mailohls_contract.AVAIL_FIELD_BY_RESOURCE.items()}
            jobs[key]=dict(job_id=jid,kernel=r['kernel_name'],arm=arm,decoder='legacy',request_number=len(jobs)+1,
                request_id=r['request_id'],source=source.name,top=top,part=r['device'],clock_ns=r['clock_period'],
                capacities=caps,budget=budgets,materialized_source_sha256=source_hash,original_source_sha256=sha(source),
                source_template_sha256=__import__('hashlib').sha256(r['input'].encode()).hexdigest(),
                headers=headers,attributions=[attribution])
    assert 0<len(jobs)<=protocol['max_physical_jobs']==4
    manifest=dict(schema='stage1-qor-dpo-validation-synthesis-v1',jobs=list(jobs.values()),
        settings=old_manifest['settings'],runner_sha256=sha(dest/'run_batch.py'),
        selection_protocol_sha256=sha(OUT/'synthesis_protocol.json'),timeout_seconds=300,
        total_tool_time_cap_seconds=1200,qor_policy='successful authenticated XML only; unmatched/timeouts/failures unknown',
        validation_only=True,test_contexts=False)
    write(dest/'manifest.json',manifest)
    response=read(OUT/'synthesis_reuse_response.json')
    assert response['manifest_sha256']==sha(old/'manifest.json')
    assert response['provenance']['runner_sha256']==sha(old/'run_batch.py')
    matches=[]
    for record in response['completed']:
        result=record['result'];receipt=record['receipt'];j=receipt['job']
        assert record['source_sha256']==j['materialized_source_sha256']
        assert record['tcl_sha256']==receipt['tcl_sha256']
        assert record['receipt_sha256']==result['receipt_sha256']
        if result['status']!='success' or not record['xml']:continue
        raw=record['xml'].encode();assert __import__('hashlib').sha256(raw).hexdigest()==result['report_sha256']
        for job in jobs.values():
            if job['part']==j['part'] and job['clock_ns']==j['clock_ns'] and job['materialized_source_sha256']==j['materialized_source_sha256']:
                matches.append(dict(new_job_id=job['job_id'],existing_job_id=j['job_id'],metrics=result['metrics'],report_sha256=result['report_sha256']))
    write(OUT/'synthesis_status.json',dict(status='SEALED_PENDING_IDLE_HOST',sealed_jobs=len(jobs),
        existing_completed_receipts=len(response['completed']),exact_successful_reuse=matches,new_jobs_launched=0,
        reason='existing Coroni HLS batch was active; run_synthesis.sh checks again and refuses concurrent launch'))
    print('Sealed validation HLS jobs:',len(jobs),'exact completed reuse:',len(matches))


if __name__=='__main__':prepare()
