"""Seal fresh V4-only GAN predictions for the existing Coroni synthesis runner."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
from experiments.gnosis_stage1_ab import gan_tcad_synthesis as s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('predictions','prepared','root'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    inf=json.loads((a.predictions/'manifest.json').read_text())
    rows=json.loads((a.predictions/'predicted_directives.json').read_text())
    assert inf['status']=='INFERENCE_COMPLETED' and len(rows)==18
    assert inf['loaded_adapters']==['mailohls_base'] and inf['all_saved_tensors_verified']
    assert inf['film_enabled'] is False and inf['residual_adapter_loaded'] is False
    assert inf['checkpoint_sha256']=='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'
    sealed=json.loads((a.prepared/'manifest.json').read_text())
    assert sealed==inf['prepared_manifest']
    assert s.sha(a.prepared/'requests.jsonl')==sealed['requests_sha256']
    sourcepath=a.prepared/'sources/GAN/GAN_placeholders.cpp'
    assert s.sha(sourcepath)==sealed['source_sha256']
    template=sourcepath.read_text();baseline=s.strip_pragmas(template)
    assert 'auto{' not in baseline and s.tokens(baseline)==s.tokens(template)
    a.root.mkdir(parents=True,exist_ok=False)
    headers={}
    for name,digest in sealed['header_sha256'].items():
        path=a.prepared/'sources/GAN'/name
        assert s.sha(path)==digest
        if name not in ('ap_fixed.h','ap_int.h'):headers[name]=path
    jobs,by_signature,requests=[],{},[]

    def add(part,clock,role,source):
        sig=hashlib.sha256(json.dumps([part,clock,source,s.SETTINGS],sort_keys=True).encode()).hexdigest()
        if sig in by_signature:return by_signature[sig]
        device='zcu104' if part.startswith('xczu') else 'u200'
        job_id='{}_{}ns_{}_{}'.format(device,str(clock).replace('.','p'),role,sig[:10])
        d=a.root/'jobs'/job_id;d.mkdir(parents=True)
        (d/'source.cpp').write_text(source)
        for name,path in headers.items():shutil.copyfile(path,d/name)
        tcl='\n'.join(['open_project -reset hls_project','set_top '+s.TOP,
            'add_files -cflags {-I/opt/Xilinx/Vitis_HLS/2021.1/include} {source.cpp}',
            'open_solution -reset solution1 -flow_target vivado','set_part {'+part+'}',
            'create_clock -period {} -name default'.format(clock),*s.SETTINGS,'csynth_design','exit',''])
        (d/'run_hls.tcl').write_text(tcl)
        jobs.append(dict(job_id=job_id,part=part,clock_ns=clock,role=role,signature=sig,
                         files={f.name:s.sha(f) for f in d.iterdir()}))
        by_signature[sig]=job_id
        return job_id

    for row in rows:
        assert row['film_enabled'] is False and row['source_domain_complete'] and len(row['directives'])==25
        assert row['checkpoint_sha256']==inf['checkpoint_sha256']
        part,clock=row['device'],row['clock_ns']
        requests.append(dict(context_id=row['context_id'],part=part,clock_ns=clock,budget_pct=row['budget_pct'],
            baseline_job=add(part,clock,'baseline',baseline),
            predicted_job=add(part,clock,'predicted',s.materialize(template,row['directives'])),directives=row['directives']))
    expected={(part,c,b) for part in ('xczu7ev-ffvc1156-2-e','xcu200-fsgd2104-2-e')
              for c in (3.33,5.,10.) for b in (20,50,100)}
    assert len({r['context_id'] for r in requests})==18
    assert {(r['part'],r['clock_ns'],r['budget_pct']) for r in requests}==expected
    assert sum(j['role']=='baseline' for j in jobs)==6
    shutil.copyfile(s.__file__,a.root/'gan_tcad_synthesis.py')
    manifest=dict(status='PREPARED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_label='V4 Stage-1 parent only; no FiLM or residual adapter',
        source_sha256=s.sha(sourcepath),predictions_sha256=s.sha(a.predictions/'predicted_directives.json'),
        inference_manifest_sha256=s.sha(a.predictions/'manifest.json'),
        inference_identity={key:inf[key] for key in ('checkpoint','checkpoint_sha256','checkpoint_step','film_enabled','loaded_adapters')},
        headers={name:s.sha(path) for name,path in headers.items()},settings=s.SETTINGS,top=s.TOP,tool=s.TOOL,
        requests=requests,jobs=jobs,runner_sha256=s.sha(a.root/'gan_tcad_synthesis.py'),
        preparer_sha256=s.sha(Path(__file__)),
        pragma_free_baseline='all pragma lines removed from top source; identical application headers and HLS settings',
        vendor_header_policy='analysis AP stub excluded; genuine Vitis 2021.1 AP headers only',
        area_definition='100 * mean(resource count / full device capacity) across BRAM18K,DSP,FF,LUT',
        speedup_definition='baseline worst-case cycles / predicted worst-case cycles, matched part and requested period',
        functional_simulation='NOT RUN',post_route_timing='NOT RUN')
    s.dump(a.root/'manifest.json',manifest)
    print('Prepared',len(jobs)-6,'distinct predicted syntheses and 6 fresh pragma-free baselines.')


if __name__=='__main__':main()
