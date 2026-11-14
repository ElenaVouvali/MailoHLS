"""Fresh V4-only Stage-1 inference on the exact GAN TCAD 18-request grid."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import warnings
import zipfile

import torch
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
from safetensors.torch import load_file
from experiments.gnosis_stage1_ab import kalman_contexts as k


def prepare(a):
    sealed = json.loads((a.prepared/'manifest.json').read_text())
    with zipfile.ZipFile(a.draft) as archive:
        assert k.sha(a.draft) == sealed['draft_zip_sha256']
        for name, digest in [('sections/05__Detailed_Design.tex',sealed['section_sha256']),
                ('figures/02__Methodology/target_conditioning_heatmap.pdf',sealed['figure_sha256'])]:
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    assert k.sha(a.prepared/'requests.jsonl') == sealed['requests_sha256']
    sources = a.prepared/'sources'
    for name, digest in [('GAN_placeholders.cpp',sealed['source_sha256']),('kernel_info.txt',sealed['metadata_sha256']),
                         *sealed['header_sha256'].items()]:
        assert k.sha(sources/'GAN'/name) == digest
    rows = [json.loads(line) for line in (a.prepared/'requests.jsonl').read_text().splitlines()]
    expected = {(d,c,b) for d in ('xczu7ev-ffvc1156-2-e','xcu200-fsgd2104-2-e')
                for c in (3.33,5.,10.) for b in (.2,.5,1.)}
    assert len(rows)==18 and {(r['device'],r['clock_period'],r['resource_budget_frac']) for r in rows}==expected
    assert len({r['context_id'] for r in rows})==18
    contract = json.loads((a.checkpoint/'training_contract.json').read_text())
    assert contract['stage']=='stage1' and contract['objective']=='PARETO_LATENCY'
    assert k.sha(a.checkpoint/'adapter_model.safetensors')==k.PARENT_SHA==sealed['checkpoint_sha256']
    assert k.sha(a.checkpoint/'training_contract.json')==sealed['contract_sha256']
    assert k.sha(a.checkpoint/'directive_domain_registry.json')==contract['directive_domain_registry_sha256']==sealed['registry_sha256']
    source = k.domains.prepare_source_template('GAN',sources/'GAN/GAN_placeholders.cpp',sources)
    k.prod.TARGET_CFG.device_mode=contract['device_mode']
    registry=k.inf.load_directive_domain_registry(str(a.checkpoint/'directive_domain_registry.json'))
    k.inf.add_missing_source_directive_domains(registry,[type('Case',(),dict(kernel_name='GAN',source_text=source))()],str(sources))
    tok=k.inf.load_checkpoint_tokenizer(str(a.checkpoint),contract)
    prior, lines={},[]
    for _,lhs in k.prod.extract_ordered_lhs_plan(source):
        _,choices=k.prod.constrained_rhs_candidates('GAN',lhs,registry,prior)
        prior[lhs.upper()]=choices[0];lines.append(f'{lhs} = {choices[0]}')
    pack=k.prod.build_deterministic_rhs_pack(source,'\n'.join(lines),tok,directive_domain_registry=registry,kernel_name='GAN')
    lengths,prompts={},{}
    for row in rows:
        assert row['input']==source and row['obj_mode']=='PARETO_LATENCY' and row['frequency_mode']=='specified'
        for resource,field in k.mc.AVAIL_FIELD_BY_RESOURCE.items():
            assert row[field]==round(k.mc.DEVICE_RESOURCES[row['device']][resource]*row['resource_budget_frac'])
        prompt=k.prompt_tokens(tok,row)
        lengths[row['context_id']]=dict(prompt=len(prompt),fixture_complete=len(prompt)+len(pack.input_ids))
        assert lengths[row['context_id']]['fixture_complete']<=a.max_length
        prompts[row['context_id']]=hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
    manifest=dict(status='CPU_PREFLIGHT_PASSED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_label='V4 Stage-1 parent only',checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=k.PARENT_SHA,
        parent_sha256=k.PARENT_SHA,checkpoint_step=None,film_enabled=False,film_sha256=None,lora_sha256=k.PARENT_SHA,
        residual_adapter_loaded=False,structural_memory_loaded=False,objective='PARETO_LATENCY',
        requests=18,directive_field_count=25,source_code=source,contexts=rows,token_lengths=lengths,prompt_sha256=prompts,
        prepared_manifest=sealed,max_length=a.max_length,candidate_batch_size=a.batch_size,
        decoder='same as FiLM experiment: production greedy full RHS/newline continuation, mean score; no cache',
        implementation_sha256=k.sha(Path(__file__)),helper_sha256=k.sha(Path(k.__file__)),
        inference_sha256=k.sha(Path(k.prod.__file__)),physical_qor='NOT MEASURED')
    state=a.checkpoint/'checkpoint-315/trainer_state.json'
    if state.exists() and k.sha(a.checkpoint/'checkpoint-315/adapter_model.safetensors')==k.PARENT_SHA:
        manifest['checkpoint_step']=json.loads(state.read_text())['global_step']
    return manifest,rows,contract,registry,tok


class ParentScorer:
    def __init__(self,model,max_length):
        self.model,self.max_length=model,max_length
    def __getattr__(self,name):
        return getattr(self.model,name)
    def bind(self,row):
        self.row=row;self.calls=0;self.started=time.monotonic()
    def __call__(self,**kwargs):
        assert kwargs['input_ids'].shape[1]<=self.max_length,'No source truncation permitted'
        with torch.autocast('cuda',dtype=torch.float16):
            result=self.model(**kwargs,use_cache=False)
        self.calls+=1
        if self.calls==1 or self.calls%5==0:
            print(json.dumps(dict(context=self.row['context_id'],batches=self.calls,
                seconds=round(time.monotonic()-self.started,2))),flush=True)
        return result


def report(out,rows):
    text=['# GAN: V4 Stage-1 parent only, FiLM absent','',
        'Arrays L1–L3: type/factor/dimension. Loops L4–L11: pipeline/unroll.',
        'L1=xbuf; L2=layer_1_out; L3=layer_2_out. Complete partition uses factor 0; unroll 0 means disabled.','',
        '|Device|Clock ns|Budget %|L1|L2|L3|L4|L5|L6|L7|L8|L9|L10|L11|','|'+'---|'*14]
    for row in rows:
        values=['ZCU104' if row['device'].startswith('xczu') else 'U200',row['clock_ns'],row['budget_pct']]
        for label in range(1,12):
            kinds=('ARRAY_T','ARRAY_F','ARRAY_D') if label<=3 else ('PIPE','UNROLL')
            values.append('/'.join(row['directives'][f'AUTO{{_{kind}_L{label}}}'] for kind in kinds))
        text.append('|'+'|'.join(map(str,values))+'|')
    text.extend(['','Full predictions follow. These are proposals; synthesis results are reported separately.',''])
    for i,row in enumerate(rows,1):text.extend([f'## Request {i}: {row["context_id"]}','','```text',row['prediction'],'```',''])
    (out/'DIRECTIVES.md').write_text('\n'.join(text))
    with (out/'directives.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['context_id','device','clock_ns','budget_pct',*rows[0]['directives']],lineterminator='\n')
        writer.writeheader()
        for row in rows:writer.writerow({**{key:row[key] for key in ('context_id','device','clock_ns','budget_pct')},**row['directives']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','prepared','draft','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--max-length',type=int,default=6144)
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest,rows,contract,registry,tok=prepare(a);k.write_json(a.out/'manifest.json',manifest)
    if a.preflight_only:print('CPU_PREFLIGHT_PASSED',flush=True);return
    uid=os.environ['CUDA_VISIBLE_DEVICES']
    assert uid=='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'
    gpu=subprocess.check_output(['nvidia-smi','-i',uid,'--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True)
    assert 'V100S' in gpu and int(gpu.split(',')[-1])>28000
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.split(',')[0].strip()==uid for line in apps.splitlines())
    torch.manual_seed(123)
    base=k.inf.load_base_model(contract['model'],revision=contract['model_revision']);base.resize_token_embeddings(len(tok))
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    with warnings.catch_warnings(record=True) as caught:
        model=PeftModel.from_pretrained(base,str(a.checkpoint),adapter_name='mailohls_base',is_trainable=False)
    assert not any('missing adapter keys' in str(w.message).lower() for w in caught)
    k.da.isolate_parent_token_serialization(model)
    assert list(model.peft_config)==['mailohls_base']
    # Verify every saved adapter tensor, including the trained token rows, loaded exactly.
    saved=load_file(str(a.checkpoint/'adapter_model.safetensors'))
    loaded=get_peft_model_state_dict(model,adapter_name='mailohls_base')
    assert set(saved)==set(loaded)
    for key,value in saved.items():torch.testing.assert_close(loaded[key].detach().cpu(),value,rtol=0,atol=0,check_dtype=True)
    del saved,loaded
    model.requires_grad_(False);model.eval()
    scorer=ParentScorer(model,a.max_length);results=[];started=time.monotonic()
    for index,row in enumerate(rows,1):
        scorer.bind(row);prompt=k.prompt_tokens(tok,row);tick=time.monotonic()
        print('BEGIN',index,row['context_id'],flush=True)
        with torch.inference_mode():
            prediction,trace=k.prod.constrained_decode_rhs_by_candidate_scoring(model=scorer,tok=tok,prompt_ids=prompt,
                source_text=row['input'],kernel_name='GAN',directive_domain_registry=registry,
                candidate_batch_size=a.batch_size,return_score_trace=True)
        assignment=k.validate_assignment(prediction,row['input'],registry,'GAN');assert len(assignment)==25
        result=dict(context_id=row['context_id'],device=row['device'],clock_ns=row['clock_period'],budget_pct=round(100*row['resource_budget_frac']),
            directives=assignment,prediction=prediction,candidate_scores=trace,film_enabled=False,source_domain_complete=True,
            checkpoint_sha256=k.PARENT_SHA,prompt_sha256=manifest['prompt_sha256'][row['context_id']],seconds=time.monotonic()-tick)
        k.write_json(a.out/f'request_{index:02d}.json',result);results.append(result)
        k.write_json(a.out/'predicted_directives.json',results);report(a.out,results)
        print('COMPLETE',index,row['context_id'],round(result['seconds'],1),flush=True)
    assert len(results)==18 and k.sha(a.checkpoint/'adapter_model.safetensors')==k.PARENT_SHA
    manifest.update(status='INFERENCE_COMPLETED',completed_requests=18,gpu=gpu.strip(),seconds=time.monotonic()-started,
        checkpoint_unchanged=True,loaded_adapters=list(model.peft_config),all_saved_tensors_verified=True,
        precision='FP16 autocast, NF4 frozen base; same backend as FiLM run')
    k.write_json(a.out/'manifest.json',manifest);print('ALL 18 REQUESTS COMPLETE',flush=True)


if __name__=='__main__':main()
