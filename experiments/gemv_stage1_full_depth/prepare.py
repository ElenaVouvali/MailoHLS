"""Freeze the 18-point GEMV target grid and three compiler-checked source controls."""
from __future__ import annotations
import argparse, ctypes, json, os, re, subprocess
from pathlib import Path
import numpy as np
from LLM_branch.common import directive_domains, source_tool_policy
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from .common import ADAPTER, DEFAULT_OUT, EXPECTED_ADAPTER, EXPECTED_CONTRACT, GPU, HISTORY, PACKAGE, ROOT, contract, digest_object, prompt_parts, read_json, sha, write_json

GNN = Path('/tmp/MailoHLS-gnn-refinement')
AUDIT = Path('/tmp/MailoHLS-stage1-mechanistic-audit/artifacts/stage1_mechanistic_audit/v1_20261106/sources')
COMPILER = '/home/elvouvali/tools/Polygeist/build-mailohls-assertions/bin/cgeist'
MLIR_PYTHON = '/home/elvouvali/.mlir-python311/bin/python'
BASE = 'xczu7ev-ffvc1156-2-e'
OTHER = 'xcu200-fsgd2104-2-e'
FOCUS = ['auto{_ARRAY_T_L1}', 'auto{_ARRAY_F_L1}', 'auto{_ARRAY_D_L1}', 'auto{_ARRAY_F_L2}', 'auto{_PIPE_L4}', 'auto{_UNROLL_L4}', 'auto{_PIPE_L5}', 'auto{_UNROLL_L5}']

def auth():
    c=read_json(ADAPTER/'training_contract.json')
    assert sha(ADAPTER/'adapter_model.safetensors')==EXPECTED_ADAPTER
    assert sha(ADAPTER/'training_contract.json')==EXPECTED_CONTRACT
    assert c['model']=='deepseek-ai/deepseek-coder-6.7b-base' and c['model_revision']=='ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912'
    tok=inf.load_checkpoint_tokenizer(str(ADAPTER),c)
    return c,tok

def diagnostic_source(kind='reduction', rename=False):
    update={'reduction':'acc += A[i][j] * x[j];','independent':'float value = A[i][j] * x[j];'}[kind]
    value='value' if kind=='independent' else 'acc'
    src=f'''#define N 32\n#define M 32\nvoid gemv(const float inputA[N][M], const float inputx[M], float output[N*M+N]) {{\n    /*L1:*/ float A[N][M];\n#pragma HLS array_partition variable=A type=auto{{_ARRAY_T_L1}} factor=auto{{_ARRAY_F_L1}} dim=auto{{_ARRAY_D_L1}}\n    /*L2:*/ float x[M];\n#pragma HLS array_partition variable=x type=auto{{_ARRAY_T_L2}} factor=auto{{_ARRAY_F_L2}} dim=auto{{_ARRAY_D_L2}}\n    /*L3:*/ float y[N];\n#pragma HLS array_partition variable=y type=auto{{_ARRAY_T_L3}} factor=auto{{_ARRAY_F_L3}} dim=auto{{_ARRAY_D_L3}}\n    for (int r=0; r<N; ++r) for (int c=0; c<M; ++c) A[r][c]=inputA[r][c];\n    for (int c=0; c<M; ++c) x[c]=inputx[c];\n    /*L4:*/ for (int i=0; i<N; ++i) {{\n#pragma HLS pipeline II=auto{{_PIPE_L4}}\n#pragma HLS unroll factor=auto{{_UNROLL_L4}}\n        float acc=0;\n        /*L5:*/ for (int j=0; j<M; ++j) {{\n#pragma HLS pipeline II=auto{{_PIPE_L5}}\n#pragma HLS unroll factor=auto{{_UNROLL_L5}}\n            {update}\n            output[i*M+j]={value};\n        }}\n        y[i]={'0' if kind=='independent' else 'acc'};\n        output[N*M+i]=y[i];\n    }}\n}}\n'''
    if rename:
        for a,b in [('A','matrix'),('x','vector'),('y','result'),('acc','total'),('i','row'),('j','col')]:
            src=re.sub(r'\b'+a+r'\b',b,src)
    return src

def compile_variants(out):
    srcroot=PACKAGE/'sources'; records=[]
    specs=[('reduce32','reduction',False),('independent32','independent',False),('reduce32_rename','reduction',True)]
    env=dict(os.environ,PYTHONHASHSEED='0',PYTHONPATH='/home/elvouvali/tools/Polygeist/build-mailohls-assertions/tools/mlir/python_packages/mlir_core:'+str(GNN))
    for name,kind,rename in specs:
        d=srcroot/name; d.mkdir(parents=True,exist_ok=True)
        source=diagnostic_source(kind,rename); (d/'gemv.c').write_text(source)
        names=('matrix','vector','result') if rename else ('A','x','y')
        meta=f'gemv\nL1,array,{names[0]},1,32,2,32\nL2,array,{names[1]},1,32\nL3,array,{names[2]},1,32\nL4,loop,32\nL5,loop,32\n'
        (d/'kernel_info.txt').write_text(meta)
        argv=[MLIR_PYTHON,'-m','GNN_branch.mlir_graph_v2','--revision','2.2','--semantic-revision','rc3','--source',str(d/'gemv.c'),'--metadata',str(d/'kernel_info.txt'),'--output',str(d/'graph.json'),'--mlir-output',str(d/'gemv.mlir'),'--cgeist',COMPILER]
        p=subprocess.run(argv,cwd=GNN,env=env,capture_output=True,text=True)
        (d/'compiler.log').write_text(p.stdout+p.stderr)
        if p.returncode: raise RuntimeError(f'RC3 compiler failed for {name}: {(p.stdout+p.stderr)[-2000:]}')
        graph=read_json(d/'graph.json')
        actions=directive_domains.load_source_actions('gemv',d)
        assert [a for a in actions]==[f'L{i}' for i in range(1,6)]
        # Compile and execute a deterministic numerical check against a float32 oracle.
        lib=d/'numeric.so'; subprocess.run(['gcc','-shared','-fPIC','-O0',str(d/'gemv.c'),'-o',str(lib)],check=True,capture_output=True,text=True)
        fn=ctypes.CDLL(str(lib)).gemv; fn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
        a=np.arange(1024,dtype=np.float32).reshape(32,32)/37; x=np.arange(32,dtype=np.float32)/13; y=np.zeros(1056,np.float32)
        fn(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if kind=='reduction': expected=np.array([np.sum(a[i]*x,dtype=np.float32) for i in range(32)],np.float32)
        else: expected=np.zeros(32,np.float32)
        error=float(np.max(np.abs(y[1024:]-expected)))
        assert error<2e-4
        rec=dict(name=name,kind=kind,syntax_control_of='reduce32' if rename else None,source_sha256=sha(d/'gemv.c'),metadata_sha256=sha(d/'kernel_info.txt'),mlir_sha256=sha(d/'gemv.mlir'),graph_sha256=sha(d/'graph.json'),graph_schema=graph.get('schema'),compiler_returncode=p.returncode,compiler_argv=argv,action_mapping=['L1','L2','L3','L4','L5'],numerical_max_abs_error=error,source_semantics='reduction' if kind=='reduction' else 'independent per-iteration product; different algorithm and output behavior')
        records.append(rec)
    write_json(out/'source_variant_manifest.json',records)
    return records

def make_row(context,device,clock,fraction,source,available=None):
    capacities=contract.DEVICE_RESOURCES[device]
    row=dict(context_id=context,kernel_name='gemv',input=source,objective='PARETO_LATENCY',obj_mode='PARETO_LATENCY',device=device,clock_period=clock,frequency_mode='specified',available_clock_periods=list(available or contract.supported_clock_periods(device)),resource_budget_id=f'uniform-{round(fraction*100)}')
    row.update({field:int(round(capacities[r]*fraction)) for r,field in contract.AVAIL_FIELD_BY_RESOURCE.items()})
    return row

def prepare(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    c,tok=auth()
    hist=read_json(HISTORY)
    original=Path('/tmp/MailoHLS-gnn-refinement/paper_figures/gemv_rc3_20261105/inputs')
    for name,expected in [('gemv.c',hist['source']['sha256']),('kernel_info.txt',hist['source']['metadata_sha256'])]:
        assert sha(PACKAGE/'inputs'/name)==expected==sha(original/name)
    source=(PACKAGE/'inputs/gemv.c').read_text()
    variants=compile_variants(out)
    rows=[]
    for dev in [BASE,OTHER]:
        for clock in [10.,5.,3.33]:
            for frac in [1.,.5,.25]:
                devname='ZCU104' if dev==BASE else 'XCU200'; clk=str(clock).replace('.','P')
                rows.append(make_row(f'GRID_{devname}_{clk}NS_{int(frac*100)}PCT',dev,clock,frac,source))
    # Preserve the old below-in-distribution request with the exact historical target fields.
    old=json.loads((Path('/tmp/MailoHLS-stage1-refinement/artifacts/gemv_framework_deep_dive/v4_20261106/requests.jsonl').read_text().splitlines()[1]))
    old['context_id']='HIST_ZCU104_10NS_20PCT'; rows.append(old)
    subset=[]
    for dev in [BASE,OTHER]:
        for clock in [10.,3.33]:
            for frac in [1.,.25]:
                devname='ZCU104' if dev==BASE else 'XCU200';clk=str(clock).replace('.','P')
                subset.append((dev,clock,frac,f'SUB_{devname}_{clk}NS_{int(frac*100)}PCT'))
    for vr in variants:
        vsrc=(PACKAGE/'sources'/vr['name']/'gemv.c').read_text()
        for dev,clock,frac,label in subset:
            rows.append(make_row(f"{vr['name']}__{label}",dev,clock,frac,vsrc))
    request_path=out/'requests.jsonl'; request_path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    reg=inf.load_directive_domain_registry(str(ADAPTER/'directive_domain_registry.json'))
    inf.add_missing_source_directive_domains(reg,[inf.InferenceCase('gemv',source,'PARETO_LATENCY',1.,0.,rows[0],None)],str(PACKAGE/'inputs'))
    write_json(out/'effective_gemv_domains.json',reg['gemv'])
    manifest_rows=[]
    for row in rows:
        prompt,ids=prompt_parts(row,tok); p=out/'prompts'/f"{row['context_id']}.txt";p.parent.mkdir(exist_ok=True);p.write_text(prompt)
        markers=[]
        for k in range(1,6):
            tid=tok.convert_tokens_to_ids(f'<SRC_L{k}>'); pos=[i for i,x in enumerate(ids) if x==tid];assert len(pos)==1
            markers.append(dict(action=f'L{k}',token_id=tid,position=pos[0],token_text=f'<SRC_L{k}>'))
        write_json(p.with_suffix('.tokens.json'),dict(token_ids=ids,source_markers=markers))
        manifest_rows.append(dict(request_id=row['context_id'],request_sha256=digest_object(row),prompt_sha256=sha(p),prompt_token_ids_sha256=digest_object(ids),prompt_token_count=len(ids),device_capacity=contract.DEVICE_RESOURCES[row['device']],available_resources={k:row[v] for k,v in contract.AVAIL_FIELD_BY_RESOURCE.items()}))
    write_json(out/'manifest.json',dict(schema='gemv-stage1-full-depth-v1',requests=manifest_rows,request_count=len(rows),grid_count=18,historical_diagnostic_count=1,structure_subset_count=24,requests_file_sha256=sha(request_path),source_sha256=sha(PACKAGE/'inputs/gemv.c'),metadata_sha256=sha(PACKAGE/'inputs/kernel_info.txt'),candidate_policy='legacy',score_reduction='mean',decode_mode='greedy',sample_count=1,clock_menu=[3.33,5.,10.],budget_fractions=[1.,.5,.25],structure_sources=[x['name'] for x in variants],focus_controlled_fields=FOCUS))
    write_json(out/'checkpoint.json',dict(path=str(ADAPTER),adapter_sha256=sha(ADAPTER/'adapter_model.safetensors'),training_contract_sha256=sha(ADAPTER/'training_contract.json'),model=c['model'],revision=c['model_revision'],tokenizer_sha256=sha(ADAPTER/'tokenizer.json'),special_tokens_map_sha256=sha(ADAPTER/'special_tokens_map.json'),registry_sha256=sha(ADAPTER/'directive_domain_registry.json'),special_token_count=len(c['special_tokens']),rank=8,quantization='existing NF4 double quantization with FP16 compute'))
    write_json(out/'environment.json',dict(gpu_uuid=GPU,python=MLIR_PYTHON,model_python='/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python',root_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gnn_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=GNN,text=True).strip(),compiler=COMPILER,rc3_graph_source='read-only GNN branch'))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT_OUT);a=ap.parse_args();rows=prepare(a.out);print(f'FROZEN requests={len(rows)} grid=18 historical=1 structure=24',flush=True)
if __name__=='__main__':main()
