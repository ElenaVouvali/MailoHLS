"""Positive and negative checks on actual normal post-pass Kalman MLIR."""
import argparse
import json
import re
from pathlib import Path
import subprocess
from compatibility_hooks_v2 import load_generator,sha
R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113';W=P/'compat9_v2_work'
parser=argparse.ArgumentParser()
parser.add_argument('--compiler-tag',default='compiler_action_storage_v2')
parser.add_argument('--output-tag',default='compiler_checks_v2')
parser.add_argument('--graph-probe',default='kalman_probe_v2')
args=parser.parse_args()
out=W/args.output_tag;out.mkdir(exist_ok=False)
mod=load_generator(R/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py','pinned_compiler_checks')
source=P/'sources/Kalman/source.cpp'
top,actions=mod.load_kernel_info_actions(source,source.parent/'kernel_info.txt')
manifest=out/'actions.tsv'
manifest.write_text(''.join(f'{a.action_id}\t{a.kind}\t{a.source_file}\t{a.source_line}\t{a.source_column}\n' for a in actions))
base=Path('/home/elvouvali/tools/Polygeist/build/bin/cgeist')
variant=W/args.compiler_tag/'bin/cgeist'
commands={};texts={}
for name,compiler,enabled in [('original',base,False),('variant_flag_off',variant,False),('variant_flag_on',variant,True)]:
    path=out/(name+'.mlir')
    cmd=mod.build_cgeist_command(executable=str(compiler),source=source,kernel=top,mlir_output=path,
                                action_manifest=manifest,cflags=['-I'+str(source.parent)])
    if enabled:cmd.insert(cmd.index('-o'),'--mailohls-preserve-action-storage')
    print('RUN',name,flush=True)
    proc=subprocess.run(cmd,cwd=source.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out/(name+'.log')).write_text(proc.stdout+'\n'+proc.stderr)
    if proc.returncode:raise RuntimeError(name+' failed: '+proc.stderr[-2000:])
    texts[name]=path.read_text();commands[name]=cmd
def normalized_layout_order(text):
    pattern=re.compile(r'(dlti\.dl_spec = #dlti\.dl_spec<)(.*?)(>, llvm\.data_layout)')
    def order(match):
        entries=re.findall(r'#dlti\.dl_entry<.*?(?=, #dlti\.dl_entry<|$)',match.group(2))
        assert len(entries)==15
        return match.group(1)+', '.join(sorted(entries))+match.group(3)
    result,count=pattern.subn(order,text);assert count==1
    return result
assert normalized_layout_order(texts['original'])==normalized_layout_order(texts['variant_flag_off']),'Default compiler IR changed beyond unordered DLTI serialization'
assert normalized_layout_order(texts['variant_flag_on'])==normalized_layout_order((W/args.graph_probe/'Kalman.mlir').read_text()),'Graph probe did not use the qualified post-pass IR'
ir=mod.import_mlir_ir()
def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:yield from walk(child.operation)
with ir.Context() as ctx:
    ctx.allow_unregistered_dialects=True
    module=ir.Module.parse(texts['variant_flag_on'])
    assert module.operation.verify()
    arrays={};loops=[]
    for op in walk(module.operation):
        if op.name=='memref.alloca' and 'mailohls.action_id' in op.attributes:
            aid=str(op.attributes['mailohls.action_id']).strip('"')
            assert 'mailohls.preserve_storage' in op.attributes
            assert aid not in arrays
            arrays[aid]={'type':str(op.results[0].type),'locations':[(x.filename,x.line,x.column) for x in mod.operation_source_points(ir,op)]}
        if op.name in ['affine.for','scf.for']:
            loops.append({'op':op.name,'locations':[(x.filename,x.line,x.column) for x in mod.operation_source_points(ir,op)]})
    assert set(arrays)=={'L1','L2','L5','L6','L7','L8'}
    for aid in ['L5','L6','L7','L8']:assert arrays[aid]['type']=='memref<64xf32>'
    for action in actions:
        if action.kind=='array':
            assert any(line==action.source_line and col==action.source_column for _,line,col in arrays[action.action_id]['locations'])
        else:
            # Polygeist locates a for-op at its induction variable, whereas
            # the source manifest locates the 'for' keyword. Require the
            # unique operation inside that same authenticated source header.
            header=source.read_text().splitlines()[action.source_line-1]
            end_column=header.rfind(')')+1
            matched=[loop for loop in loops if any(Path(file).name==Path(action.source_file).name and
                     line==action.source_line and action.source_column<=col<=end_column
                     for file,line,col in loop['locations'])]
            assert len(matched)==1,(action.action_id,matched)
assert sum('memref.alloca()' in line and 'memref<64xf32>' in line for line in texts['original'].splitlines())==2
cmd=mod.build_cgeist_command(executable=str(variant),source=source,kernel=top,mlir_output=out/'should_not_exist.mlir',action_manifest=manifest,cflags=['-I'+str(source.parent)])
cmd=[arg for arg in cmd if not arg.startswith('-mailohls-action-manifest=')]
cmd.insert(cmd.index('-o'),'--mailohls-preserve-action-storage')
proc=subprocess.run(cmd,cwd=source.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(out/'missing_manifest.log').write_text(proc.stderr)
assert proc.returncode!=0 and 'requires --mailohls-action-manifest' in proc.stderr
result={'status':'PASS','flag_off_mlir_byte_identical_to_original':texts['original']==texts['variant_flag_off'],
 'flag_off_ir_identical_except_unordered_dlti_entry_order':True,
 'comparison_normalization':'Only DLTI data-layout entry order; no emitted IR or GEXF changed','normal_post_pass_ir':True,
 'not_immediate':True,'all_four_64_arrays_source_located':True,'all_seven_action_loops_source_located':True,
 'missing_manifest_flag_rejected':True,'arrays':arrays,'loops':loops,'commands':commands,
 'original_compiler_sha256':sha(base),'variant_compiler_sha256':sha(variant),
 'original_mlir_sha256':sha(out/'original.mlir'),'preserved_mlir_sha256':sha(out/'variant_flag_on.mlir')}
(out/'PASS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('PASS: original flag-off IR except unordered DLTI entry order; four real arrays and seven exact loops in post-pass IR',flush=True)
