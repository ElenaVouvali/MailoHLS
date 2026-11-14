"""Build an isolated opt-in cgeist overlay from actual local sources/objects.

Reuses existing static dependency libraries; never modifies the original build.
Only driver.cc and PolygeistMem2Reg.cpp are recompiled. Commands, source diffs,
input hashes and resulting binary hash are retained beside the versioned binary.
"""
import argparse
import concurrent.futures
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
SRC=Path('/home/elvouvali/tools/Polygeist')
BASE=SRC/'build'
parser=argparse.ArgumentParser()
parser.add_argument('--tag',default='compiler_action_storage_v1')
parser.add_argument('--reconstructed-mem2reg',type=Path)
args=parser.parse_args()
if not args.tag.replace('_','').isalnum():raise ValueError('Unsafe build tag')
OUT=HERE/args.tag
OUT.mkdir(exist_ok=False)
for directory in ['original','source','objects','lib','bin','logs']:(OUT/directory).mkdir()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
paths={'driver.cc':SRC/'tools/cgeist/driver.cc',
       'PolygeistMem2Reg.cpp':SRC/'lib/polygeist/Passes/PolygeistMem2Reg.cpp'}
texts={name:p.read_text() for name,p in paths.items()}
reconstruction=None
if args.reconstructed_mem2reg:
    reconstruction=json.loads((args.reconstructed_mem2reg.parent/'RESULT.json').read_text())
    if not reconstruction['object_byte_identical'] or sha(args.reconstructed_mem2reg)!=reconstruction['source_sha256']:
        raise RuntimeError('Reconstructed source was not authenticated against the actual base object')
    original_object=BASE/'tools/polygeist/lib/polygeist/Passes/CMakeFiles/obj.MLIRPolygeistTransforms.dir/PolygeistMem2Reg.cpp.o'
    if sha(original_object)!=reconstruction['original_object_sha256']:raise RuntimeError('Base object changed')
    texts['PolygeistMem2Reg.cpp']=args.reconstructed_mem2reg.read_text()
for name,text in texts.items():(OUT/'original'/name).write_text(text)
driver=texts['driver.cc']
anchor='static cl::opt<std::string> MailoHLSActionManifest('
assert driver.count(anchor)==1
driver=driver.replace(anchor,'''static cl::opt<bool> MailoHLSPreserveActionStorage(
    "mailohls-preserve-action-storage", cl::init(false),
    cl::desc("Evaluation-only: preserve exact manifest-marked array storage through mem2reg"),
    cl::cat(toolOptions));

'''+anchor)
old='''      std::shared_ptr<const std::vector<MailoHLSArrayAction>> actions)
      : actions(std::move(actions)) {}'''
new='''      std::shared_ptr<const std::vector<MailoHLSArrayAction>> actions,
      bool preserveStorage = false)
      : actions(std::move(actions)), preserveStorage(preserveStorage) {}'''
assert driver.count(old)==1;driver=driver.replace(old,new)
old='''        op->setAttr("mailohls.action_kind",
                    mlir::StringAttr::get(op->getContext(), "array"));'''
new=old+'''
        if (preserveStorage)
          op->setAttr("mailohls.preserve_storage", mlir::UnitAttr::get(op->getContext()));'''
assert driver.count(old)==1;driver=driver.replace(old,new)
old='''  std::shared_ptr<const std::vector<MailoHLSArrayAction>> actions;
};'''
assert driver.count(old)==1;driver=driver.replace(old,'''  std::shared_ptr<const std::vector<MailoHLSArrayAction>> actions;
  bool preserveStorage;
};''')
old='''  std::shared_ptr<const std::vector<MailoHLSArrayAction>> mailoHLSActions;'''
assert driver.count(old)==1
driver=driver.replace(old,'''  if (MailoHLSPreserveActionStorage && MailoHLSActionManifest.empty()) {
    llvm::errs() << "error: --mailohls-preserve-action-storage requires --mailohls-action-manifest\\n";
    return 1;
  }
'''+old)
old='''  if (true) {
    optPM.addPass(mlir::createCSEPass());'''
assert driver.count(old)==1
driver=driver.replace(old,'''  if (true) {
    // The original marker runs after early promotion. Evaluation-only mode
    // marks exact source allocations before any canonicalization/mem2reg.
    if (MailoHLSPreserveActionStorage)
      optPM.addPass(std::make_unique<MarkMailoHLSArrayActionsPass>(mailoHLSActions, true));
    optPM.addPass(mlir::createCSEPass());''')
mem=texts['PolygeistMem2Reg.cpp']
old='''bool isPromotable(mlir::Value AI) {
  std::deque<mlir::Value> list = {AI};'''
assert mem.count(old)==1
mem=mem.replace(old,'''bool isPromotable(mlir::Value AI) {
  // Set only by the explicit evaluation flag on an exact manifest match.
  // All unmarked allocation promotion and normal control-flow passes remain.
  if (auto *allocation = AI.getDefiningOp())
    if (allocation->hasAttr("mailohls.preserve_storage") &&
        allocation->hasAttr("mailohls.action_id"))
      return false;
  std::deque<mlir::Value> list = {AI};''')
modified={'driver.cc':driver,'PolygeistMem2Reg.cpp':mem}
patch=[]
for name,text in modified.items():
    (OUT/'source'/name).write_text(text)
    patch.extend(difflib.unified_diff(texts[name].splitlines(True),text.splitlines(True),
                                    'original/'+name,'source/'+name))
(OUT/'action_storage_v1.patch').write_text(''.join(patch))
entries=json.loads((BASE/'compile_commands.json').read_text())
commands=[]
for name,p in paths.items():
    e=next(e for e in entries if e['file']==str(p))
    args=shlex.split(e['command'])
    args[args.index('-o')+1]=str(OUT/'objects'/(name+'.o'))
    args[args.index('-c')+1]=str(OUT/'source'/name)
    commands.append({'file':name,'argv':args,'cwd':str(BASE)})
(OUT/'compile_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
def compile_one(command):
    print('COMPILE',command['file'],flush=True)
    with (OUT/'logs'/(command['file']+'.log')).open('w') as log:
        subprocess.run(command['argv'],cwd=BASE,stdout=log,stderr=subprocess.STDOUT,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(compile_one,commands))
archive=OUT/'lib/libMLIRPolygeistTransforms.a'
assert (BASE/'lib/libMLIRPolygeistTransforms.a').read_bytes()[:8]==b'!<arch>\n'
shutil.copy2(BASE/'lib/libMLIRPolygeistTransforms.a',archive)
# Archive member names must match the existing member, not the source basename.
member=OUT/'objects/PolygeistMem2Reg.cpp.o'
assert member.is_file()
subprocess.run(['/usr/bin/ar','r',str(archive),str(member)],check=True)
subprocess.run(['/usr/bin/ranlib',str(archive)],check=True)
raw=(HERE/'original_cgeist_link_command.txt').read_text()
args=shlex.split(raw)
assert args[:2]==[':','&&'] and args[-2:]==['&&',':']
args=args[2:-2]
args[args.index('-o')+1]=str(OUT/'bin/cgeist')
replacements={'tools/polygeist/tools/cgeist/CMakeFiles/cgeist.dir/driver.cc.o':str(OUT/'objects/driver.cc.o'),
              'lib/libMLIRPolygeistTransforms.a':str(archive)}
args=[replacements.get(arg,arg) for arg in args]
(OUT/'link_argv.json').write_text(json.dumps(args,indent=2)+'\n')
inputs={str((BASE/a).resolve()):sha(BASE/a) for a in args[1:]
        if not a.startswith('-') and (BASE/a).is_file() and str(OUT) not in str((BASE/a).resolve())}
(OUT/'base_link_inputs_sha256.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
print('LINK',len(inputs),'original dependency files; isolated archive and objects',flush=True)
with (OUT/'logs/link.log').open('w') as log:
    subprocess.run(args,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,check=True)
# Clang resources follow the relocated compiler through an explicit link.
(OUT/'lib/clang').symlink_to(BASE/'lib/clang',target_is_directory=True)
receipt={'schema':'mailohls-cgeist-action-storage-overlay-v1','binary':str(OUT/'bin/cgeist'),
 'binary_sha256':sha(OUT/'bin/cgeist'),'base_binary_sha256':sha(BASE/'bin/cgeist'),
 'source_before':{name:sha(OUT/'original'/name) for name in paths},
 'source_after':{name:sha(OUT/'source'/name) for name in paths},
 'patch_sha256':sha(OUT/'action_storage_v1.patch'),'base_link_inputs_sha256':sha(OUT/'base_link_inputs_sha256.json'),
 'compile_commands_sha256':sha(OUT/'compile_commands.json'),'link_argv_sha256':sha(OUT/'link_argv.json'),
 'opt_in_flag':'--mailohls-preserve-action-storage','original_source_build_modified':False,
 'builder_sha256':sha(__file__),'mem2reg_original_object_reconstruction':reconstruction,
 'excluded_unbuilt_local_mem2reg_edits':bool(reconstruction),
 'changes':['mark exact action allocations before early passes','skip promotion only for explicitly marked storage'],
 'status':'BUILT_NOT_YET_QUALIFIED'}
(OUT/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
