"""Reusable CLI for either repaired representation; fresh output directory only."""
import argparse
import json
from pathlib import Path
from compatibility_hooks_v2 import load_generator,install_graph_hooks,sha
R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113'
parser=argparse.ArgumentParser()
parser.add_argument('--kernel',choices=['Kalman','rosetta-3d-rendering'],required=True)
parser.add_argument('--out',type=Path,required=True)
a=parser.parse_args();a.out=a.out.resolve()
if a.out.exists():raise FileExistsError('Fresh output directory required')
sources=json.loads((P/'sources/SOURCES_MANIFEST.json').read_text())
source=next((P/'sources'/a.kernel).glob('source.*'))
assert sha(source)==sources[a.kernel]['source_sha256']
assert sha(source.parent/'kernel_info.txt')==sources[a.kernel]['metadata_sha256']
compiler=P/'compat9_v2_work/compiler_action_storage_v2/bin/cgeist' if a.kernel=='Kalman' else Path('/home/elvouvali/tools/Polygeist/build/bin/cgeist')
if a.kernel=='Kalman':
    build=json.loads((compiler.parent.parent/'BUILD_RECEIPT.json').read_text())
    assert sha(compiler)==build['binary_sha256']
mod=load_generator(R/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py','pinned_eval_generator_cli')
audit=[];install_graph_hooks(mod,a.kernel,audit)
a.out.mkdir()
mod.run(mod.build_arg_parser().parse_args([str(source),'--cgeist',str(compiler),
 '--cflag=-I'+str(source.parent),'--output',str(a.out/(a.kernel+'.gexf')),
 '--analysis-output',str(a.out/(a.kernel+'.analysis.json')),'--mlir-output',str(a.out/(a.kernel+'.mlir'))]))
(a.out/'RUNTIME_PROVENANCE.json').write_text(json.dumps({'kernel':a.kernel,'compiler_sha256':sha(compiler),
 'generator_sha256':sha(mod.__file__),'hook_sha256':sha(Path(__file__).with_name('compatibility_hooks_v2.py')),
 'representation':'evaluation compatibility; not identical historical preprocessing','native_global_proof':audit},indent=2)+'\n')
