from pathlib import Path
import json
from compatibility_hooks_v2 import load_generator,install_graph_hooks,sha
R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113'
out=P/'compat9_v2_work/rosetta_verified_v3'
out.mkdir(exist_ok=False)
generator=R/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py'
mod=load_generator(generator,'pinned_rosetta_verified_v3')
proof=[];install_graph_hooks(mod,'rosetta-3d-rendering',proof)
source=P/'sources/rosetta-3d-rendering/source.cpp'
args=mod.build_arg_parser().parse_args([str(source),'--cgeist','/home/elvouvali/tools/Polygeist/build/bin/cgeist',
 '--cflag=-I'+str(source.parent),'--output',str(out/'rosetta-3d-rendering.gexf'),
 '--analysis-output',str(out/'rosetta.analysis.json'),'--mlir-output',str(out/'rosetta.mlir')])
mod.run(args)
assert len(proof)==1
assert proof[0]['mlir_sha256']==sha(out/'rosetta.mlir')
(out/'NATIVE_BINDING.json').write_text(json.dumps(proof[0],indent=2,sort_keys=True)+'\n')
