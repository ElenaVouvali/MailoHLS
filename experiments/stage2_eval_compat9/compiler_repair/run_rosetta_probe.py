from pathlib import Path
import importlib.util
import sys

r=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
p=r/'artifacts/stage2_eval_9_20261113'
spec=importlib.util.spec_from_file_location('prior_repair','/home/elvouvali/repair_stage2_eval_graphs.py')
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
generator=r/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py'
assert repair.sha(generator)==repair.EXPECTED_GENERATOR_SHA
mod=repair.compiler_compat_module(generator)
repair.install_compatibility_hooks(mod,'rosetta-3d-rendering')
out=p/'compat9_v2_work/rosetta_probe_v2'
out.mkdir(exist_ok=False)
source=p/'sources/rosetta-3d-rendering/source.cpp'
args=mod.build_arg_parser().parse_args([str(source),'--cgeist','/home/elvouvali/tools/Polygeist/build/bin/cgeist',
 '--cflag=-I'+str(source.parent),'--output',str(out/'rosetta-3d-rendering.gexf'),
 '--analysis-output',str(out/'rosetta.analysis.json'),'--mlir-output',str(out/'rosetta.mlir')])
mod.run(args)
