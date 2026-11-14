from pathlib import Path
import importlib.util
import sys

R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113'
spec=importlib.util.spec_from_file_location('prior_repair','/home/elvouvali/repair_stage2_eval_graphs.py')
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
generator=R/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py'
assert repair.sha(generator)==repair.EXPECTED_GENERATOR_SHA
mod=repair.compiler_compat_module(generator)
old=mod.build_cgeist_command
def command(**kwargs):
    cmd=old(**kwargs)
    cmd.insert(cmd.index('-o'),'--mailohls-preserve-action-storage')
    return cmd
mod.build_cgeist_command=command
out=P/'compat9_v2_work/kalman_probe_v2'
out.mkdir(exist_ok=False)
source=P/'sources/Kalman/source.cpp'
compiler=P/'compat9_v2_work/compiler_action_storage_v2/bin/cgeist'
args=mod.build_arg_parser().parse_args([str(source),'--cgeist',str(compiler),
 '--cflag=-I'+str(source.parent),'--output',str(out/'Kalman.gexf'),
 '--analysis-output',str(out/'Kalman.analysis.json'),'--mlir-output',str(out/'Kalman.mlir')])
mod.run(args)
