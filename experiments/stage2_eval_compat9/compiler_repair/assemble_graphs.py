"""Assemble nine qualified graphs without changing protected sources/controls."""
import json
from pathlib import Path
import shutil
import subprocess
import networkx as nx
from compatibility_hooks_v2 import sha,GENERATOR_SHA

R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113';W=P/'compat9_v2_work'
OUT=P/'graphs_compat9_v2'
KERNELS=('GAN','Kalman','chstone-aes','chstone-jpeg','covariance','jacobi-2d','rosetta-3d-rendering','syr2k','trmm-opt')
SEVEN=tuple(k for k in KERNELS if k not in ['Kalman','rosetta-3d-rendering'])
before=json.loads((W/'before_sha256.json').read_text())
for relative,digest in before.items():
    path=Path(relative) if relative.startswith('/') else P/relative
    if sha(path)!=digest:raise RuntimeError('Protected original changed: '+str(path))
generator=R/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py'
assert sha(generator)==GENERATOR_SHA
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()=='6ee1390611c9047a4b8e2b07200d223b93ee8d16'
check=json.loads((W/'compiler_checks_final/PASS.json').read_text())
assert check['status']=='PASS'
compiler=W/'compiler_action_storage_v2/bin/cgeist'
build=json.loads((compiler.parent.parent/'BUILD_RECEIPT.json').read_text())
assert sha(compiler)==build['binary_sha256']==check['variant_compiler_sha256']
assert build['mem2reg_original_object_reconstruction']['object_byte_identical']
sources=json.loads((P/'sources/SOURCES_MANIFEST.json').read_text())
for key in ['kalman_probe_v2','rosetta_verified_v3']:
    assert json.loads((W/key/'FROZEN_PASS.json').read_text())['status']=='PASS_FROZEN_GNN_GMT_SINGLE_KERNEL'
OUT.mkdir(exist_ok=False)
for k in SEVEN:shutil.copy2(P/'graphs_retry1'/(k+'.gexf'),OUT/(k+'.gexf'))
shutil.copy2(W/'kalman_probe_v2/Kalman.gexf',OUT/'Kalman.gexf')
shutil.copy2(W/'kalman_probe_v2/Kalman.mlir',OUT/'Kalman.mlir')
shutil.copy2(W/'kalman_probe_v2/Kalman.analysis.json',OUT/'Kalman.analysis.json')
for old,new in [('rosetta-3d-rendering.gexf','rosetta-3d-rendering.gexf'),('rosetta.mlir','rosetta-3d-rendering.mlir'),
                ('rosetta.analysis.json','rosetta-3d-rendering.analysis.json'),('NATIVE_BINDING.json','ROSETTA_NATIVE_BINDING.json')]:
    shutil.copy2(W/'rosetta_verified_v3'/old,OUT/new)
records={}
for k in KERNELS:
    path=OUT/(k+'.gexf');g=nx.read_gexf(path)
    gp=json.loads(g.graph['name'].split(':',1)[1])
    assert gp['generator_sha256']==GENERATOR_SHA
    assert gp['source_sha256']==sources[k]['source_sha256']
    assert gp['action_sha256']==sources[k]['metadata_sha256']
    anchors={a['action_id']:a for _,a in g.nodes(data=True) if a.get('action_resolution')}
    assert set(anchors)==set(sources[k]['actions'])
    if k=='Kalman':
        assert len(anchors)==13 and all(a['action_resolution']=='source_location' for a in anchors.values())
        assert gp['cgeist_sha256']==sha(compiler)
    elif k=='rosetta-3d-rendering':
        assert len(anchors)==12 and anchors['L1']['action_resolution']=='global_symbol_and_access'
    else:assert sha(path)==before['graphs_retry1/'+k+'.gexf']
    records[k]={'graph_sha256':sha(path),'nodes':len(g),'edges':g.number_of_edges(),'actions':len(anchors),
        'source_sha256':gp['source_sha256'],'action_sha256':gp['action_sha256'],'generator_sha256':gp['generator_sha256'],
        'mlir_sha256':gp['mlir_sha256'],'cgeist_sha256':gp['cgeist_sha256'],'action_resolutions':gp['action_resolutions'],
        'variant':'early_action_storage_preservation_post_pass' if k=='Kalman' else
                  'verified_static_global_source_symbol_and_access' if k=='rosetta-3d-rendering' else 'unchanged_seven_control',
        'no_source_or_metadata_edits':True}
    if k=='rosetta-3d-rendering':
        proof=json.loads((OUT/'ROSETTA_NATIVE_BINDING.json').read_text())
        assert proof['mlir_sha256']==gp['mlir_sha256']
        records[k]['native_binding']=proof
        records[k]['native_binding_sha256']=sha(OUT/'ROSETTA_NATIVE_BINDING.json')
        records[k]['scope_limitations']=['L8 fragment and L9 pixels have real allocation/call evidence but no resolved direct array-access scope edges in pinned builder; no edges invented']
receipt={'schema':'mailohls-stage2-eval-compat9-v2','status':'NINE_GRAPHS_QUALIFIED_FOR_FROZEN_EXPORT',
 'representation':'explicit heterogeneous evaluation compiler compatibility, not identical historical preprocessing',
 'generator_sha256':GENERATOR_SHA,'source_manifest_sha256':sha(P/'sources/SOURCES_MANIFEST.json'),
 'original_exporter_sha256':before['/home/elvouvali/export_stage2_eval_gmt.py'],
 'runtime_hook_sha256':sha(W/'compatibility_hooks_v2.py'),'assembler_sha256':sha(__file__),
 'kalman_graph_runner_sha256':sha(W/'run_kalman_probe.py'),'rosetta_graph_runner_sha256':sha(W/'run_rosetta_verified.py'),
 'compiler_build_receipt_sha256':sha(compiler.parent.parent/'BUILD_RECEIPT.json'),
 'compiler_checks_sha256':sha(W/'compiler_checks_final/PASS.json'),
 'seven_original_gexfs_unchanged':True,'protected_before_manifest_sha256':sha(W/'before_sha256.json'),
 'kernels':records,'full_export_performed':False,'source_templates_or_metadata_modified':False}
(OUT/'EVAL_COMPAT9_V2_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('Nine graphs assembled and authenticated; full nine-kernel export still required',flush=True)
