"""Export small tracked provenance contracts from the ignored build outputs."""
import argparse, hashlib, json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
from experiments.rc3_fusion_foundation.rc3_action_relations import RELATION_SCHEMA_ID

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifacts',default=str(ROOT/'artifacts/rc3_fusion_foundation/v1_20261107'))
    a=ap.parse_args(); root=Path(a.artifacts)
    build=json.loads((root/'manifest.json').read_text())
    bundle=root/'synthesis_bundle_v2'
    synthesis=json.loads((bundle/'manifest.json').read_text())
    provenance={
      'schema':'salt-hls-rc3-foundation-provenance-v1',
      'source_commits':{
        'gnn_refinement':'172e05a69650a666c323955ea78cb501fd9364c2',
        'rc3_structural_memory_audit':'bb4bc8f8261e33f7a229d92f30e36a0a53f986a7',
        'stage1_refinement':'63afcfc7a54e573a544678e728cdb7d99d865981',
        'gemv_stage1_full_depth':'97ed75ed3f41ad80247d739888d9ee488881872c'},
      'relation_schema':{'name':'salt-hls-rc3-action-relations-v1','policy':'proven_loop_scope_array_access_and_loop_hierarchy_v1',
        'schema_id':RELATION_SCHEMA_ID},
      'implementation_sha256':{str(p):sha(ROOT/p) for p in [
        Path('experiments/rc3_fusion_foundation/rc3_action_relations.py'),
        Path('experiments/rc3_fusion_foundation/structural_memory_v1.py'),
        Path('experiments/rc3_fusion_foundation/build_memory.py'),
        Path('experiments/rc3_fusion_foundation/prepare_synthesis.py'),
        Path('experiments/rc3_fusion_foundation/summarize_synthesis.py'),
        Path('experiments/rc3_fusion_foundation/test_rc3_memory.py'),
        Path('GNN_branch/mlir_graph_v2.py'),Path('GNN_branch/action_point_relations.py'),
        Path('GNN_branch/action_demand_v22.py'),Path('GNN_branch/semantic_tensor_features_v22.py'),
        Path('GNN_branch/target_graph_v22.py'),Path('GNN_branch/target_graph_tensor_v22.py'),
        Path('experiments/v22_gnn_training/v4b/models.py'),Path('LLM_branch/common/structural_xattn.py'),
        Path('LLM_branch/common/structural_memory.py')]},
      'gnn_checkpoint':build['gnn'], 'inputs':build['inputs'],
      'memory_artifacts':build['memory_files'],
      'synthesis_manifest':{'path':'synthesis_bundle_v2/manifest.json','sha256':sha(bundle/'manifest.json'),
        'runner_path':'synthesis_bundle_v2/run_batch.py','runner_sha256':sha(bundle/'run_batch.py'),
        'physical_jobs':synthesis['physical_job_count'],'tool_invoked':False,
        'tool_available':json.loads((bundle/'execution_status.json').read_text())['exists']}}
    out=PACKAGE/'manifests'; out.mkdir(exist_ok=True)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    schemas={'relation_schema':{'name':'salt-hls-rc3-action-relations-v1','id':provenance['relation_schema']['schema_id'],
      'absolute_slot_rule':'Lk -> slot k-1; no compaction','evidence_status_values':['proven','possible','not_established','missing'],
      'relation_types':['self','loop_child','loop_parent','array_accessed_by_loop','loop_accesses_array','loop_dependence'],
      'absence_semantics':'not established is not independence'},
      'memory_schemas':{'A':'salt-hls-xattn-action-memory-v1','B':'salt-hls-xattn-action-memory-v1 + RC3 relation mask',
       'C':'salt-hls-rc3-action-evidence-memory-v1+xattn-mask-v1'},
      'evidence_types':['array_memory_root','array_ssa_root','loop_node','control_region','affine_memory_access','arithmetic_operation',
       'loop_carried_ssa_value','memory_access_witness','loop_carried_witness','dependence_witness','loop_hierarchy_witness','operand_witness','definition_witness'],
      'evidence_known_fields':['class','proven','shape','trip_count','affine_map','access_role','operation_family','loop_carried','dependence_kind'],
      'edge_vector':'mean(endpoint node vectors) plus frozen relation embedding in dims 0:16 and frozen edge-feature embedding in dims 16:32; remaining dimensions are endpoint mean; diagnostic design, not trained'}
    (out/'schemas.json').write_text(json.dumps(schemas,indent=2,sort_keys=True)+'\n')
    print(f'Wrote {out}')

if __name__=='__main__': main()
