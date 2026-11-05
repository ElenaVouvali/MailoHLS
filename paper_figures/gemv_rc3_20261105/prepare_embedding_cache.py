"""Prepare standalone RC3 GEMV structural tensors for CPU embedding inspection.

This is out-of-cohort inference, not a measured V4b training/evaluation cache.
No directive candidates, QoR labels or split memberships are introduced.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import target_graph_v22 as overlay
from GNN_branch import target_graph_tensor_v22 as tensors
from experiments.v22_gnn_training.directive_preference_probe import capability_vector


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 here=Path(__file__).resolve().parent;root=here.parents[1]
 manifest=json.loads((here/'manifest.json').read_text());graph=compiler.read_graph(here/'gemv.canonical.json')
 compiler.validate(graph)
 assert graph.graph['graph_identity']==manifest['graph_identity']
 assert compiler.digest(root/manifest['selected_source'])==manifest['source_sha256']
 assert compiler.digest(here/'inputs/kernel_info.txt')==manifest['metadata_sha256']
 for name,expected in manifest['construction_code_matches_frozen_rc3'].items():
  assert compiler.digest(root/'GNN_branch'/name)==expected
 registry=overlay.load_registry();graphs={};features={}
 for profile in ('zcu104','u200'):
  joint=overlay.enrich(graph,profile,10.,registry=registry)
  data=tensors.tensorize(joint)
  assert data.matched_mask.sum()==5
  graphs[('gemv',profile,10.)]=data
  features[(profile,10.)]=capability_vector(registry,profile,10.)
 provenance={'scope':'standalone GEMV out-of-cohort structural inference only',
  'frozen_release_member':False,'measured_examples':0,'training_examples_added':0,
  'graph_manifest_sha256':compiler.digest(here/'manifest.json'),
  'canonical_graph_sha256':compiler.digest(here/'gemv.canonical.json'),
  'graph_identity':graph.graph['graph_identity'],'profile_registry_sha256':compiler.digest(overlay.PROFILES),
  'tensorizer_sha256':compiler.digest(root/'GNN_branch/target_graph_tensor_v22.py')}
 prepared={'graph_cache':graphs,'profile_features':features,'profile_feature_dim':next(iter(features.values())).numel()}
 torch.save({'schema':'standalone-gemv-rc3-embedding-input-v1','provenance':provenance,'prepared':prepared},a.output)
 a.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print(str(a.output))


if __name__=='__main__':main()
