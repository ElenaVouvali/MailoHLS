"""RC3-native proven-pair view for the existing refinement proposal policies.
No historical numeric/GEXF traversal or inferred independence is used.
"""
from collections import defaultdict
from pathlib import Path
from GNN_branch.mlir_graph_v2 import read_graph
from experiments.rc3_fusion_foundation.rc3_action_relations import build_action_relations
from experiments.refinement import configurations as c

def from_rc3(path,expected_sha256,actions):
 path=Path(path)
 if c.sha256(path)!=expected_sha256:raise ValueError('RC3 graph changed')
 graph=read_graph(path);pack=build_action_relations(graph);groups=defaultdict(list)
 for record in pack['ledger']:
  left,right=record['query_action'],record['key_action'];kind=record.get('source_relation_type',record['relation_type'])
  if left==right or not record['routing_eligible'] or record['evidence_status']!='proven':continue
  if left not in actions or right not in actions:raise ValueError('Compiler route contains an unknown absolute action')
  if kind not in ('memory_access','loop_hierarchy','memory_dependence'):continue
  pair=tuple(sorted((left,right),key=lambda label:int(label[1:])));groups[pair].append(record)
 result=[]
 for pair,records in sorted(groups.items(),key=lambda kv:tuple(int(x[1:]) for x in kv[0])):
  types=sorted({r.get('source_relation_type',r['relation_type']) for r in records})
  result.append({'schema':'salt-rc3-refinement-proven-pair-v1','action_i':pair[0],'action_j':pair[1],'verified_relation_class':'direct_verified','verified_relation_types':types,'compiler_graph_sha256':expected_sha256,'compiler_graph_identity':graph.graph['graph_identity'],'relation_schema_id':pack['relation_schema_id'],'witnesses':sorted(records,key=c.object_hash),'absence_semantics':'No relation means not_established, not independence','scope':'eligibility only; no performance or hardware binding claim'})
 return result
