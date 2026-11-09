"""Regression for the RC3 archive's duplicate program names."""
import io,json,tarfile
from experiments.rc3_stage2_multikernel.audit import graph_index

def test_graph_variants_are_preserved_not_overwritten(tmp_path):
 path=tmp_path/'graphs.tar.gz'
 with tarfile.open(path,'w:gz') as tar:
  for suffix in ('aaa','bbb','ccc'):
   data=json.dumps({'metadata':{'action_point_coverage':{'records':[]},'graph_identity':suffix,'provenance':{'source_sha256':'source'}},'nodes':[],'edges':[]}).encode()
   info=tarfile.TarInfo('canonical/program-'+suffix+'.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
 index=graph_index(path)
 assert len(index)==1
 assert [g['graph_identity'] for g in index['program']['archive_variants']]==['aaa','bbb','ccc']
