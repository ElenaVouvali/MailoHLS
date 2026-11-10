import copy,hashlib,io,json,tarfile
from pathlib import Path
import pytest
from LLM_branch.common.memory_extension import validate_heldout_extension

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(root):
 root.mkdir();(root/'train.memory.pt').write_bytes(b'unit-test-only original');(root/'GAN.memory.pt').write_bytes(b'unit-test-only extension');(root/'graphs').mkdir();source='void GAN() {}'
 graph={'metadata':{'graph_identity':'g','provenance':{'source_sha256':'a','metadata_sha256':'m','local_header_sha256':{}},'action_point_coverage':{'records':[{'compiler_matched':True}]}}};raw=json.dumps(graph).encode();(root/'graphs/GAN.json').write_bytes(raw)
 archive=root/'archive.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  item=tarfile.TarInfo('canonical/GAN.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 parent={'memory_interface':'rc3_action_evidence_c_v1','max_slots':64,'mem_dim':128,'evidence_schema':'e','evidence_capacity':1024,'action_relation_schema':'r','gnn_checkpoint_sha256':'gnn','compiler_graph_archive_sha256':sha(archive),'target_independent':True,'hardware_messages':False,'memory_files':[{'kernel':'train','file':'train.memory.pt','sha256':sha(root/'train.memory.pt')}]};(root/'training_memory_manifest.json').write_text(json.dumps(parent));digest=sha(root/'training_memory_manifest.json')
 extra={'kernel':'GAN','file':'GAN.memory.pt','sha256':sha(root/'GAN.memory.pt'),'extension_graph_file':'graphs/GAN.json','graph_file_sha256':sha(root/'graphs/GAN.json'),'graph_identity':'g','source_sha256':'a','action_slots':1,'evidence_records':1,'key_count':65};extension={**parent,'schema':'salt-hls-authenticated-heldout-memory-extension-v1','extension_kind':'authenticated_heldout_programs','parent_memory_manifest_file':'training_memory_manifest.json','parent_memory_manifest_sha256':digest,'extension_kernels':['GAN'],'memory_files':parent['memory_files']+[extra],'authenticated_archive_file':'archive.tar.gz','extension_source_receipts':{'gan':{'source_sha256':'a','metadata_sha256':'m','header_sha256':{},'archive_member':'canonical/GAN.json','prompt_source_sha256':hashlib.sha256(source.encode()).hexdigest()}}};write(root,extension);return extension,digest,source

def write(root,m):(root/'memory_manifest.json').write_text(json.dumps(m))
def test_accepts_append_and_exact_prompt(tmp_path):
 root=tmp_path/'bank';m,h,s=fixture(root);r=validate_heldout_extension(root,h,[('GAN',s)]);assert r['extension_kernels']==['gan']
@pytest.mark.parametrize('field,value',[('mem_dim',64),('gnn_checkpoint_sha256','wrong'),('hardware_messages',True)])
def test_rejects_contract_change(tmp_path,field,value):
 root=tmp_path/'bank';m,h,s=fixture(root);m[field]=value;write(root,m)
 with pytest.raises(ValueError):validate_heldout_extension(root,h)
def test_rejects_modified_original_tensor(tmp_path):
 root=tmp_path/'bank';m,h,s=fixture(root);(root/'train.memory.pt').write_bytes(b'changed')
 with pytest.raises(ValueError):validate_heldout_extension(root,h)
def test_rejects_deleted_parent(tmp_path):
 root=tmp_path/'bank';m,h,s=fixture(root);m['memory_files']=m['memory_files'][1:];write(root,m)
 with pytest.raises(ValueError):validate_heldout_extension(root,h)
def test_rejects_wrong_source_even_valid_memory(tmp_path):
 root=tmp_path/'bank';m,h,s=fixture(root)
 with pytest.raises(ValueError):validate_heldout_extension(root,h,[('GAN','different program')])
def test_rejects_rehashed_but_unarchived_graph(tmp_path):
 root=tmp_path/'bank';m,h,s=fixture(root);g=json.loads((root/'graphs/GAN.json').read_text());g['metadata']['graph_identity']='changed';(root/'graphs/GAN.json').write_text(json.dumps(g));m['memory_files'][-1]['graph_file_sha256']=sha(root/'graphs/GAN.json');write(root,m)
 with pytest.raises(ValueError):validate_heldout_extension(root,h)
def test_rejects_alias_collision(tmp_path):
 root=tmp_path/'bank';m,h,s=fixture(root);m['memory_files'].append({**m['memory_files'][-1],'kernel':' GAN '});write(root,m)
 with pytest.raises(ValueError):validate_heldout_extension(root,h)
def test_rejects_unregistered_tensor(tmp_path):
 root=tmp_path/'bank';m,h,s=fixture(root);(root/'alien.memory.pt').write_bytes(b'unit-test-only')
 with pytest.raises(ValueError):validate_heldout_extension(root,h)
