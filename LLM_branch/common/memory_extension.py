"""Explicit authenticated held-out RC3 memory extension; historical defaults unchanged."""
import hashlib,json,tarfile,re
from pathlib import Path

def _norm(name):return re.sub(r'[-\s]+','_',str(name).strip().lower())

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _file(root,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Extension paths must stay inside the bank')
 path=root/p
 if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Extension path escapes bank')
 return path

def validate_heldout_extension(memory_dir,expected_parent_sha256,case_sources=()):
 root=Path(memory_dir);extension=json.loads((root/'memory_manifest.json').read_text())
 if extension.get('schema')!='salt-hls-authenticated-heldout-memory-extension-v1' or extension.get('extension_kind')!='authenticated_heldout_programs':raise ValueError('Not an authenticated held-out memory extension')
 if extension.get('memory_ablation'):raise ValueError('An extension cannot disguise an ablation')
 parentfile=_file(root,extension['parent_memory_manifest_file'])
 if extension.get('parent_memory_manifest_sha256')!=expected_parent_sha256 or _sha(parentfile)!=expected_parent_sha256:raise ValueError('Extension parent differs from frozen checkpoint')
 parent=json.loads(parentfile.read_text())
 for key in ('memory_interface','max_slots','mem_dim','evidence_schema','evidence_capacity','action_relation_schema','gnn_checkpoint_sha256','compiler_graph_archive_sha256','target_independent','hardware_messages'):
  if extension.get(key)!=parent.get(key):raise ValueError('Extension changes frozen interface/encoder field: '+key)
 def index(records):
  result={}
  for r in records:
   key=_norm(r['kernel'])
   if key in result:raise ValueError('Duplicate normalized kernel in extension')
   result[key]=r
  return result
 base=index(parent['memory_files']);allfiles=index(extension['memory_files']);new=set(allfiles)-set(base);declared={_norm(k) for k in extension['extension_kernels']}
 if not new or new!=declared or not set(base)<=set(allfiles):raise ValueError('Extension must append exactly the declared new programs')
 for key,record in base.items():
  if allfiles[key]!=record:raise ValueError('Original training memory entry changed: '+key)
 files=[]
 for record in allfiles.values():
  name=record['file']
  if Path(name).name!=name or not name.endswith('.memory.pt') or _norm(name[:-len('.memory.pt')])!=_norm(record['kernel']):raise ValueError('Tensor filename must match its declared normalized kernel')
  path=_file(root,name);files.append(path.name)
  if _sha(path)!=record['sha256']:raise ValueError('Memory tensor checksum mismatch: '+record['kernel'])
 if len(files)!=len(set(files)) or set(files)!={p.name for p in root.glob('*.memory.pt')}:raise ValueError('Unregistered or duplicate memory tensor files')
 archive=_file(root,extension['authenticated_archive_file'])
 if _sha(archive)!=parent['compiler_graph_archive_sha256']:raise ValueError('RC3 archive differs from authenticated parent')
 receipts=extension['extension_source_receipts'];verified={}
 with tarfile.open(archive) as tar:
  for key in sorted(new):
   record=allfiles[key];receipt=receipts[key];graphfile=_file(root,record['extension_graph_file']);raw=graphfile.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=record['graph_file_sha256']:raise ValueError('Extension graph checksum mismatch')
   archived=tar.extractfile(receipt['archive_member'])
   if archived is None or archived.read()!=raw:raise ValueError('Extension graph does not match pinned RC3 archive')
   metadata=json.loads(raw)['metadata'];provenance=metadata['provenance']
   if metadata['graph_identity']!=record['graph_identity'] or provenance['source_sha256']!=record['source_sha256']:raise ValueError('Extension graph/source identity mismatch')
   for field,pfield in [('source_sha256','source_sha256'),('metadata_sha256','metadata_sha256'),('header_sha256','local_header_sha256')]:
    if receipt[field]!=provenance[pfield]:raise ValueError('Extension source/action/header receipt mismatch')
   coverage=metadata['action_point_coverage']['records']
   if len(coverage)!=record['action_slots'] or not all(r['compiler_matched'] for r in coverage):raise ValueError('Held-out action mapping incomplete')
   if record['key_count']!=64+record['evidence_records'] or record['evidence_records']>parent['evidence_capacity']:raise ValueError('Evidence capacity/absolute-slot contract changed')
   verified[key]=receipt['prompt_source_sha256']
 for kernel,source in case_sources:
  key=_norm(kernel)
  if key not in allfiles:raise ValueError('Input source has no authenticated memory')
  if key in verified and hashlib.sha256(source.encode()).hexdigest()!=verified[key]:raise ValueError('Held-out prompt source differs from the authenticated extension')
 return {'parent_memory_manifest_sha256':expected_parent_sha256,'extension_kernels':sorted(new),'memory_interface':parent['memory_interface'],'gnn_checkpoint_sha256':parent['gnn_checkpoint_sha256']}
