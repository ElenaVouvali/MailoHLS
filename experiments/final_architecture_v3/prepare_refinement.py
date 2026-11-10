"""Seal one real decoder request for the normal Stage3 refinement interface.
No predictions, preferences or synthesis are run by this preparation step.
"""
import argparse,json,re,shutil
from pathlib import Path
from GNN_branch.insert_placeholders import insert_placeholders
from LLM_branch.common import directive_domains,mailohls_contract as contract
from experiments.refinement import configurations as c

def main():
 p=argparse.ArgumentParser();p.add_argument('--row',type=Path,required=True,help='One authenticated model-input JSON object');p.add_argument('--source',type=Path,required=True);p.add_argument('--tool-provenance',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 row=json.loads(a.row.read_text());kernel=row['kernel_name'];metadata=a.source.parent/'kernel_info.txt';actions=directive_domains.load_source_actions(kernel,a.source.parent);top=metadata.read_text().splitlines()[0].strip()
 if not re.fullmatch(r'[A-Za-z_]\w*',top):raise ValueError('Top must be explicit and canonical')
 if row.get('obj_mode',row.get('objective'))!='PARETO_LATENCY':raise ValueError('Latency qualification only')
 part=row['device'];clock=float(row.get('selected_clock_period',row.get('clock_period')));caps=contract.DEVICE_RESOURCES[part];budgets={res:int(row[contract.AVAIL_FIELD_BY_RESOURCE[res]]) for res in c.RESOURCE_KEYS}
 a.out.mkdir(parents=True);snapshot=a.out/'source';snapshot.mkdir();shutil.copyfile(metadata,snapshot/'kernel_info.txt')
 for header in a.source.parent.glob('*.h'):shutil.copyfile(header,snapshot/header.name)
 template=snapshot/a.source.name;template.write_text(''.join(insert_placeholders(str(a.source),allowed_labels=set(actions))))
 tool=json.loads(a.tool_provenance.read_text());identity={k:tool[k] for k in ('executable_sha256','version_output','vendor_headers')}
 if '2021.1' not in identity['version_output']:raise ValueError('Incompatible Vitis version')
 request={'schema':'salt-v3-refinement-request-v1','kernel':kernel,'objective':'PARETO_LATENCY','part':part,'clock_ns':clock,'capacities':caps,'budgets':budgets,'top':top,'source_template':str(template.resolve()),'source_directory':str(snapshot.resolve()),'source_sha256':c.sha256(template),'original_source_sha256':c.sha256(a.source),'prompt_source_sha256':__import__('hashlib').sha256(row['input'].encode()).hexdigest(),'headers_sha256':{p.name:c.sha256(p) for p in snapshot.glob('*.h')},'action_metadata_sha256':c.sha256(metadata),'domain_identity':c.tcad.SOURCE_DOMAIN_POLICY,'tool_identity':identity,'flow_settings':list(c.SETTINGS),'verified_relations':[],'inactive_actions':[],'provenance':{'model_request_file':str(a.row.resolve()),'model_request_sha256':c.sha256(a.row),'original_source':str(a.source.resolve()),'tool_receipt_sha256':c.sha256(a.tool_provenance)},'qualification':'AVAILABLE_EXPERIMENTAL; no functional or physical result implied'}
 c.write_json(a.out/'request.json',request);(a.out/'SHA256SUMS').write_text(''.join(c.sha256(p)+'  '+str(p.relative_to(a.out))+'\n' for p in sorted(a.out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'));print(a.out/'request.json')
if __name__=='__main__':main()
