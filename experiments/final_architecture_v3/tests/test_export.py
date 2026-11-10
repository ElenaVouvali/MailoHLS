import hashlib,json
import pytest
from experiments.final_architecture_v3.export_stage2 import export,FILES,sha

def test_checkpoint_missing_registry_export_preserves_contract(tmp_path):
 source=tmp_path/'checkpoint';source.mkdir();registry=tmp_path/'registry.json';registry.write_text('{}')
 for n in FILES:(source/n).write_text('fixture-'+n)
 contract={'stage':'stage2','directive_domain_registry_sha256':sha(registry)};(source/'training_contract.json').write_text(json.dumps(contract));before=sha(source/'training_contract.json')
 out=export(source,registry,tmp_path/'portable')
 assert sha(out/'training_contract.json')==before and not (source/'directive_domain_registry.json').exists()
 assert export(source,registry,out)==out
 registry.write_text('{"wrong":true}')
 with pytest.raises(ValueError):export(source,registry,out)
