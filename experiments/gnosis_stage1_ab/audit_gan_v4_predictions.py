"""Audit fresh V4 GAN predictions against their inputs and archived inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference',type=Path,required=True)
    p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--film-inference',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.inference/'manifest.json').read_text())
    rows=json.loads((a.inference/'predicted_directives.json').read_text())
    previous=[json.loads(line) for line in a.historical.read_text().splitlines()]
    film=json.loads((a.film_inference/'predicted_directives.json').read_text())
    film_manifest=json.loads((a.film_inference/'manifest.json').read_text())
    assert manifest['status']=='INFERENCE_COMPLETED' and len(rows)==18
    assert len({r['context_id'] for r in rows})==18 and len(manifest['prompt_sha256'])==18
    assert len(set(manifest['prompt_sha256'].values()))==18
    assert all(not r['film_enabled'] and len(r['directives'])==25 for r in rows)
    assert manifest['checkpoint_sha256']==film_manifest['parent_sha256']
    assert manifest['prepared_manifest']==film_manifest['prepared_manifest']
    assert len(previous)==18 and len(film)==18
    by_old={r['context_id']:r for r in previous};by_film={r['context_id']:r for r in film}
    groups={};differences=[]
    for row in rows:
        signature=json.dumps(row['directives'],sort_keys=True)
        config=groups.setdefault(signature,'V'+str(len(groups)+1))
        prior={m[1].upper():m[2] for m in re.finditer(r'(auto\{[^}]+\})\s*=\s*(\S+)',
               by_old[row['context_id']]['candidates'][0]['prediction'],re.I)}
        assert set(prior)==set(row['directives'])
        hist_diff={key:dict(historical=prior[key],fresh=value) for key,value in row['directives'].items() if value!=prior[key]}
        other=by_film[row['context_id']]['directives']
        differences.append(dict(context_id=row['context_id'],configuration=config,
            historical_changed_fields=hist_diff,
            film_changed_fields={key:dict(v4=value,film=other[key]) for key,value in row['directives'].items() if value!=other[key]}))
    result=dict(status='COMPLETE',requests=18,fields_per_request=25,unique_configurations=len(groups),
        distinct_input_prompts=18,historical_exact_matches=sum(not d['historical_changed_fields'] for d in differences),
        historical_predictions_sha256=hashlib.sha256(a.historical.read_bytes()).hexdigest(),
        film_checkpoint_step=film_manifest['checkpoint_step'],same_source_requests_and_parent_as_film=True,
        scope='Descriptive assignment comparison only; parent versus residual-LoRA-plus-FiLM is not a causal FiLM ablation',
        requests_comparison=differences,configurations={'V'+str(i+1):json.loads(key) for i,key in enumerate(groups)})
    (a.inference/'PREDICTION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('requests','unique_configurations','historical_exact_matches','distinct_input_prompts')}))


if __name__=='__main__':main()
