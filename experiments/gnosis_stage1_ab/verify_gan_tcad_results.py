"""Verify the published GAN evidence and recompute each paired measurement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from experiments.gnosis_stage1_ab import gan_tcad_synthesis as synth


def same(left, right):
    if isinstance(left, dict):
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).parent/'results/gan_tcad_film_20261112')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    index = json.loads((a.root/'EVIDENCE_INDEX.json').read_text())
    for relative, digest in index.items():
        assert hashlib.sha256((a.root/relative).read_bytes()).hexdigest() == digest, relative
    inference = json.loads((a.root/'inference/manifest.json').read_text())
    predictions = json.loads((a.root/'inference/predicted_directives.json').read_text())
    assert inference['status'] == 'INFERENCE_COMPLETED' and inference['checkpoint_step'] == 600
    assert len(predictions) == 18 and len({r['context_id'] for r in predictions}) == 18
    assert all(r['film_enabled'] and len(r['directives']) == 25 for r in predictions)
    manifest = json.loads((a.root/'synthesis/manifest.json').read_text())
    assert manifest['predictions_sha256'] == synth.sha(a.root/'inference/predicted_directives.json')
    assert manifest['inference_manifest_sha256'] == synth.sha(a.root/'inference/manifest.json')
    assert manifest['runner_sha256'] == synth.sha(Path(synth.__file__))
    assert manifest['source_sha256'] == synth.sha(repo/'evaluation_set/GAN/code_to_memory_outputs/meta/GAN_placeholders.cpp')
    assert len(manifest['jobs']) == 18
    results = {}
    for job in manifest['jobs']:
        d = a.root/'synthesis/jobs'/job['job_id']
        for name, digest in job['files'].items():
            path = d/name if (d/name).exists() else repo/'evaluation_set/GAN'/name
            assert synth.sha(path) == digest, (job['job_id'], name)
        result = json.loads((d/'result.json').read_text())
        assert result['status'] == 'success'
        xml = d/'hls_project/solution1/syn/report/forward_propagation_csynth.xml'
        assert synth.sha(xml) == result['report_sha256']
        for key, value in synth.parse_report(xml, job).items():
            assert same(value, result[key]), (job['job_id'], key)
        results[job['job_id']] = result
    rows = json.loads((a.root/'synthesis/request_results.json').read_text())
    assert len(rows) == 18
    by_id = {r['context_id']: r for r in predictions}
    for row in rows:
        baseline, prediction = results[row['baseline_job']], results[row['predicted_job']]
        assert row['directives'] == by_id[row['context_id']]['directives']
        assert baseline['part'] == prediction['part'] == row['part']
        assert baseline['clock_ns'] == prediction['clock_ns'] == row['clock_ns']
        assert same(row['speedup'], baseline['latency_cycles']/prediction['latency_cycles'])
        assert same(row['mean_normalized_utilization_pct'], prediction['mean_normalized_utilization_pct'])
        assert row['budget_met'] == all(prediction['resources'][k] <= round(prediction['capacities'][k]*row['budget_pct']/100)
                                       for k in synth.RESOURCES)
    print('Verified {} artifact hashes, 18 XML reports, and all 18 request comparisons.'.format(len(index)))


if __name__ == '__main__':
    main()
