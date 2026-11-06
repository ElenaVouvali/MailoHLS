"""Final integrity receipt for frozen GEMV full-depth evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import ADAPTER, DEFAULT_OUT, EXPECTED_ADAPTER, EXPECTED_CONTRACT, read_json, sha, write_json
from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES


def main(out: Path) -> None:
    out = Path(out)
    done = read_json(out / 'completed.json')
    analysis = read_json(out / 'analysis_summary.json')
    checkpoint = read_json(out / 'checkpoint.json')
    contracts = read_json(out / 'manifest.json')
    results = read_json(out / 'candidate_scores.json')
    official = [json.loads(line) for line in (out / 'official.jsonl').read_text().splitlines()]
    requests = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
    assert len(requests) == len(results) == len(official) == 43
    assert len({r['context_id'] for r in requests}) == 43
    assert done['grid'] == 18 and done['historical'] == 1 and done['structure'] == 24
    assert analysis['capture_layers'] == list(range(1, 33))
    assert analysis['vector_shape'] == [4096]
    assert analysis['source_validation_all'] and analysis['v2_validation_all']
    assert checkpoint['adapter_sha256'] == EXPECTED_ADAPTER == sha(ADAPTER / 'adapter_model.safetensors')
    assert checkpoint['training_contract_sha256'] == EXPECTED_CONTRACT
    assert done['weights_unchanged'] and done['hidden_states_are_selected_positions']
    assert done['official_prediction_count'] == 43 and done['full_depth_layers'] == 32

    by_id = {r['context_id']: r for r in requests}
    for device_key, device in [('ZCU104', 'xczu7ev-ffvc1156-2-e'), ('XCU200', 'xcu200-fsgd2104-2-e')]:
        for clock in (10.0, 5.0, 3.33):
            for fraction in (100, 50, 25):
                rid = f'GRID_{device_key}_{str(clock).replace(".", "P")}NS_{fraction}PCT'
                row = by_id[rid]
                capacities = DEVICE_RESOURCES[device]
                assert row['device'] == device and row['clock_period'] == clock
                assert row['objective'] == row['obj_mode'] == 'PARETO_LATENCY'
                assert row['frequency_mode'] == 'specified'
                assert row['available_clock_periods'] == [3.33, 5.0, 10.0]
                expected = {key: int(round(value * fraction / 100)) for key, value in capacities.items()}
                actual = {'BRAM_18K': row['avail_bram'], 'DSP': row['avail_dsp'], 'FF': row['avail_ff'], 'LUT': row['avail_lut']}
                assert actual == expected, (rid, actual, expected)
    historical = by_id['HIST_ZCU104_10NS_20PCT']
    assert historical['device'] == 'xczu7ev-ffvc1156-2-e' and historical['clock_period'] == 10.0
    assert historical['avail_bram'] == 125 and historical['avail_dsp'] == 346
    assert historical['avail_ff'] == 92160 and historical['avail_lut'] == 46080
    baseline_source = by_id['GRID_ZCU104_10P0NS_100PCT']['input']
    assert all(by_id[rid]['input'] == baseline_source for rid in by_id if rid.startswith(('GRID_', 'HIST_')))

    for result in results:
        assignment = result['assignment']
        assert len(assignment) == 13, (result['request_id'], len(assignment))
        natural = [x for x in result['trace_records'] if x['prefix_mode'] == 'natural']
        assert len(natural) == 13
    assert all(x['validation']['source'] and x['validation']['source_tool_v2'] for x in results)

    chunks = sorted((out / 'hidden_states').glob('*.npz'))
    assert len(chunks) == 43
    index_count = 0
    for chunk in chunks:
        index = read_json(chunk.with_suffix('.index.json'))
        vectors = np.load(chunk)['vectors']
        assert vectors.ndim == 2 and vectors.shape[1] == 4096 and vectors.dtype == np.float32
        assert len(index) == vectors.shape[0]
        index_count += len(index)
    assert all(x in analysis['boundaries'] for x in ('post_self_attention_residual', 'post_decoder_residual', 'final_rms_normalized_lm_head_input'))

    patches = read_json(out / 'activation_patching.json')['records']
    assert len(patches) == 189
    active = [x for x in patches if x['status'] == 'PATCHED']
    assert len(active) == 189
    assert max(abs(x['placebo_delta']) for x in active) == 0
    assert all(x['placebo_scores'] == x['reference_scores'] if 'reference_scores' in x else len(x['placebo_scores']) == len(x['patched_scores']) for x in active)

    report_names = ['README.md', 'DIRECTIVE_GRID.md', 'STAGE1_32_LAYER_ANALYSIS.md',
                    'HARDWARE_ENCODING_AUDIT.md', 'GEMV_STRUCTURE_TARGET_ANALYSIS.md',
                    'XATTN_PLACEMENT_RECOMMENDATION.md', 'PRESENTATION_NOTES.md']
    package = Path(__file__).resolve().parent
    for name in report_names:
        assert (package / name).is_file(), name
    figures = sorted((out / 'figures').glob('fig*'))
    assert len(figures) == 18
    for stem in range(1, 10):
        assert len(list((out / 'figures').glob(f'fig{stem}_*.png'))) == 1
        assert len(list((out / 'figures').glob(f'fig{stem}_*.svg'))) == 1
    assert (out / 'stage2_interface_receipt.json').is_file()

    # Hash manifest covers compact code/reports and all generated evidence files.
    root = package.parent.parent
    receipt = {'status': 'PASS', 'request_count': len(requests), 'assignments_complete_13_fields': True,
               'source_and_v2_validation': True, 'layers_1_to_32': True, 'hidden_state_dimension': 4096,
               'hidden_state_chunks': len(chunks), 'hidden_state_vectors': index_count,
               'activation_patch_records': len(active), 'placebo_max_abs_margin_delta': max(abs(x['placebo_delta']) for x in active),
               'v4_adapter_hash_verified': True, 'training_contract_hash_verified': True,
               'official_weights_unchanged': True, 'artifact_manifest': 'artifact_manifest.json'}
    write_json(out / 'checks.json', receipt)
    files = []
    for base in [package, out]:
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name != 'artifact_manifest.json':
                files.append({'path': str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
                              'size_bytes': p.stat().st_size, 'sha256': sha(p)})
    manifest = {'run_id': out.name, 'request_count': len(requests), 'hidden_state_chunk_count': len(chunks),
                'hidden_state_vector_rows': index_count, 'artifact_count': len(files), 'files': files}
    write_json(out / 'artifact_manifest.json', manifest)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().out)
