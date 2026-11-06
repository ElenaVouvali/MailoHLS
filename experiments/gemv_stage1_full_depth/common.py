from __future__ import annotations

import hashlib
import json
from pathlib import Path

from LLM_branch.common import mailohls_contract as contract

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'artifacts/gemv_stage1_full_depth/v1_20261106'
ADAPTER = Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')
GPU = 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'
EXPECTED_ADAPTER = '28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'
EXPECTED_CONTRACT = '1c7a2034dba45030e51525fd415367e6fcf276cceddd89eb1132edfd66840671'
LAYERS = tuple(range(1, 33))  # one-based, native modules layers[0..31]
POLICY = ROOT / 'experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json'
HISTORY = ROOT / 'experiments/stage1_quality_sft/results/gemv_v4_20261105_report.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_object(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def prompt_parts(row, tok):
    """Exactly predict_case's section-wise tokenization and fixed clock context."""
    sections = contract.build_prompt_sections(row['input'], row.get('objective', row.get('obj_mode')),
                                             contract.target_prompt_fields(row))
    base = [token for section in sections for token in tok(section, add_special_tokens=False)['input_ids']]
    clock = contract.selected_clock_response_token_ids(row, tok)
    return ''.join(sections) + contract.selected_clock_response_prefix(row), base + clock


def decision_prefixes(row, tok, assignments):
    """Replay fixed prefix tokens only; selection stays in the production decoder."""
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference

    _, ids = prompt_parts(row, tok)
    prompt_length = len(ids)
    prior_ids, current = [], None
    for label, lhs in inference.extract_ordered_lhs_plan(row['input']):
        if label != current:
            chunk = tok(f'{inference.target_placeholder_token(label)}\n', add_special_tokens=False)['input_ids']
            ids += chunk
            prior_ids += chunk
            current = label
        chunk = tok(f'{lhs} = ', add_special_tokens=False)['input_ids']
        ids += chunk
        prior_ids += chunk
        yield dict(action=label, field=lhs, ids=list(ids), prior_output_ids=list(prior_ids),
                   prompt_length=prompt_length)
        chunk = tok(assignments[lhs.upper()] + '\n', add_special_tokens=False)['input_ids']
        ids += chunk
        prior_ids += chunk
