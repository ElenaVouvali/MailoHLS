from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREPARED = Path('/tmp/stage1_quality_sft_v1')
PARENT_WORKTREE = Path('/tmp/MailoHLS-stage1-refinement')
V4 = Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')
OUT = ROOT / 'artifacts/stage1_qor_preference/v1_20261106'
MODEL_OUT = Path('/home/elvouvali/mailohls_runs/stage1_qor_dpo_latency_v1')
GPU = 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'
ADAPTER_SHA = '28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'
CONTRACT_SHA = '1c7a2034dba45030e51525fd415367e6fcf276cceddd89eb1132edfd66840671'
POLICY = ROOT / 'experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def rows(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as f: return [json.loads(line) for line in f if line.strip()]


def jsonl(path, records):
    Path(path).write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))


def authenticate():
    c = read(V4 / 'training_contract.json')
    assert sha(V4 / 'adapter_model.safetensors') == ADAPTER_SHA
    assert sha(V4 / 'training_contract.json') == CONTRACT_SHA
    assert c['stage'] == 'stage1' and c['objective'] == 'PARETO_LATENCY'
    assert sha(V4 / 'directive_domain_registry.json') == c['directive_domain_registry_sha256']
    assert c['model_revision'] == 'ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912'
    from LLM_branch.inference.eval_stage1_stage2_stage3 import load_checkpoint_tokenizer
    return c, load_checkpoint_tokenizer(str(V4), c)
