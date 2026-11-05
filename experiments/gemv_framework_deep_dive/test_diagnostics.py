"""CPU fixtures validate hooks and real decoder alignment; no V4 CPU inference."""
import json

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from LLM_branch.common import directive_domains, source_tool_policy
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from .analyze import pairwise
from .common import DEFAULT_OUT, PACKAGE, POLICY, contract, decision_prefixes, digest_object, prompt_parts, read_json, sha
from .prepare import authenticate, requests
from .trace import SelectedStates, assert_same_prediction, replay


@pytest.fixture(scope='module')
def evidence():
    c, tok = authenticate()
    rows = [json.loads(line) for line in (DEFAULT_OUT / 'requests.jsonl').read_text().splitlines()]
    historical = read_json(DEFAULT_OUT / 'historical_reference.json')['requests']
    return c, tok, rows, historical


def test_frozen_requests_and_authentication(evidence):
    c, tok, rows, history = evidence
    assert rows == requests((PACKAGE / 'inputs/gemv.c').read_text())
    loaded = inference.load_inference_cases_jsonl(str(DEFAULT_OUT / 'requests.jsonl'))
    assert [case.obj_mode for case in loaded] == ['PARETO_LATENCY'] * 4
    assert len({r['input'] for r in rows}) == 1
    assert [r['context_id'] for r in rows] == ['R1', 'R2', 'R3', 'R4']
    assert [rows[1][k] for k in ('avail_bram', 'avail_dsp', 'avail_ff', 'avail_lut')] == [125, 346, 92160, 46080]
    assert rows[3]['device'] == 'xcu200-fsgd2104-2-e'
    assert [rows[3][k] for k in ('avail_bram', 'avail_dsp', 'avail_ff', 'avail_lut')] == [4320, 6840, 2364480, 1182240]
    assert '<CLK=3P33NS>' in prompt_parts(rows[2], tok)[0]
    assert c['objective'] == 'PARETO_LATENCY' and c['prompt_schema_version'] == 2
    assert [len(prompt_parts(r, tok)[1]) for r in rows] == [604, 597, 608, 608]
    for r, frozen in zip(rows, read_json(DEFAULT_OUT / 'manifest.json')['requests'], strict=True):
        assert sha(DEFAULT_OUT / 'prompts' / (r['context_id'] + '.txt')) == frozen['prompt_sha256']


def test_historical_prompts_and_policy_v2(evidence):
    _, tok, rows, history = evidence
    old = [json.loads(line) for line in (PACKAGE.parent / 'stage1_quality_sft/results/gemv_v4_20261105_requests.jsonl').read_text().splitlines()]
    actions = directive_domains.load_source_actions('gemv', PACKAGE / 'inputs')
    for current, previous in [(rows[0], old[0]), (rows[3], old[1])]:
        assert prompt_parts(current, tok) == prompt_parts(previous, tok)
        assignments = inference.parse_assignment_dict(history[current['context_id']]['prediction'])
        assert len(assignments) == 13
        source_tool_policy.validate_source_assignment(assignments, actions)
        source_tool_policy.validate_complete_assignment(assignments, actions, source_tool_policy.load_policy(POLICY))
    # Explicit all-dimensions semantics: factor applies to each dimension.
    a = directive_domains.SourceAction('array', dimensions=((1, 8), (2, 32)))
    assert source_tool_policy.tuple_rejection(('block', '16', '0'), a, source_tool_policy.load_policy(POLICY))
    assert source_tool_policy.tuple_rejection(('block', '8', '0'), a, source_tool_policy.load_policy(POLICY)) is None


def test_native_llama_boundaries_and_noninterference():
    torch.manual_seed(13)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2,
                         attention_dropout=0., use_cache=False)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).cpu().eval()  # small random CPU fixture, NOT checkpoint inference
    ids = torch.tensor([[1, 4, 8, 3, 7]])
    with torch.no_grad():
        off = model(ids).logits
        oracle, handles = {}, []
        for number in (8, 16, 24, 32):
            layer = model.model.layers[number - 1]
            def before(module, args, n=number):
                oracle[(n, 'input')] = args[0].clone()
            def attention(module, args, output, n=number):
                oracle[(n, 'attention')] = output[0].clone()
            def mlp(module, args, output, n=number):
                oracle[(n, 'mlp')] = output.clone()
            handles += [layer.register_forward_pre_hook(before), layer.self_attn.register_forward_hook(attention), layer.mlp.register_forward_hook(mlp)]
        trace = SelectedStates(model)
        class Tok:
            def decode(self, ids):
                return str(ids[0])
        trace.arm({'request_id': 'fixture'}, [{'token_index': 3, 'action': 'L5', 'field': 'fixture'}], ids[0].tolist(), Tok())
        on = model(ids).logits
        assert torch.equal(on, off)
        assert len(trace.vectors) == 8
        for record, vector in zip(trace.index, trace.vectors, strict=True):
            n = record['layer']
            u = oracle[(n, 'input')] + oracle[(n, 'attention')]
            expected = u if record['boundary'] == 'post_self_attention_residual' else u + oracle[(n, 'mlp')]
            np.testing.assert_array_equal(vector, expected[0, 3].numpy())
            assert record['tensor_shape'] == [16]
        trace.close()
        for handle in handles:
            handle.remove()


def test_real_decoder_prefix_replay_alignment_and_control(evidence, monkeypatch):
    _, tok, rows, historical = evidence
    source = rows[0]['input']
    registry = {'gemv': read_json(DEFAULT_OUT / 'effective_gemv_domains.json')}
    observed = []
    def scores(**kwargs):
        observed.append(kwargs['base_input_ids'][0].tolist())
        rhs = kwargs['candidate_text'].strip()
        value = {'block': -1., 'cyclic': -2., 'complete': -3., 'none': -4., '2': -1., '1': -2., '0': -3.}.get(rhs, -10.)
        return dict(mean_logprob=value, sum_logprob=value * 2, token_count=2)
    monkeypatch.setattr(inference, 'score_rhs_candidate_suffix', scores)
    model = torch.nn.Linear(1, 1).cpu().eval()
    result = inference.constrained_decode_rhs_by_candidate_scoring(
        model=model, tok=tok, prompt_ids=prompt_parts(rows[0], tok)[1], source_text=source,
        kernel_name='gemv', directive_domain_registry=registry, debug_topk=10000)
    assignments = inference.parse_assignment_dict(result['canonical_prediction'])
    prefix_ids = {tuple(p['ids']) for p in decision_prefixes(rows[0], tok, assignments)}
    assert len(assignments) == 13 and len(prefix_ids) == 13
    assert all(tuple(ids) in prefix_ids for ids in observed)
    assert len(set(map(tuple, observed))) == sum(not r['forced_by_semantics'] for r in result['site_debug'])
    official = {'candidates': [dict(result)]}
    assert_same_prediction(official, official)
    reference = inference.parse_assignment_dict(historical['R1']['prediction'])
    frozen = [[p['prior_output_ids'] for p in decision_prefixes(row, tok, reference)] for row in rows]
    assert all(prefixes == frozen[0] for prefixes in frozen)
    class TraceFixture:
        def disarm(self): pass
        def arm(self, *args): pass
    record = replay(rows[0], official, reference, model, tok, registry, TraceFixture())
    assert len(record) == 26
    controlled = [r for r in record if r['prefix_mode'] == 'controlled']
    assert [r['prior_output_ids'] for r in controlled] == frozen[0]
    assert all(r['extraction_equivalence'] for r in record)


def test_float32_pairwise_metrics_and_prefix_guard():
    index = [dict(request_id=f'R{i}', prefix_mode='controlled', action='L5', field='unroll',
                  layer=8, boundary='post_decoder_residual', vector_index=i - 1, prior_output_ids=[4, 5]) for i in range(1, 5)]
    vectors = np.asarray([[3, 4], [3, 4], [6, 8], [0, 5]], dtype=np.float32)
    rows = pairwise(vectors, index)
    assert rows[0]['cosine'] == pytest.approx(1.)
    assert rows[0]['relative_l2'] == 0.
    assert rows[1]['relative_l2'] == pytest.approx(1.)
    assert rows[1]['relative_norm_difference'] == pytest.approx(1.)
    index[-1]['prior_output_ids'] = [99]
    with pytest.raises(AssertionError):
        pairwise(vectors, index)


def test_launcher_stops_before_loader_on_gpu_failure(tmp_path, monkeypatch):
    from . import run as launcher
    from .common import GPU
    for name in ('manifest.json', 'requests.jsonl', 'effective_gemv_domains.json'):
        (tmp_path / name).write_bytes((DEFAULT_OUT / name).read_bytes())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', GPU)
    def unavailable(_):
        raise RuntimeError('fixture NVIDIA driver inaccessible')
    monkeypatch.setattr(launcher, 'check', unavailable)
    monkeypatch.setattr(launcher, 'authenticate', lambda: pytest.fail('GPU check must precede model authentication/loading'))
    monkeypatch.setattr(inference, 'load_stage_model', lambda *a, **k: pytest.fail('must not load weights'))
    with pytest.raises(RuntimeError, match='fixture NVIDIA'):
        launcher.run(tmp_path)
    status = read_json(tmp_path / 'launch_status.json')
    assert status['official_requests_completed'] == 0 and status['model_loaded'] is False
    assert not (tmp_path / 'official.jsonl').exists()


def test_saved_gpu_evidence_shape_and_semantic_alignment(evidence):
    path = DEFAULT_OUT / 'hidden_states.npz'
    if not path.exists():
        pytest.skip('Actual GPU evidence has not been collected')
    _, tok, requests, history = evidence
    assert read_json(DEFAULT_OUT / 'launch_status.json')['status'] == 'COMPLETED'
    vectors = np.load(path)['vectors']
    assert vectors.shape == (992, 4096) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    index = read_json(DEFAULT_OUT / 'hidden_index.json')
    assert [r['vector_index'] for r in index] == list(range(992))
    official = [json.loads(line) for line in (DEFAULT_OUT / 'official.jsonl').read_text().splitlines()]
    assignments = {r['context_id']: inference.parse_assignment_dict(r['candidates'][0]['canonical_prediction']) for r in official}
    prefixes = {}
    for row in requests:
        for mode in ('natural', 'controlled'):
            values = assignments[row['context_id']] if mode == 'natural' else assignments['R1']
            for p in decision_prefixes(row, tok, values):
                prefixes[(row['context_id'], mode, p['field'])] = p
    for r in index:
        assert r['layer'] in (8, 16, 24, 32) and r['tensor_shape'] == [4096]
        if r['position_kind'] == 'source':
            expected_id = tok.convert_tokens_to_ids(f'<SRC_{r["action"]}>')
            assert r['token_id'] == expected_id
        else:
            prefix = prefixes[(r['request_id'], r['prefix_mode'], r['field'])]
            assert r['token_index'] == len(prefix['ids']) - 1
            assert r['token_id'] == prefix['ids'][-1]
            assert r['prefix_ids_sha256'] == digest_object(prefix['ids'])
            assert r['prior_output_ids'] == prefix['prior_output_ids']
    assert len(pairwise(vectors, index)) == 744
    for request in ('R1', 'R4'):
        assert assignments[request] == inference.parse_assignment_dict(history[request]['prediction'])
