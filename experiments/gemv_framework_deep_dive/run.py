"""Launch the unchanged production entrypoint and intercept its completed cases."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from LLM_branch.common import directive_domains, source_tool_policy
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from experiments.stage1_quality_sft.gpu_preflight import check
from .common import (ADAPTER, DEFAULT_OUT, GPU, PACKAGE, POLICY, contract,
                     digest_object, read_json, sha, write_json)
from .prepare import authenticate
from .trace import SelectedStates, assert_same_prediction, replay


def run(out):
    manifest = read_json(out / 'manifest.json')
    assert sha(out / 'requests.jsonl') == manifest['requests_file_sha256']
    assert sha(PACKAGE / 'inputs/gemv.c') == manifest['source_sha256']
    assert sha(PACKAGE / 'inputs/kernel_info.txt') == manifest['metadata_sha256']
    assert sha(out / 'effective_gemv_domains.json') == manifest['effective_domains_sha256']
    if ((out / 'official.jsonl').exists() or (out / 'hidden_states.npz').exists()
            or any(out.glob('R*_official.json'))):
        raise FileExistsError('Use a fresh --out to preserve existing inference evidence')
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Map CUDA_VISIBLE_DEVICES to the mandatory UUID'
        gpu = check(GPU)
        assert gpu['name'] == 'Tesla V100S-PCIE-32GB'
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert torch.cuda.get_device_name(0) == 'Tesla V100S-PCIE-32GB'
        assert torch.cuda.get_device_capability(0) == (7, 0)
    except (AssertionError, RuntimeError) as error:
        write_json(out / 'launch_status.json', dict(status='BLOCKED_GPU', blocker=str(error),
                   cuda_available=torch.cuda.is_available(), cuda_count=torch.cuda.device_count(),
                   official_requests_completed=0, model_loaded=False))
        raise
    authenticate()
    frozen_checkpoint = read_json(out / 'checkpoint.json')
    assert all(sha(ADAPTER / name) == value for name, value in frozen_checkpoint['files'].items())
    state = {}
    original_load, original_predict = inference.load_stage_model, inference.predict_case
    history = read_json(out / 'historical_reference.json')['requests']
    records, official_rows, validations = [], [], []
    controlled_prefixes = {}
    actions = directive_domains.load_source_actions('gemv', PACKAGE / 'inputs')
    policy = source_tool_policy.load_policy(POLICY)

    def loaded(args, tok):
        model = original_load(args, tok)
        model.eval()
        assert not getattr(model, 'initialized_structural_xattn', False)
        state.update(model=model, tok=tok)
        return model

    def predict(**kwargs):
        request_id = kwargs['case'].platform_row['context_id']
        row = kwargs['case'].platform_row
        expected = next(r for r in manifest['requests'] if r['request_id'] == request_id)
        assert digest_object(row) == expected['request_sha256']
        # Remove the preceding request's hooks before the authentic decode.
        previous_trace = state.get('trace')
        if previous_trace is not None:
            previous_trace.close()
        first = original_predict(**kwargs)
        write_json(out / f'{request_id}_official.json', first)
        assignment = inference.parse_assignment_dict(first['candidates'][0]['canonical_prediction'])
        assert len(assignment) == 13
        validation = dict(request_id=request_id, complete=True)
        for label, validator in [('source', lambda: source_tool_policy.validate_source_assignment(assignment, actions)),
                                 ('source_tool_v2', lambda: source_tool_policy.validate_complete_assignment(assignment, actions, policy))]:
            try:
                validator()
                validation[label] = dict(valid=True)
            except ValueError as error:
                validation[label] = dict(valid=False, reason=str(error))
        if request_id in history:
            if assignment != inference.parse_assignment_dict(history[request_id]['prediction']):
                write_json(out / 'historical_mismatch.json', dict(request_id=request_id,
                    new_prediction=first['candidates'][0]['canonical_prediction'],
                    historical_prediction=history[request_id]['prediction'],
                    status='STOPPED_BEFORE_TRACE_ANALYSIS'))
                raise AssertionError(f'{request_id} differs from historical evidence; inspect saved output and contracts')
            validation['historical_reproduction'] = True
        if request_id == 'R1':
            state['reference_assignments'] = assignment
        # Full decoder equivalence with read-only hooks attached, collection disarmed.
        trace = state['trace'] = SelectedStates(state['model'])
        if previous_trace is not None:
            trace.index, trace.vectors = previous_trace.index, previous_trace.vectors
        second = original_predict(**kwargs)
        assert_same_prediction(first, second)
        validation['full_decode_hook_equivalence'] = True
        new_records = replay(row, first, state['reference_assignments'], kwargs['model'],
                             kwargs['tok'], kwargs['directive_domain_registry'], trace)
        for record in new_records:
            if record['prefix_mode'] == 'controlled':
                previous = controlled_prefixes.setdefault(record['field'], record['prior_output_ids'])
                assert previous == record['prior_output_ids']
        records.extend(new_records)
        official_rows.append(first)
        validations.append(validation)
        # Incremental evidence survives later failures; production main writes its
        # official JSONL only when all cases finish.
        write_json(out / f'{request_id}_official.json', first)
        write_json(out / 'candidate_scores.json', records)
        write_json(out / 'validation.json', validations)
        trace.save(out / 'hidden_states.npz')
        write_json(out / 'hidden_index.json', trace.index)
        return first

    argv = ['--stage', 'stage1', '--adapter_dir', str(ADAPTER),
            '--directive_domain_registry_json', str(ADAPTER / 'directive_domain_registry.json'),
            '--application_dataset_dir', str(PACKAGE / 'inputs'),
            '--input_jsonl', str(out / 'requests.jsonl'), '--output_jsonl', str(out / 'official.jsonl'),
            '--admissibility_policy', 'legacy', '--score_reduction', 'mean', '--num_samples', '1',
            '--sample_seed', '123', '--max_prompt_tokens', '7168', '--debug_topk', '10000',
            '--candidate_max_prefix_tokens', '0', '--candidate_keep_head_tokens', '256']
    write_json(out / 'launch.json', dict(entrypoint='LLM_branch.inference.eval_stage1_stage2_stage3',
               argv=argv, gpu=gpu, manifest_sha256=sha(out / 'manifest.json')))
    saved_argv = sys.argv
    try:
        write_json(out / 'launch_status.json', dict(status='RUNNING', official_requests_completed=0))
        inference.load_stage_model, inference.predict_case = loaded, predict
        sys.argv = [inference.__file__, *argv]
        with torch.inference_mode():
            inference.main()
        trace = state['trace']
        assert len(official_rows) == 4 and len(trace.index) == 992
        assert len({tuple(item['tensor_shape']) for item in trace.index}) == 1
        assert trace.index[0]['tensor_shape'] == [4096]
        assert all(sha(ADAPTER / name) == value for name, value in frozen_checkpoint['files'].items())
        write_json(out / 'launch_status.json', dict(status='COMPLETED', official_requests_completed=4,
                   vector_count=len(trace.index), tensor_shape=[992, 4096],
                   instrumentation_equivalent=True, controlled_prefixes_identical=True,
                   checkpoint_files_unchanged=True))
    except Exception as error:
        write_json(out / 'launch_status.json', dict(status='FAILED', error_type=type(error).__name__,
                   error=str(error), official_requests_completed=len(official_rows),
                   partial_evidence_preserved=True))
        raise
    finally:
        inference.load_stage_model, inference.predict_case = original_load, original_predict
        sys.argv = saved_argv
        if state.get('trace') is not None:
            state['trace'].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.out)


if __name__ == '__main__':
    main()
