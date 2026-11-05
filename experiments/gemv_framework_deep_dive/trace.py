"""Read-only selected-position hooks and exact candidate-prefix replay."""
from __future__ import annotations

import math

import numpy as np
import torch

from LLM_branch.common import structural_xattn
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from .common import LAYERS, contract, decision_prefixes, digest_object


def equal_scores(a, b):
    # Matches the focused Stage-1 roundtrip tests' 1e-5 absolute/relative tolerance.
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)


class SelectedStates:
    """Hooks never return replacement tensors. Copy only explicitly armed positions."""
    def __init__(self, model, layers=LAYERS):
        self.model = model
        self.layer_path = structural_xattn.infer_decoder_layers_attr_name(model)
        native = structural_xattn.getattr_recursive(model, self.layer_path)
        self.positions = []
        self.context = {}
        self.index, self.vectors, self.handles = [], [], []
        for number in layers:
            layer = native[number - 1]
            if not hasattr(layer, 'post_attention_layernorm'):
                raise TypeError('Native post_attention_layernorm missing')
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                self._pre_hook(number)))
            self.handles.append(layer.register_forward_hook(self._output_hook(number)))

    def _copy(self, tensor, number, boundary):
        if not self.positions:
            return
        assert tensor.ndim == 3 and tensor.shape[0] == 1
        for site in self.positions:
            vector = tensor[0, site['token_index']].detach().float().cpu().numpy().copy()
            self.index.append(dict(self.context, **site, layer=number, boundary=boundary,
                                   vector_index=len(self.vectors), tensor_shape=list(vector.shape),
                                   native_module=f'{self.layer_path}.{number - 1}'))
            self.vectors.append(vector)

    def _pre_hook(self, number):
        def hook(module, inputs):
            self._copy(inputs[0], number, 'post_self_attention_residual')
        return hook

    def _output_hook(self, number):
        def hook(module, inputs, output):
            self._copy(output[0] if isinstance(output, tuple) else output,
                       number, 'post_decoder_residual')
        return hook

    def arm(self, context, sites, ids, tok):
        self.context = context
        self.positions = [dict(site, token_id=ids[site['token_index']],
                               token_text=tok.decode([ids[site['token_index']]])) for site in sites]

    def disarm(self):
        self.positions = []

    def close(self):
        for handle in self.handles:
            handle.remove()

    def save(self, path):
        if not self.vectors:
            raise ValueError('No vectors captured; refusing to create an empty evidence tensor')
        np.savez_compressed(path, vectors=np.stack(self.vectors).astype(np.float32))


def assert_same_prediction(first, second):
    a, b = first['candidates'][0], second['candidates'][0]
    assert a['canonical_prediction'] == b['canonical_prediction']
    assert equal_scores(a['sequence_score'], b['sequence_score'])
    for left, right in zip(a['site_debug'], b['site_debug'], strict=True):
        assert left['lhs'] == right['lhs']
        assert left['chosen_rhs'] == right['chosen_rhs']
        assert [r['rhs'] for r in left['top_candidates']] == [r['rhs'] for r in right['top_candidates']]
        for x, y in zip(left['top_candidates'], right['top_candidates'], strict=True):
            assert all(equal_scores(x[k], y[k]) for k in ('score', 'sum_logprob', 'mean_logprob'))


@torch.no_grad()
def replay(row, official, reference_assignments, model, tok, registry, trace):
    """Use production score/filter/select functions. Never generate a new trajectory."""
    natural_assignments = inference.parse_assignment_dict(official['candidates'][0]['canonical_prediction'])
    official_sites = official['candidates'][0]['site_debug']
    records = []
    for mode, assignments in [('natural', natural_assignments), ('controlled', reference_assignments)]:
        prior = {}
        for site_number, prefix in enumerate(decision_prefixes(row, tok, assignments)):
            lhs, label, ids = prefix['field'], prefix['action'], prefix['ids']
            domain = inference.get_rhs_candidates_for_lhs('gemv', lhs, registry)
            candidates = contract.filter_semantic_candidates(lhs, domain, prior, registry['gemv'])
            tensor = torch.tensor([ids], device=inference.get_first_real_device(model))
            kwargs = dict(model=model, tok=tok, base_input_ids=tensor,
                          base_attention_mask=torch.ones_like(tensor), use_structural_memory=False)
            trace.disarm()
            if mode == 'natural':
                saved = official_sites[site_number]
                assert saved['lhs'] == lhs
                scores = saved['top_candidates']
                assert set(candidates) == {item['rhs'] for item in scores}
                selected = saved['chosen_rhs']
            else:
                scores = []
                for rhs in candidates:
                    if len(candidates) == 1:
                        stats = dict(mean_logprob=0., sum_logprob=0., token_count=0)
                    else:
                        stats = inference.score_rhs_candidate_suffix(**kwargs, candidate_text=rhs + '\n')
                    scores.append(dict(rhs=rhs, score=stats['mean_logprob'], **stats))
                chosen, scores = inference.select_candidate_from_scored(
                    scores, decode_mode='greedy', sample_temperature=1., sample_top_p=1.,
                    sample_top_k=0, sample_generator=None)
                selected = chosen['rhs']
            sites = [dict(action=label, field=lhs, position_kind='decision', token_index=len(ids) - 1)]
            # Source vectors are captured once, from the first natural scoring
            # input, preserving that forward pass's sequence length and suffix.
            if mode == 'natural' and site_number == 0:
                for k in range(1, 6):
                    token_id = tok.convert_tokens_to_ids(f'<SRC_L{k}>')
                    locations = [i for i, value in enumerate(ids) if value == token_id]
                    assert len(locations) == 1
                    sites.append(dict(action=f'L{k}', field='source', position_kind='source', token_index=locations[0]))
            context = dict(request_id=row['context_id'], prefix_mode=mode,
                           prefix_ids_sha256=digest_object(ids), prior_output_ids=prefix['prior_output_ids'],
                           prior_output_sha256=digest_object(prefix['prior_output_ids']),
                           prefix_length=len(ids), candidate_first_token_offset=len(ids),
                           capture_suffix=selected + '\n',
                           candidate_token_ids=tok(selected + '\n', add_special_tokens=False)['input_ids'])
            # Compare the *same* forward input with extraction off and on, even
            # for semantically forced fields whose official score is synthetic 0.
            off = inference.score_rhs_candidate_suffix(**kwargs, candidate_text=selected + '\n')
            trace.arm(context, sites, ids, tok)
            on = inference.score_rhs_candidate_suffix(**kwargs, candidate_text=selected + '\n')
            trace.disarm()
            assert all(equal_scores(on[key], off[key]) for key in ('mean_logprob', 'sum_logprob'))
            if len(candidates) > 1:
                expected = next(s for s in scores if s['rhs'] == selected)
                assert equal_scores(on['mean_logprob'], expected['score'])
            records.append(dict(context, action=label, field=lhs, selected=selected,
                                static_candidates=domain, active_candidates=candidates, scores=scores,
                                forced_by_semantics=len(candidates) == 1,
                                margin=scores[0]['score'] - scores[1]['score'] if len(scores) > 1 else None,
                                extraction_equivalence=True))
            # This update is the frozen trajectory, NEVER the controlled argmax.
            prior[lhs.upper()] = assignments[lhs.upper()]
    return records
