"""Regression test for Stage-1 special-token adapter persistence."""

from __future__ import annotations

import copy

import pytest
import torch

peft = pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from LLM_branch.train.train_SFT_xattn_new import (
    save_mailohls_adapter,
    score_rhs_candidate_batch,
    score_rhs_candidate_suffix,
)


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "<pad>": 1, "alpha": 2, "beta": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
    )


def _base_model(tie_word_embeddings: bool) -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            use_cache=False,
            tie_word_embeddings=tie_word_embeddings,
        )
    )


def test_candidate_batch_matches_scalar_scoring():
    torch.manual_seed(123)
    tokenizer = _tokenizer()
    model = _base_model(False).eval()
    base_input_ids = torch.tensor([[2, 3]])
    base_attention_mask = torch.ones_like(base_input_ids)
    candidate_texts = ["alpha", "beta alpha", "alpha beta alpha"]

    scalar = [
        score_rhs_candidate_suffix(
            model=model,
            tok=tokenizer,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            candidate_text=text,
        )
        for text in candidate_texts
    ]
    batched = score_rhs_candidate_batch(
        model=model,
        tok=tokenizer,
        base_input_ids=base_input_ids,
        base_attention_mask=base_attention_mask,
        candidate_texts=candidate_texts,
    )
    for expected, actual in zip(scalar, batched):
        assert actual["sum_logprob"] == pytest.approx(
            expected["sum_logprob"], abs=1e-6
        )
        assert actual["mean_logprob"] == pytest.approx(
            expected["mean_logprob"], abs=1e-6
        )


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_stage1_adapter_roundtrip(tmp_path, tie_word_embeddings):
    torch.manual_seed(123)
    tokenizer = _tokenizer()
    special_tokens = ["<MAILOHLS_A>", "<MAILOHLS_B>"]
    assert tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    ) == 2
    special_ids = sorted(tokenizer.convert_tokens_to_ids(special_tokens))

    base = _base_model(tie_word_embeddings)
    base.resize_token_embeddings(len(tokenizer))
    weights_tied = (
        base.get_input_embeddings().weight.data_ptr()
        == base.get_output_embeddings().weight.data_ptr()
    )
    assert weights_tied is tie_word_embeddings
    trainable_token_spec = special_ids
    initial_state = copy.deepcopy(base.state_dict())
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            trainable_token_indices=trainable_token_spec,
        ),
    )
    token_delta_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name and parameter.requires_grad
    )
    assert token_delta_params == len(special_ids) * 32

    input_ids = torch.tensor([[special_ids[0], 2, special_ids[1], 3]])
    labels = input_ids.clone()
    for special_id in special_ids:
        labels[labels == special_id] = -100
    ordinary_output_before = (
        model.get_output_embeddings().weight[2].detach().clone()
    )
    output_weights_before = (
        model.get_output_embeddings().weight.detach().clone()
    )
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    token_delta_gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name
    }
    assert any(
        "embed_tokens" in name
        and gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for name, gradient in token_delta_gradients.items()
    )
    assert not any(
        "lm_head" in name for name in token_delta_gradients
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(
        ordinary_output_before,
        model.get_output_embeddings().weight[2],
        atol=0.0,
        rtol=0.0,
    )
    if not weights_tied:
        torch.testing.assert_close(
            output_weights_before,
            model.get_output_embeddings().weight,
            atol=0.0,
            rtol=0.0,
        )

    model.eval()
    with torch.no_grad():
        logits_before_save = model(input_ids=input_ids).logits

    adapter_dir = tmp_path / "stage1_adapter"
    save_mailohls_adapter(
        model,
        tokenizer,
        str(adapter_dir),
        {"schema": "mailohls-training-contract-test-v1"},
    )

    reloaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(adapter_dir)
    fresh_base = _base_model(tie_word_embeddings)
    fresh_base.resize_token_embeddings(len(reloaded_tokenizer))
    fresh_base.load_state_dict(initial_state)
    assert (
        fresh_base.get_input_embeddings().weight.data_ptr()
        == fresh_base.get_output_embeddings().weight.data_ptr()
    ) is tie_word_embeddings
    reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir).eval()
    if not tie_word_embeddings:
        assert (
            reloaded.get_input_embeddings().weight.data_ptr()
            != reloaded.get_output_embeddings().weight.data_ptr()
        )

    with torch.no_grad():
        logits_after_reload = reloaded(input_ids=input_ids).logits

    torch.testing.assert_close(
        logits_before_save,
        logits_after_reload,
        atol=1e-5,
        rtol=1e-5,
    )
