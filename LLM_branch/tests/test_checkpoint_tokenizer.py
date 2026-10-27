"""Checkpoint token identity must survive permutations with the same ID set."""
import copy

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from LLM_branch.inference import eval_stage1_stage2_stage3 as inference


def base_tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>")),
        unk_token="<unk>", eos_token="<eos>",
    )


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    tokens = ["<DEV=U200>", "<DEV=ZCU104>"]
    saved = base_tokenizer()
    saved.add_special_tokens({"additional_special_tokens": tokens})
    saved.save_pretrained(tmp_path)
    mapping = {s: saved.convert_tokens_to_ids(s) for s in tokens}
    contract = dict(model="local-test-base", tokenizer="local-test-base", model_revision="pinned",
                    special_tokens=tokens, special_token_ids=sorted(mapping.values()),
                    tokenizer_size=len(saved))
    original = inference.AutoTokenizer.from_pretrained

    def load(source, **kwargs):
        if str(source) == "local-test-base":
            assert kwargs["revision"] == "pinned"
            return base_tokenizer()
        return original(source, **kwargs)

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", load)
    return tmp_path, contract, mapping


def test_load_saved_legacy_tokenizer(checkpoint):
    path, contract, mapping = checkpoint
    tok = inference.load_checkpoint_tokenizer(str(path), contract)
    assert {s: tok.convert_tokens_to_ids(s) for s in mapping} == mapping
    assert tok.pad_token == tok.eos_token
    assert tok.padding_side == "right"


def test_reject_permutation_even_when_id_set_and_vocab_size_match(checkpoint):
    path, contract, mapping = checkpoint
    swapped = base_tokenizer()
    swapped.add_special_tokens({"additional_special_tokens": list(reversed(contract["special_tokens"]))})
    assert len(swapped) == contract["tokenizer_size"]
    assert sorted(swapped.convert_tokens_to_ids(contract["special_tokens"])) == contract["special_token_ids"]
    swapped.save_pretrained(path)
    with pytest.raises(ValueError, match="token-to-ID mapping"):
        inference.load_checkpoint_tokenizer(str(path), contract)


def test_explicit_mapping_does_not_need_base_tokenizer(checkpoint, monkeypatch):
    path, contract, mapping = checkpoint
    contract["special_token_to_id"] = mapping
    monkeypatch.setattr(inference, "build_tokenizer", lambda *a, **kw: pytest.fail("must use saved mapping"))
    inference.load_checkpoint_tokenizer(str(path), contract)


def test_reconstruct_missing_saved_tokenizer_from_recorded_order(checkpoint, tmp_path):
    _, contract, mapping = checkpoint
    missing = tmp_path / "no-tokenizer"
    missing.mkdir()
    tok = inference.load_checkpoint_tokenizer(str(missing), contract)
    assert {s: tok.convert_tokens_to_ids(s) for s in mapping} == mapping


def test_default_registration_matches_training_order(checkpoint):
    tok = inference.build_tokenizer("local-test-base", revision="pinned")
    tokens = inference.mailohls_contract.TARGET_PLATFORM_TOKENS
    ids = tok.convert_tokens_to_ids(tokens)
    assert ids == sorted(ids)


@pytest.mark.parametrize("field,value", [("tokenizer_size", 999), ("special_token_ids", [9, 10])])
def test_reject_corrupt_contract(checkpoint, field, value):
    path, contract, mapping = checkpoint
    contract = copy.deepcopy(contract)
    contract[field] = value
    with pytest.raises(ValueError):
        inference.load_checkpoint_tokenizer(str(path), contract)
