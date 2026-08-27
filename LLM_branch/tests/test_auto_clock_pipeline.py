"""AUTO selector boundary regressions for the frozen Stage-2 decode path."""

from __future__ import annotations

import json

from LLM_branch.clock_adapt import infer_auto
from LLM_branch.train import train_SFT_xattn_new as stage2


def _bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def test_specified_clock_request_remains_byte_identical():
    request = {
        "frequency_mode": "specified",
        "device": "xczu7ev",
        "selected_clock_period": 5.0,
        "selected_clock_period_ns": 5.0,
        "resource_budget": {"BRAM_18K": 100, "DSP": 200, "FF": 300, "LUT": 400},
        "avail_bram": 100,
        "avail_dsp": 200,
        "avail_ff": 300,
        "avail_lut": 400,
    }
    before = _bytes(request)
    converted = infer_auto.auto_to_specified_request(request, object(), object())
    assert _bytes(converted) == before
    assert _bytes(request) == before


def test_auto_request_uses_unchanged_stage2_prompt_and_one_constrained_decode(monkeypatch):
    request = {
        "frequency_mode": "auto",
        "device": "xczu7ev",
        "resource_budget": {"BRAM_18K": 100, "DSP": 200, "FF": 300, "LUT": 400},
        "avail_bram": 100,
        "avail_dsp": 200,
        "avail_ff": 300,
        "avail_lut": 400,
        "source": "L1: for (;;) {}",
    }
    monkeypatch.setattr(infer_auto, "select_clock", lambda *_args, **_kwargs: (5.0, None))
    calls = []

    def stage2_build_prompt(specified):
        calls.append(("prompt", dict(specified)))
        return stage2.build_prompt(specified["source"], "PARETO_ADP", row=specified)

    def constrained_directive_decode(prompt, specified):
        calls.append(("decode", prompt, dict(specified)))
        return "auto{_PIPE_L1} = 1"

    specified, prompt, decoded = infer_auto.auto_select_then_decode(
        request,
        selector=object(),
        memory_pack=object(),
        build_prompt=stage2_build_prompt,
        constrained_decode=constrained_directive_decode,
    )
    assert specified["frequency_mode"] == "specified"
    assert specified["selected_clock_period_ns"] == 5.0
    expected_request = dict(request)
    expected_request.update({
        "frequency_mode": "specified",
        "selected_clock_period": 5.0,
        "selected_clock_period_ns": 5.0,
    })
    assert prompt == stage2.build_prompt(
        expected_request["source"], "PARETO_ADP", row=expected_request
    )
    assert decoded == "auto{_PIPE_L1} = 1"
    assert [call[0] for call in calls] == ["prompt", "decode"]
