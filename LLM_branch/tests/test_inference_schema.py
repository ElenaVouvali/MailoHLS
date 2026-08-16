from LLM_branch.inference.eval_stage1_stage2_stage3 import (
    directive_schema_signature,
)


def test_directive_schema_signature_accepts_lk_anchor():
    text = (
        "<L1>\n"
        "auto{_PIPE_L1} = 1\n"
        "auto{_UNROLL_L1} = 0"
    )

    assert directive_schema_signature(text) == [
        ("anchor", "L1"),
        ("assignment", "AUTO{_PIPE_L1}"),
        ("assignment", "AUTO{_UNROLL_L1}"),
    ]