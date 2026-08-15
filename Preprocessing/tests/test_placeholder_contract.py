from pathlib import Path

import pytest

from GNN_branch.insert_placeholders import insert_placeholders
from Preprocessing.create_jsonl import Action, source_with_placeholders


def _write_source(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "kernel.cpp"
    path.write_text(text, encoding="utf-8")
    return path


def test_allowed_labels_strip_inactive_marker_but_preserve_code(tmp_path):
    source = _write_source(
        tmp_path,
        "  L1: for (int i = 0; i < 4; ++i) {}\n"
        "  /* L2: */ for (int j = 0; j < 8; ++j) {}\n",
    )
    output = "".join(insert_placeholders(source, allowed_labels={"l1"}))

    assert "auto{_PIPE_L1}" in output
    assert "auto{_UNROLL_L1}" in output
    assert "L2:" not in output
    assert "for (int j = 0; j < 8; ++j) {}" in output
    assert "auto{_PIPE_L2}" not in output


def test_omitted_allowed_labels_preserves_legacy_behavior(tmp_path):
    source = _write_source(
        tmp_path,
        "L1: for (int i = 0; i < 4; ++i) {}\n"
        "L2: for (int j = 0; j < 8; ++j) {}\n",
    )
    output = "".join(insert_placeholders(source))
    assert "auto{_PIPE_L1}" in output
    assert "auto{_PIPE_L2}" in output
    assert "L2:" in output


def test_source_contract_accepts_exact_loop_and_array_sites(tmp_path):
    source = _write_source(
        tmp_path,
        "L1: for (int i = 0; i < 4; ++i) {}\n"
        "L2: float buffer[16];\n",
    )
    actions = [
        Action("L1", "loop", None, trip_count=4),
        Action("L2", "array", None, dimensions=frozenset({1})),
    ]
    template = source_with_placeholders(source, actions)
    assert "auto{_PIPE_L1}" in template
    assert "auto{_UNROLL_L1}" in template
    assert "auto{_ARRAY_T_L2}" in template
    assert "auto{_ARRAY_F_L2}" in template
    assert "auto{_ARRAY_D_L2}" in template


def test_source_contract_rejects_missing_and_extra_kind_sites(tmp_path):
    source = _write_source(tmp_path, "L1: float buffer[16];\n")
    actions = [Action("L1", "loop", None, trip_count=16)]
    with pytest.raises(ValueError, match="missing=.*PIPE_L1.*extra=.*ARRAY"):
        source_with_placeholders(source, actions)
