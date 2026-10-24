"""Compiler top selection without an MLIR installation or kernel rewrites."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pytest

from GNN_branch import mlir_graph_gen as generator


SYMBOL = "_Z19forward_propagationPfS_"


def command_options(tmp_path):
    return dict(executable="/fake/cgeist", source=tmp_path / "network.cpp",
                kernel="forward_propagation", mlir_output=tmp_path / "network.mlir",
                action_manifest=tmp_path / "actions.tsv", cflags=[])


def test_explicit_symbol_changes_only_frontend_selector(tmp_path):
    options = command_options(tmp_path)
    original = generator.build_cgeist_command(**options)
    selected = generator.build_cgeist_command(**options, compiler_symbol=SYMBOL)
    assert "-function=forward_propagation" in original
    assert "-function=" + SYMBOL in selected
    assert [x for x in original if not x.startswith("-function=")] == [
        x for x in selected if not x.startswith("-function=")]
    args = generator.build_arg_parser().parse_args([
        str(options["source"]), "--kernel", "forward_propagation", "--compiler-symbol", SYMBOL])
    assert args.kernel == "forward_propagation" and args.compiler_symbol == SYMBOL
    assert generator.build_arg_parser().parse_args([str(options["source"])]).compiler_symbol is None


@pytest.mark.parametrize("symbol", ["", "*", "one,two", "name --flag", "name\nother"])
def test_ambiguous_or_invalid_symbol_is_rejected(tmp_path, symbol):
    with pytest.raises(ValueError, match="exactly one symbol"):
        generator.build_cgeist_command(**command_options(tmp_path), compiler_symbol=symbol)


@pytest.mark.parametrize("emitted_symbol", [SYMBOL, '"' + SYMBOL + '"', "unrelated_helper"])
def test_compilation_checks_the_requested_top_without_rewriting_source(tmp_path, emitted_symbol):
    options = command_options(tmp_path)
    options.pop("executable")
    options["source"].write_text("void forward_propagation(float *x, float *y) {}\n")
    before = options["source"].read_bytes()

    def compile_fake(argv, **kwargs):
        assert "-function=" + SYMBOL in argv
        options["mlir_output"].write_text(f"module {{\n  func.func @{emitted_symbol}() {{\n    return\n  }}\n}}\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch.object(generator, "resolve_cgeist", return_value="/fake/cgeist"), \
            patch.object(generator.subprocess, "run", side_effect=compile_fake):
        if emitted_symbol == "unrelated_helper":
            with pytest.raises(RuntimeError, match="did not emit the selected compiler symbol"):
                generator.compile_source_to_mlir(**options, cgeist="fake", compiler_symbol=SYMBOL)
        else:
            command, text = generator.compile_source_to_mlir(**options, cgeist="fake", compiler_symbol=SYMBOL)
            assert SYMBOL in text and "-function=" + SYMBOL in command
    assert options["source"].read_bytes() == before


@pytest.mark.parametrize("explicit_symbol", [False, True])
def test_graph_records_both_source_identity_and_optional_compiler_symbol(tmp_path, explicit_symbol):
    graph = nx.MultiDiGraph()
    graph.graph.update({key: "test-only" for key in generator.PERSISTED_GRAPH_METADATA_KEYS})
    graph.graph["kernel"] = "forward_propagation"
    if explicit_symbol:
        graph.graph["compiler_symbol"] = SYMBOL
    path = tmp_path / "graph.gexf"
    generator.write_gexf_deterministic(graph, path)
    metadata = json.loads(nx.read_gexf(path).graph["name"].split(":", 1)[1])
    assert metadata["kernel"] == "forward_propagation"
    assert metadata.get("compiler_symbol") == (SYMBOL if explicit_symbol else None)
