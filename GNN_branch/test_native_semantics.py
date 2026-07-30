#!/usr/bin/env python3
"""Dataset-independent native alias, fallback, and feature regressions."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import networkx as nx

import mlir_graph_gen as graph_gen


OUTPUT = Path("/tmp/mailohls-native-analysis-smoke/native-semantics-effects.gexf")
PT_OUTPUT = OUTPUT.with_suffix(".pt")
POLYGEIST_TEST = Path(
    "/home/elvouvali/tools/Polygeist/python/test_mailohls_analysis.py"
)


def update_coverage(analysis):
    effects = analysis["effects"]
    coverage = analysis["coverage"]
    coverage["effects_with_effects"] = sum(
        item["status"] == "known" for item in effects
    )
    coverage["effects_known_no_effect"] = sum(
        item["status"] == "known_no_effect" for item in effects
    )
    coverage["effects_unknown"] = sum(
        item["status"] == "unknown" for item in effects
    )
    coverage["view_edges"] = len(analysis["views"])
    coverage["alias_queries"] = len(analysis["aliases"])
    for result in (
        "dependence", "no_dependence", "unknown", "not_applicable"
    ):
        coverage[result] = sum(
            item["result"] == result for item in analysis["dependences"]
        )


def build(module, mlir_text, analysis):
    update_coverage(analysis)
    builder = graph_gen.MlirGraphBuilder(
        module, mlir_text, [], native_analysis=analysis
    )
    return builder, builder.build().graph


def key_by_ssa(builder):
    return {
        value_id: key for key, value_id in builder.value_ids_by_key.items()
    }


def roots(builder, value_id):
    return builder.memory_roots_by_value[key_by_ssa(builder)[value_id]]


def dependence_edges(graph, query):
    source = (
        f"{query['source']['function']}:op{int(query['source']['ordinal'])}"
    )
    target = (
        f"{query['target']['function']}:op{int(query['target']['ordinal'])}"
    )
    nodes = {
        str(data.get("op_uid")): node
        for node, data in graph.nodes(data=True)
        if data.get("op_uid")
    }
    return [
        data
        for _, edge_target, data in graph.edges(
            nodes[source], data=True
        )
        if str(data.get("role")) == query["kind"]
        and int(data.get("flow", -1)) == graph_gen.FLOW_MEMORY_DEPENDENCE
        and edge_target == nodes[target]
    ]


def stamp_graph(graph, analysis):
    graph.graph.update(
        kernel="native-semantics",
        source_sha256="0" * 64,
        action_sha256="0" * 64,
        cgeist_sha256="0" * 64,
        generator_sha256="0" * 64,
        mlir_sha256="0" * 64,
        mlir_level="synthetic",
        frontend_policy="synthetic",
        action_resolutions={},
        native_analysis_schema=analysis["schema"],
        binding_sha256="0" * 64,
        native_analysis_coverage=analysis["coverage"],
    )


def python_phase():
    from mlir import ir
    from mlir._mlir_libs import _mailohls_analysis

    namespace = {}
    exec(POLYGEIST_TEST.read_text(encoding="utf-8"), namespace)
    mlir_text = namespace["MLIR"]
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(mlir_text)
        base = _mailohls_analysis.analyze_module(module.operation)
        probe = graph_gen.MlirGraphBuilder(module, mlir_text, [])
        declaration = dict(probe._discover_functions())["a_decl"]
        before_block = probe.next_block_id
        assert probe._entry_block_id(declaration) == -1
        assert probe.next_block_id == before_block

        isolated = copy.deepcopy(base)
        isolated["views"] = []
        isolated["aliases"] = []
        base_builder, _ = build(module, mlir_text, isolated)
        affine_id = base_builder.function_name_to_id["affine_cases"]
        assert [
            base_builder.value_ids_by_key[key]
            for key in base_builder.function_argument_keys[affine_id]
        ] == ["affine_cases:b0:a0", "affine_cases:b0:a1"]
        assert any(
            effect.get("value") == "affine_cases:b0:a0"
            for record in base["effects"]
            for effect in record["effects"]
        )
        candidates = []
        for value_id, key in key_by_ssa(base_builder).items():
            value_roots = base_builder.memory_roots_by_value.get(key, set())
            if value_roots:
                candidates.append((value_id, value_roots))
        selected = []
        for value_id, value_roots in candidates:
            if all(value_roots.isdisjoint(other) for _, other in selected):
                selected.append((value_id, value_roots))
            if len(selected) == 3:
                break
        assert len(selected) == 3, candidates
        a, b, c = (item[0] for item in selected)

        for classification in ("may_alias", "partial_alias"):
            analysis = copy.deepcopy(isolated)
            analysis["aliases"] = [
                {"lhs": a, "rhs": b, "result": classification},
                {"lhs": b, "rhs": c, "result": classification},
            ]
            builder, graph = build(module, mlir_text, analysis)
            assert roots(builder, a).isdisjoint(roots(builder, b))
            assert roots(builder, b).isdisjoint(roots(builder, c))
            assert roots(builder, a).isdisjoint(roots(builder, c))
            assert sum(
                data.get("role") == classification
                for _, _, data in graph.edges(data=True)
            ) == 4

        must = copy.deepcopy(isolated)
        must["aliases"] = [{"lhs": a, "rhs": b, "result": "must_alias"}]
        must_builder, _ = build(module, mlir_text, must)
        assert roots(must_builder, a) == roots(must_builder, b)
        assert roots(must_builder, a).isdisjoint(roots(must_builder, c))
        assert 2 in must_builder.graph.graph["must_alias_component_sizes"]

        no_alias = copy.deepcopy(isolated)
        no_alias["aliases"] = [{"lhs": a, "rhs": b, "result": "no_alias"}]
        no_builder, no_graph = build(module, mlir_text, no_alias)
        assert roots(no_builder, a).isdisjoint(roots(no_builder, b))
        assert not any(
            data.get("role") == "no_alias"
            for _, _, data in no_graph.edges(data=True)
        )

        fallback_query = next(
            item for item in base["dependences"]
            if item["fallback_required"]
            and item["source"] != item["target"]
        )
        _, fallback_graph = build(module, mlir_text, copy.deepcopy(base))
        fallback = dependence_edges(fallback_graph, fallback_query)
        assert len(fallback) == 1
        assert fallback[0]["certainty"] == "may"
        assert fallback[0]["fallback_reason"] == fallback_query["reason"]

        irrelevant_analysis = copy.deepcopy(base)
        irrelevant = next(
            item for item in irrelevant_analysis["dependences"]
            if item["source"] == fallback_query["source"]
            and item["target"] == fallback_query["target"]
            and item["kind"] == fallback_query["kind"]
        )
        irrelevant.clear()
        irrelevant.update(
            source=fallback_query["source"],
            target=fallback_query["target"],
            kind=fallback_query["kind"],
            result="not_applicable",
            reason="proven_no_alias",
            fallback_required=False,
            possible_loop_carried=False,
        )
        _, irrelevant_graph = build(
            module, mlir_text, irrelevant_analysis
        )
        assert dependence_edges(irrelevant_graph, irrelevant) == []

        self_query = next(
            item for item in base["dependences"]
            if item["source"] == item["target"]
        )
        noncarried = copy.deepcopy(base)
        replacement = next(
            item for item in noncarried["dependences"]
            if item["source"] == self_query["source"]
            and item["target"] == self_query["target"]
            and item["kind"] == self_query["kind"]
        )
        replacement.clear()
        replacement.update(
            source=self_query["source"],
            target=self_query["target"],
            kind=self_query["kind"],
            result="unknown",
            reason="aliased_storage_not_canonicalized",
            fallback_required=True,
            possible_loop_carried=False,
        )
        _, noncarried_graph = build(module, mlir_text, noncarried)
        assert dependence_edges(noncarried_graph, replacement) == []

        duplicate = copy.deepcopy(base)
        duplicate["effects"].append(copy.deepcopy(duplicate["effects"][0]))
        try:
            build(module, mlir_text, duplicate)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate native effect record was accepted")

        effects = copy.deepcopy(base)
        target = next(
            item for item in effects["effects"]
            if item["op_name"] == "test.unknown_effect"
        )
        nonoperand_value = next(
            item["value"]
            for record in effects["effects"]
            for item in record["effects"]
            if "value" in item
        )
        target["status"] = "known"
        target["effects"] = [
            {"kind": kind, "value": nonoperand_value}
            for kind in ("read", "write", "allocate", "free", "unknown_effect")
        ]
        _, effect_graph = build(module, mlir_text, effects)
        feature_node = next(
            data for _, data in effect_graph.nodes(data=True)
            if data.get("native_operation_effect_count") == 5
        )
        assert feature_node["native_operation_effects"] == (
            "allocate,free,read,unknown_effect,write"
        )
        assert all(
            feature_node[f"native_operation_effect_{kind}"] == 1
            for kind in ("read", "write", "allocate", "free", "unknown")
        )
        stamp_graph(effect_graph, effects)
        graph_gen.write_gexf_deterministic(effect_graph, OUTPUT)

    subprocess.run(
        ["/home/elvouvali/.hls-llm/bin/python", __file__, "--pyg"],
        check=True,
    )
    print("MailoHLS native semantic regressions: PASS")


def pyg_phase():
    import torch

    sys.argv = [sys.argv[0]]
    import mlir_data

    graph = nx.read_gexf(OUTPUT)
    node = next(
        data for _, data in graph.nodes(data=True)
        if data.get("native_operation_effect_count") == 5
    )
    numeric = mlir_data.node_numeric_features(node)
    assert len(mlir_data.NODE_NUMERIC_NAMES) == len(numeric)
    assert mlir_data.NODE_NUMERIC_NAMES[-6:] == [
        "native_operation_effect_read",
        "native_operation_effect_write",
        "native_operation_effect_allocate",
        "native_operation_effect_free",
        "native_operation_effect_unknown",
        "bounded_native_operation_effect_count",
    ]
    assert numeric[-6:] == [1.0, 1.0, 1.0, 1.0, 1.0, 5.0 / 8.0]
    encoders = mlir_data.fit_encoders([OUTPUT])
    payload = mlir_data.encode_static_graph(
        graph, OUTPUT.name, "native-semantics", encoders
    )
    torch.save(payload, PT_OUTPUT)
    loaded = torch.load(PT_OUTPUT)
    assert loaded["x"].shape[1] >= len(mlir_data.NODE_NUMERIC_NAMES)
    assert any(
        torch.allclose(
            row[-6:],
            torch.tensor([1, 1, 1, 1, 1, 5 / 8], dtype=row.dtype),
        )
        for row in loaded["x"]
    )


if __name__ == "__main__":
    if "--pyg" in sys.argv:
        pyg_phase()
    else:
        python_phase()
