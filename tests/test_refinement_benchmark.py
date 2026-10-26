"""Protocol, matched controls and reporting around the unchanged R1 engine."""
from argparse import Namespace
from pathlib import Path
import struct

import pytest

from test_refinement_search import source, oracle
from experiments.refinement import configurations as c, experiment as old, synthesis as hls, relations
from experiments.refinement_search import benchmark as b, controller as ctl, policies as p


@pytest.fixture
def prepared(source, tmp_path, monkeypatch, request):
    kernel, problem = source
    original = {"context": c.CONTEXT, "domain_policy": "tcad", "kernels": {"toy": kernel}, "native_domains": {"toy": problem["native_domains"]}}
    directory = tmp_path / "historical"; directory.mkdir(); c.write_json(directory / "manifest.json", {"original": True})
    monkeypatch.setattr(old, "load_output", lambda path: (original, []))
    monkeypatch.setattr(old, "read_measurements", lambda path: ({}, [], 0))
    monkeypatch.setattr(c, "from_paper", lambda row, actions: problem["baseline"])
    monkeypatch.setattr(c, "compiler_actions", lambda kernel, actions: actions)
    monkeypatch.setattr(relations, "audit_kernel", lambda kernel, actions: (problem["relations"], []))
    args = Namespace(input_dir=str(directory), output_dir=str(tmp_path / "engine"), measurements=[], kernels=["toy"],
                     policies=list(b.MAIN), max_evaluations=8, seed=123, execution=getattr(request, "param", "replay"), fresh_synthesis=False)
    with b.runtime():
        ctl.prepare(args)
    args.search_dir = args.output_dir; args.output_dir = str(tmp_path / "benchmark")
    args.seeds = [123]; args.import_search = []; args.protocol_version = "test"; args.timeout = 60
    b.prepare(args)
    return Path(args.output_dir), args, problem


def execution_args(output, **changes):
    return Namespace(output_dir=str(output), mode="replay", kernels=None, policies=None, groups="all", allow_batch=False,
                     limit=None, vitis_hls="unused", **changes)


def test_protocol_same_baseline_budget_and_engine_files_preserved(prepared):
    output, _, _ = prepared; protocol = b.load_protocol(output)
    assert protocol["maximum_main_logical_queries"] == 40 and protocol["maximum_control_logical_queries"] == 8
    assert "initial_worktree_status" in protocol and "benchmark_sha256" in protocol
    assert not any(path.endswith("benchmark.py") for path in protocol["core_sha256"])
    manifests = []
    with b.runtime(protocol):
        for entry in protocol["runs"]:
            manifest, _ = ctl.load_run(c.repo_path(entry["path"])); manifests.append(manifest)
    assert {m["max_evaluations"] for m in manifests} == {8}
    assert len({m["baseline_signature"] for m in manifests}) == 1
    assert set(manifests[0]["request"]["resource_budgets"]) == set(c.RESOURCE_KEYS)
    assert all(c.sha256(c.repo_path(path)) == digest for path, digest in protocol["core_sha256"].items())


def test_matched_control_legal_deterministic_and_not_claimed_independent(source):
    _, problem = source
    rows = b.matched_proposals(problem, problem["baseline"])
    assert rows and rows == b.matched_proposals(problem, problem["baseline"], 123)
    lookup = {row["signature"]: row for row in p.structural_proposals(problem, problem["baseline"])}
    for row in rows:
        assert len(row["reference_changed_actions"]) == 2 and row["control_label"] == "no_verified_relation"
        if row["relation_evidence"]:
            assert row["relation_evidence"]["verified_relation_class"] not in b.VERIFIED
        template = lookup[row["matching_template_signature"]]
        assert b._profile_key(b._edit_profile(problem, problem["baseline"], row)) == b._profile_key(b._edit_profile(problem, problem["baseline"], template))
        c.policy_record(row["assignment"], problem["baseline"], problem["actions"], problem["native_domains"], "tcad")
    problem["relations"][0]["verified_relation_class"] = "conservative_unknown"
    assert not b.matched_proposals(problem, problem["baseline"])


def test_protocol_cannot_be_rewritten_and_tampering_blocks_resume(prepared):
    output, args, _ = prepared
    digest = c.sha256(output / "protocol.json")
    with pytest.raises(ValueError, match="nonempty"):
        b.prepare(args)
    assert digest == c.sha256(output / "protocol.json")
    payload = ctl._read_sealed(output / "protocol.json"); payload["benchmark_sha256"] = "wrong"; payload.pop("seal")
    c.write_json(output / "protocol.json", ctl._sealed(payload))
    with pytest.raises(ValueError, match="Frozen benchmark"):
        b.load_protocol(output)


def test_all_policies_private_matched_budget_resume_and_outputs(prepared, oracle):
    output, _, _ = prepared; digest = c.sha256(output / "protocol.json")
    args = execution_args(output); args.limit = 3
    b.run(args)
    assert len(oracle) == 3
    prefix = c.read_jsonl(output / "search_traces.jsonl")
    assert all(r["proposal_type"] == "baseline" for r in prefix)
    args.limit = None; b.run(args)
    rows = b.summarize(output)
    assert len(rows) == 6 and len(oracle) == sum(r["logical_HLS_calls"] for r in rows)
    assert all(r["logical_HLS_calls"] <= 8 for r in rows)
    assert sum(r["logical_HLS_calls"] for r in rows if r["group"] == "main") <= 40
    assert all(r["ADP@1"] == 100 and r["ADP@8"] == 100 for r in rows)
    assert all(r["validation_status"] == "not_tested" for r in rows)
    before = len(oracle); b.run(args); assert len(oracle) == before
    assert c.sha256(output / "protocol.json") == digest
    assert all((output / name).is_file() for name in ("summary.csv", "report.md", "measurements.jsonl", "search_traces.jsonl"))


def test_missing_replay_curve_remains_incomplete(prepared, oracle):
    output, _, _ = prepared
    oracle.values = [100] + [{"synthesis_status": "unavailable", "ADP": None, "rankable": False, "feasible": None}] * 7
    args = execution_args(output); args.policies = ["coordinate"]
    b.run(args)
    row = next(r for r in b.summarize(output) if r["policy"] == "coordinate")
    assert row["ADP@1"] == 100 and row["ADP@2"] is None and row["ADP@8"] is None
    assert row["failed"] == 0 and row["unavailable"] == 7


def test_infeasible_worse_results_cannot_replace_best_and_small_gain_flag(prepared, oracle):
    output, _, _ = prepared
    oracle.values = [100, 99.8, 120, {"ADP": 1, "feasible": False}, 110, 110, 110, 110]
    args = execution_args(output); args.policies = ["coordinate"]
    b.run(args)
    row = next(r for r in b.summarize(output) if r["policy"] == "coordinate")
    assert row["ADP@8"] == 99.8 and row["small_gain_below_0.5pct"]
    assert row["infeasible"] == 1 and row["incumbent_improvement_fraction"] == pytest.approx(1 / 7)


def test_trace_qor_tampering_is_detected(prepared, oracle):
    output, _, _ = prepared
    args = execution_args(output); args.policies = ["coordinate"]; args.limit = 1; b.run(args)
    protocol = b.load_protocol(output); entry = next(r for r in protocol["runs"] if r["policy"] == "coordinate")
    path = c.repo_path(entry["path"]) / "search_trace.jsonl"
    rows = c.read_jsonl(path); rows[0]["measured_ADP"] = 1; c.write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Trace QoR differs"):
        b.summarize(output)


@pytest.mark.parametrize("prepared", ["synthesis"], indirect=True)
def test_batch_requires_explicit_opt_in(prepared):
    output, _, _ = prepared
    # Exercise the production guard before any Vitis/measurement interface.
    args = execution_args(output); args.mode = "run"
    with pytest.raises(ValueError, match="require --allow-batch"):
        b.run(args)


def test_benchmark_cache_reveals_only_after_paid_query(prepared, oracle, monkeypatch):
    output, _, _ = prepared
    original = ctl.measure
    accesses = []
    def hidden(protocol, manifest, shared, proposal, root):
        path = Path(root) / manifest["kernel"] / manifest["policy"]
        trace = c.read_jsonl(path / "search_trace.jsonl")[-1]
        assert trace["logical_evaluation_cost"] == 1 and trace["status"] == "pending"
        accesses.append(proposal["signature"])
        result, _ = original(manifest, shared, proposal, trace["query_id"], root, "unused", 1)
        return result
    monkeypatch.setattr(b, "_cached_measurement", hidden)
    args = execution_args(output); args.policies = ["coordinate"]; args.limit = 2
    b.run(args)
    row = next(r for r in b.summarize(output) if r["policy"] == "coordinate")
    assert row["logical_HLS_calls"] == row["authenticated_reuses"] == len(accesses) == 2
    assert row["new_physical_syntheses"] == 0


def test_context_scope_restores_core_functions_and_does_not_skip_hash_checks(prepared, monkeypatch):
    output, _, _ = prepared
    before = (ctl.code_hashes, p.choose, p.POLICIES, ctl.measure)
    with b.runtime(b.load_protocol(output)):
        assert "matched" in p.POLICIES
    assert before == (ctl.code_hashes, p.choose, p.POLICIES, ctl.measure)
    hashes = b.core_hashes()
    monkeypatch.setattr(b, "core_hashes", lambda: {**hashes, "extra-unreviewed-file": "bad"})
    with pytest.raises(ValueError, match="Frozen benchmark"):
        b.load_protocol(output)


def test_directional_audit_includes_both_sides_without_qor(source):
    _, problem = source
    audit = b._initial_direction_audit(problem, 123)
    assert audit["coordinate"]["counts"]["increase"] > 0
    assert audit["coordinate"]["counts"]["decrease"] > 0
    assert audit == b._initial_direction_audit(problem, 123)


@pytest.mark.parametrize("name", ["GAN", "covariance", "Kalman"])
def test_functional_harness_has_explicit_inputs_and_no_synthesis(name):
    code, plan = b._testbench(name, "test_top")
    assert "test_top(" in code and "pattern<3" in code and "validation_output.bin" in code
    assert plan["output_values"] > 0 and plan["absolute_tolerance"] > 0


def test_functional_comparison_uses_tolerances_rejects_missing_and_nan(tmp_path):
    reference = tmp_path / "a"; candidate = tmp_path / "b"
    reference.write_bytes(struct.pack("=dd", 1., 2.)); candidate.write_bytes(struct.pack("=dd", 1., 2. + 1e-10))
    plan = {"output_values": 2, "element_bytes": 8, "absolute_tolerance": 1e-9, "relative_tolerance": 1e-9}
    assert b._compare_outputs(reference, candidate, plan)["mismatches"] == 0
    candidate.write_bytes(struct.pack("=dd", 1., 4.))
    assert b._compare_outputs(reference, candidate, plan)["mismatches"] == 1
    candidate.write_bytes(struct.pack("=dd", 1., float("nan")))
    with pytest.raises(ValueError, match="Nonfinite"):
        b._compare_outputs(reference, candidate, plan)
    candidate.write_bytes(b"")
    with pytest.raises(ValueError, match="expected number"):
        b._compare_outputs(reference, candidate, plan)


def test_validation_status_requires_execution_evidence(tmp_path):
    c.write_jsonl(tmp_path / "functional/receipts.jsonl", [{"kernel": "toy", "signature": "x", "status": "csim_pass", "executed": False}])
    with pytest.raises(ValueError, match="lacks an executed"):
        b._validation_records(tmp_path)
