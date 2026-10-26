"""CPU-only search tests: the synthetic oracle is never available to a policy."""
from argparse import Namespace
from pathlib import Path
import stat

import pytest

from experiments.refinement import configurations as c, experiment as old, synthesis as hls, relations
from experiments.refinement_search import controller as ctl, policies as p


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "input"; directory.mkdir()
    metadata = directory / "kernel_info.txt"
    metadata.write_text("top\nL1,array,a,1,30\nL3,array,b,1,50\nL9,loop,30\nL11,loop,50\n")
    path = directory / "kernel.cpp"
    path.write_text("void top(int *x) {\nL1: int a[30];\nL3: int b[50];\nL9: for(int i=0;i<30;i++) {a[i]=x[i];}\nL11: for(int i=0;i<50;i++) {x[i]=b[i];}\n}\n")
    kernel, actions, _ = old.snapshot_source("toy", {"source": str(path), "metadata": str(metadata), "top": "top"}, tmp_path / "snapshot")
    native = c.legacy_domains(actions, directory)
    base = {key: "none" if "ARRAY_T" in key else "0" for key in native}
    for label in ("L1", "L3"):
        base.update({c.field_key("ARRAY_T", label): "cyclic", c.field_key("ARRAY_F", label): "5", c.field_key("ARRAY_D", label): "1"})
    for label in ("L9", "L11"):
        base[c.field_key("UNROLL", label)] = "5"
    kernel.update(initial_assignment=base, relations={"status": "blocked", "reason": "test-only fixture"}, paper_baseline={})
    evidence = [{"action_i": "L1", "action_j": "L9", "verified_relation_class": "direct_verified",
                 "original_relation_types": ["loop_array_access"], "evidence_artifact_hash": "test-only", "reason": "fixture compiler access"},
                {"action_i": "L3", "action_j": "L11", "verified_relation_class": "conservative_unknown", "reason": "unresolved helper binding"}]
    return kernel, {"baseline": base, "actions": actions, "native_domains": native, "relations": evidence, "inactive_actions": []}


def observed(proposal, value=100, **changes):
    return {**proposal, "measurement": {"ADP": value, "synthesis_status": "success", "feasible": True,
                                      "rankable": True, "comparison_context_id": "same", **changes}}


def initial(problem):
    return observed(p.choose("coordinate", problem, [], problem["baseline"]))


def test_single_proposals_balanced_complete_deterministic(source):
    _, problem = source; base = problem["baseline"]
    rows = p.single_proposals(problem, base)
    assert rows == p.single_proposals(problem, base, 123)
    assert len({r["signature"] for r in rows}) == len(rows)
    assert {r["direction"] for r in rows if r["family"] == "UNROLL"} >= {"lower", "upper", "disable"}
    assert {r["family"] for r in rows} >= {"PIPE", "UNROLL", "ARRAY_F", "ARRAY_T"}
    for row in rows:
        assert len(row["reference_changed_actions"]) == 1
        c.policy_record(row["assignment"], base, problem["actions"], problem["native_domains"], "tcad")
        assert c.parse_assignment(c.serialize(row["assignment"], problem["actions"])) == row["assignment"]
    problem["inactive_actions"] = ["L9"]
    assert all("L9" not in row["reference_changed_actions"] for row in p.single_proposals(problem, base))
    assert any("L11" in row["reference_changed_actions"] for row in p.single_proposals(problem, base))


def test_coordinate_reanchors_immediately_after_improvement(source):
    _, problem = source; base = problem["baseline"]
    history = [initial(problem)]
    first = p.choose("coordinate", problem, history, base)
    history.append(observed(first, 80))
    following = p.choose("coordinate", problem, history, first["assignment"])
    assert following["reference_signature"] == first["signature"]


def test_composition_common_reference_and_stale_rejection(source):
    _, problem = source; base = problem["baseline"]
    history = [initial(problem)]
    a = p.choose("compose", problem, history, base)
    history.append(observed(a, 90))
    b = p.choose("compose", problem, history, a["assignment"])
    assert b["reference_signature"] == history[0]["signature"]
    assert a["reference_changed_actions"] != b["reference_changed_actions"]
    history.append(observed(b, 80))
    combined = p.choose("compose", problem, history, b["assignment"])
    assert combined["proposal_type"] == "composition"
    assert len(combined["reference_changed_actions"]) == 2
    assert combined["constituent_signatures"] == [a["signature"], b["signature"]]
    assert combined["reference_signature"] == c.signature(base)
    stale = dict(b["assignment"])
    unrelated = next(r for r in p.single_proposals(problem, stale) if r["reference_changed_actions"][0] not in combined["reference_changed_actions"])
    candidates, rejected = p.composition_proposals(problem, history, unrelated["assignment"])
    assert not candidates and rejected[0]["reason"] == "stale_or_unrelated_incumbent_edits"
    history[2]["measurement"]["comparison_context_id"] = "different"
    assert not p.composition_proposals(problem, history, b["assignment"])[0]


def test_composition_does_not_merge_edits_to_same_action(source):
    _, problem = source; base = problem["baseline"]
    rows = [r for r in p.single_proposals(problem, base) if r["reference_changed_actions"] == ["L9"]][:2]
    history = [initial(problem), *[observed(row, 90) for row in rows]]
    assert not p.composition_proposals(problem, history, rows[-1]["assignment"])[0]


def test_structural_requires_verified_evidence_not_successful_singles(source):
    _, problem = source; base = problem["baseline"]
    history = [initial(problem)]
    rows = p.structural_proposals(problem, base)
    assert rows
    for row in rows:
        assert set(row["reference_changed_actions"]) == {"L1", "L9"}
        assert row["relation_evidence"]["verified_relation_class"] == "direct_verified"
    a = p.choose("structural", problem, history, base)
    assert a["proposal_type"] == "structural"  # No constituent was evaluated.
    problem["relations"][0]["verified_relation_class"] = "conservative_unknown"
    assert p.choose("structural", problem, history, base) is None


def test_random_seed_reproducible_and_hybrid_fixed_slots(source):
    _, problem = source; base = problem["baseline"]
    history = [initial(problem)]
    assert p.choose("random", problem, history, base, 123) == p.choose("random", problem, history, base, 123)
    for step in range(2, 9):
        row = p.choose("hybrid", problem, history, base)
        assert (row["proposal_type"] == "structural") == (step % 4 == 0)
        history.append(observed(row, 110))
    problem["relations"] = []
    row = p.choose("hybrid", problem, history[:3], base)
    assert row["proposal_type"] == "coordinate"


@pytest.fixture
def prepared(source, tmp_path, monkeypatch):
    kernel, problem = source
    old_dir = tmp_path / "historical"; old_dir.mkdir()
    c.write_json(old_dir / "manifest.json", {"immutable": "old archive"})
    original = {"context": c.CONTEXT, "domain_policy": "tcad", "kernels": {"toy": kernel},
                "native_domains": {"toy": problem["native_domains"]}}
    monkeypatch.setattr(old, "load_output", lambda path: (original, []))
    monkeypatch.setattr(old, "read_measurements", lambda path: ({}, [], 0))
    monkeypatch.setattr(c, "from_paper", lambda row, actions: problem["baseline"])
    monkeypatch.setattr(c, "compiler_actions", lambda kernel, actions: actions)
    monkeypatch.setattr(relations, "audit_kernel", lambda kernel, actions: (problem["relations"], []))
    args = Namespace(input_dir=str(old_dir), output_dir=str(tmp_path / "search"), measurements=[], kernels=["toy"],
                     policies=list(p.POLICIES), max_evaluations=8, seed=123, execution="replay", fresh_synthesis=False)
    ctl.prepare(args)
    return Path(args.output_dir), args


@pytest.fixture
def oracle(monkeypatch):
    class Calls(list):
        values = []
    calls = Calls()
    def measure(manifest, shared, proposal, query_id, root, executable, timeout):
        path = Path(root) / manifest["kernel"] / manifest["policy"]
        trace = c.read_jsonl(path / "search_trace.jsonl")[-1]
        assert trace["status"] == "pending" and trace["logical_evaluation_cost"] == 1
        assert trace["candidate_signature"] == proposal["signature"]
        calls.append((manifest["policy"], proposal["signature"]))
        value = calls.values[len(c.read_jsonl(path / "measurements.jsonl"))] if calls.values else 100
        changes = value if isinstance(value, dict) else {"ADP": value}
        result = {"synthesis_status": "success", "ADP": 100, "rankable": True, "feasible": True,
                  "comparison_context_id": "same", "latency_cycles": 100, "resources": {r: 1 for r in c.CAPACITIES}, **changes}
        return result, {"physical_synthesis_status": "historical_reuse", "new_physical_synthesis_runs": 0,
                        "authenticated_measurement_reuses": int(result["synthesis_status"] != "unavailable"), "physical_synthesis_runtime_seconds": 0}
    monkeypatch.setattr(ctl, "measure", measure)
    # Test-only measurement adapter; real cache authentication is tested below.
    monkeypatch.setattr(ctl, "_verify_result", lambda *args: None)
    return calls


def state(path):
    manifest, shared = ctl.load_run(path)
    return ctl.read_state(path, manifest, shared)


def test_budget_cache_queries_incumbent_monotonic_and_resume(prepared, oracle):
    output, _ = prepared; path = output / "toy" / "coordinate"
    oracle.values = [100, 80, 120, 80, {"ADP": 1, "feasible": False}, {"synthesis_status": "failed", "ADP": None, "rankable": False}, 90, 70]
    ctl.search(path, limit=3)
    prefix = c.read_jsonl(path / "search_trace.jsonl")
    ctl.search(path)
    traces, records, history, pending, terminal = state(path)
    assert len(history) == len(oracle) == 8 and terminal == "budget_exhausted" and pending is None
    assert traces[:3] == prefix
    bests = [r["best_ADP_so_far"] for r in traces if r["event_type"] == "query"]
    assert bests == [100, 80, 80, 80, 80, 80, 80, 70]
    assert len({row["signature"] for row in history}) == 8
    assert history[1]["signature"] == traces[3]["incumbent_after"]  # Equal ADP keeps first incumbent.
    ctl.search(path)
    assert len(oracle) == 8
    rows = ctl.summarize(path)
    assert rows[3]["logical_evaluations"] == rows[3]["authenticated_measurement_reuses"] == 8
    assert rows[3]["new_physical_synthesis_runs"] == 0 and rows[3]["failures"] == 1


def test_missing_replay_does_not_become_fake_qor_or_failure(prepared, oracle):
    output, _ = prepared; path = output / "toy" / "coordinate"
    oracle.values = [100] + [{"synthesis_status": "unavailable", "ADP": None, "feasible": None, "rankable": False}] * 7
    ctl.search(path)
    rows = ctl.summarize(path)
    assert rows[0]["best_ADP"] == 100 and rows[0]["status"] == "measured"
    assert rows[3]["best_ADP"] is None and rows[3]["best_observed_ADP"] == 100
    assert rows[3]["unavailable"] == 7 and rows[3]["failures"] == 0


def test_policy_histories_isolated_with_same_baseline(prepared, oracle):
    output, _ = prepared
    first = output / "toy" / "coordinate"; other = output / "toy" / "compose"
    ctl.search(first, limit=3)
    assert not c.read_jsonl(other / "measurements.jsonl")
    ctl.search(other, limit=2)
    a = state(first)[2]; b = state(other)[2]
    assert a[0]["signature"] == b[0]["signature"]
    manifest, shared = ctl.load_run(other)
    facts = ctl.problem_for(manifest, shared)
    assert set(facts) == {"baseline", "actions", "native_domains", "relations", "inactive_actions"}
    assert a[1]["signature"] == b[1]["signature"]


def test_crash_after_measurement_resume_does_not_query_twice(prepared, oracle, monkeypatch):
    output, _ = prepared; path = output / "toy" / "coordinate"
    original_write = c.write_jsonl
    def interrupted_write(filename, rows):
        if Path(filename) == path / "search_trace.jsonl" and rows[-1].get("status") == "completed":
            raise RuntimeError("simulated crash after durable measurement")
        original_write(filename, rows)
    monkeypatch.setattr(c, "write_jsonl", interrupted_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ctl.search(path, limit=1)
    assert len(oracle) == 1 and state(path)[3]["status"] == "pending"
    monkeypatch.setattr(c, "write_jsonl", original_write)
    ctl.search(path, limit=1)
    assert len(oracle) == 1 and len(state(path)[2]) == 1


def test_no_eligible_structural_pairs_early_termination_carries(prepared, oracle, monkeypatch):
    output, _ = prepared; path = output / "toy" / "structural"
    monkeypatch.setattr(p, "structural_proposals", lambda *args: [])
    ctl.search(path)
    assert len(oracle) == 1 and state(path)[4] == "neighborhood_exhausted"
    assert all(row["best_ADP"] == 100 for row in ctl.summarize(path))
    assert ctl.summarize(path)[-1]["status"] == "early_exhausted_carried"


def test_mismatched_evaluation_context_blocks_comparison(prepared, oracle):
    output, _ = prepared; path = output / "toy" / "coordinate"
    oracle.values = [100, {"ADP": 1, "comparison_context_id": "wrong"}]
    with pytest.raises(ValueError, match="mismatched evaluation contexts"):
        ctl.search(path)
    traces = c.read_jsonl(path / "search_trace.jsonl")
    assert traces[0]["best_ADP_so_far"] == 100 and traces[-1]["status"] == "pending"


@pytest.mark.parametrize("field,value", [("part", "wrong"), ("clock_period_ns", 5), ("objective", "Latency")])
def test_unsupported_context_rejected_even_if_resealed(prepared, field, value):
    output, _ = prepared; path = output / "toy" / "coordinate"
    manifest = ctl._read_sealed(path / "manifest.json")
    manifest["request"][field] = value; manifest.pop("seal")
    c.write_json(path / "manifest.json", ctl._sealed(manifest))
    with pytest.raises(ValueError, match="Unsupported request"):
        ctl.load_run(path)


def test_tampered_source_and_existing_output_rejected(prepared):
    output, args = prepared; path = output / "toy" / "coordinate"
    with pytest.raises(ValueError, match="nonempty"):
        ctl.prepare(args)
    _, shared = ctl.load_run(path)
    filename = c.repo_path(shared["kernels"]["toy"]["source_provenance"]["snapshot_source"])
    filename.write_text(filename.read_text() + "// changed\n")
    with pytest.raises(ValueError, match="Frozen source/cache evidence changed"):
        ctl.load_run(path)


def test_replay_cache_miss_and_failed_result_validation(prepared):
    output, _ = prepared; path = output / "toy" / "coordinate"
    manifest, shared = ctl.load_run(path); problem = ctl.problem_for(manifest, shared)
    proposal = p.choose("coordinate", problem, [], problem["baseline"])
    result, cost = ctl.measure(manifest, shared, proposal, "paid-test-query", output, "unused", 1)
    assert result["synthesis_status"] == "unavailable" and result["ADP"] is None
    assert cost["new_physical_synthesis_runs"] == cost["authenticated_measurement_reuses"] == 0
    candidate = ctl.physical_candidate(manifest, shared, proposal)
    for invalid in ({"synthesis_status": "failed", "ADP": 1}, {"synthesis_status": "unavailable", "ADP": 1, "feasible": None}):
        with pytest.raises(ValueError):
            ctl._verify_result({"result": invalid}, candidate, shared["kernels"]["toy"], None)


def test_unchanged_historical_hash_contract():
    hashes = old.implementation_hashes()
    assert not any("refinement_search" in path for path in hashes)
    assert all(c.sha256(c.repo_path(path)) == digest for path, digest in hashes.items())


def test_sixteen_query_budget_and_default_prepare_never_calls_oracle(prepared, oracle):
    output, _ = prepared
    assert not oracle
    path = output / "toy" / "coordinate"
    manifest = ctl._read_sealed(path / "manifest.json"); manifest.pop("seal")
    manifest["max_evaluations"] = manifest["request"]["maximum_evaluations"] = 16
    c.write_json(path / "manifest.json", ctl._sealed(manifest))
    ctl.search(path)
    assert len(oracle) == 16 and ctl.summarize(path)[-1]["logical_evaluations"] == 16


def test_policy_decision_does_not_depend_on_hidden_cache(prepared, oracle):
    output, _ = prepared; path = output / "toy" / "coordinate"
    manifest, shared = ctl.load_run(path)
    before = ctl.problem_for(manifest, shared)
    shared["cache"] = {"future-winner": {"ADP": -123456}}
    after = ctl.problem_for(manifest, shared)
    assert before == after
    assert p.choose("coordinate", before, [], before["baseline"]) == p.choose("coordinate", after, [], after["baseline"])


def test_controller_rejects_invalid_proposals_before_oracle(prepared, oracle, monkeypatch):
    output, _ = prepared; path = output / "toy" / "coordinate"
    choose = p.choose
    def invalid(*args):
        row = choose(*args)
        row["assignment"] = {**row["assignment"], c.field_key("ARRAY_F", "L1"): "7"}
        row["signature"] = c.signature(row["assignment"])
        return row
    monkeypatch.setattr(p, "choose", invalid)
    with pytest.raises(ValueError):
        ctl.search(path)
    assert not oracle


@pytest.mark.parametrize("policy", p.POLICIES)
def test_every_policy_obeys_its_private_budget(prepared, oracle, policy):
    output, _ = prepared; path = output / "toy" / policy
    ctl.search(path)
    _, _, history, pending, terminal = state(path)
    assert len(history) == len(oracle) <= 8 and history[0]["proposal_type"] == "baseline"
    assert len({r["signature"] for r in history}) == len(history)
    assert pending is None and terminal in ("budget_exhausted", "neighborhood_exhausted")
    assert all(not c.read_jsonl(output / "toy" / other / "measurements.jsonl") for other in p.POLICIES if other != policy)


def test_cpu_vendor_adapter_authentication_global_cache_and_resume(prepared, tmp_path, monkeypatch):
    output, args = prepared
    original_input_hash = c.sha256(Path(args.input_dir) / "manifest.json")
    # The real shared source/Tcl/report/receipt pipeline executes a tiny CPU-only
    # test program. It does not pretend these synthetic QoR values are research results.
    xml = '<profile><ReportVersion><Version>2021.1</Version></ReportVersion><UserAssignments><unit>ns</unit><Part>xczu7ev-ffvc1156-2-e</Part><TopModelName>top</TopModelName><TargetClockPeriod>10</TargetClockPeriod><FlowTarget>vivado</FlowTarget></UserAssignments><PerformanceEstimates><SummaryOfOverallLatency><unit>clock cycles</unit><Worst-caseLatency>100</Worst-caseLatency></SummaryOfOverallLatency></PerformanceEstimates><AreaEstimates><Resources><BRAM_18K>1</BRAM_18K><DSP>1</DSP><FF>1</FF><LUT>1</LUT></Resources><AvailableResources><BRAM_18K>624</BRAM_18K><DSP>1728</DSP><FF>460800</FF><LUT>230400</LUT></AvailableResources></AreaEstimates></profile>'
    executable = tmp_path / "test_hls"
    executable.write_text('#!/usr/bin/python3\nimport pathlib\np=pathlib.Path("hls_project/solution1/syn/report/top_csynth.xml")\np.parent.mkdir(parents=True,exist_ok=True)\np.write_text(' + repr(xml) + ')\nprint("Finished Command csynth_design")\nprint("Exiting vitis_hls")\n')
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    tool = {"executable": str(executable), "version": "2021.1", "executable_sha256": c.sha256(executable),
            "vendor_headers": {name: {"path": str(tmp_path / name), "sha256": "test-only"} for name in hls.VENDOR_HEADERS}}
    monkeypatch.setattr(hls, "tool_info", lambda *args: tool)
    for policy in ("coordinate", "compose"):
        path = output / "toy" / policy
        manifest = ctl._read_sealed(path / "manifest.json"); manifest.pop("seal"); manifest["execution"] = "synthesis"
        c.write_json(path / "manifest.json", ctl._sealed(manifest))
        ctl.search(path, executable=str(executable), limit=1)
        ctl.search(path, executable=str(executable), limit=1)
    assert len(c.read_jsonl(output / "physical" / "attempts.jsonl")) == 2
    assert len(state(output / "toy" / "coordinate")[2]) == 2
    second = ctl.summarize(output / "toy" / "compose")[1]
    assert second["logical_evaluations"] == second["authenticated_measurement_reuses"] == 2
    assert second["new_physical_synthesis_runs"] == 0
    assert c.sha256(Path(args.input_dir) / "manifest.json") == original_input_hash
    path = output / "toy" / "coordinate"
    manifest, shared = ctl.load_run(path)
    history = state(path)[2]; problem = ctl.problem_for(manifest, shared)
    proposal = p.choose("coordinate", problem, history, history[0]["assignment"])
    attempts = c.read_jsonl(output / "physical" / "attempts.jsonl")
    attempts.append({"query_id": "interrupted-physical-reservation", "key": "toy:" + proposal["signature"],
                     "work_attempt": 1, "tool_identity": hls.tool_identity(tool), "status": "reserved"})
    c.write_jsonl(output / "physical" / "attempts.jsonl", attempts)
    def forbidden(*args):
        raise AssertionError("A reserved job must not be launched twice")
    monkeypatch.setattr(hls, "run_candidate", forbidden)
    with pytest.raises(RuntimeError, match="reserved without a durable result"):
        ctl.measure(manifest, shared, proposal, "interrupted-physical-reservation", output, str(executable), 1)
    assert len(c.read_jsonl(output / "physical" / "attempts.jsonl")) == 3
