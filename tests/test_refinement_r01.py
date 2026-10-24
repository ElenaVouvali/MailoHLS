"""CPU-only R0.1 attribution, immutable provenance and shared-helper regressions."""
import copy
import json
from pathlib import Path
import stat
from argparse import Namespace
from unittest.mock import patch

import networkx as nx
import pytest

from LLM_branch.common import directive_domains as dd
from experiments.refinement import r0_candidates as c, r0_synthesis as hls
from experiments.refinement.r0_common import object_hash, sha256, read_jsonl, write_json
from experiments.refinement.r0_local_headroom import code_hashes as r0_hashes, snapshot_source
from experiments.refinement_r01 import r01_attribution as cli
from experiments.refinement_r01.r01_analysis import factorial_candidates, factorial_analysis, domain_classification
from experiments.refinement_r01.r01_provenance import authenticate_work, source_context, tool_identity, R0Archive, snapshot_kernel
from experiments.refinement_r01.r01_relation_audit import verify_pair


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "kernel_info.txt").write_text("top\nL1,array,a,1,8\nL2,array,b,1,8\nL3,array,d,1,8\nL4,loop,8\nL11,loop,8\n")
    path = directory / "network.cpp"
    path.write_text("""void top(int *x) {
L1: int a[8];
L2: int b[8];
L3: int d[8];
L4: for (int i=0;i<8;i++) { a[i]=x[i]; }
L11: for (int i=0;i<8;i++) { x[i]=d[i]; }
}
""")
    output = tmp_path / "r01"
    kernel, actions, template = snapshot_source("GAN", {"source": str(path), "metadata": str(directory / "kernel_info.txt"), "top": "top"}, output)
    native = kernel["domains"]
    base = {key: "none" if "ARRAY_T" in key else "0" for key in native}
    for label, factor in [("L1", "2"), ("L2", "5"), ("L3", "5")]:
        base.update({c.key("ARRAY_T", label): "cyclic", c.key("ARRAY_F", label): factor, c.key("ARRAY_D", label): "1"})
    base[c.key("UNROLL", "L11")] = "5"
    base[c.key("PIPE", "L4")] = "1"
    domains = c.published_factor_domains(base, actions, native)
    kernel["domains"] = domains
    return output, kernel, actions, base, domains, native


def test_exact_complete_factorial_and_canonical_signatures(source):
    _, _, actions, base, domains, _ = source
    a, b = {c.key("ARRAY_F", "L2"): "6"}, {c.key("PIPE", "L11"): "1"}
    cells = factorial_candidates(base, actions, domains, a, b)
    expected = {"d0": set(), "dX": set(a), "dY": set(b), "dXY": set(a) | set(b)}
    for label, values in cells.items():
        assert set(c.changed(base, values, actions)[1]) == expected[label]
        c.validate(values, actions, domains)
        assert c.parse_assignment(c.serialize(values, actions)) == values
        assert c.signature(dict(reversed(list(values.items())))) == c.signature(values)
    assert cells["dXY"][c.key("UNROLL", "L11")] == "5"
    assert base[c.key("ARRAY_F", "L2")] == "5"
    with pytest.raises(ValueError): factorial_candidates(base, actions, domains, a, a)


def test_domains_distinguish_historical_baseline_and_new_edit(source):
    _, _, actions, base, domains, native = source
    original = copy.deepcopy(native)
    a = {**base, c.key("ARRAY_F", "L2"): "6"}
    b = {**base, c.key("PIPE", "L11"): "1"}
    assert domain_classification(base, base, native)["configuration_domain"] == "extended_domain"
    assert domain_classification(a, base, native)["changed_value_requires_extension"]
    assert not domain_classification(b, base, native)["changed_value_requires_extension"]
    fully_native = {key: "none" if "ARRAY_T" in key else "0" for key in native}
    assert domain_classification(fully_native, fully_native, native)["configuration_domain"] == "native_domain"
    assert "6" not in native[c.key("ARRAY_F", "L2")]
    assert native == original


def measurement(adp, context="same", feasible=True):
    return {"ADP": adp, "comparison_context_id": context, "authentication": "authenticated_new",
        "synthesis_status": "success", "feasible": feasible, "latency_cycles": 100,
        "resources": {key: 1 for key in cli.CONTEXT["device_capacities"]}}


@pytest.mark.parametrize("joint,expected,label", [(70, 0, "additive"), (60, -10, "favorable_nonadditive"), (90, 20, "unfavorable_nonadditive")])
def test_factorial_signs(joint, expected, label):
    rows = {"d0": measurement(100), "dX": measurement(80), "dY": measurement(90), "dXY": measurement(joint)}
    result = factorial_analysis(rows)
    assert result["interaction_ADP"] == expected
    assert result["interaction_class"] == label
    assert result["improvement_pct"]["dX"] == 20


@pytest.mark.parametrize("bad", [None, measurement(1, context="other"), measurement(1, feasible=False), {**measurement(1), "authentication": "historical_unverified"}])
def test_incomplete_or_incomparable_factorial_is_unavailable(bad):
    rows = {"d0": measurement(100), "dX": measurement(80), "dY": measurement(90), "dXY": bad}
    assert factorial_analysis(rows)["status"] == "unavailable"


def graph_fixture(shared=True):
    graph = nx.MultiDiGraph()
    graph.add_node("1", type=104, action_id="L3", array_var="d", action_resolution="source_location", function=0)
    graph.add_node("2", type=1, text="memref", is_memory_root=1, function=0)
    graph.add_node("3", type=0, is_loop=1, action_id="L11", action_resolution="source_location", function=0)
    graph.add_node("4", type=4, function=0)
    graph.add_node("5", type=0, text="func.call", function=0, op_uid="caller:call")
    graph.add_node("6", type=0, text="affine.load", function=1, op_uid="helper:load")
    graph.add_node("7", type=4, function=1)
    graph.add_node("8", type=0, text="func.func", function=1)
    graph.add_edge("1", "2", flow=7, role="array_root")
    graph.add_edge("3", "4", flow=8, role="region_entry")
    graph.add_edge("4", "5", flow=0, role="next_in_block")
    graph.add_edge("5", "8", flow=2, role="calls")
    graph.add_edge("8", "7", flow=8, role="region_entry")
    graph.add_edge("7", "6", flow=0, role="next_in_block")
    if shared:graph.add_edge("1", "6", flow=7, role="array_read")
    return graph


def test_shared_helper_not_direct_and_unknown_bindings_stay_uncertain():
    graph = graph_fixture()
    original = {"relation_bits": {"i_to_j": 8, "j_to_i": 8}, "relation_type": ["loop_array_access"]}
    result = verify_pair(graph, "L3", "L11", original, "artifact-hash")
    assert result["verified_relation_class"] == "conservative_unknown"
    assert result["shared_helper_nodes"] == ["6"]
    assert result["evidence_artifact_hash"] == "artifact-hash"
    assert result["original_relation_bits"] == original["relation_bits"]
    assert result["action_j"] == "L11"  # Absolute labels never renumbered.


def test_verified_own_region_root_access():
    graph = graph_fixture()
    graph.add_edge("1", "5", flow=7, role="array_read")
    graph.add_edge("2", "5", flow=10, role="read", operand_index=0)
    result = verify_pair(graph, "L3", "L11", artifact_hash="proof")
    assert result["verified_relation_class"] == "direct_verified"
    assert result["callsite_provenance"][0]["root_node"] == "2"


def test_GAN_two_buffers_shared_operator_regression():
    graph = graph_fixture()
    graph.nodes['1']['array_var'] = 'layer_2_out'
    graph.add_node('20', type=104, action_id='L2', array_var='layer_1_out', action_resolution='source_location', function=0)
    graph.add_node('21', type=1, text='memref', is_memory_root=1, function=0)
    graph.add_edge('20','21',flow=7,role='array_root')
    # A shared helper body is associated with both arrays across callers.
    graph.add_edge('20','6',flow=7,role='array_read')
    # This particular L11 call reads only the L3 root.
    graph.add_edge('1','5',flow=7,role='array_read')
    graph.add_edge('2','5',flow=10,role='read',operand_index=0)
    legacy={'relation_bits':{'i_to_j':8,'j_to_i':8},'relation_type':['loop_array_access']}
    assert verify_pair(graph,'L3','L11',legacy)['verified_relation_class']=='direct_verified'
    result=verify_pair(graph,'L2','L11',legacy)
    assert result['verified_relation_class']=='conservative_unknown'
    assert result['supporting_graph_nodes']==['6']


def test_indirect_requires_specific_call_and_formal_memory_binding():
    graph = graph_fixture()
    graph.add_node("9", type=1, function=1)
    graph.add_edge("2", "5", flow=1, role="operand", position=0)
    graph.add_edge("2", "9", flow=2, role="actual_to_formal", callsite="different_call")
    graph.add_edge("9", "6", flow=10, role="read")
    assert verify_pair(graph, "L3", "L11")["verified_relation_class"] == "conservative_unknown"
    graph.add_edge("2", "9", flow=2, role="actual_to_formal", callsite="5")
    assert verify_pair(graph, "L3", "L11")["verified_relation_class"] == "indirect_verified"


def test_indirect_binding_uses_observed_operand_position_and_callee_identity():
    graph = graph_fixture()
    graph.add_node('9',type=1,function=1)
    graph.add_edge('2','5',flow=1,role='operand',position=0)
    graph.add_edge('2','9',flow=2,role='actual_to_formal',position=1)
    graph.add_edge('9','6',flow=10,role='read')
    result=verify_pair(graph,'L3','L11')
    assert result['verified_relation_class']=='indirect_verified'
    assert result['callsite_provenance'][0]['actual_operand_position']==0
    graph.nodes['9']['function']=99
    assert verify_pair(graph,'L3','L11')['verified_relation_class']=='conservative_unknown'


def test_factorial_reporting_does_not_use_sorted_json_cell_order(prepared,monkeypatch):
    output,_,_=prepared
    manifest,candidates=cli.load_output(output)
    values={'d0':100,'dA':80,'dB':90,'dAB':65,'dC':85,'dCB':75,'dM':95,'dMB':85,'dA_native4':97}
    rows=[]
    for candidate in candidates:
        value=values[candidate['aliases'][0]]
        rows.append({**measurement(value),'candidate_id':candidate['candidate_id'],'area_proxy':1,'kernel':'GAN'})
    from experiments.refinement.r0_common import write_jsonl
    write_jsonl(output/'R01_MEASUREMENTS.jsonl',rows)
    monkeypatch.setattr(cli,'verify_measurement',lambda *args:None)
    result=cli.summarize(output)
    assert result['AB']['interaction_ADP']==-5
    assert result['AB']['improvement_pct']['dY']==10
    assert result['CB']['interaction_ADP']==0
    assert result['CB']['improvement_pct']['dX']==15


def test_unknown_dependence_is_not_access_proof_and_absence_not_independence():
    graph = graph_fixture(shared=False)
    graph.add_edge("5", "6", flow=12, certainty="unknown")
    assert verify_pair(graph, "L3", "L11")["verified_relation_class"] == "no_evidenced_relation"
    graph.nodes["3"]["action_resolution"] = "fallback"
    with pytest.raises(ValueError):verify_pair(graph, "L3", "L11")


def synthetic_xml():
    return '''<profile><ReportVersion><Version>2021.1</Version></ReportVersion>
<UserAssignments><unit>ns</unit><Part>xczu7ev-ffvc1156-2-e</Part><TopModelName>top</TopModelName><TargetClockPeriod>10</TargetClockPeriod><FlowTarget>vivado</FlowTarget></UserAssignments>
<PerformanceEstimates><SummaryOfOverallLatency><unit>clock cycles</unit><Worst-caseLatency>100</Worst-caseLatency></SummaryOfOverallLatency><SummaryOfTimingAnalysis><EstimatedClockPeriod>7</EstimatedClockPeriod></SummaryOfTimingAnalysis></PerformanceEstimates>
<AreaEstimates><Resources><BRAM_18K>1</BRAM_18K><DSP>1</DSP><FF>1</FF><LUT>1</LUT></Resources><AvailableResources><BRAM_18K>624</BRAM_18K><DSP>1728</DSP><FF>460800</FF><LUT>230400</LUT></AvailableResources></AreaEstimates></profile>'''


@pytest.fixture
def prepared(source, tmp_path, monkeypatch):
    output, kernel, actions, base, domains, native = source
    relation = [{"action_i": "L3", "action_j": "L11", "verified_relation_class": "direct_verified"},
        {"action_i": "L1", "action_j": "L11", "verified_relation_class": "no_evidenced_relation"}]
    recovery = {"status": "fixture", "measurements": []}
    fake_inputs = (None, kernel, actions, base, domains, native, {}, recovery, {"status": "fixture"}, relation)
    monkeypatch.setattr(cli, "_audit_inputs", lambda args, target: fake_inputs)
    args = Namespace(output_dir=str(output), r0_output=str(tmp_path / "old-r0"), fresh_measurements=True)
    cli.prepare(args)
    executable = tmp_path / "fake_hls"
    executable.write_text('#!/usr/bin/python3\nimport pathlib\np=pathlib.Path("r0_project/solution1/syn/report/top_csynth.xml")\np.parent.mkdir(parents=True,exist_ok=True)\np.write_text('+repr(synthetic_xml())+')\nprint("Finished Command csynth_design")\nprint("Exiting vitis_hls")\n')
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    tool = {"executable": str(executable), "version": "2021.1", "executable_sha256": sha256(executable),
        "vendor_headers": {name: {"path": str(tmp_path / name), "sha256": "test-only"} for name in hls.VENDOR_HEADERS}}
    monkeypatch.setattr(hls, "tool_info", lambda executable: tool)
    return output, tool, args


def test_real_runner_resume_and_factorial_cell_identity(prepared):
    output, _, _ = prepared
    cli.run(output, "fake")
    first = (output / "R01_MEASUREMENTS.jsonl").read_bytes()
    cli.run(output, "fake")
    assert (output / "R01_MEASUREMENTS.jsonl").read_bytes() == first
    assert len(read_jsonl(output / "R01_ATTEMPTS.jsonl")) == 9
    assert len(read_jsonl(output / "R01_MEASUREMENTS.jsonl")) == 9
    interactions = cli.summarize(output)
    assert all(r["interaction_ADP"] == 0 for r in interactions.values())
    # Explicit labels survive JSON sort order: never associate dAB with dY.
    manifest, _ = cli.load_output(output)
    assert manifest["factorials"]["AB"]["labels"] == ["d0", "dA", "dB", "dAB"]


def test_immutable_inputs_and_original_archive_are_not_overwritten(prepared):
    output, _, args = prepared
    assert not Path(args.r0_output).exists()
    with pytest.raises(ValueError, match="Immutable"):cli.prepare(args)
    with pytest.raises(ValueError):cli._safe_output(args.r0_output, args.r0_output)
    candidate = output / "R01_CANDIDATES.jsonl"
    candidate.write_text(candidate.read_text() + "\n")
    with pytest.raises(ValueError, match="input changed"):cli.load_output(output)


def test_budget_cap_and_wrong_tool_cannot_launch(prepared):
    output, tool, _ = prepared
    wrong = {**tool, "executable_sha256": "wrong"}
    # Seal an incompatible tool without modifying the immutable manifest.
    write_json(output / "R01_TOOL.json", tool_identity(wrong))
    with pytest.raises(ValueError, match="tool changed"):cli.run(output, "fake")
    (output / "R01_TOOL.json").unlink()
    from experiments.refinement.r0_common import write_jsonl
    write_jsonl(output / "R01_ATTEMPTS.jsonl", [{"candidate_id": "previous", "status": "failed"}] * 12)
    with patch.object(hls, "run_candidate", side_effect=AssertionError("must not launch")):
        cli.run(output, "fake")
    assert not read_jsonl(output / "R01_MEASUREMENTS.jsonl")


def test_report_or_source_tampering_blocks_comparison(prepared):
    output, _, _ = prepared
    cli.run(output, "fake", limit=1)
    manifest, candidates = cli.load_output(output)
    row = read_jsonl(output / "R01_MEASUREMENTS.jsonl")[0]
    candidate = next(c for c in candidates if c["candidate_id"] == row["candidate_id"])
    kernel = manifest["kernel"]
    bad = copy.deepcopy(kernel)
    bad["source_provenance"]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        authenticate_work(candidate, row, bad, cli.repo_path, expected_evaluation_hash=candidate["evaluation_context_hash"])
    Path(row["report_path"]).write_text("modified")
    with pytest.raises(ValueError, match="integrity mismatch"):cli.summarize(output)


def test_r0_hash_contract_ignores_sibling_package():
    before = r0_hashes()
    assert all("refinement_r01" not in path for path in before)
    assert before == r0_hashes()


def test_authentication_of_read_only_archived_R0_and_context_rejection(prepared, source, tmp_path):
    output, tool, _ = prepared
    _, current_kernel, actions, base, domains, native = source
    old = tmp_path/'old-authenticated-r0'
    kernel = snapshot_kernel(current_kernel, cli.repo_path, old)
    directives=[]
    for label,action in actions.items():
        if action.kind=='loop':
            directives.append({'action_id':label,'kind':'loop','pipeline_ii':int(base[c.key('PIPE',label)]),'unroll_factor':int(base[c.key('UNROLL',label)])})
        else:
            directives.append({'action_id':label,'kind':'array','partition_type':base[c.key('ARRAY_T',label)],'partition_factor':int(base[c.key('ARRAY_F',label)]),'partition_dim':int(base[c.key('ARRAY_D',label)])})
    kernel.update(initial_assignment=base,paper_baseline={'directives':directives},domains=domains)
    candidate=c.candidate_record('GAN','baseline',base,base,actions,domains)
    candidate['evaluation_context_hash']=object_hash({'context':cli.CONTEXT,'source':kernel['source_provenance'],
        'template_sha256':kernel['template_sha256'],'top':kernel['top'],'assignment':candidate['canonical_signature'],'code':r0_hashes()})
    result=hls.run_candidate(candidate,kernel,old,tool)
    from experiments.refinement.r0_common import write_jsonl
    write_json(old/'manifest.json',{'context':cli.CONTEXT,'kernels':{'GAN':kernel},'code_sha256':r0_hashes(),'candidates_sha256':object_hash([candidate])})
    write_jsonl(old/'candidates.jsonl',[candidate]);write_jsonl(old/'candidate_pool.jsonl',[])
    write_jsonl(old/'synthesis_results.jsonl',[result])
    before={str(p):sha256(p) for p in old.rglob('*') if p.is_file()}
    archive=R0Archive(old)
    audits,reused=archive.authenticate()
    assert audits[0]['status']=='authenticated_reuse'
    assert candidate['canonical_signature'] in reused
    assert before=={str(p):sha256(p) for p in old.rglob('*') if p.is_file()}
    # A different target is rejected at recovery; the original is left untouched.
    manifest=copy.deepcopy(archive.manifest)
    manifest['context']['requested_clock_ns']=5
    other=tmp_path/'wrong-context'
    import shutil
    shutil.copytree(old,other)
    write_json(other/'manifest.json',manifest)
    with pytest.raises(ValueError,match='context'):R0Archive(other)
    # Failed authentication remains historical_unverified, never a comparison cell.
    bad=copy.deepcopy(result);bad['report_sha256']='wrong'
    archive.results=[bad]
    audits,reused=archive.authenticate()
    assert audits[0]['status']=='historical_unverified' and not reused
    assert before=={str(p):sha256(p) for p in old.rglob('*') if p.is_file()}


def test_real_GAN_native_domain_and_sparse_source_contract():
    spec = cli.SPECS["GAN"]
    info = cli.repo_path(spec["metadata"]).parent
    actions = dd.load_source_actions("GAN", info)
    template = dd.prepare_source_template("GAN", cli.repo_path(spec["source"]), info)
    native = dd.source_site_domains("GAN", template, info)
    assert "6" not in native[c.key("ARRAY_F", "L2")]
    assert "4" in native[c.key("ARRAY_F", "L2")]
    assert "L11" in actions and actions["L11"].kind == "loop"
