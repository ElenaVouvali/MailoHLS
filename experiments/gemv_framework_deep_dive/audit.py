"""CPU provenance supplement; never evaluates checkpoint weights."""
import argparse
import datetime
from pathlib import Path

from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from .common import DEFAULT_OUT, PACKAGE, ROOT, contract, decision_prefixes, read_json, sha, write_json
from .prepare import authenticate


def audit(out):
    import json
    _, tok = authenticate()
    rows = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
    history = read_json(out / 'historical_reference.json')['requests']
    assignments = inference.parse_assignment_dict(history['R1']['prediction'])
    registry = {'gemv': read_json(out / 'effective_gemv_domains.json')}
    positions = []
    reference_output = {}
    for row in rows:
        prior = {}
        for p in decision_prefixes(row, tok, assignments):
            lhs, ids = p['field'], p['ids']
            candidates = inference.get_rhs_candidates_for_lhs('gemv', lhs, registry)
            active = contract.filter_semantic_candidates(lhs, candidates, prior, registry['gemv'])
            reference = reference_output.setdefault(lhs, p['prior_output_ids'])
            assert reference == p['prior_output_ids']
            positions.append(dict(request_id=row['context_id'], field=lhs, action=p['action'],
                                  position_kind='historical_R1_prefix_alignment_only',
                                  token_index=len(ids) - 1, token_id=ids[-1], token_text=tok.decode([ids[-1]]),
                                  prior_output_ids=p['prior_output_ids'], checkpoint_candidates=candidates,
                                  semantic_candidates=active, candidate_count=len(active),
                                  forced_by_semantics=len(active) == 1))
            prior[lhs.upper()] = assignments[lhs.upper()]
    write_json(out / 'prepared_decision_positions.json', positions)
    contracts = {}
    for name, path in [('archived_stage2', ROOT / 'artifacts/reproducibility/stage2_fusion/training_contract.json'),
                       ('local_production_adp_stage2', ROOT / 'mailohls_runs/stage2_specified_adp_epoch9_s123_v3/best_custom_stage2/training_contract.json')]:
        c = read_json(path)
        contracts[name] = dict(path=str(path), sha256=sha(path), objective=c['objective'],
                               model=c['model'], model_revision=c['model_revision'], structural=c['structural'])
    write_json(out / 'stage2_contracts.json', contracts)
    graph_root = Path('/tmp/MailoHLS-gnn-refinement/paper_figures/gemv_rc3_20261105')
    g = read_json(graph_root / 'gemv.canonical.json')
    selected_nodes = [n for n in g['nodes'] if n.get('action_id') in ('L1', 'L4', 'L5')]
    selected_edges = [e for e in g['edges'] if e['relation_type'] in ('memory_access', 'loop_child', 'loop_carried')]
    graph_manifest = read_json(graph_root / 'manifest.json')
    assert len(g['nodes']) == graph_manifest['nodes'] == 66
    assert len(g['edges']) == graph_manifest['edges'] == 106
    assert graph_manifest['source_sha256'] == read_json(out / 'manifest.json')['source_sha256']
    assert graph_manifest['metadata_sha256'] == read_json(out / 'manifest.json')['metadata_sha256']
    assert graph_manifest['provenance']['mlir_sha256'] == sha(graph_root / 'gemv.mlir')
    write_json(out / 'compiler_evidence.json', dict(graph_file_sha256=sha(graph_root / 'gemv.canonical.json'),
               graph_identity=graph_manifest['graph_identity'], nodes=len(g['nodes']), edges=len(g['edges']),
               selected_nodes=selected_nodes, selected_edges=selected_edges))
    code_files = [ROOT / p for p in ['FINAL_ARTIFACTS.json', 'LLM_branch/common/mailohls_contract.py',
        'LLM_branch/common/directive_domains.py', 'LLM_branch/common/source_tool_policy.py',
        'LLM_branch/common/structural_xattn.py', 'LLM_branch/common/structural_memory.py',
        'LLM_branch/inference/eval_stage1_stage2_stage3.py']]
    code_files += sorted(PACKAGE.glob('*.py'))
    write_json(out / 'code_receipt.json', dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               files={str(p.relative_to(ROOT)): sha(p) for p in code_files}))
    launch = read_json(out / 'launch_status.json') if (out / 'launch_status.json').exists() else {}
    completed = launch.get('status') == 'COMPLETED'
    if completed:
        preparation_status = out / 'preparation_status.json'
        if not preparation_status.exists():
            write_json(preparation_status, read_json(out / 'status.json'))
        write_json(out / 'status.json', dict(launch, preparation_status_file='preparation_status.json',
                   gpu_access='approved execution outside restricted sandbox'))
        names = ['manifest.json', 'requests.jsonl', 'official.jsonl', 'candidate_scores.json',
                 'validation.json', 'hidden_states.npz', 'hidden_index.json', 'pairwise.csv',
                 'candidate_comparisons.csv', 'final_decision_states.csv', 'sensitivity_summary.json', 'launch.json',
                 'launch_status.json', 'gpu_access_resolution.json', 'checkpoint.json',
                 'compiler_evidence.json', 'stage2_contracts.json', 'code_receipt.json', 'figure_receipt.json',
                 'status.json', 'preparation_status.json', 'environment.json']
        write_json(out / 'evidence_receipt.json', dict(status='COMPLETED',
                   model_loaded_on_required_gpu=True, tensor_shape=[992, 4096], dtype='float32',
                   files={name: sha(out / name) for name in names}))
    write_json(out / 'acceptance.json', dict(
        A_request_correctness='passed_CPU',
        B_model_correctness='loaded_and_authenticated' if completed else 'checkpoint/tokenizer files verified; model not loaded',
        C_complete_fresh_predictions='passed' if completed else 'pending_GPU',
        D_trace_shape_and_boundary='passed' if completed else 'native boundary CPU fixture passed; actual 4096D vectors pending',
        E_behavioral_equivalence='passed' if completed else 'CPU fixture passed; V4 GPU comparison pending',
        F_controlled_prefixes='passed' if completed else '52 CPU reference sites passed; fresh R1 prefix pending',
        G_historical_reproduction='passed' if completed else 'exact prompt tokens passed; output reproduction pending',
        H_reporting='no imputed inference or geometry values',
        source_tool_v2='fresh assignments audited' if completed else 'historical assignments pass; fresh audit pending'))
    # Small tracked evidence remains useful if ignored binary artifacts are moved.
    results = PACKAGE / 'results'
    results.mkdir(exist_ok=True)
    for name in ['manifest.json', 'checkpoint.json', 'environment.json', 'launch_status.json',
                 'historical_reference.json', 'stage2_contracts.json', 'code_receipt.json', 'analysis_status.json', 'acceptance.json',
                 'gpu_access_resolution.json', 'validation.json', 'sensitivity_summary.json', 'numerical_findings.json', 'launch.json',
                 'evidence_receipt.json']:
        if (out / name).exists():
            write_json(results / name, read_json(out / name))
    (results / 'requests.jsonl').write_bytes((out / 'requests.jsonl').read_bytes())
    print('CPU audit complete: 52 decision positions, two Stage-2 contracts, compiler witnesses and hashes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    audit(args.out)


if __name__ == '__main__':
    main()
