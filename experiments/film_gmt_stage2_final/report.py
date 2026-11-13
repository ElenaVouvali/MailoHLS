"""Seal final preparation receipts; never starts model training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.film_gmt_stage2_final.bank_contract import ROOT,sha,write


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(run):
    run=run.resolve();proof=ROOT/'docs/provenance/film_gmt_stage2_final_20261113_original_anchors'
    if proof.exists():raise FileExistsError('Preserve sealed preparation evidence')
    smoke=json.loads((run/'final_native_cached_smoke/smoke/smoke.json').read_text())
    cpu=json.loads((run/'final_native_cached_audit/audit/preflight.json').read_text())
    gmt=json.loads((run/'fallback_memory_original_anchors/PREPARATION.json').read_text())
    expanded=json.loads((run/'expanded_qualification/qualification.json').read_text())
    static=json.loads((run/'historical_all_sources_static_audit.json').read_text())
    if smoke['status']!='SMOKE_PASS' or smoke['dtype']!='torch.float16':raise ValueError('Native FP16 smoke not qualified')
    if '75 passed' not in (run/'final_tests_original_anchors.log').read_text():raise ValueError('Final CPU regression checks incomplete')
    proof.mkdir(parents=True)
    bank=run/'fallback_bank_original_anchors';bank_m=json.loads((bank/'manifest.json').read_text())
    memory=run/'fallback_memory_original_anchors/memory_B';memory_m=json.loads((memory/'memory_manifest.json').read_text())
    identity=json.loads((run/'fallback_memory_original_anchors/historical_identity.json').read_text())
    source_files={str(p.relative_to(ROOT)):sha(p) for directory in ['experiments/film_gmt_stage2_final',
        'experiments/historical_mlir_gmt_gemv','experiments/historical_gmt_stage2_ab'] for p in sorted((ROOT/directory).glob('*.py'))}
    versioned={'schema':'film-gmt-stage2-supervision-snapshot-v1','version':'20261113-original-film-31+2-original-anchors-v2',
        'status':'QUALIFIED_EXPLICIT_FALLBACK','cohort':'original-film-31+2',
        'complete_recovered_49_4_bank':False,'source_contract':'canonical-csv-apl-v1',
        'recovered_sidecars_applied':False,'locked_test_measurements_read':False,
        'base_stage1_bank_manifest_sha256':sha(bank/'manifest.json'),'outputs_sha256':bank_m['outputs_sha256'],
        'dataset_identity_sha256':object_sha(bank_m['outputs_sha256']),
        'coverage':cpu['training_context_coverage'],'source_files_sha256':source_files,
        'full_source_action_assignments_validated':True,'unknown_states_imputed':False}
    write(bank/'stage2_manifest.json',versioned)
    source_copy={'expanded_qualification.json':run/'expanded_qualification/qualification.json',
        'historical_all_sources_static_audit.json':run/'historical_all_sources_static_audit.json',
        'cpu_audit.json':run/'final_native_cached_audit/audit/preflight.json','v100s_fp16_smoke.json':run/'final_native_cached_smoke/smoke/smoke.json',
        'gmt_preparation.json':run/'fallback_memory_original_anchors/PREPARATION.json','gmt_history.json':run/'fallback_memory_original_anchors/gmt_history.json',
        'historical_identity.json':run/'fallback_memory_original_anchors/historical_identity.json',
        'original_cache_anchors.json':run/'fallback_memory_original_anchors/original_cache_anchors.json',
        'original_cached_anchor_comparison.json':run/'original_cached_anchor_comparison.json',
        'target_conditioning_audit.json':run/'target_conditioning_audit.json',
        'memory_manifest.json':memory/'memory_manifest.json','stage2_manifest.json':bank/'stage2_manifest.json',
        'final_assets.json':bank/'final_assets.json','checkout_materialization.json':run/'checkout_materialization.json'}
    for name,src in source_copy.items():shutil.copyfile(src,proof/name)
    logs={p.name:{'sha256':sha(p),'text':p.read_text()} for p in run.glob('*.log')}
    write(proof/'validation_commands.json',{'schema':'film-gmt-final-validation-v1','logs':logs,
        'commands':{'expanded':'python -m experiments.film_gmt_stage2_final.bank_contract --parent-registry PARENT/directive_domain_registry.json --out RUN/expanded_qualification',
            'memory':'CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb PYTHONHASHSEED=0 /home/elvouvali/.hls-llm/bin/python -u -m experiments.film_gmt_stage2_final.prepare_memory --bank ORIGINAL_BANK --graphs /home/elvouvali/MailoHLS/GNN_branch/MLIR_graphs --out RUN/fallback_memory --train-gmt',
            'original_anchor_export':'/home/elvouvali/.hls-llm/bin/python -m experiments.film_gmt_stage2_final.pin_cached_anchors --prepared RUN/fallback_memory --cache /home/elvouvali/MailoHLS/artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned --bank RUN/fallback_bank --out RUN/fallback_memory_original_anchors',
            'smoke':'CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHON -m experiments.film_gmt_stage2_final.train --mode smoke --cohort original-film-31+2 --repo ROOT --parent PARENT --film-checkpoint FILM_CP --gnosis-bank GNOSIS_BANK --gmt-memory GMT_MEMORY --gpu-uuid GPU_UUID --output RUN/v100s_smoke_fp16',
            'canonical_cache':'python -m Preprocessing.consolidate_data verify-cache',
            'recovery':'python -m Preprocessing.gnosis_recovery --verify',
            'compilation':'python -m Preprocessing.gnosis_compilation_recovery --verify',
            'tests':'PYTHON -m pytest experiments/film_gmt_stage2_final/test_integration.py LLM_branch/tests/test_structural_xattn.py Preprocessing/tests/test_canonical_data.py Preprocessing/tests/test_canonical_preprocessing.py Preprocessing/tests/test_final_experiment_contract.py Preprocessing/tests/test_placeholder_contract.py Preprocessing/tests/test_gnosis_recovery.py Preprocessing/tests/test_gnosis_compilation_recovery.py -q',
            'historical_tests':'/home/elvouvali/.hls-llm/bin/python -m unittest discover -s experiments/historical_mlir_gmt_gemv -p test_historical_gmt_slots.py -v'},
        'notes':['One GMT fit, 600 auxiliary updates. No 450-update Stage-2 run.',
                 'Final memory uses original historical cached anchors byte-for-byte, not fresh allclose reconstructions; original and fresh exports are retained.',
                 'First smoke failed during an eval-mode backward diagnostic; fixed checkpointed train-mode probe.',
                 'Next smoke passed under BF16 emulation; native hardware gating corrected and final FP16 smoke passed.',
                 'Four previously ignored recovery log extracts restored by their existing SHA-256, never by replacing stored hashes.']})
    checkpoint_paths={
        'V4_parent':Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run/adapter_model.safetensors'),
        'FiLM_residual':Path(cpu['configuration']['film_checkpoint'])/'device_adapt/adapter_model.safetensors',
        'FiLM':Path(cpu['configuration']['film_checkpoint'])/'film.pt',
        'historical_GNN':Path(identity['checkpoint_path']),'selected_GMT':run/'fallback_memory_original_anchors/gmt_selected.pt'}
    checkpoint_manifest={k:{'path':str(p),'sha256':sha(p)} for k,p in checkpoint_paths.items()}
    inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    summary={'schema':'film-gmt-final-readiness-v1','requested_expanded_cohort':'NOT_READY',
        'explicit_original_31_2_fallback':'READY','reason':'Nine complete canonical sources lack frozen V4 action domains and matching historical compiler action inventories.',
        'canonical_source_eligible':54,'requested_non_test_sources':{'train':49,'val':4},
        'actual_supervised_sources':{'train':31,'val':2},'memory_files':33,
        'dataset_identity_sha256':versioned['dataset_identity_sha256'],
        'dataset_manifest_sha256':sha(bank/'stage2_manifest.json'),
        'base_stage1_bank_manifest_sha256':sha(bank/'manifest.json'),
        'memory_manifest_sha256':sha(memory/'memory_manifest.json'),
        'memory_identity_sha256':object_sha(memory_m),
        'frozen_checkpoints':checkpoint_manifest,'gmt_updates':600,'selected_gmt_step':gmt['gmt_best_step'],
        'historical_static_compatible':sum(e['status']=='STATIC_COMPATIBLE' for e in static['kernels'].values()),
        'historical_sources_requiring_regeneration':[k for k,e in static['kernels'].items() if e['status']!='STATIC_COMPATIBLE'],
        'historical_GNN_fit_validation_family_overlap':[],
        'historical_GNN_checkpoint_selection_used_validation':True,
        'normalization_fit_policy':identity['normalization']['fit_policy'],
        'locked_test_qor_rows_read':False,'full_stage2_updates_launched':0,
        'original_cached_action_anchors_bit_identical':True,'GMT_retrained_for_cache_reexport':False,
        'source_files_sha256':source_files,
        'provided_v2_files_sha256':{str(p):sha(p) for p in [Path('/home/elvouvali/train_film_gmt_stage2_v2.py'),Path('/home/elvouvali/film_gmt_stage2_commands_v2.sh')]},
        'input_revisions':{'recovery':'2e72d2671f75ef476ab23a80ac0c86ae8e077c31','historical_wrappers':'da24e50','selected_FiLM_tree':'5fa4700'},
        'artifacts_sha256':inventory}
    write(proof/'readiness.json',summary)
    report=f'''# Final FiLM / historical GMT Stage-2 preparation

**NOT READY for the requested recovered 49-train + 4-validation cohort.**
**READY for an explicitly selected, independently authenticated original 31+2 fallback.**
No full 450-update Stage-2 campaign was launched. The selected V4 parent, Arm B
step-600 residual LoRA/FiLM, historical epoch-9 GNN, raw measurements, and split
are unchanged. All work is isolated on `research/film-gmt-stage2-final-20261113`.

## Expanded-cohort blockers

Canonical source-level eligibility remains 54: 49 train, four validation, one locked
test. Qualification stopped **before aggregation, Pareto selection or serialization**;
no partial recovered bank was published. The frozen V4 registry lacks 27 RHS keys
for the following nine kernels. The historical graphs also lack the corresponding
complete action inventories / metadata identities. Source hashes themselves match.

| Kernel | Missing action IDs | Fold |
|---|---|---|
'''
    import re
    for k,e in expanded['blockers'].items():
        labels=sorted({re.search(r'_L(\d+)\}',lhs).group(1) for lhs in e['missing_lhs']},key=int)
        report+=f"| `{k}` | "+', '.join('L'+x for x in labels)+f" | {e['split']} |\n"
    report+='''
These are contract / compiler-graph blockers, not unknown historical directive
states. No old registry entries, source actions, graph identities or hashes were
changed to bypass them. Completing this cohort requires a separately declared
source-authenticated Stage-2 domain contract and authentic historical graph
regeneration/qualification. No partial-cohort override was used.

All 53 non-test sources received a static **historical** graph audit: 44 pass
source/action/generator checks; nine need regeneration. Only the complete original
31+2 cohort received numerical GNN/GMT qualification. RC3 compatibility is a
different release contract and is not used to qualify this historical encoder.
The measured-source coalescing KNN version remains separately namespaced under
the existing recovery provenance; it was not substituted, fitted or exported.
Locked-test memory export was omitted; no locked-test QoR rows were parsed.

## Qualified fallback and sampling

The new immutable fallback directory contains `train_sft.jsonl`, `train_pairs.jsonl`,
`val_sft.jsonl`, `val_pairs.jsonl`, the byte-identical selected Stage-1 manifest,
and a separate versioned `stage2_manifest.json`. It is explicitly CSV/APL complete;
it does not pretend to contain recovered sidecar data. The recovery join API is
tested to join complete historical states before configuration grouping; unknown,
incomplete and locked-test assignments fail closed. The expanded-bank loader has
a distinct schema, full-cohort/source/contract/hash gates and preserves original
Stage-1 authentication.

| Quantity | Training | Validation |
|---|---:|---:|
| Supervised kernels / families | 31 / 15 | 2 / 2 |
| SFT references | 3,276 | 216 |
| Budget contexts in bank | 1,237 | 72 |
| Device/clock targets in bank | 181 | 12 |
| Preference examples / complete two-context groups | 498 / 249 | 28 / 14 |

The old 112-context cap and `steps // 3` ceiling are removed. At 450 updates and
four SFT microbatches/update, **all 1,237 training contexts receive one or two
visits**. Every preference group (128 budget, 121 clock) is visited with **both**
target contexts; its two ranking losses are averaged to preserve objective scale.
Twenty-four fixed validation SFT references cover all 12 device/clock targets,
both validation kernels and 20 distinct budget contexts; eight complete preference
groups are used. This is prospective schedule coverage, not a claim that training
has happened. The five missing Cartesian training targets were already absent
from the original authenticated bank: CFD baseline U200/10 ns and four SPCL-01
targets; no new outcomes are invented to fill them.

Exact prompt/RHS token audit covers every SFT reference and both paired alternatives:
maximum 6,660 tokens, below 7,168, with no truncation. The GPU smoke uses the longest
training SFT example (CFD coalescing, 6,659 tokens). Full budget remains 450 updates.

## Historical GNN and GMT receipts

The exact epoch-9 four-layer GNN uses its captured 403-node / 82-edge tensorizer,
training artifacts and normalization; no GNN retraining or normalization refit.
All original 64 x 128 action anchors are copied **byte-for-byte from the existing
historical caches**, with pinned checkpoint/graph/normalization identities and
per-file SHA-256 receipts. A stronger check found fresh GNN inference differs by
up to 3.875e-6 because of floating-point reconstruction. The final export therefore
pins the original cached bytes, recomputes summaries with the same frozen GMT,
and preserves the earlier fresh-inference export and both hashes for review.
No extra GMT fit was run. Appended summaries preserve memory, compute/dependence and control roles;
every active action has genuine compiler evidence and valid relational routing.
No empty, action-only, truncated or mismatched-source fallback is accepted.

One `HistoricalRoleGMT` run completed **600 auxiliary updates**; internal selection
chose **update 330**. The model fit 27 original training kernels; the four KMeans
variants were a family-disjoint internal development group within Stage-2 training.
The two Stage-2 validation families did not participate in GMT fit/selection or
historical GNN gradient fitting. Historical GNN checkpoint selection did use its
held-out validation ranking, and this inherited selection exposure is explicit.
Frozen normalization has `training-kernels-only` fit policy.

Evidence cap is now **16,384**, with the strict overflow rejection retained. The
two largest qualified CFD cases contain 14,477 and 14,281 scoped evidence records;
both pass without truncation. Initial source-size preflight also checked CFD flux,
which correctly failed its historical action-metadata identity. No stored graph
was rewritten to hide that failure.

| Kernel | Fold | Active actions | Evidence records | Memory rows (64 anchors included) |
|---|---|---:|---:|---:|
'''
    val=set(gmt['expected_val_kernels'])
    for k,e in gmt['kernels'].items():
        report+=f"| `{k}` | {'val' if k in val else 'train'} | {e['actions']} | {e['evidence_records']} | {e['memory_tokens']} |\n"
    report+='''
## Validation and engineering changes

CPU tests: **75 passed, 19 subtests passed**; one historical GMT test also passes.
Canonical/cache integrity and both recovery verifiers pass. All original raw CSV,
source/header, mapping, split and checkpoint identities are unchanged.
Four exact `graph.log` extracts referenced by recovery commit `2e72d26` had been
omitted by the global `*.log` ignore. They were restored from the existing local
recovery worktree by their already stored hashes, and a scoped ignore exception
now makes fresh checkouts complete. No stored receipt/hash was regenerated.

The first composite smoke exposed an eval-mode backward diagnostic that disabled
checkpointing on the long CFD example; its failed output is retained. The fixed
probe uses checkpointed training mode. A subsequent successful BF16-emulation
run exposed PyTorch's permissive V100 BF16 detection; capability gating now enforces
native **FP16** on Volta, and the final native-FP16 smoke passes. These are execution
fixes, not architecture experiments. Only standalone smoke updates occurred, with
no trained Stage-2 checkpoint promoted or full campaign started.

Four structural layers remain at 8/16/24/32, with zero initial gates and the original
60-update gate-opening curriculum. V4, residual LoRA, FiLM and historical memories
remain frozen; only 8,683,524 structural attention/gate parameters train. Action-
normalized RHS SFT, target-aware ranking and structural ON/OFF diagnostics remain.

'''
    report+=f"Native-FP16 zero-gate parity absolute delta: **{smoke['zero_gate_film_parity_abs_delta']}**. "
    report+=f"All four gate gradients are finite/nonzero. Opened-branch projection gradients span {min(smoke['projection_gradient_norms_at_probe_gate'].values()):.3g} to {max(smoke['projection_gradient_norms_at_probe_gate'].values()):.3g}. "
    report+=f"At the declared diagnostic gate `tanh(g)=0.05`, mean absolute ON/OFF score delta is {smoke['structural_effect_probe']['mean_abs_score_delta']:.7f}; this is a pathway test, not learned QoR improvement.\n\n"
    report+='| Layer | Actual post-rounding residual RMS / hidden RMS |\n|---|---:|\n'
    for e in smoke['structural_effect_probe']['per_layer']:report+=f"| {e['layer']} | {e['effective_post_rounding_rms_ratio_mean']:.8f} |\n"
    report+='\n## Paths and manual commands\n\nThese paths select the **qualified original fallback**, not a recovered 49+4 bank.\n\n```bash\n'
    variables={'ROOT':str(ROOT),'PARENT':str(checkpoint_paths['V4_parent'].parent),
        'FILM_CP':str(checkpoint_paths['FiLM'].parent),'GNOSIS_BANK':str(bank),'GMT_MEMORY':str(memory),
        'GPU_UUID':'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb',
        'ENTRY':str(ROOT/'experiments/film_gmt_stage2_final/train.py'),'RUN_ROOT':str(run/'campaign')}
    for k,v in variables.items():report+=k+'='+v+'\n'
    report+='```\n\n`GPU_UUID` is a device identifier. `campaign` has no training output yet.\n\n```bash\n'
    for mode in ['audit','smoke','train']:report+=f'COHORT=original-film-31+2 MODE={mode} bash {ROOT}/experiments/film_gmt_stage2_final/commands.sh\n'
    report+='''```

Outputs must be fresh; resume requires an authenticated saved optimizer checkpoint.
The final `MODE=train` command is supplied for manual use and was **not executed**.
Do not set `ALLOW_EXPLICIT_SUBSET=1` (the launcher rejects it).

## SHA-256 identities and recommendation

'''
    for label,value in [('Dataset four-file identity',summary['dataset_identity_sha256']),
        ('Versioned Stage-2 dataset manifest',summary['dataset_manifest_sha256']),
        ('Original selected Stage-1 bank manifest',summary['base_stage1_bank_manifest_sha256']),
        ('GMT memory manifest',summary['memory_manifest_sha256']),
        ('Complete memory identity',summary['memory_identity_sha256'])]:report+=f'- {label}: `{value}`\n'
    for k,e in checkpoint_manifest.items():report+=f"- {k}: `{e['sha256']}`\n"
    report+='''
Complete per-file checkpoint, graph, source, action, memory and runtime hashes,
selection receipts and failed/successful validation output are in
[`docs/provenance/film_gmt_stage2_final_20261113_original_anchors`](provenance/film_gmt_stage2_final_20261113_original_anchors/).
Existing FiLM/GNN/GMT and paper results were not overwritten. No push was made.

**Choose the explicitly qualified original 31+2 cohort for a manual campaign, or
keep the expanded campaign stopped until the nine domain/graph contracts are
resolved. Do not label this fallback as the requested 49+4 integration.**
'''
    (ROOT/'docs/film_gmt_stage2_final.md').write_text(report)
    print(json.dumps({'expanded':'NOT_READY','original_31_2':'READY','report':str(ROOT/'docs/film_gmt_stage2_final.md')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
