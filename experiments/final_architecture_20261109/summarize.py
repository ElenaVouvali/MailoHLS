"""Rebuild compact reports and figures solely from recorded CPU evidence."""
from pathlib import Path
import json,csv,shutil,hashlib,subprocess
from .audit import ROOT,OUT,sha
D=ROOT/'docs/architecture_freeze';PK=ROOT/'experiments/final_architecture_20261109'
def write(name,text):(D/name).write_text(text.strip()+'\n')
def main():
 D.mkdir(exist_ok=True); evidence=PK/'results';evidence.mkdir(exist_ok=True)
 for src in ('reaudit/v4b_reaudit.json','reaudit/supplement.json','contrasts/target_contrasts.json','contrasts/target_contrasts.csv','paper_inventory.json'):
  f=OUT/src
  if f.exists():
   if f.name=='target_contrasts.json':
    compact=json.loads(f.read_text())
    for row in compact['rows']:
     for field in ('left_unique_hashes','right_unique_hashes','left_acceptable_hashes','right_acceptable_hashes'):
      values=row.pop(field);row[field+'_count']=len(values);row[field+'_sha256']=hashlib.sha256(json.dumps(values).encode()).hexdigest()
    compact['full_hash_lists']={'local_path':str(f),'sha256':sha(f),'note':'Large raw set ledger retained outside Git; rerun prepare from authenticated data to regenerate'}
    (evidence/f.name).write_text(json.dumps(compact,indent=2)+'\n')
   else:shutil.copy2(f,evidence/f.name)
 a=json.loads((OUT/'reaudit/v4b_reaudit.json').read_text());c=json.loads((OUT/'contrasts/target_contrasts.json').read_text());h=json.loads((OUT/'reaudit/historical_replay.json').read_text());p=a['path_tests']
 (evidence/'adp_replay_summary.json').write_text(json.dumps({k:v['correct']['family_macro'] for k,v in h.items()},indent=2)+'\n')
 decisions=[]
 for component,decision,status,reason in [
 ('research Stage1','V4 near-optimal multi-reference latency','retained baseline','Specified research parent; no new end-task comparison qualifies replacement'),
 ('production registries','unchanged independent V1 paths','retained baseline','Latency production fallback remains Stage1; ADP Stage2/3 identities are separate'),
 ('target prompt','device/clock/objective tokens and component budgets','retained baseline','Canonical production contract; physical adaptation remains unqualified'),
 ('compiler encoder','frozen RC3 V4b two-layer 64-to-128','unresolved','Authenticated feature extractor; historical ADP head selection is not latency Stage2 selection'),
 ('hardware messages','no promotion or rejection','unresolved','Real path verified; historical incremental ADP advantage absent, validation latency reversals absent'),
 ('FiLM/gates','no promotion or rejection','unresolved','Gate learns; comparable wider head is stronger historically; convergence unresolved'),
 ('fusion','static-memory compatible fresh XAttn first','unresolved','No new convergence-qualified static baseline; 20 updates insufficient'),
 ('fusion layers','no newly optimal schedule selected','unresolved','Historical 8/16/24/32 anchor only; no matched convergence comparison'),
 ('DPO','optional, excluded from selected latency endpoint','unresolved','Archived held-out preference ranking fell 19/54 to 18/54; fused DPO not tested'),
 ('refinement','separate optional mode; NOT_RUN','unresolved','One-pass endpoint preserved; no qualified core arm yet')]:
  decisions.append({'component':component,'selected':decision,'status':status,'reason':reason,'test_status':'NOT_RUN','confidence':'limited development evidence','hardware_outcome_evidence':'no new generated-configuration QoR','risk_of_leakage':'locked test excluded','reversal_condition':'matched development/confirmation benefit after correctness and convergence gates'})
 (D/'architecture_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
 records=[]
 for eid,hyp,status,metrics,limit in [
 ('C4','hardware/cache/FiLM implementation path','PASS',{'cache_comparisons':96,'source_hashes_equal':all(v['equal'] for v in a['authentication']['source_checks'].values()),'reauthenticated_measurements':3664},'one real training kernel across six contexts for path diagnostics'),
 ('C4_ADP_replay','historical incremental hardware/FiLM advantage','INCONCLUSIVE',{k:v['correct']['selection_score'] for k,v in h.items()},'ADP, one seed, two validation families; not latency generation'),
 ('D5','latency common-configuration contrasts','PASS',c['counts'],'resource proxies; no timing-qualified contexts; 2 device views correlated'),
 ('Stage1_reference','real V4 constrained reference decode','BLOCKED',{},'Lionо V100S occupied; Davinci SSH host key unverified'),
 ('static_XAttn','convergence-qualified static fusion','BLOCKED',{},'safe GPU unavailable; no convergence experiment performed'),
 ('location_training','matched H0-H5/F1-F3 latency selection','NOT_RUN',{},'no trustworthy validation reversal cohort or convergence baseline'),
 ('new_synthesis','confirm development target switches','NOT_RUN',{},'prior bank anchor finite latency unresolved; no blind queue'),
 ('locked_test','frozen architecture final validation','NOT_RUN',{},'choices unresolved; test remains locked')]:
  records.append({'experiment_id':eid,'hypothesis':hyp,'baseline':'V4 latency for Stage2; corrected V4b for predictor audit','variant':eid,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dataset_identity':a['authentication']['cache_sha256'],'split_identity':a['authentication']['manifest_sha256'],'checkpoint_identity':a['historical_identities'],'seed':123,'GPU_host':None,'GPU_UUID':None,'training_updates':0,'training_runtime':0,'evaluation_contexts':18 if eid=='C4_ADP_replay' else None,'available_metrics':metrics,'confidence_intervals_if_applicable':None,'synthesis_receipts':[],'status':status,'limitations':limit,'decision':'retain established path; no qualified new architecture'})
 (D/'experiment_results.json').write_text(json.dumps(records,indent=2)+'\n')
 with (D/'experiment_results.csv').open('w',newline='') as f:
  fields=['experiment_id','hypothesis','status','training_updates','training_runtime','limitations','decision'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
 # Add exact historical score replay values with explicit failures and denominators.
 lines=['| Predictor | Feasible-selection failure (family macro) | Conditional ADP regret | Combined |','|---|---:|---:|---:|']
 for arm,v in h.items():
  q=v['correct'];m=q['family_macro'];lines.append(f"| {arm} | {m['selection_failure']:.6f} | {m['top1_regret']:.6f} | {q['selection_score']:.6f} |")
 supplement=OUT/'reaudit/supplement.json'
 if supplement.exists():
  q=json.loads(supplement.read_text())['capacity_replay'];m=q['family_macro'];lines.append(f"| wider compiler head | {m['selection_failure']:.6f} | {m['top1_regret']:.6f} | {q['selection_score']:.6f} |")
 write('V4B_IMPLEMENTATION_REAUDIT.md',f'''# Corrected V4b implementation re-audit

Verdict: no incremental hardware-message or FiLM advantage demonstrated under the historical ADP protocol. This does not reject hardware-aware memory for latency Stage2.

All saved source hashes in the corrected full37 protocol match this implementation. All 3,664 cache measurements were reauthenticated against source/table/graph/split identities. The cache SHA is `{a['authentication']['cache_sha256']}`. See [raw re-audit](../../experiments/final_architecture_20261109/results/v4b_reaudit.json).

`V4bGraphModel.encode` removes hardware edges only for compiler memory; `from_memory` concatenates `graph.hardware_context` into both heads for both arms. Budget is separate and absent from physical predictions. `SmallGraphEncoder` executes exactly two residual typed TransformerConv layers with reverse derived edges. On machsuite-gemm-blocked, all five action nodes are one message hop from hardware nodes. Removing hardware edges leaves 256 edges versus 322. Hardware-feature gradient norm is {p['input_gradient_tests']['hardware']['hardware_node_gradient_norm']:.6f} with messages and exactly zero without them. This verifies transmission, not useful optimization.

The fast `(kernel,config_hash)` cache equals fresh `(kernel,device,clock,config_hash)` tensors exactly for 96 candidate contexts across both devices and three clocks. Candidate demand remains source/candidate-derived. No cache fix was warranted. Compiler memory is invariant; hardware and FiLM memory vary. The direct numeric head ablation changes predictions with memory fixed. H0/H1/H2/H3 therefore have distinct information paths; H2/H3 here are inference diagnostics, not separately trained arms.

FiLM modulates matched action vectors; pooled program vectors remain unchanged. Its selected raw gate is {p['film_selected_gate']:.8f}, effective gate {p['film_effective_gate']:.8f}; modulation/base norm ratio {p['film_modulation_relative_norm']:.8f} on the first audited context. In two disposable CPU diagnostic updates, gate gradient at zero is {p['film_two_step_diagnostic'][0]['gate_gradient']:.9f}; MLP gradient norm moves from zero to {p['film_two_step_diagnostic'][1]['modulation_gradient_norm']:.9g}. These two updates establish gradient mechanics only. They are excluded from GPU/full-training spend and no model is saved or promoted.

Historical 1,500-update arms share exact sampled identities and compatible initial states. FiLM selects step250; a plateau was not established. The capacity head uses191,604 parameters versus FiLM191,667 and selects step500. Full model compiler/hardware each127,538. Recovered historical weights and receipts are hashed, with source commits in the JSON. No gate-gradient trajectory was logged historically; do not reconstruct one.

{chr(10).join(lines)}

Each result has three validation kernels, two families,18 physical contexts,90 correlated budget views. Lower combined score means failure+conditional regret; it is not unconditional latency. No graph improves both coverage and regret over every control. Zero/shuffled/wrong-device memory controls were replayed; wrong-device memories retain the true numeric head target. These models evaluate supplied designs and do not generate directives.

305 training and5 validation selected observations have all-zero resource percentages; the corrected ADP ordering excludes them globally while physical regression retains proxy zero labels. Supplement reports normalization sensitivity after excluding them, without retraining. Exact integer counts and timing compliance remain unavailable here.

| Concern | Verdict | Limitation |
|---|---|---|
| Target path and hardware gradients | VERIFIED | One real kernel, six physical contexts |
| Shared initialization and schedule | VERIFIED | One historical seed |
| Real candidate cache equivalence | VERIFIED | 96 comparisons; source version pinned |
| Global zero-ADP correction | VERIFIED | Regression still includes ambiguous proxy labels |
| FiLM gate mechanism | VERIFIED | Historical convergence INCONCLUSIVE |
| Latency-oriented matched training | NOT_TESTED | Requires informative development cohort |
| Latency Stage2 advantage | NOT_TESTED | Standalone head is a different task |
''')
 write('TARGET_CONDITIONING_ANALYSIS.md',f'''# Target-conditioning data audit

[All per-contrast rows](../../experiments/final_architecture_20261109/results/target_contrasts.csv) and [complete sets/counts](../../experiments/final_architecture_20261109/results/target_contrasts.json) derive from97,442 authenticated non-test complete measurements across40 kernels. There are no conflicting duplicate physical labels in this reconstruction. No test labels were loaded.

The audit intersects complete assignment hashes before classifying response. It records original target-specific pool sizes, feasible proxy sets, common support,1% acceptable sets and both-direction transfer regret. A reversal needs at least5% loss from reusing the opposite winner in both directions. Budget changes reuse physical measurements. Device views explicitly distinguish full fractions from common absolute ZCU limits; these are correlated views.

Counts: `{json.dumps(c['counts'],sort_keys=True)}`.

Training has3 clock-reversal rows and2 device-reversal views. The device views describe the same physical contrast, not two independent examples. Reversals involve lavaMD and spcl_example families. The spcl contrast has an extreme latency ratio (>36,000); authenticate original report units before using it as training emphasis. These are latency-specific candidate-bank signals, not qualified generated hardware improvements. Resource feasibility uses percentage-derived counts; rounded zero observations and timing uncertainty prevent exact feasibility claims.

Validation contains no latency reversal on shared pools:10 supported clock contrasts and16 device views retain an acceptable common winner. Two clock and two device views lack support. Budget views show11 proxy boundaries,43 invariants. These are3 kernels in2 families, not independent budget examples. No timing-qualified context is available from these tables. A changed feasible proxy set is insufficient proof of physical resource binding.

Conclusion: input variation exists, but validation lacks trustworthy switch coverage needed to rank target-memory architectures. Retain invariant contexts. Confirm one development switch with unrounded resource reports and finite latency before escalating. Missing metrics are unknown; unchanged directives may be correct on invariant contexts.
''')
 write('XATTN_ARCHITECTURE_ANALYSIS.md','''# Cross-attention qualification

The existing gated block and `post_self_attention_residual` native layernorm pre-hook preserve the decoder MLP. RC3 already corrected native residual dtype handling. Selected RHS positions use forward-filled absolute action IDs, including the causal position scoring the first RHS token. A single prefix vector does not determine a multi-token candidate score.

The focused suite passed56 tests including historical V4b, structural attention and multikernel contracts, plus two new matched-pool regressions. This is CPU correctness evidence. The inherited real-decoder zero-gate parity receipt is historical; no new real LLM parity run occurred here.

The previous C pilot20 updates selected step0 with unchanged validation accuracy. It does not establish convergence or fusion inferiority. No gate, placement, routing or memory format is newly selected. A/B action memory and variable-length C evidence are separate schemas. Metadata type/status is not automatically consumed as learned features.

First Davinci experiment: exact V4 zero-gate decoder parity, one-batch gradient/update smoke, then one resumable static-memory run to a validation plateau or declared500-update/8-GPU-hour bound. Record gate and Q/K/V gradients after the first update, skipped updates, structural/residual norms, natural complete decode, and controlled-prefix authentic/gate-off/shuffled/witness-disabled interventions. Compare small nonzero initialization only with identical projection weights and schedule. Then compare matched placements; do not reject untrained alternatives. No convergence-qualified baseline completed in this run.
''')
 write('GNN_ARCHITECTURE_ANALYSIS.md','''# GNN architecture and supervision

The V1 paper encoder is four TransformerConv layers with JK-max and concatenated first/JK states. RC3 V4b is two typed64D layers and a learned64-to-128 projection. Equal64x128 export shape does not establish compatibility. V1 attention weights must never read RC3 memory without fresh matched fusion training.

The corrected V4b frozen compiler checkpoint is authenticated by SHA ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162. It is a legitimate experimental feature extractor, selected by an ADP predictor criterion. No evidence here selects it as the optimal latency Stage2 encoder. New hardware-message or FiLM rejection would be unjustified; see V4B_IMPLEMENTATION_REAUDIT.md.

RC3 graphs preserve action slots, loop hierarchy, affine accesses, SSA recurrence witnesses and explicit knownness. Edge witnesses in C are deterministic composites; a node embedding is not proof that every relation is recoverable. Static memory must contain neither candidate directives, budgets, objective labels nor measured QoR. Physical context may change hardware overlay embeddings; requested budget changes ranking eligibility, not fixed-design physical facts.

Archive reconciliation inherited from the prior branch:63 canonical variants /61 names,55 measured-source matches,40 curated V4 kernels,38 complete mappings,20 pilot packs. These are different denominators. Ambiguous variant names are preserved rather than silently overwritten. Hotspot repairs staged in the user's other checkout were not imported.

No depth/JK or latency-supervision training was launched. Test compiler/hardware/FiLM under matched latency feasible-ranking labels before considering depth. Auxiliary predictor superiority cannot select exported memory without downstream matched generated-design evaluation.
''')
 write('TRAINING_OBJECTIVE_ANALYSIS.md','''# Training objective decision

The research parent remains V4 latency near-optimal multi-reference SFT. It uses rank8 attention Q/K/V/O LoRA and trained added-token embeddings; the saved adapters do not target FFN projections. Preserve action-normalized nonforced RHS supervision and complete correlated assignments. Do not select labels field by field from different measured designs.

This run did not compare matched continued SFT and switch-enriched SFT. Training-only latency contrasts are candidates for authenticated supervision; proxy resource boundaries and extreme spcl ratios need confirmation. Validation switch coverage must improve before architecture selection can claim adaptation quality.

Archived text DPO136 training pairs/17 families and54 validation pairs/2 families improved training ranking82.35% to84.56%, but validation fell35.19% to33.33%. It does not qualify a replacement. DPO necessity is INCONCLUSIVE, and it remains excluded from the selected latency endpoint. Standard summed nonforced completion log-probabilities differ from production mean RHS candidate scores and historical semantic-action DPO reductions. A new compatible Stage2 DPO arm requires matched reference memory, masks, padding, causal prefixes and a chosen-only SFT control.
''')
 write('SYNTHESIS_VALIDATION.md','''# Synthesis evidence and limits

Coroni access was verified; Vitis resolves to /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls with322GiB disk free at preflight. No new synthesis was launched in Revision2. No post-route checks ran. The prior architecture-finalization receipts are retained independently:2 new physical calls, one300s timeout and one completed Pathfinder synthesis with undefined worst-case latency. The latter reports14 BRAM18K,0DSP,1515FF,3418LUT and7.3ns HLS estimate at requested10ns. It cannot support latency ranking. Parser failure due to undefined latency is distinct from synthesis failure.

The previous sealed10-job continuation remains gated; do not launch it blindly. New jobs must be development-only, checksum sealed, deduplicated by source/headers/full assignment/part/clock/tool/settings, transferred to fresh Coroni directories, and retrieved with raw XML/log hashes. Budgets reuse physical receipts. Validate finite latency for an anchor first. HLS estimated clock is not post-route timing, and synthesis is not functional verification.

Next high-information physical work: verify the lavaMD training clock reversal with the identical two complete assignments at both clocks, including original units, finite cycle reports and unrounded resource counts. The suspicious spcl latency ratio must be resolved before amplification. Validation needs an independent-family meaningful switch; do not move test/external workloads into selection.
''')
 write('STRUCTURAL_HARDWARE_COMPLEMENTARITY.md','''# Information availability versus decision use

Compiler facts are explicit and action-aligned in RC3. Real hardware-message gradients and direct numeric-head ablations verify information paths. They do not show that an LLM decodes a recurrence fact or uses it to improve latency. Compiler-only cached memory is target invariant by design; a target-aware LLM query may still retrieve it differently.

Existing GEMV full-depth measurements show contextual sensitivity and autoregressive-prefix dependence. They remain descriptive historical evidence, not family-held-out semantic probes or synthesis-grounded quality. This run did not execute new LLM hidden states, intervention logits or fused generation. Thus compiler fact availability is observed, simple path influence is observed in the standalone predictor, LLM semantic recoverability and useful fused decision influence remain unresolved.

A meaningful joint test needs the same frozen V4 model, compatible freshly trained fusion, authentic/wrong/zero evidence controls and common measured feasible candidate pools. Do not infer complementary information solely because graph relations are explicit or latent vectors differ.
''')
 write('FINAL_RECOMMENDATION.md','''# Final recommendation

Architecture freeze: **NO** for a newly qualified target-conditioned Stage2. Preserve the existing registered production paths. Use frozen V4 multi-reference latency as the research Stage1 parent and static RC3 memory with fresh compatible fusion as the next experimental baseline. No alternative won a matched end-task comparison here.

Three hypotheses remain separate: numeric hardware information reaches the predictors (verified); incremental hardware graph messages improve an already target-aware predictor (not demonstrated under historical ADP); target-conditioned memory improves latency Stage2 generation (not tested). Corrected implementation paths are functional; lack of validation winner reversals limits architectural conclusions.

Do not increase DPO weight, promote FiLM, reject hardware messages, deepen the GNN or freeze four fusion layers from these results. First obtain a convergence-qualified static-memory baseline and a trustworthy development switch/stay cohort. The single most informative next physical experiment is the paired lavaMD clock-winner confirmation described in SYNTHESIS_VALIDATION.md. Davinci can run the GPU baseline after artifacts and GPU preflight pass; pulling Git alone does not transfer checkpoints, caches or memory banks.
''')
 write('ARCHITECTURE_DECISIONS.md','''# Architecture decisions and executable scope

This Revision2 package extends the prior [architecture report](../architecture-finalization/README.md). It corrects the interpretation of V4b and completes CPU cache/message/gradient and latency-common-pool audits. It does not claim a newly optimized model.

The policy remains complete constrained one-pass directive proposal. Source registration determines absolute L1-L64 slots; source-derived domains provide local legality. Target prompts carry objective, canonical part, fixed clock and four available counts/fractions. Production RHS candidates are ranked by mean continuation token log-probability, greedily serialized in field order. Local legality does not guarantee resource feasibility or timing.

The selected research parent is DeepSeek-Coder6.7B revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912, saved V4 latency tokenizer and rank8 attention LoRA with trained special-token deltas, adapter SHA28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164. Its choice is a retained research baseline, not a newly measured superiority claim. Reference request is PARETO_LATENCY / xczu7ev-ffvc1156-2-e /10ns /624BRAM18K,1728DSP,460800FF,230400LUT.

Compiler graphs preserve structured loop/access/SSA evidence and explicit missingness. Frozen V4b produces128D action/node features. A exposes exact action keys; B adds compiler relations; C exposes eligible typed witnesses with provenance and padding. Future compatible fusion uses a hardware-conditioned LLM query and static memory, injecting tanh(gate)*attention output after self-attention residual and before the native MLP. No fresh fusion weights have qualified here. Four-layer V1/ADP and RC3 pilot checkpoints remain independent.

See architecture_decisions.json for component choices, risks and conditions for reversal; V4B_IMPLEMENTATION_REAUDIT.md for controlled predictor evidence; TARGET_CONDITIONING_ANALYSIS.md for measured-bank coverage; REPRODUCIBILITY.md for commands. All new model architecture choices remain unresolved rather than rejected. Parameter cost, generated QoR, inference cost and held-out adaptation require GPU and physical follow-up.

Objective support: registered Stage1 supports latency/area/ADP; registered Stage2 is ADP, latency and area fall back to Stage1; ADP Stage3 is separately recorded with ablation lineage. This research package changes no registry/default. Automatic clock selection, U50 adaptation and optional synthesis-feedback refinement are outside this fixed-clock qualification.
''')
 write('REPRODUCIBILITY.md','''# Reproduction and Davinci handoff

Pull research/final-architecture-decision-20261109 into an isolated persistent worktree. Git contains code and compact receipts, not large checkpoints/tables/caches/memory packs. Verify SHA identities in artifact_manifest.json before use. Install pinned dependencies in an isolated environment; do not alter shared server packages.

```bash
git fetch origin research/final-architecture-decision-20261109
git worktree add ~/MailoHLS-final-architecture-20261109 origin/research/final-architecture-decision-20261109
cd ~/MailoHLS-final-architecture-20261109
export SALT_V4B_RUN_ROOT=/local/path/to/v4b-v2-20261102-8f3593b
export SALT_PREPROCESSED_DIR=/local/path/to/preprocessed_CSVS
python -m experiments.final_architecture_20261109.run audit
python -m experiments.final_architecture_20261109.run audit --execute
python -m experiments.final_architecture_20261109.run prepare --execute
python -m experiments.final_architecture_20261109.evaluate
python -m experiments.final_architecture_20261109.run screen --execute
python -m experiments.final_architecture_20261109.summarize
```

CPU environment used Python3.10, Torch2.6.0+cu124, PyG2.6.1, SciPy1.15.3. PyG/SciPy were restored into an isolated venv with shared LLM packages read-only. `audit.py` verifies cache/graph/model/source hashes, rebuilds fresh candidate tensors and replays historical ADP selection. Artifacts live under artifacts/final_architecture_20261109. Existing complete receipts are immutable; a rerun must use a new output when results differ. The report generator is deterministic and loads no LLM. Full candidate-hash set ledgers remain in the ignored raw JSON; compact Git receipts contain exact set hashes and counts.

GPU preflight:
```bash
hostname
nvidia-smi --query-gpu=uuid,name,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
python -m experiments.final_architecture_20261109.run baseline \
  --stage1 /local/v4/multi_reference/run --data-dir /local/frozen_requests \
  --run-dir /local/new_baseline --gpu-uuid GPU-VERIFIED-ON-THIS-HOST
```
Add --execute only after reviewing the dry-run. `requests.jsonl` must use the canonical frozen request schema. The wrapper verifies V4 hash, rejects occupied/insufficient GPUs, checks CUDA, then uses the production decoder. It neither guesses UUIDs nor terminates Ray. Liono V100S had6105MiB free and an active26382MiB allocation; Davinci SSH failed host-key verification. No GPU forward/optimizer update occurred in this run.

For static training use the portable wrapper around `experiments.rc3_stage2_multikernel.train.command`:
```bash
python -m experiments.final_architecture_20261109.run static_smoke \
 --stage1 /local/v4/multi_reference/run --data-dir /local/audit \
 --memory-dir /local/memory_c/memory --run-dir /local/new_run/smoke \
 --gpu-uuid GPU-VERIFIED-ON-THIS-HOST
python -m experiments.final_architecture_20261109.run static_train \
 --stage1 /local/v4/multi_reference/run --data-dir /local/audit \
 --memory-dir /local/memory_c/memory --run-dir /local/new_run/static \
 --gpu-uuid GPU-VERIFIED-ON-THIS-HOST --steps 500
```
Both commands default to dry-run. Add --execute after review and successful exact-decoder parity/gradient smoke. Resume static training with --resume /local/new_run/static/checkpoint-N; the production Trainer restores optimizer/scheduler/RNG state. The smoke uses one layer32 update; static training uses the historical four-layer schedule as an unqualified anchor. No automatic early plateau claim is made. Memory manifest/source alignment must pass the production loader; no old V1 memory is substituted. Changing data/layer/precision contracts requires a new run directory.

The wrapper uses explicit local V4/data/memory/output paths. First run the existing smoke command dry-run. Its historical launcher is liono-specific; do not execute its hard-coded UUID on Davinci. The new portable preflight can validate a Davinci UUID. A bounded convergence protocol must expose steps, validation curves, optimizer/RNG resume and exact decoder parity; the historical20-step pilot is insufficient. This package deliberately refuses train_finalists while those gates are unmet. It does not provide an untested automatic convergence launcher.

Synthesis remains Coroni-only. Existing sealed bundle instructions are in ../architecture-finalization/README.md. Preserve tool/version/source identities and all failed jobs. Do not run the pending bank until finite anchor latency is resolved. Revision2 spent0 new HLS jobs,0 GPU hours and0 post-route checks;2 disposable CPU FiLM optimizer steps tested gradient mechanics. Limits remain8GPUh/30initial new HLS/48maximum/6configured post-route.

No locked-test metrics or external workload outcomes were read for selection. No new checkpoint was trained or production model overwritten.
''')
 # Exact active-paper claim anchors, without attributing figures to unverified checkpoints.
 write('TCAD_CLAIM_CROSSWALK.md','''# Active TCAD draft to evidence

The ZIP is now present and was read. Exact ZIP/member hashes are in results/paper_inventory.json. Historical report saying ZIP absent applies to the earlier run only. Manuscript files are extracted under ignored artifacts, unchanged. Active lines exclude commented alternatives.

| Active section / anchor | Claim | Code / artifact | Status and suggested correction |
|---|---|---|---|
|04 Design Overview; fig:design-overview | Three-stage hybrid for objectives | FINAL_ARTIFACTS.json, saved Stage1/Stage2/Stage3 contracts | Supported by older architecture only for compatible registered paths; describe latency research as Stage1 baseline until fusion qualifies |
|05 Detailed Design; subsubsec:peft, line665 | Attention and FFN LoRA | adapter_config.json and trained parameter receipts | Contradicted for authenticated adapters: list Q/K/V/O attention targets; do not claim FFN adaptation |
|05 gnn-backbone;06 training | Four-layer JK,128D first+JK export | V1 encoder and production memory lineage | Older architecture only; RC3 V4b uses two64D layers and128D projection |
|06 evaluation data |130310/13011/6705 raw rows, family split | split manifest; prior reconciliation | Historical full cohort; distinguish2524/54 V4 curated rows and1164/54 RC3 pilot from raw counts |
|05 target-conditioning; fig:target-conditioning | GAN target response | historical target-conditioning figure and synthesis identity | Locked external descriptive result; not architecture-selection evidence or RC3 proof; checkpoint attribution unresolved |
|05 structural-fusion; tab:memory-ablation | Fixed Kalman latency intervention | frozen historical experiment | Historical-only locked case; validity does not register a production latency Stage2 |
|06 Stage2 training | layers8/16/24/32;3epochs;8.68M | saved ADP contracts vs RC3 20-update receipt | Registered/older objective-specific contract; do not apply to RC3 convergence or all objectives |
|05 DPO equation;06 Stage3 | beta0.5, NLL0.02, completion likelihood | train_DPO_harp_xattn.py; saved reduction contract | Historical semantic-action reduction differs from standard summed completion DPO; state exact implementation and objective lineage |
|07 graph-representation-comparison; tab:representation-predictive | Matched graph superiority | original graph comparison receipt required | Not reproduced by cross-schema V1/RC3 comparisons; table attribution pending |
|07 target-generalization; fig:target-generalization |6.67ns/U50 transfer and post-route | historical external transfer receipts | Historical-only; no new transfer/post-route execution here |
|07 SoTA/cost figures;08 limitations | End-task gains and optimization cost | matching source/tool/checkpoint receipts required | Do not transfer figures to newly selected RC3 policy; new matched quality comparison pending |

Suggested wording: “Corrected RC3 experiments verify hardware-message transmission but do not demonstrate an incremental benefit under the historical ADP protocol. Latency-oriented Stage2 target adaptation remains under qualification.”

Source, checkpoint and objective attribution for every historical figure has not been recovered. Mark those pending rather than inventing identities. Preserve all original figures and measurements. The draft's constrained action space is local admissibility, not proof of synthesis, numerical correctness or timing closure.
''')
 (PK/'README.md').write_text('# Revision2 architecture investigation\n\nCompleted CPU evidence:96 real cache comparisons;3664 reauthenticated selected measurements; corrected ADP replay;97442 non-test physical-record target audit;56 focused tests. GPU/convergence/physical qualification remain blocked or not run.\n\nStart with [architecture decisions](../../docs/architecture_freeze/ARCHITECTURE_DECISIONS.md), [V4b re-audit](../../docs/architecture_freeze/V4B_IMPLEMENTATION_REAUDIT.md) and [Davinci reproduction](../../docs/architecture_freeze/REPRODUCIBILITY.md).\n\n`python -m experiments.final_architecture_20261109.run --help` provides guarded CPU/baseline entry points; dry-run is default. New training/final synthesis are explicitly gated, not implemented as an automatic sweep.\n')
 manifest={'schema':'architecture-r2-artifacts-v1','source_parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'large_artifacts':{'root':str(OUT),'v4b_runs':str(__import__('experiments.final_architecture_20261109.audit',fromlist=['RUN']).RUN)},'files':{str(f.relative_to(ROOT)):sha(f) for f in sorted(evidence.iterdir())},'experimental_source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in sorted(PK.glob('*.py'))},'code_commit_note':'code_commit fields record parent HEAD during execution; exact experimental source hashes accompany this receipt','test_labels_loaded':False}
 (D/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('reports generated',D)
if __name__=='__main__':main()
