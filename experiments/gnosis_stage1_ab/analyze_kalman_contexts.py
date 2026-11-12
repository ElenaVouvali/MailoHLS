"""Summarize executed Kalman predictions and fixed-prefix context contrasts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODELS = ['parent', 'A_selected', 'B_selected_ON', 'B_selected_OFF']
CONTEXTS = ['zcu104_10ns_100pct', 'u200_10ns_100pct',
            'zcu104_3p33ns_100pct', 'zcu104_10ns_20pct']


def differences(a, b):
    if set(a) != set(b):
        raise ValueError('Compared configurations have different directive fields')
    return {key: {'before': a[key], 'after': b[key]}
            for key in a if a[key] != b[key]}


def compare_scores(before, after):
    if [s['lhs'] for s in before] != [s['lhs'] for s in after]:
        raise ValueError('Fixed-prefix directive plans differ')
    changes = []
    for a, b in zip(before, after):
        av = {x['rhs']: float(x['mean_logprob']) for x in a['candidates']}
        bv = {x['rhs']: float(x['mean_logprob']) for x in b['candidates']}
        if set(av) != set(bv):
            raise ValueError('Fixed-prefix candidate sets differ')
        if not all(math.isfinite(x) for x in list(av.values())+list(bv.values())):
            raise ValueError('Nonfinite candidate score')
        if len(av) < 2:
            continue  # Forced singleton scores are decoder sentinels, not likelihoods.
        ar = [x['rhs'] for x in a['candidates']]
        br = [x['rhs'] for x in b['candidates']]
        reference = ar[0]
        changes.append(dict(lhs=a['lhs'], before_winner=reference, after_winner=br[0],
            winner_changed=reference != br[0], reference_rank_after=br.index(reference)+1,
            before_top_two_margin=av[ar[0]]-av[ar[1]],
            after_top_two_margin=bv[br[0]]-bv[br[1]],
            delta_reference_mean_logprob=bv[reference]-av[reference],
            delta_reference_signed_margin=(bv[reference]-max(v for k,v in bv.items() if k!=reference))
                -(av[reference]-max(v for k,v in av.items() if k!=reference)),
            delta_top_two_margin=(bv[br[0]]-bv[br[1]])-(av[ar[0]]-av[ar[1]]),
            max_abs_candidate_mean_logprob_change=max(abs(bv[k]-av[k]) for k in av)))
    return changes


def shared_generation_prefix(before, after):
    """Compare actual generation scores up to and including first divergence.

    Earlier chosen values are identical here. Later free-running scores have
    different preceding directives and cannot be treated as fixed-prefix effects.
    """
    if before['prompt_sha256'] != after['prompt_sha256']:
        raise ValueError('On/off generation comparison requires the same prompt')
    earlier = {}
    rows = []
    for a,b in zip(before['candidate_scores'],after['candidate_scores']):
        if a['lhs'] != b['lhs']:
            raise ValueError('Generation plans differ')
        compared = compare_scores([a],[b])
        for row in compared:
            rows.append(dict(row, shared_preceding_directives=dict(earlier)))
        arhs,brhs=a['candidates'][0]['rhs'],b['candidates'][0]['rhs']
        if arhs != brhs:
            break
        earlier[a['lhs'].upper()] = arhs
    return rows


def analyze(run):
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest['status'] != 'INFERENCE_COMPLETED':
        raise ValueError('Inference is incomplete; a final comparison cannot be published')
    provenance = json.loads((run/'execution_provenance.json').read_text())
    repo = Path(__file__).resolve().parents[2]
    for name,digest in provenance['backend_file_sha256'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Local reproduction backend differs from executed code: '+name)
    if hashlib.sha256(Path(__file__).with_name('kalman_contexts.py').read_bytes()).hexdigest() != provenance['inference_implementation_sha256']:
        raise ValueError('Inference implementation differs from executed code')
    records = {}
    for model in MODELS:
        for context in CONTEXTS:
            p = run/f'{model}__{context}.json'
            record = json.loads(p.read_text())
            if (record['model'],record['context_id']) != (model,context):
                raise ValueError('Result identity mismatch')
            records[model,context] = record
    reference = records['parent',CONTEXTS[0]]['directives']
    for record in records.values():
        if record['fixed_prefix_reference'] != reference:
            raise ValueError('Comparison uses different preceding-directive references')
        if not record['source_domain_complete']:
            raise ValueError('Incomplete source domains')
    context_effects, model_effects = [], []
    for model in MODELS:
        baseline = records[model,CONTEXTS[0]]
        for context in CONTEXTS[1:]:
            other = records[model,context]
            score = compare_scores(baseline['fixed_prefix_candidate_scores'],
                                   other['fixed_prefix_candidate_scores'])
            context_effects.append(dict(model=model, contrast=context,
                changed_directives=differences(baseline['directives'],other['directives']),
                fixed_prefix_winner_changes=sum(x['winner_changed'] for x in score),
                max_abs_candidate_mean_logprob_change=max(x['max_abs_candidate_mean_logprob_change'] for x in score),
                fixed_prefix_fields=score))
    for first,second in [('parent','A_selected'),('parent','B_selected_ON'),
                         ('A_selected','B_selected_ON'),('B_selected_ON','B_selected_OFF')]:
        for context in CONTEXTS:
            a,b=records[first,context],records[second,context]
            score=compare_scores(a['fixed_prefix_candidate_scores'],b['fixed_prefix_candidate_scores'])
            effect=dict(before=first,after=second,context=context,
                changed_directives=differences(a['directives'],b['directives']),
                fixed_prefix_winner_changes=sum(x['winner_changed'] for x in score),
                max_abs_candidate_mean_logprob_change=max(x['max_abs_candidate_mean_logprob_change'] for x in score),
                fixed_prefix_fields=score)
            if first == 'B_selected_ON' and second == 'B_selected_OFF':
                effect['shared_generated_prefix_fields']=shared_generation_prefix(a,b)
            model_effects.append(effect)
    report=dict(status='COMPLETE_EXECUTED_KALMAN_COMPARISON',predictions=len(records),
        directive_fields=len(reference),source_domain_complete=True,
        context_effects=context_effects,model_effects=model_effects,
        physical_qor_improvement='NOT EVALUATED',
        analysis_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Free-running configuration changes and fixed-prefix ranking sensitivity are separate. Neither establishes good directives or hardware QoR.')
    (run/'observations.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    with (run/'directive_comparison.csv').open('w',newline='') as stream:
        columns=['directive']+[m+'/'+c for m in MODELS for c in CONTEXTS]
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for field in [s['lhs'].upper() for s in records['parent',CONTEXTS[0]]['candidate_scores']]:
            writer.writerow(dict(directive=field,**{m+'/'+c:records[m,c]['directives'][field] for m in MODELS for c in CONTEXTS}))
    observed=[]
    for model in MODELS[:3]:
        entries=[x for x in context_effects if x['model']==model]
        observed += [model+': '+ '; '.join(
            f"{x['contrast']} changes {len(x['changed_directives'])} generated fields"
            for x in entries)+'.','']
    controls=[x for x in model_effects if x['before']=='B_selected_ON']
    for x in controls:
        fields=x['changed_directives']
        if fields:
            observed += [x['context']+': turning B FiLM off changes '+ '; '.join(
                f"{k}: {v['before']} (on) → {v['after']} (off)" for k,v in fields.items())+'.','']
            first=next((s for s in x['shared_generated_prefix_fields'] if s['winner_changed']),None)
            if first:
                observed += [f"The first difference is {first['lhs']}, with identical preceding directive tokens. "
                    f"Its top-two margins are {first['before_top_two_margin']:.6f} (on) "
                    f"and {first['after_top_two_margin']:.6f} (off).",'']
        else:
            observed += [x['context']+': B on/off generates the same complete assignment.','']
    observed += [f"Under the separately controlled common-parent reference prefix, B on/off has "
        f"{sum(x['fixed_prefix_winner_changes'] for x in controls)} candidate-winner changes across four contexts. "
        f"Candidate mean log-probabilities shift by up to "
        f"{max(x['max_abs_candidate_mean_logprob_change'] for x in controls):.6f}. "
        'This fixed-prefix result and the actual-generation first-divergence result answer different questions.','']
    parent_cohort=json.loads((run/'parent_training_cohort.json').read_text())
    if not parent_cohort['kalman_present'] and all(not x['kalman_present'] for x in provenance['training_cohort'].values()):
        observed += ['Kalman is absent from the authenticated parent training JSONL and from the A/B train/validation banks. Training-set identity checks are recorded in the companion provenance files. No measured Kalman labels were used for inference or checkpoint selection.','']
    lines=['# Kalman: selected Stage-1 checkpoints under four target contexts','',
        f'All {len(records)} configurations were computed: parent, selected A, selected B with FiLM on, and the B-off control. Each has {len(reference)} source-domain-valid directive fields. These are Stage-1 predictions; no GNN structural memory is used.','',
        'The common parent is the previously trained V4 SFT adapter from which both matched continuations began. A adds a residual LoRA; B adds a residual LoRA and intermediate-layer FiLM. '
        f"Selected steps are A={manifest['checkpoints']['A']['receipt']['step']} and B={manifest['checkpoints']['B']['receipt']['step']}.",'',
        '## Observed behavior','',
        *observed,
        'No quality ordering of parent/A/B follows from the number of changed fields. The same configuration can be appropriate for two resource budgets, and a changed configuration can be infeasible. These checkpoints were selected at different updates; B-on/off is the control that keeps B weights fixed.','',
        '## Changes relative to ZCU104, 10 ns, full budget','',
        '| Model | Changed context | Changed generated fields | Fixed-prefix candidate-winner changes |',
        '| --- | --- | ---: | ---: |']
    for x in context_effects:
        lines.append(f"| {x['model']} | {x['contrast']} | {len(x['changed_directives'])} | {x['fixed_prefix_winner_changes']} |")
    lines += ['', 'Fixed-prefix comparisons score the same legal continuations under the same earlier directives, taken from the parent baseline prediction. They do not use measured Kalman labels. Full scores and signed margin changes are in `observations.json` and the per-case files.','',
        '## Checkpoint and FiLM comparisons','',
        '| Comparison | Context | Changed generated fields | Fixed-prefix candidate-winner changes |',
        '| --- | --- | ---: | ---: |']
    for x in model_effects:
        lines.append(f"| {x['before']} → {x['after']} | {x['context']} | {len(x['changed_directives'])} | {x['fixed_prefix_winner_changes']} |")
    lines += ['', 'A changed field does not establish a better decision. An unchanged configuration can coexist with changed candidate scores. B-on/off keeps B weights fixed and toggles its FiLM contribution; free-running effects may propagate through earlier choices, while the fixed-prefix scores isolate local ranking changes. The report also compares actual B-on/off generation scores up to the first differing value: those earlier directive tokens are identical, providing an additional matched-prefix check at the first changed decision.','',
        '**Physical QoR improvement: NOT EVALUATED.** There are no matched synthesis results for these assignments, so feasibility under the 20% resource budget, achieved timing at 3.33 ns, latency and FPGA resource usage remain unverified.','',
        'The JSON manifest records checkpoint/source identities and complete target contexts. `predicted_directives.json` contains all assignments; `directive_comparison.csv` provides a field-by-field view.']
    (run/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.run)
    print(json.dumps({k:report[k] for k in ['status','predictions','directive_fields','physical_qor_improvement']},indent=2))
