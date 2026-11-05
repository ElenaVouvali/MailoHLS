"""CPU-only analysis and deterministic PNG/SVG figures from recorded evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import DEFAULT_OUT, PACKAGE, ROOT, read_json, sha, write_json


def pairwise(vectors, index):
    """Corresponding semantic sites, within one authenticated checkpoint only."""
    lookup = {(r['request_id'], r['prefix_mode'], r['action'], r['field'], r['layer'], r['boundary']): r
              for r in index}
    assert len(lookup) == len(index)
    rows = []
    for key, baseline in lookup.items():
        if key[0] != 'R1':
            continue
        a = vectors[baseline['vector_index']].astype(np.float32)
        na = float(np.linalg.norm(a))
        for request in ('R2', 'R3', 'R4'):
            compared = lookup[(request, *key[1:])]
            if baseline['prefix_mode'] == 'controlled':
                assert baseline['prior_output_ids'] == compared['prior_output_ids']
            b = vectors[compared['vector_index']].astype(np.float32)
            nb = float(np.linalg.norm(b))
            rows.append(dict(prefix_mode=baseline['prefix_mode'], action=baseline['action'],
                             field=baseline['field'], layer=baseline['layer'], boundary=baseline['boundary'],
                             comparison=f'R1-vs-{request}', cosine=float(np.dot(a, b) / (na * nb + 1e-12)),
                             relative_l2=float(np.linalg.norm(b - a) / (na + 1e-12)),
                             baseline_norm=na, compared_norm=nb,
                             relative_norm_difference=(nb - na) / (na + 1e-12)))
    return rows


def candidate_comparisons(records):
    rows = []
    lookup = {(r['request_id'], r['prefix_mode'], r['field']): r for r in records}
    for (request, mode, field), a in lookup.items():
        if request != 'R1':
            continue
        for other in ('R2', 'R3', 'R4'):
            b = lookup[(other, mode, field)]
            rows.append(dict(field=field, prefix_mode=mode, comparison=f'R1-vs-{other}',
                             baseline=a['selected'], compared=b['selected'], choice_changed=a['selected'] != b['selected'],
                             baseline_margin=a['margin'], compared_margin=b['margin'],
                             domains_identical=set(a['active_candidates']) == set(b['active_candidates']),
                             ranking_changed=[s['rhs'] for s in a['scores']] != [s['rhs'] for s in b['scores']]))
    return rows


def csv_file(path, rows):
    if rows:
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def save(fig, root, name):
    fig.savefig(root / (name + '.png'), dpi=250, bbox_inches='tight', metadata={'Software': 'GEMV Stage-1 diagnostics'})
    fig.savefig(root / (name + '.svg'), bbox_inches='tight', metadata={'Date': None})
    svg = root / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def missing(root, name, title, detail):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis('off')
    ax.set_title(title, loc='left', weight='bold', pad=18)
    ax.text(.5, .62, 'NO FRESH V100S DATA', ha='center', va='center', fontsize=20, color='#8b3a3a')
    ax.text(.5, .30, detail + '\nNVIDIA driver communication failed; no values imputed.',
            ha='center', va='center', fontsize=11)
    save(fig, root, name)


def action_values(prediction):
    # Pure analysis parser, no tokenizer/model import or GPU access.
    import re
    values = {k.upper(): v for k, v in re.findall(r'(auto\{_[A-Z0-9_]+\})\s*=\s*(\S+)', prediction, re.I)}
    result = []
    for k in range(1, 6):
        names = ('ARRAY_T', 'ARRAY_F', 'ARRAY_D') if k <= 3 else ('PIPE', 'UNROLL')
        result.append(', '.join(values[f'AUTO{{_{name}_L{k}}}'] for name in names))
    return result


def analyze(out):
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.hashsalt': 'gemv-v4-stage1',
                         'axes.spines.top': False, 'axes.spines.right': False})
    requests = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
    manifest = read_json(out / 'manifest.json')
    historical = read_json(out / 'historical_reference.json')['requests']
    official_path = out / 'official.jsonl'
    official = [json.loads(line) for line in official_path.read_text().splitlines()] if official_path.exists() else []
    complete = len(official) == 4 and (out / 'hidden_states.npz').exists()
    metrics, preferences = [], []
    if complete:
        assert read_json(out / 'launch_status.json')['status'] == 'COMPLETED'
        vectors = np.load(out / 'hidden_states.npz')['vectors']
        metrics = pairwise(vectors, read_json(out / 'hidden_index.json'))
        records = read_json(out / 'candidate_scores.json')
        preferences = candidate_comparisons(records)
        csv_file(out / 'pairwise.csv', metrics)
        csv_file(out / 'candidate_comparisons.csv', preferences)
        csv_file(out / 'final_decision_states.csv', [r for r in metrics if r['layer'] == 32
                 and r['boundary'] == 'post_decoder_residual' and r['prefix_mode'] == 'natural'
                 and r['field'] != 'source'])
    write_json(out / 'analysis_status.json', dict(fresh_inference_complete=complete,
               official_requests=len(official), pairwise_rows=len(metrics),
               missing=[] if complete else ['official four-request predictions', 'candidate scores',
                                           'hidden states', 'pairwise metrics', 'GPU behavioral equivalence']))

    # Figure 1: presentation excerpt; source itself remains byte-identical.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axis('off')
    ax.set_title('GEMV: five registered source actions', loc='left', weight='bold')
    lines = [('L1', 'float A[32][32];', 'partition type / factor / dimension'),
             ('L2', 'float x[32];', 'partition type / factor / dimension'),
             ('L3', 'float y[32];', 'partition type / factor / dimension'),
             ('L4', 'for (int i=0; i<32; ++i) {', 'outer: PIPE / UNROLL'),
             ('', '    float acc = 0;', ''),
             ('L5', '    for (int j=0; j<32; ++j) {', 'inner: PIPE / UNROLL'),
             ('', '        acc += A[i][j] * x[j];', 'loop-carried accumulator'),
             ('', '    }', ''), ('', '    y[i] = acc;', ''), ('', '}', ''), ('', 'return y[0];', '')]
    for i, (action, code, note) in enumerate(lines):
        y = .91 - i * .066
        ax.text(.01, y, action, color='#176b8a', weight='bold', transform=ax.transAxes)
        ax.text(.08, y, code, family='monospace', fontsize=12, transform=ax.transAxes)
        ax.text(.66, y, note, fontsize=10, color='#555555', transform=ax.transAxes)
    ax.text(.01, .055, 'Presentation excerpt; exact inputs preserved separately.\nUninitialized A/x and only y[0] observed: representation study, not a numerical GEMV benchmark.', fontsize=10)
    save(fig, figures, 'fig1_actions')

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis('off')
    ax.set_title('Four requests; R1 is the reference', loc='left', weight='bold')
    cells = []
    for r in requests:
        cells.append([r['context_id'], 'ZCU104' if r['context_id'] != 'R4' else 'XCU200',
                      f"{r['clock_period']:g} ns", '20%' if r['context_id'] == 'R2' else '100%',
                      ' / '.join(str(r[k]) for k in ('avail_bram', 'avail_dsp', 'avail_ff', 'avail_lut'))])
    table = ax.table(cellText=cells, colLabels=['ID', 'FPGA', 'Clock', 'Budget', 'BRAM / DSP / FF / LUT'],
                     loc='center', colWidths=[.07, .12, .12, .1, .59], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for rc in [(2, 3), (2, 4), (3, 2), (4, 1), (4, 4)]:
        table[rc].set_facecolor('#d9edf5')
    ax.text(.01, .025, 'Identical authenticated source, metadata, PARETO_LATENCY, specified clock, legacy greedy decoder.', fontsize=10)
    save(fig, figures, 'fig2_requests')

    if complete:
        by_id = {r['context_id']: {s['lhs']: s['chosen_rhs'] for s in r['candidates'][0]['site_debug']} for r in official}
        fields = [s['lhs'] for s in official[0]['candidates'][0]['site_debug']]
        def short_label(field):
            return field.replace('auto{_', '').replace('}', '').replace('ARRAY_T', 'partition type').replace('ARRAY_F', 'partition factor').replace('ARRAY_D', 'partition dimension')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.set_title('V4 multi-reference: official directive predictions', loc='left', weight='bold')
        table = ax.table(cellText=[[short_label(field), *[by_id[k][field] for k in ('R1', 'R2', 'R3', 'R4')]] for field in fields],
                         colLabels=['Action / field', 'R1', 'R2', 'R3', 'R4'], loc='center', cellLoc='center',
                         colWidths=[.40, .15, .15, .15, .15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.7)
        for i, field in enumerate(fields):
            for j, k in enumerate(('R2', 'R3', 'R4'), 2):
                if by_id[k][field] != by_id['R1'][field]:
                    table[(i + 1, j)].set_facecolor('#f4d5a4')
        ax.text(.01, .015, 'Highlighted cells differ from R1. All assignments pass the separate V2 source/tool audit.', fontsize=10)
        save(fig, figures, 'fig3_directives')
    else:
        missing(figures, 'fig3_directives', 'Official four-request directive matrix',
                'R1 / R2 / R3 / R4: not executed in this session.\nHistorical R1/R4 evidence is reported separately.')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_title('Authenticated November 5 V4 outputs — historical evidence only', loc='left', weight='bold')
    hvalues = {k: action_values(v['prediction']) for k, v in historical.items()}
    table = ax.table(cellText=[[f'L{i+1}', hvalues['R1'][i], 'not recorded', 'not recorded', hvalues['R4'][i]] for i in range(5)],
                     colLabels=['Action', 'R1 historical', 'R2', 'R3', 'R4 historical'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.text(.01, .02, 'Arrays: type, factor, dimension. Loops: PIPE, UNROLL. Zero changed fields in R1 vs R4.\nAll 13 assignments revalidated against source and V2 tool policy; no fresh inference or QoR.', fontsize=9)
    save(fig, figures, 'historical_directives')

    if complete:
        for action, suffix in [('L1', 'ARRAY_F'), ('L4', 'UNROLL'), ('L5', 'UNROLL')]:
            field = f'auto{{_{suffix}_{action}}}'
            fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
            for ax, mode, selected_field in zip(axes, ['natural', 'natural', 'controlled'], ['source', field, field]):
                data = [r for r in metrics if r['action'] == action and r['field'].upper() == selected_field.upper() and r['prefix_mode'] == mode]
                matrix = [[next(r['cosine'] for r in data if r['layer'] == layer and r['boundary'] == boundary and r['comparison'] == f'R1-vs-R{i}') for i in (2, 3, 4)]
                          for boundary in ('post_self_attention_residual', 'post_decoder_residual') for layer in (8, 16, 24, 32)]
                # No rounded annotations: preserve small but nonzero differences.
                im = ax.imshow(matrix, vmin=min(map(min, matrix)), vmax=1., cmap='viridis', aspect='auto')
                for y, values in enumerate(matrix):
                    for x, value in enumerate(values):
                        ax.text(x, y, f'{value:.5f}', ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', alpha=.7, edgecolor='none'))
                ax.set_yticks(range(8), [f'{b} {l}' for b in ('post-attn', 'post-layer') for l in (8, 16, 24, 32)])
                ax.set_xticks(range(3), ['R2', 'R3', 'R4'])
                ax.set_title(f'{mode}: {selected_field}')
                fig.colorbar(im, ax=ax, label='cosine similarity', shrink=.7)
            fig.suptitle(f'{action}: corresponding positions vs R1; each panel has its own labeled range')
            save(fig, figures, f'fig4_similarity_{action}')
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True, sharex=True, sharey='col')
        for row, mode in enumerate(('natural', 'controlled')):
            for col, action in enumerate(('L1', 'L4', 'L5')):
                ax = axes[row, col]
                field = f'auto{{_{"ARRAY_F" if action == "L1" else "UNROLL"}_{action}}}'
                for i, color in zip((2, 3, 4), ('#0072b2', '#d55e00', '#009e73')):
                    for boundary, style in [('post_self_attention_residual', '-'), ('post_decoder_residual', '--')]:
                        data = [r for r in metrics if r['prefix_mode'] == mode and r['field'].upper() == field.upper() and r['boundary'] == boundary and r['comparison'] == f'R1-vs-R{i}']
                        data.sort(key=lambda r: r['layer'])
                        ax.plot([r['layer'] for r in data], [r['relative_l2'] for r in data], style, marker='o', color=color,
                                label=f'R{i}: {"post-attn" if style == "-" else "post-layer"}')
                ax.set_title(f'{mode}: {action}')
                ax.set_ylabel('relative L2 movement')
                ax.set_xlabel('decoder layer (1-based)')
                ax.set_xticks([8, 16, 24, 32])
                ax.set_ylim(bottom=0)
        axes[0, 0].legend(fontsize=7)
        save(fig, figures, 'fig5_movement')
        fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
        for row, mode in enumerate(('natural', 'controlled')):
            for col, field in enumerate(('auto{_ARRAY_F_L1}', 'auto{_ARRAY_D_L1}', 'auto{_UNROLL_L5}')):
                ax = axes[row, col]
                selected = [r for r in records if r['prefix_mode'] == mode and r['field'] == field]
                keys = sorted({s['rhs'] for r in selected for s in r['scores']}, key=int)
                for r in selected:
                    scores = {s['rhs']: s['score'] for s in r['scores']}
                    ax.plot(range(len(keys)), [scores.get(k, float('nan')) for k in keys], marker='o', label=r['request_id'])
                    ax.set_xticks(range(len(keys)), keys)
                ax.set_title(f'{mode}: {field}')
                ax.set_xlabel('candidate RHS (domain can depend on earlier choices)')
                ax.set_ylabel('mean token log-score incl. newline')
                ax.legend()
        save(fig, figures, 'fig6_candidates')
        final_rows = [r for r in metrics if r['layer'] == 32 and r['boundary'] == 'post_decoder_residual'
                      and r['prefix_mode'] == 'natural' and r['field'] != 'source']
        ordered_fields = [s['lhs'] for s in official[0]['candidates'][0]['site_debug']]
        matrix = [[100. * next(r['relative_l2'] for r in final_rows if r['field'] == field
                    and r['comparison'] == f'R1-vs-R{i}') for i in (2, 3, 4)] for field in ordered_fields]
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(map(max, matrix)), aspect='auto')
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                ax.text(x, y, f'{value:.2f}%', ha='center', va='center', fontsize=10,
                        color='white' if value > 35 else 'black')
        ax.set_yticks(range(len(ordered_fields)), [short_label(field) for field in ordered_fields])
        ax.set_xticks(range(3), ['R2: budget', 'R3: clock', 'R4: device'])
        ax.set_title('Final decoder residual before each directive RHS\nNatural prefixes; before final RMS normalization')
        fig.colorbar(im, ax=ax, label='relative L2 movement vs R1 (%)')
        save(fig, figures, 'fig8_final_decision_states')
    else:
        for action in ('L1', 'L4', 'L5'):
            missing(figures, f'fig4_similarity_{action}', f'{action}: hidden-state cosine similarity',
                    'No selected hidden-state tensors exist; no heatmap or color scale can be estimated.')
        missing(figures, 'fig5_movement', 'Natural vs controlled prefix: relative L2 movement',
                'No measured hidden-state norms or movements are available.')
        missing(figures, 'fig6_candidates', 'L1 partition factor and L5 unroll: candidate scores',
                'Historical outputs saved no per-field scores; no candidate margins can be recovered.')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    ax.set_title('Stage-1 diagnostics and the future structural path', loc='left', weight='bold')
    def box(x, y, text, color):
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=.65', fc=color, ec='#536572'), transform=ax.transAxes)
    def arrow(a, b, dashed=False):
        ax.annotate('', xy=b, xytext=a, xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#536572', linestyle='--' if dashed else '-'))
    for x, label in [(.12, 'C source +\ntarget constraints'), (.38, 'V4 Stage-1 LLM'), (.64, 'selected residual\nstates'), (.88, 'constrained RHS\nscoring → directives')]:
        box(x, .72, label, '#dceef5')
    for a, b in [((.23, .72), (.29, .72)), ((.49, .72), (.55, .72)), ((.73, .72), (.79, .72))]:
        arrow(a, b)
    for x, label in [(.12, 'C → MLIR'), (.38, 'compiler graph\n→ GNN'), (.64, 'action-aligned\nmemory'), (.88, 'cross-attention\n(future experiment)')]:
        box(x, .30, label, '#f2eee4')
    for a, b in [((.22, .30), (.29, .30)), ((.49, .30), (.55, .30)), ((.73, .30), (.78, .30)), ((.88, .43), (.65, .61))]:
        arrow(a, b, True)
    status_text = ('Fresh Stage-1 traces recorded; no structural injection.' if complete else
                   'This session: GPU blocked; no fresh LLM forward pass or structural injection.')
    ax.text(.01, .04, 'Solid: existing Stage-1 path and prepared diagnostics. Dashed: proposed GEMV fusion, not executed.\n' + status_text, fontsize=10)
    save(fig, figures, 'fig7_architecture')

    # Tables are generated even when all fresh cells are explicitly missing.
    table = ['| Action | R1 | R2 | R3 | R4 |', '|---|---|---|---|---|']
    predictions = {r['context_id']: action_values(r['candidates'][0]['canonical_prediction']) for r in official}
    for i, action in enumerate(['L1 — A partition', 'L2 — x partition', 'L3 — y partition', 'L4 — outer (PIPE, UNROLL)', 'L5 — inner (PIPE, UNROLL)']):
        vals = [predictions[k][i] if k in predictions else 'not run' for k in ('R1', 'R2', 'R3', 'R4')]
        if 'R1' in predictions:
            vals = [v if j == 0 or v == vals[0] else f'**{v}**' for j, v in enumerate(vals)]
        table.append('| ' + action + ' | ' + ' | '.join(vals) + ' |')
    (out / 'directive_table.md').write_text('\n'.join(table) + '\n')
    if metrics:
        largest = max(metrics, key=lambda r: r['relative_l2'])
        contrast_summary = []
        for scope in ('source', 'natural_decision', 'controlled_decision'):
            for contrast in ('R1-vs-R2', 'R1-vs-R3', 'R1-vs-R4'):
                selected = [r for r in metrics if r['comparison'] == contrast and
                            ((scope == 'source' and r['field'] == 'source') or
                             (scope != 'source' and r['field'] != 'source' and
                              r['prefix_mode'] == scope.split('_')[0]))]
                contrast_summary.append(dict(scope=scope, comparison=contrast, sites=len(selected),
                    mean_relative_l2=float(np.mean([r['relative_l2'] for r in selected])),
                    maximum=max(selected, key=lambda r: r['relative_l2'])))
        write_json(out / 'sensitivity_summary.json', contrast_summary)
        write_json(out / 'numerical_findings.json', dict(largest_relative_l2=largest,
                   changed_natural_choices=sum(r['choice_changed'] for r in preferences if r['prefix_mode'] == 'natural'),
                   changed_controlled_choices=sum(r['choice_changed'] for r in preferences if r['prefix_mode'] == 'controlled')))
    findings = ['Fresh V4 Stage-1 observations', '============================', '', (out / 'directive_table.md').read_text()]
    if not metrics:
        findings += ['No fresh inference or hidden-state measurements. Largest contrast/layer, cosine, relative L2, norms, and ranking changes are unknown.']
    else:
        findings += ['All values below come from saved vectors and candidate scores; inference checkpoint identity is in checkpoint.json.', '',
                     '| Mode | Action | Field | Boundary | Largest relative L2 | Layer | Contrast | Cosine |',
                     '|---|---|---|---|---:|---:|---|---:|']
        keys = sorted({(r['prefix_mode'], r['action'], r['field'], r['boundary']) for r in metrics})
        for key in keys:
            subset = [r for r in metrics if tuple(r[k] for k in ('prefix_mode', 'action', 'field', 'boundary')) == key]
            best = max(subset, key=lambda r: r['relative_l2'])
            findings.append('| ' + ' | '.join(key) + f" | {best['relative_l2']:.7f} | {best['layer']} | {best['comparison']} | {best['cosine']:.7f} |")
        findings += ['', 'Candidate preferences (13 fields × 3 contrasts per mode):', '']
        for mode in ('natural', 'controlled'):
            subset = [r for r in preferences if r['prefix_mode'] == mode]
            findings.append(f"- {mode}: {sum(r['choice_changed'] for r in subset)}/{len(subset)} choices change; "
                            f"{sum(r['ranking_changed'] for r in subset)} rankings change; "
                            f"{sum(not r['domains_identical'] for r in subset)} domains differ.")
        findings += ['', 'Consult pairwise.csv for all cosine, relative L2, baseline/compared norms and signed relative norm changes; candidate_comparisons.csv for field-level margins and domain identity. These describe sensitivity, not optimization quality.']
    (out / 'findings.md').write_text('\n'.join(findings) + '\n')
    write_json(out / 'figure_receipt.json', dict(manifest_sha256=sha(out / 'manifest.json'),
        fresh_data=complete, figures={p.name: sha(p) for p in sorted(figures.iterdir()) if p.suffix in ('.png', '.svg')}))
    print(f'Wrote {len(list(figures.glob("*.svg")))} figure pairs; fresh inference={complete}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    analyze(args.out)


if __name__ == '__main__':
    main()
