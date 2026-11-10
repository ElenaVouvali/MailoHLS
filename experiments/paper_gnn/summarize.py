"""Summarize authentic completed runs and render standalone training curves."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[]
    for tag in ('latency','area','adp'):
        directory=a.root/tag
        if not (directory/'completion.json').exists():
            results.append({'objective':tag,'status':'not_completed'});continue
        completion=json.loads((directory/'completion.json').read_text())
        if sha(directory/'selected.pt')!=completion['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        metrics=json.loads((directory/'selected_metrics.json').read_text())
        identity=json.loads((directory/'data_identity.json').read_text())
        protocol=json.loads((directory/'protocol.json').read_text())
        checks={p.name:json.loads(p.read_text()) for p in directory.glob('xattn_*.json')}
        curves=json.loads((directory/'curves.json').read_text())
        with (a.out/(tag+'_curves.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
        fig,axes=plt.subplots(1,3,figsize=(13,3.6))
        for field in ('loss','qor_loss','resource_loss','rank_loss'):
            axes[0].plot([r['epoch']+1 for r in curves],[r[field] for r in curves],label=field)
        axes[0].legend(fontsize=7);axes[0].set_ylabel('Training loss')
        for field in ('kendall_tau_b','feasible_selection_rate'):
            axes[1].plot([r['epoch']+1 for r in curves],[r[field] for r in curves],label=field)
        axes[1].legend(fontsize=7);axes[1].set_ylabel('Development family macro')
        for field in ('feasible_regret_at_1','log_qor_mae'):
            axes[2].plot([r['epoch']+1 for r in curves],[r[field] for r in curves],label=field)
        axes[2].legend(fontsize=7);axes[2].set_ylabel('Development error / regret')
        for ax in axes:
            ax.set_xlabel('Epoch');ax.axvline(completion['selected_epoch']+1,color='black',linestyle=':',alpha=.5)
        fig.suptitle(protocol['objective']);fig.tight_layout()
        fig.savefig(a.out/(tag+'_curves.png'),dpi=180);fig.savefig(a.out/(tag+'_curves.pdf'));plt.close(fig)
        results.append({'objective':protocol['objective'],'seed':protocol['seed'],
            'cohort_qualification':identity.get('measurement_policy_id','legacy_resource_filtered_cohort'),
            'status':'trained_exported_checked' if checks and all(v['status']=='passed' for v in checks.values()) else 'compatibility_pending',
            'selected_epoch_zero_based':completion['selected_epoch'],'updates_at_selection':completion['updates'],
            'total_updates':completion['total_updates'],'metrics':metrics['family_macro'],
            'family_metrics':metrics['families'],'measurement_coverage':identity['coverage'],
            'evaluation_coverage':metrics['coverage'],'candidate_pool_count':len(metrics['pools']),
            'eligible_development_families':len(metrics['families']),
            'eligible_development_kernels':sorted({r['kernel'] for r in metrics['pools']}),
            'split_sha256':identity['split_sha256'],'feature_schema':identity['feature_schema']['schema'],
            'checkpoint_sha256':completion['checkpoint_sha256'],
            'memory_manifest_sha256':completion['memory_manifest_sha256'],
            'compatibility':{k:{'status':v['status'],'gpu':v['gpu'],'kernel_checks':len(v['checks'])} for k,v in checks.items()}})
    report={'schema':'salt-paper-gnn-results-v1','objectives':results,
            'resource_filter_correction_pending':True,
            'locked_test_evaluated':False,'downstream_qor_superiority_established':False,
            'limitations':['Only eligible held-out families contribute; the locked split has three development families, but missing measured action columns exclude sort-radix, leaving two eligible families.',
              'Feasible regret is conditional on a measured-feasible selection; read alongside feasible-selection rate.',
              'Resources reconstructed from reported measured percentages and authenticated physical capacities.',
              'Timing qualification unknown in these accepted tables; physical feasibility is not timing qualification.',
              'Whole-design measured candidate pools do not establish quality of LLM-generated completions.']}
    (a.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Paper GNN execution results','',
           'These completed runs use the legacy resource-filtered cohort. Corrected-cohort integration/reruns remain pending. No downstream LLM/synthesis improvement is established.','',
           '| Objective | Status | Seed | tau-b | Feasible regret@1 (ratio) | Log2 QoR MAE | Feasible selection |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        m=row.get('metrics',{})
        fmt=lambda k:f'{m[k]:.6g}' if m.get(k) is not None else '—'
        lines.append(f"| {row['objective']} | {row['status']} | {row.get('seed','—')} | {fmt('kendall_tau_b')} | {fmt('feasible_regret_at_1')} | {fmt('log_qor_mae')} | {fmt('feasible_selection_rate')} |")
    lines += ['','See results.json for family-level metrics, coverage, hashes and qualification limits.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    files=[p for p in sorted(a.out.iterdir()) if p.is_file() and p.name!='SHA256SUMS']
    (a.out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in files))


if __name__=='__main__':main()
