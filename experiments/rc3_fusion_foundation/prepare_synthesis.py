"""Freeze a small executable Vitis HLS bundle; this script never launches HLS."""
from __future__ import annotations

import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
VARIANT_ROOT = PACKAGE / 'benchmark/variants/reduce32'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-worktree', default='/tmp/MailoHLS-stage1-refinement')
    ap.add_argument('--out', default=str(ROOT / 'artifacts/rc3_fusion_foundation/v1_20261107/synthesis_bundle_v2'))
    args = ap.parse_args()
    stage1 = Path(args.stage1_worktree).resolve()
    sys.path.insert(0, str(stage1))
    from LLM_branch.common import directive_domains, mailohls_contract as contract, source_tool_policy
    from experiments.refinement.synthesis import materialize

    source = VARIANT_ROOT / 'gemv.c'; metadata = VARIANT_ROOT / 'kernel_info.txt'
    if not source.exists() or not metadata.exists(): raise FileNotFoundError('Authenticated reduce32 variant unavailable')
    policy_path = stage1 / 'experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json'
    policy = source_tool_policy.load_policy(policy_path)
    actions = directive_domains.load_source_actions('gemv', VARIANT_ROOT)
    baseline = {
        'L1': ('block', '2', '1'), 'L2': ('cyclic', '2', '1'), 'L3': ('cyclic', '4', '1'),
        'L4': ('0', '2'), 'L5': ('0', '2'),
    }
    tight = {**baseline, 'L2': ('cyclic','4','1'), 'L3': ('complete','0','1'), 'L5': ('0','4')}
    configs = {'stage1_r1': baseline, 'stage1_zcu104_3p33_25pct': tight}
    for key, action, fields in [
        ('alt_l1_dim2', 'L1', ('block','2','2')),
        ('alt_l2_factor4', 'L2', ('cyclic','4','1')),
        ('alt_l5_pipeline1', 'L5', ('1','2')),
        ('alt_l5_unroll4', 'L5', ('0','4')),
    ]:
        value = dict(baseline); value[action] = fields; configs[key] = value

    def mapping(config):
        out = {}
        for label, vals in config.items():
            if label in ('L1','L2','L3'):
                out.update({f'AUTO{{_ARRAY_T_{label}}}': vals[0], f'AUTO{{_ARRAY_F_{label}}}': vals[1], f'AUTO{{_ARRAY_D_{label}}}': vals[2]})
            else:
                out.update({f'AUTO{{_PIPE_{label}}}': vals[0], f'AUTO{{_UNROLL_{label}}}': vals[1]})
        return out

    outputs = []
    for name, config in configs.items():
        assignment = mapping(config)
        source_tool_policy.validate_source_assignment(assignment, actions)
        source_tool_policy.validate_complete_assignment(assignment, actions, policy)
        serial = '\n'.join(f'{key} = {value}' for key, value in assignment.items())
        text = materialize(source.read_text(), serial)
        if 'auto{' in text.lower(): raise AssertionError('Unresolved directive placeholder')
        outputs.append((name, config, text))

    bundle = Path(args.out)
    if bundle.exists() and any(bundle.iterdir()): raise FileExistsError(f'Refusing to overwrite nonempty bundle {bundle}')
    (bundle / 'jobs').mkdir(parents=True, exist_ok=True)
    runner_source = stage1 / 'artifacts/synthesis/quality_v4_gan_kalman_20261105/bundle/run_batch.py'
    shutil.copy2(runner_source, bundle / 'run_batch.py')
    contexts = [
        ('zcu104_10ns', 'xczu7ev-ffvc1156-2-e', 10.0),
        ('zcu104_3p33ns', 'xczu7ev-ffvc1156-2-e', 3.33),
        ('xcu200_10ns', 'xcu200-fsgd2104-2-e', 10.0),
        ('xcu200_3p33ns', 'xcu200-fsgd2104-2-e', 3.33),
    ]
    jobs = []
    for ci, (label, part, clock) in enumerate(contexts, 1):
        capacities = contract.DEVICE_RESOURCES[part]
        for config_name, config, text in outputs:
            job_id = f'{config_name}_{label}'
            jobdir = bundle / 'jobs' / job_id; jobdir.mkdir(parents=True)
            materialized = jobdir / 'gemv.c'; materialized.write_text(text)
            budget_views = []
            for pct in (100, 25):
                budget_views.append({'fraction': pct / 100,
                                     'resources': {k: int(round(v * pct / 100)) for k, v in capacities.items()}})
            job = {'job_id': job_id, 'kernel': 'GEMV_reduce32_diagnostic', 'arm': config_name,
                   'request_number': ci, 'request_id': label, 'source': 'gemv.c', 'top': 'gemv',
                   'part': part, 'clock_ns': clock, 'capacities': capacities,
                   'budget': capacities, 'budget_views_for_posthoc_feasibility': budget_views,
                   'directive_assignment': mapping(config), 'source_action_map': {k: v.kind for k,v in actions.items()},
                   'materialized_source_sha256': sha(materialized), 'original_source_sha256': sha(source),
                   'metadata_sha256': sha(metadata), 'headers': {}, 'functional_validation': 'not performed by csynth runner'}
            jobs.append(job)
    manifest = {'schema': 'stage1-v4-synthesis-batch-v1', 'objective': 'PARETO_LATENCY',
                'task_note': 'Small measured candidate pool; HLS estimates only. Budget views deduplicate physical jobs.',
                'settings': ['config_array_partition -complete_threshold 0 -throughput_driven off',
                             'config_compile -pipeline_loops 0'], 'jobs': jobs,
                'source_variant': {'name': 'reduce32', 'source_sha256': sha(source),
                                   'metadata_sha256': sha(metadata), 'graph_sha256': sha(VARIANT_ROOT/'graph.json'),
                                   'mlir_sha256': sha(VARIANT_ROOT/'gemv.mlir'),
                                   'semantics': 'initialized inputs, scalar reduction, all reduction outputs observable; independent from historical original GEMV'} ,
                'candidate_count': len(configs), 'physical_job_count': len(jobs),
                'validation': {'source_valid': True, 'vitis_2021_1_tool_policy_valid': True,
                               'assignments_validated': len(configs), 'tool_invocation_performed': False},
                'validation_provenance': {'stage1_worktree': str(stage1),
                    'stage1_commit': subprocess.check_output(['git','-C',str(stage1),'rev-parse','HEAD'],text=True).strip(),
                    'policy_id': policy.policy_id, 'policy_sha256': sha(policy_path),
                    'source_tool_policy_sha256': sha(stage1/'LLM_branch/common/source_tool_policy.py'),
                    'directive_domains_sha256': sha(stage1/'LLM_branch/common/directive_domains.py'),
                    'materializer_sha256': sha(stage1/'experiments/refinement/synthesis.py'),
                    'reduce32_source_sha256': sha(source), 'reduce32_metadata_sha256': sha(metadata)}}
    dump(bundle / 'manifest.json', manifest)
    tool = Path('/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls')
    state = {'vitis_executable': str(tool), 'exists': tool.exists(),
             'runner_sha256': sha(bundle/'run_batch.py'), 'manifest_sha256': sha(bundle/'manifest.json'),
             'run_command': f"python {bundle/'run_batch.py'} --bundle {bundle} --output {bundle.parent/'synthesis_results'} --tool {tool}"}
    dump(bundle / 'execution_status.json', state)
    print(json.dumps({'manifest': str(bundle/'manifest.json'), 'jobs': len(jobs), 'tool_exists': tool.exists()}, indent=2))


if __name__ == '__main__': main()
