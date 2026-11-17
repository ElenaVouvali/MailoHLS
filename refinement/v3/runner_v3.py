#!/usr/bin/env python3
"""liono -> Coroni driver; local preparation/preview are entirely offline."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import cg_bdr as cg
ROOT = HERE.parents[1]
NAMES = ('GAN','jacobi-2d','Kalman','trmm-opt','covariance','syr2k')
REMOTE = 'bonus24@coroni.microlab.ntua.gr'
PYTHON = '/usr/bin/python3.8'


def name(kernel):
    return kernel+'_LATENCY_10ns_4eval'


def archive(kernel):
    return HERE.parent/('evidence' if kernel=='GAN' else 'v2/evidence')/name(kernel)


def bundle(root,kernel):
    return root/'bundles'/name(kernel)


def view_bundle(root,kernel):
    collected = sorted((root/'collected').glob('*/'+name(kernel)))
    return collected[-1] if collected else bundle(root,kernel)


def command(argv,capture=False):
    print('> '+' '.join(shlex.quote(str(x)) for x in argv),file=sys.stderr,flush=True)
    return subprocess.run([str(x) for x in argv],check=True,universal_newlines=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def remote(argv):
    return command(['ssh','-o','BatchMode=yes',REMOTE,' '.join(shlex.quote(str(x)) for x in argv)],True)


def remote_root(root):
    # Unique per local runs root; no v2 namespace or bundle can be resumed.
    slug = re.sub('[^A-Za-z0-9_.-]','_',root.name)
    return '/home/bonus24/cg_bdr_v3_runs/'+slug+'-'+cg.signature(str(root))[:12]


def audit():
    results = {}
    # Separate subprocesses avoid module-name collisions between frozen audits.
    for version, path in [('legacy',HERE.parent/'audit_evidence.py'),('v2',HERE.parent/'v2/audit_evidence.py')]:
        results[version] = json.loads(command([sys.executable,'-B',path],True))
    return dict(archived=results, actual_new_vitis_calls=0)


def prepare(args):
    audit()  # Verifies both immutable inventories before any extraction/import.
    spec = importlib.util.spec_from_file_location('prediction_selector',HERE.parent/'select_initial.py')
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    output = []
    for kernel in args.kernels:
        old = archive(kernel)
        dest = bundle(args.runs_root,kernel)
        check_new = args.runs_root/'extracted'/name(kernel)
        cg.check(not dest.exists() and not check_new.exists(), 'Use a fresh runs-root; refusing existing bundle/extraction')
        check_new.mkdir(parents=True)
        for src in (old/'inputs').iterdir():
            if src.name=='llvm_graph.gexf.gz':
                with gzip.open(src,'rb') as stream,(check_new/'llvm_graph.gexf').open('xb') as target:
                    shutil.copyfileobj(stream,target)
            else:
                shutil.copy2(str(src),str(check_new/src.name))
        prediction = selector.select(HERE.parent/'evidence_plot_rows.jsonl',kernel,'LATENCY',cg.PART,10,100)
        initial = check_new/'initial.json'
        cg.jdump(initial,prediction)
        params = argparse.Namespace(bundle=dest,kernel=kernel,source_dir=check_new,
                template=check_new/'template.cpp',kernel_info=check_new/'kernel_info.txt',
                graph=check_new/'llvm_graph.gexf',initial=initial,policy=args.policy,
                extraction_provenance=dict(archive=str(old.resolve()), archive_manifest_sha256=cg.digest(old/'manifest.json'),
                                          compressed_graph_sha256=cg.digest(old/'inputs/llvm_graph.gexf.gz'),
                                          archive_unmodified=True))
        cg.prepare(params)
        status = 'not requested: next approved call will synthesize initial baseline'
        if args.reuse_verified_baselines:
            try:
                status = cg.import_baseline(dest,old)
            except (RuntimeError,ValueError,OSError) as error:
                # No budget charge on rejected import. Must explicitly approve fresh baseline.
                status = 'Reuse rejected; fresh baseline requires typed approval: '+str(error)
        output.append(dict(kernel=kernel,bundle=str(dest),baseline_import=status,summary=cg.summary(dest)))
    return output


def paths(root,kernel):
    rroot = remote_root(root)
    return rroot,rroot+'/code/refinement/v3/cg_bdr.py',rroot+'/bundles/'+name(kernel)


def transfer(root,kernel):
    rroot,controller,rbundle = paths(root,kernel)
    local = bundle(root,kernel)
    cg.verify_bundle(local)
    print('REMOTE paths: '+json.dumps(dict(host=REMOTE,root=rroot,python=PYTHON,vitis=cg.VITIS,
                                          controller=controller,bundle=rbundle)),file=sys.stderr)
    # Check all required binaries and vendor headers without invoking Vitis.
    remote(['test','-x',PYTHON])
    remote(['test','-x',cg.VITIS])
    for header in ('ap_fixed.h','ap_int.h'):
        remote(['test','-f','/opt/Xilinx/Vitis_HLS/2021.1/include/'+header])
    version = remote([PYTHON,'-c','import sys; assert sys.version_info[:2] == (3,8); print(sys.version)'])
    print(version.strip(),file=sys.stderr)
    for rel in cg.CODE:
        local_code = (HERE/rel).resolve()
        target = str(Path(rroot+'/code/refinement/v3')/rel)
        target = str(Path(target))  # .. is safe here: fixed CODE list, never operator supplied.
        remote(['mkdir','-p',str(Path(target).parent)])
        response = remote([PYTHON,'-c',
            'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING")',target]).strip()
        if response=='MISSING':
            command(['scp',local_code,REMOTE+':'+shlex.quote(target)])
        else:
            cg.check(response==cg.digest(local_code), 'Remote sealed code differs; refusing overwrite')
        verified = remote(['sha256sum',target]).split()[0]
        cg.check(verified==cg.digest(local_code), 'Transferred code hash mismatch')
    remote(['mkdir','-p',rroot+'/bundles'])
    response = remote([PYTHON,'-c','import pathlib,sys; print(pathlib.Path(sys.argv[1]).exists())',rbundle]).strip()
    if response=='False':
        command(['scp','-r',local,REMOTE+':'+shlex.quote(rroot+'/bundles/')])
    other = json.loads(remote(['cat',rbundle+'/manifest.json']))
    cg.check(other==cg.jload(local/'manifest.json'), 'Remote/local frozen manifest mismatch')
    return controller,rbundle


def pilot(args):
    rows = []
    for kernel in args.kernels:
        if args.no_hls:
            # Local dry-run intentionally needs neither SSH nor an available Coroni.
            rows.append(dict(kernel=kernel,no_hls=True,remote_paths=dict(host=REMOTE,root=paths(args.runs_root,kernel)[0],controller=paths(args.runs_root,kernel)[1],bundle=paths(args.runs_root,kernel)[2],python=PYTHON,vitis=cg.VITIS),
                             preview=cg.preview(view_bundle(args.runs_root,kernel),args.mode=='scout')))
            continue
        controller,rbundle = transfer(args.runs_root,kernel)
        extra = ['--enable-one-scout'] if args.mode=='scout' else []
        p = json.loads(remote([PYTHON,controller,'preview','--bundle',rbundle,'--explain']+extra))
        cg.emit(p)
        if p['stop_reason']:
            rows.append(p)
            continue
        token = cg.approval_token(Path(rbundle),p)
        print('Resource/latency effects are uncertain; approval covers exactly ONE call, not a campaign.',file=sys.stderr)
        try:
            reply = input('Type '+token+'\n> ')
        except EOFError:
            reply = ''
        if reply!=token:
            rows.append(dict(kernel=kernel,stop_reason='Fresh typed approval absent; zero new calls'))
            break
        # Exactly one invocation, revalidated under remote lock against current state.
        rows.append(json.loads(remote([PYTHON,controller,'run','--bundle',rbundle,
                    '--vitis-hls',cg.VITIS,'--approval',reply]+extra).split('V3_RESULT\n')[-1]))
    return rows


def collect(args):
    destination = args.runs_root/'collected'/time.strftime('%Y%m%dT%H%M%S')
    destination.mkdir(parents=True,exist_ok=False)
    rows = []
    for kernel in args.kernels:
        _,_,rbundle = paths(args.runs_root,kernel)
        exists = remote([PYTHON,'-c','import pathlib,sys; print(pathlib.Path(sys.argv[1]).exists())',rbundle]).strip()
        if exists!='True':
            rows.append(dict(kernel=kernel,status='not present remotely'))
            continue
        command(['scp','-r',REMOTE+':'+shlex.quote(rbundle),destination])
        local = destination/name(kernel)
        cg.verify_bundle(local)
        rows.append(dict(kernel=kernel,collected=str(local),summary=cg.summary(local)))
    return rows


def report(args):
    output = []
    for kernel in args.kernels:
        path = view_bundle(args.runs_root,kernel)
        if not path.exists():
            continue
        row = cg.summary(path)
        row['read_only_preview'] = cg.preview(path,args.mode=='scout')
        row['historical_comparators'] = []
        for label,old in [('v0.1 historical',HERE.parent/'evidence'/name(kernel)),
                          ('v2 historical',HERE.parent/'v2/evidence'/name(kernel))]:
            if not old.exists():
                continue
            m = cg.jload(old/'manifest.json')
            state = cg.jload(old/'state.json')
            metrics = [cg.read_report(old/r['result']['work_dir']/'hls_project/solution1/syn/report'/
                                     (m['top']+'_csynth.xml'),m) for r in state['evaluations']]
            # Historical selection metric is labelled, never silently converted/merged.
            index = cg.jload(old/'summary.json')['best']['best_evaluation']
            row['historical_comparators'].append(dict(label=label, original_selected_evaluation=index,
                    original_selection_metric='Worst-caseLatency' if label.startswith('v0') else 'Best-caseLatency',
                    trajectory=metrics, record_count=len(metrics), baseline_shared_with_v3=bool(row['reused_verified_evaluations']),
                    best_case_replay_after_budget=[dict(budget=n, metrics=min([v for v in metrics[:n] if cg.eligible(v)],
                       key=lambda v:v['latency_best_cycles'],default=None), carried_forward=n>len(metrics)) for n in range(1,5)]))
        output.append(row)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','prepare','preview','pilot','collect','report'])
    p.add_argument('--kernels',nargs='+',choices=NAMES,default=list(NAMES))
    p.add_argument('--runs-root',type=Path,default=Path.home()/'cg_bdr_v3_runs')
    p.add_argument('--mode',choices=['strict','scout'],default='strict')
    p.add_argument('--policy',choices=['graph','feedback-only'],default='graph')
    p.add_argument('--reuse-verified-baselines',action='store_true')
    p.add_argument('--explain',action='store_true')
    p.add_argument('--no-hls',action='store_true')
    args = p.parse_args()
    args.runs_root = args.runs_root.expanduser().resolve()
    if args.command=='audit':
        output = audit()
    elif args.command=='prepare':
        output = prepare(args)
    elif args.command=='preview':
        output = [dict(cg.preview(view_bundle(args.runs_root,k),args.mode=='scout'),
                       bundle_read=str(view_bundle(args.runs_root,k)),
                       state_scope='local snapshot; collect to refresh remote measurements') for k in args.kernels]
    elif args.command=='pilot':
        output = pilot(args)
    elif args.command=='collect':
        output = collect(args)
    else:
        output = report(args)
    cg.emit(output)


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,ValueError,OSError,subprocess.CalledProcessError) as error:
        print('RUNNER V3 ERROR: '+str(error),file=sys.stderr)
        sys.exit(2)
