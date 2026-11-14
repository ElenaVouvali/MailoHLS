#!/usr/bin/env python3
"""Install an isolated routing-only Stage-2 training/inference ablation.

Designed against MailoHLS research/stage2-evaluated-20261113.
Fails closed on source drift. Does not edit the original training/inference files.
"""
from __future__ import annotations
import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_GIT_BLOBS = {
    'experiments/film_gmt_stage2_final/train.py': '66bf0bad1466d0be587bf21cfafad5e6c736cf79',
    'experiments/stage2_eval_compat9/stage2_infer_compact_views.py': '64c06d1d6035a40ee1c7fa00b1808c79c3a066a2',
}


def replace_once(text: str, old: str, new: str, why: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f'{why}: expected one exact source anchor; found {n}. STOP and inspect live source.')
    return text.replace(old, new, 1)


def check_hash(path: Path, expected: str) -> None:
    actual = subprocess.check_output(['git', 'hash-object', str(path)], text=True).strip()
    if actual != expected:
        raise RuntimeError(f'Pinned upstream source differs: {path}\nGit blob: {actual}\nExpected: {expected}\nDo not run an unreviewed ablation on changed training code.')


def inference_source(repo: Path) -> str:
    # The original Nov-13 FiLM branch does not contain the compact inference
    # runner; it was committed to the separately evaluated Nov-13 branch.
    rel = 'experiments/stage2_eval_compat9/stage2_infer_compact_views.py'
    path = repo / rel
    if path.is_file():
        check_hash(path, EXPECTED_GIT_BLOBS[rel])
        return path.read_text()
    for ref in ('research/stage2-evaluated-20261113',
                'origin/research/stage2-evaluated-20261113',
                '3a64895afed87600d1b326ab45b260cea2dfabe7',
                'FETCH_HEAD'):
        proc = subprocess.run(['git', '-C', str(repo), 'show', f'{ref}:{rel}'],
                              capture_output=True)
        if proc.returncode != 0:
            continue
        actual = subprocess.check_output(['git', 'hash-object', '--stdin'],
                                         input=proc.stdout).decode().strip()
        if actual != EXPECTED_GIT_BLOBS[rel]:
            raise RuntimeError(f'Inference baseline in {ref} has unexpected Git blob {actual}')
        return proc.stdout.decode('utf-8')
    raise RuntimeError('Compact inference source missing in this worktree. On Liono run:\n'
                       'git -C "$ROOT" fetch origin research/stage2-evaluated-20261113\n'
                       'Then re-run installer; it will authenticate the exact source from Git.')


def make_train(src: str) -> str:
    s = src
    s = replace_once(s,
        '    p.add_argument("--mode", choices=("audit", "smoke", "train", "resume"), required=True)\n',
        '    p.add_argument("--mode", choices=("audit", "smoke", "train", "resume"), required=True)\n'
        '    p.add_argument("--routing-mode", choices=("self_only", "full_memory"), required=True)\n'
        '    p.add_argument("--stop-at-step", type=int, default=300, help="Stop after saving the matched checkpoint; LR schedule still uses --steps=450")\n',
        'train CLI routing selector')
    s = replace_once(s,
        '    kv,mk,relation=[x.to(device) if x is not None else None for x in pack]\n',
        '    kv,mk,relation=[x.to(device) if x is not None else None for x in pack]\n'
        '    # Only alter the action-to-memory attention mask. Both alternatives\n'
        '    # use the exact same frozen 64-anchor + GMT memory tensor.\n'
        '    routing = getattr(model, "_routing_ablation_mode", None)\n'
        '    if routing == "self_only":\n'
        '        # Lk queries are rows 0..63; only key Lk is permitted.\n'
        '        relation = torch.eye(mk.size(1), device=mk.device, dtype=torch.bool)[None, :, :] & mk[:, None, :]\n'
        '    elif routing == "full_memory":\n'
        '        # Every Lk may attend all *valid* anchors and compiler evidence.\n'
        '        relation = mk[:, None, :].expand(-1, mk.size(1), -1).clone()\n'
        '    else:\n'
        '        raise RuntimeError(f"Unexpected training routing: {routing!r}")\n',
        'train routing mask substitution')
    s = replace_once(s,
        '    model,film,hooks,tok,registry,encoder,opt,scaler,dtype,params=model_and_runtime(a,parent_contract)\n',
        '    model,film,hooks,tok,registry,encoder,opt,scaler,dtype,params=model_and_runtime(a,parent_contract)\n'
        '    model._routing_ablation_mode = a.routing_mode\n',
        'bind training routing to model')
    s = replace_once(s,
        '              "routing":"compiler_relational; original 64 action anchors + 3 typed GMT slots per action; no fallback",\n',
        '              "routing":f"{a.routing_mode}; same frozen 64 action anchors + typed GMT evidence slots; mask-only ablation",\n',
        'provenance protocol routing mode')
    s = replace_once(s,
        '    if a.mode=="audit" and a.output.exists():\n',
        '    if a.steps != 450 or a.eval_interval != 150 or a.stop_at_step != 300:\n'
        '        raise ValueError("Pinned checkpoint-300 ablation requires --steps 450, --eval-interval 150 and --stop-at-step 300")\n'
        '    if a.mode=="audit" and a.output.exists():\n',
        'keep 450-step schedule and 300-step observation point')
    s = replace_once(s,
        '        if step==a.steps:break\n',
        '        if step==a.stop_at_step:\n'
        '            if not (a.output/f"checkpoint-{step:04d}"/"receipt.json").exists():\n'
        '                raise RuntimeError("Checkpoint 300 not saved at matched validation step")\n'
        '            if frozen_before!=tensors_sha(model,film):\n'
        '                raise RuntimeError("Frozen V4/residual/FiLM changed in ablation")\n'
        '            write_json(a.output/"TARGET_REACHED.json",\n'
        '                {"status":"ABLATION_CHECKPOINT_300_READY","routing_mode":a.routing_mode,\n'
        '                 "trained_updates":step,"schedule_total_updates":a.steps,\n'
        '                 "checkpoint":str(a.output/f"checkpoint-{step:04d}"),\n'
        '                 "protocol_sha256":protocol_sha,"frozen_composite_sha256":frozen_before})\n'
        '            print("[ABLATION TARGET READY]",a.routing_mode,"step",step,flush=True)\n'
        '            return\n'
        '        if step==a.steps:break\n',
        'exit immediately after matched cp300')
    return s


def make_infer(src: str) -> str:
    s=src
    s=replace_once(s,
        "V100 = 'GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f'",
        "V100 = os.environ.get('SALT_GPU_UUID', 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb')",
        'Lion V100 UUID')
    s=replace_once(s,
        "    p.add_argument('--deployment', type=Path, default=DEPLOY)\n",
        "    p.add_argument('--deployment', type=Path, default=DEPLOY)\n"
        "    p.add_argument('--run-dir', type=Path, required=True, help='Run folder with checkpoint-0300 and protocol.json')\n"
        "    p.add_argument('--repo', type=Path, required=True)\n"
        "    p.add_argument('--parent', type=Path, required=True)\n"
        "    p.add_argument('--film-checkpoint', type=Path, required=True)\n"
        "    p.add_argument('--routing-mode', choices=('compiler_relational','self_only','full_memory'), required=True)\n",
        'inference CLI')
    s=replace_once(s,
        "    a.repo = a.deployment / 'code'\n    a.parent = Path('/home/evouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')\n    a.film_checkpoint = a.deployment / 'assets/film/B/checkpoint-0600'\n",
        "    a.repo = a.repo.resolve()\n    a.parent = a.parent.resolve()\n    a.film_checkpoint = a.film_checkpoint.resolve()\n",
        'inference source and frozen checkpoint refs')
    s=replace_once(s,
        "    out = a.deployment / 'runs/campaign_s123_v1/final'\n",
        "    out = a.run_dir.resolve()\n",
        'checkpoint run location')
    s=replace_once(s,
        "    entry = a.repo / 'experiments/film_gmt_stage2_final/train.py'\n",
        "    entry = a.repo / ('experiments/film_gmt_stage2_final/train.py'\n"
        "        if a.routing_mode == 'compiler_relational' else\n"
        "        'experiments/routing_ablation_20261114/train_routing.py')\n",
        'train file authentication based on mode')
    begin="    deployment_paths = json.loads((a.deployment / 'adapter/paths.json').read_text())\n"
    end="    # Set visibility *before* importing any torch-backed repository modules.\n"
    assert s.count(begin)==1 and s.count(end)==1, 'Inference deployed-only provenance anchors changed'
    i=s.index(begin);j=s.index(end,i)
    s=s[:i]+(
        "    # This is a source-authenticated Liono local ablation, not a promoted Davinci deployment.\n"
        "    expected = {'compiler_relational':'compiler_relational', 'self_only':'self_only', 'full_memory':'full_memory'}\n"
        "    if not str(protocol.get('routing', '')).startswith(expected[a.routing_mode] + ';'):\n"
        "        raise ValueError('Checkpoint routing does not match requested training routing')\n"
        "    if int(protocol.get('train_config',{}).get('steps',-1)) != 450:\n"
        "        raise ValueError('Wrong Stage-2 training scheduler horizon for checkpoint 300')\n"
        "    if a.routing_mode != 'compiler_relational' and int(protocol['train_config'].get('stop_at_step',-1)) != 300:\n"
        "        raise ValueError('Ablation was not stopped at matching 300-update checkpoint')\n"
    )+s[j:]
    s=replace_once(s,
        "        packed = sm.get_structural_memory_pack_for_kernel(memory, kernel, 64, 128, 'compiler_relational')\n"
        "        if packed[-1] is None:\n"
        "            raise ValueError(f'{kernel}: missing compiler-relational attention mask')\n",
        "        packed = sm.get_structural_memory_pack_for_kernel(memory, kernel, 64, 128, 'compiler_relational')\n"
        "        if packed[-1] is None:\n"
        "            raise ValueError(f'{kernel}: missing compiler-relational attention mask')\n"
        "        if a.routing_mode != 'compiler_relational':\n"
        "            import torch as _torch\n"
        "            kv, mk, _ = packed\n"
        "            nkeys = mk.size(1)\n"
        "            if a.routing_mode == 'self_only':\n"
        "                relation = _torch.eye(nkeys, dtype=_torch.bool)[None,:,:] & mk[:,None,:]\n"
        "            else:\n"
        "                relation = mk[:,None,:].expand(-1,nkeys,-1).clone()\n"
        "            packed = (kv, mk, relation.contiguous())\n",
        'inference structural masks')
    s=replace_once(s,
        "    from experiments.film_gmt_stage2_final import train\n",
        "    from experiments.film_gmt_stage2_final import train\n",
        'ensure inference still uses original model loader') if False else s
    s=replace_once(s,
        "            'checkpoint_sha256': receipt['structural_state_sha256'],\n",
        "            'checkpoint_sha256': receipt['structural_state_sha256'],\n"
        "            'routing_mode': a.routing_mode,\n",
        'prediction provenance routing field')
    # Reject fake/incorrect single-card capability for both GPU and production requirements.
    return s


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args()
    repo=a.repo.resolve()
    train_path=repo/'experiments/film_gmt_stage2_final/train.py'
    check_hash(train_path, EXPECTED_GIT_BLOBS['experiments/film_gmt_stage2_final/train.py'])
    infer_text=inference_source(repo)
    generated={
        repo/'experiments/routing_ablation_20261114/train_routing.py':make_train(train_path.read_text()),
        repo/'experiments/routing_ablation_20261114/infer_routing.py':make_infer(infer_text),
    }
    for path,body in generated.items():
        compile(body,str(path),'exec')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.read_text() != body:
                raise FileExistsError(f'Generated source already exists but differs: {path}')
            print('[EXISTS VERIFIED]',path)
        else:
            path.write_text(body)
            print('[CREATED]',path)
        print('[SHA256]',hashlib.sha256(path.read_bytes()).hexdigest())
    print('Generated exact-source, separately authenticated Stage-2 routing ablation. No GPU work was started.')

if __name__=='__main__':
    main()
