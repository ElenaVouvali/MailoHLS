"""Load authenticated historical code; never import the current RC3 encoder."""
import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import subprocess
import sys
import types
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REV = 'bf82c648273a5cd5eb30363059a3809e7b3979db'
RAW = ROOT / 'artifacts/historical_mlir_gmt_gemv'
SNAPSHOT = RAW / 'historical'
ASSETS = Path('/home/elvouvali/MailoHLS')
CKPT_REL = Path('Checkpoints/gnn_final_absolute_direct_rank_s123/run1')
CKPT = ASSETS / CKPT_REL / 'val_structural_rank_model_state_dict.pth'
EXPECTED = '34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623'
NORM_REL = Path('artifacts/gnn/absolute_direct_rank_epoch9_s123')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def prepare():
    if not (SNAPSHOT / CKPT_REL / 'gnn_checkpoint_contract.json').exists():
        SNAPSHOT.mkdir(parents=True, exist_ok=True)
        archive = subprocess.check_output(['git', 'archive', REV, 'GNN_branch',
            str(CKPT_REL.parent), 'paper_examples/gemv', 'evaluation_set/gemv'], cwd=ROOT)
        subprocess.run(['tar', '-x', '-C', str(SNAPSHOT)], input=archive, check=True)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load():
    prepare()
    contract_path = SNAPSHOT / CKPT_REL / 'gnn_checkpoint_contract.json'
    contract = json.loads(contract_path.read_text())
    sidecar = json.loads((SNAPSHOT / CKPT_REL / 'val_structural_rank_model_state_dict.pth.json').read_text())
    assert sha(CKPT) == EXPECTED == sidecar['checkpoint_sha256']
    assert sha(contract_path) == sidecar['contract_sha256']
    provenance = SNAPSHOT / CKPT_REL / 'provenance'
    authenticated = {}
    for name, record in contract['training_artifacts']['files'].items():
        p = provenance / name
        if not p.exists():
            p = ASSETS / CKPT_REL / 'provenance' / name
        assert sha(p) == record['sha256'], (name, sha(p), record['sha256'])
        authenticated[name] = {'path': str(p), 'sha256': sha(p)}
    # Original config parser, followed by every captured resolved flag. Saver
    # is logging only; suppress its import-time filesystem writes for inference.
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(provenance))
    config = import_file('config', provenance / 'config.py')
    sys.argv = old_argv
    for name, value in contract['resolved_flags'].items():
        setattr(config.FLAGS, name, value)
    config.FLAGS.device = 'cpu'
    logging = types.ModuleType('saver')
    logging.saver = types.SimpleNamespace()
    sys.modules['saver'] = logging
    utils = import_file('utils', provenance / 'utils.py')
    import_file('nn_att', provenance / 'nn_att.py')
    implementation = import_file('model', provenance / 'model.py')
    tensorizer = import_file('mlir_data', provenance / 'mlir_data.py')
    model = implementation.Net(403, edge_dim=82, init_pragma_dict=None,
        target_stats=contract['target_stats'], resource_stats=contract['resource_stats'])
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    # Import exact historical helper functions without executing exporter main.
    sys.argv = [sys.argv[0], '--pt_path', '.', '--ckpt', str(CKPT),
        '--checkpoint_contract', str(contract_path), '--checkpoint_sidecar', '.',
        '--feature_schema', '.', '--out', '.', '--embedding_mode', 'static_pre_npt']
    builder = import_file('historical_memory_builder', SNAPSHOT / 'GNN_branch/build_structural_memory.py')
    sys.argv = old_argv
    encoders = pickle.loads(Path(authenticated['encoders.pkl']['path']).read_bytes())
    norm_path = ASSETS / NORM_REL / 'multiscale_normalization_stats.json'
    norm = json.loads(norm_path.read_text())
    norm_manifest = json.loads((ASSETS / NORM_REL / 'multiscale_aligned/memory_manifest.json').read_text())
    pinned_norm = subprocess.check_output(['git','show',REV+':'+str(NORM_REL/'multiscale_normalization_stats.json')],cwd=ROOT)
    assert hashlib.sha256(pinned_norm).hexdigest() == sha(norm_path)
    assert sha(norm_path) == norm_manifest['multiscale']['normalization_stats_sha256']
    assert norm_manifest['gnn_checkpoint_sha256'] == EXPECTED
    assert norm_manifest['gnn_contract_sha256'] == sha(contract_path)
    return model, tensorizer, builder, encoders, norm, dict(
        source_revision=REV, checkpoint_sha256=EXPECTED, checkpoint_path=str(CKPT),
        selection_epoch=9, strict_load=True, model_class='historical model.Net',
        model_implementation_sha256=sha(provenance / 'model.py'),
        branch_model_sha256=sha(SNAPSHOT/'GNN_branch/model.py'),
        model_architecture=str(model),
        model_parameter_count=sum(p.numel() for p in model.parameters()),
        static_layers=4,post_pragma_graph_layers_excluded_from_memory=1,
        reviewed_pinned_sources={str(p.relative_to(SNAPSHOT)):sha(p) for p in [
            SNAPSHOT/'GNN_branch'/name for name in ('model.py','config.py','mlir_data.py',
                'mlir_graph_gen.py','build_structural_memory.py','build_multiscale_memory.py',
                'diagnose_layerwise_memory.py')]},
        contract_sha256=sha(contract_path), authenticated_training_files=authenticated,
        model_init_flags=contract['model_init_flags'], model_init=contract['model_init'],
        normalization=norm, normalization_sha256=sha(norm_path),
        normalization_manifest_sha256=sha(ASSETS / NORM_REL / 'multiscale_aligned/memory_manifest.json'),
        component_layout=['0:64 JK-max / jkn_rms / sqrt(2)',
                          '64:128 action-centered conv1 / centered_conv1_rms / sqrt(2)'],
        clipping_policy='per-layer and final norm capped at 20, historical exporter',
        logger_suppressed=True)


def clip(x):
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite encoder output; refusing silent sanitization')
    return x * (20. / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.)


@torch.no_grad()
def export_graph(gexf, name, runtime):
    from torch_geometric.data import Batch, Data
    model, tensorizer, builder, encoders, norm, identity = runtime
    g = nx.read_gexf(gexf)
    static = tensorizer.encode_static_graph(g, name, name, encoders)
    assert static['x'].shape[1] == 403 and static['edge_attr'].shape[1] == 82
    data = Data(**{k: v for k, v in static.items() if torch.is_tensor(v)})
    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()
    data.X_pragma_per_node = torch.zeros(data.x.shape[0], 5)
    data = builder._build_llm_slots_from_gexf(data, g, max_slots=64)
    data = builder._disable_pragma_conditioning(data)
    batch = Batch.from_data_list([data])
    layers = model.forward_static_node_embed_layers(batch)
    torch.testing.assert_close(layers['jkn'], model.forward_static_node_embed(batch), rtol=1e-5, atol=1e-6)
    selected = torch.where(data.X_llm_scopeids)[0]
    slots = data.X_llm_labelid[selected] - 1
    active = torch.zeros(64, dtype=torch.bool)
    active[slots] = True
    ids = [-1] * 64
    for slot, node in zip(slots.tolist(), selected.tolist()):
        ids[slot] = node
    first, jk = clip(layers['conv_1'][selected]), clip(layers['jkn'][selected])
    center = first.mean(0, keepdim=True)
    action = torch.zeros(64, 128)
    action[slots] = clip(torch.cat([jk / norm['jkn_rms'],
        (first - center) / norm['centered_conv1_rms']], -1) / math.sqrt(2))
    relation, bits, stats = builder.build_action_relation_mask(g,
        node_ids=ids, active_mask=active, max_slots=64)
    # Evidence uses frozen node states. Center against the identical action
    # conv1 mean, preserving the historical action-coordinate convention.
    node_states = torch.cat([clip(layers['jkn']) / norm['jkn_rms'],
        (clip(layers['conv_1']) - center) / norm['centered_conv1_rms']], -1) / math.sqrt(2)
    pack = dict(node_embs=action, node_embs_mask=active, slot_ids=torch.arange(1,65),
        slot_cats=torch.zeros(64,dtype=torch.long), labels=[i+1 if active[i] else -1 for i in range(64)],
        node_ids=ids, max_slots=64, gnn_dim=128, disable_pragma_injection=True,
        gnn_checkpoint_sha256=EXPECTED, feature_schema_sha256=identity['authenticated_training_files']['feature_schema.json']['sha256'],
        gnn_contract_sha256=identity['contract_sha256'],
        embedding_mode='multiscale::jkn+centered_conv1',
        action_relation_schema=builder.ACTION_RELATION_SCHEMA,
        action_relation_mask=relation, action_relation_bits=bits, action_relation_stats=stats,
        source_gexf_sha256=sha(gexf), graph_provenance=static['graph_provenance'],
        node_states=node_states, static=static, layer_action_conv1=first, layer_action_jkn=jk)
    for slot, node in zip(slots.tolist(), selected.tolist()):
        pack['slot_cats'][slot] = data.X_llm_scopecat[node]
    return g, pack


def metrics(matrix):
    x = matrix.double()
    centered = x - x.mean(0)
    sv = torch.linalg.svdvals(centered)
    p = sv / sv.sum().clamp_min(1e-30)
    effective = torch.exp(-(p * p.clamp_min(1e-30).log()).sum()) if sv.sum() > 0 else torch.tensor(0.)
    unit = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return dict(pairwise_cosine=(unit @ unit.T).tolist(),
        pairwise_euclidean=torch.cdist(x,x).tolist(), norms=x.norm(dim=-1).tolist(),
        mean_centered_variance=centered.square().mean().item(), singular_values=sv.tolist(),
        effective_rank_singular_entropy=effective.item(),
        participation_rank=(sv.square().sum().square() / sv.pow(4).sum().clamp_min(1e-30)).item(),
        centered_rank_upper_bound=min(x.shape[0]-1,x.shape[1]))


def main():
    import numpy as np
    torch.set_num_threads(4)
    runtime = load()
    gexf = SNAPSHOT / 'paper_examples/gemv/gemv_mlir.gexf'
    g, pack = export_graph(gexf, 'gemv', runtime)
    source = SNAPSHOT / 'evaluation_set/gemv/code_to_memory_outputs/meta/gemv.c'
    kernel_info = source.with_name('kernel_info.txt')
    mlir = SNAPSHOT / 'paper_examples/gemv/gemv.mlir'
    provenance = pack['graph_provenance']
    assert sha(source) == provenance['source_sha256']
    assert sha(kernel_info) == provenance['action_sha256']
    assert sha(mlir) == provenance['mlir_sha256']
    assert provenance['generator_sha256'] == runtime[-1]['authenticated_training_files']['mlir_graph_gen.py']['sha256']
    generator = import_file('historical_graph_generator', SNAPSHOT / 'GNN_branch/mlir_graph_gen.py')
    _, actions = generator.load_kernel_info_actions(source, kernel_info)
    action_records = []
    for action in actions:
        slot = int(action.action_id[1:])-1
        attrs = g.nodes[str(pack['node_ids'][slot])]
        assert attrs['action_resolution'] == 'source_location'
        assert attrs['action_source_line'] == action.source_line
        assert attrs['action_source_column'] == action.source_column
        if action.kind == 'array':
            assert attrs['array_var'] == {'L1':'A','L2':'x','L3':'y'}[action.action_id]
        action_records.append(dict(action=action.action_id, absolute_memory_slot=slot,
            source_file=source.name, source_line=action.source_line, source_column=action.source_column,
            source_text=source.read_text().splitlines()[action.source_line-1],
            action_type=action.kind, compiler_node_id=pack['node_ids'][slot], compiler_anchor=attrs))
    assert pack['node_embs_mask'].nonzero().flatten().tolist() == list(range(5))
    RAW.mkdir(parents=True, exist_ok=True)
    torch.save(pack, RAW / 'gemv.pt')
    np.savez_compressed(HERE / 'results/original_embeddings.npz',
        full_memory=pack['node_embs'].numpy(), active_vectors=pack['node_embs'][:5].numpy(),
        active=pack['node_embs_mask'].numpy(), slot_ids=np.arange(1,65),
        first=pack['layer_action_conv1'].numpy(), jk=pack['layer_action_jkn'].numpy())
    torch.save({k:v for k,v in pack.items() if k not in ('static','node_states')}, HERE / 'results/original_memory.pt')
    write(HERE / 'results/action_mapping.json', action_records)
    write(HERE / 'results/original_metrics.json', metrics(pack['node_embs'][:5]))
    identity = runtime[-1]
    identity.update(graph_sha256=sha(gexf), source_sha256=sha(source), mlir_sha256=sha(mlir),
        kernel_info_sha256=sha(kernel_info), graph_provenance=provenance,
        node_count=len(g), edge_count=g.number_of_edges(), tensor_pack_sha256=sha(RAW/'gemv.pt'),
        original_npz_sha256=sha(HERE/'results/original_embeddings.npz'),
        graph_compatible=True, actions_fully_mapped=True,
        source_variant='labeled float gemv(); return y[0]; compiler scalarizes y',
        torch_version=torch.__version__, inference_device='cpu', inference_dtype='float32')
    write(HERE / 'results/manifest.json', identity)
    print(json.dumps(dict(strict_load=True, actions=5, shape=[5,128], checkpoint_sha256=EXPECTED)))


if __name__ == '__main__':
    main()
