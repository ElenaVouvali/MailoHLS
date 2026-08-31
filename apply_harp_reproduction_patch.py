#!/usr/bin/env python3
"""Apply the minimal MailoHLS source patch needed by the HARP reproduction.

Target: the stage2-analysis-refactor source around commit d0364adc.
This script is intentionally conservative: every source edit is anchored to the
exact old block and aborts if that block cannot be found.

It patches:
  * GNN_branch/config.py      -- adds explicit HARP graph/cache paths
  * GNN_branch/main_GNN.py    -- dispatches mlir_data vs harp_data by --dataset
  * GNN_branch/train_GNN.py   -- uses the same selected backend in training

Run from the repository root after copying GNN_branch/harp_data.py into place.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# If this helper was copied into a repo root, use cwd/repo; otherwise it may be
# run from the resend bundle with an explicit repository path as argv[1].
import argparse
p = argparse.ArgumentParser()
p.add_argument("repo", nargs="?", default=".")
args = p.parse_args()
REPO = Path(args.repo).expanduser().resolve()


def patch_once(path: Path, old: str, new: str, sentinel: str) -> None:
    text = path.read_text(encoding="utf-8")
    if sentinel in text:
        print(f"[SKIP] {path}: already patched ({sentinel})")
        return
    if old not in text:
        raise RuntimeError(
            f"Refusing to patch {path}: expected source anchor was not found. "
            "Use the stage2-analysis-refactor source near d0364adc or apply the "
            "equivalent edit manually."
        )
    backup = path.with_suffix(path.suffix + ".pre_harp_repro.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[PATCH] {path}")


config = REPO / "GNN_branch" / "config.py"
main = REPO / "GNN_branch" / "main_GNN.py"
train = REPO / "GNN_branch" / "train_GNN.py"
harp = REPO / "GNN_branch" / "harp_data.py"
for required in (config, main, train, harp):
    if not required.is_file():
        raise FileNotFoundError(required)

# 1) Config: add explicit HARP graph/cache locations.  Insert immediately before
# the model-architecture section so this stays next to the MLIR path options.
config_anchor = "\n################## model architecture ##################\n"
config_insert = r'''

# HARP/GNOSIS reproduction paths.  These are separate from the MLIR cache so a
# paired experiment can guarantee that only the static graph representation
# changes.
parser.add_argument(
    "--harp_graph_dir",
    default=None,
    help="Directory containing deterministic GNOSIS HARP GEXF graphs.",
)
parser.add_argument(
    "--harp_dataset_cache_dir",
    default=None,
    help="Explicit processed HARP tensor cache used by --dataset harp.",
)

################## model architecture ##################
'''
patch_once(config, config_anchor, config_insert, "--harp_dataset_cache_dir")

# 2) main_GNN.py: select the backend once from FLAGS.dataset.
main_old = '''# from data import get_data_list, MyOwnDataset\n# import data\n\nfrom mlir_data import get_data_list, MyOwnDataset\nimport mlir_data as data\n\nSAVE_DIR = data.SAVE_DIR\n'''
main_new = '''# Paired representation backend selection.\n# The model/training stack stays identical; only the tensorized static graph\n# module changes between HARP-Rep and Structured-MLIR.\nif FLAGS.dataset == "harp":\n    from harp_data import get_data_list, MyOwnDataset\n    import harp_data as data\nelif FLAGS.dataset == "mlir":\n    from mlir_data import get_data_list, MyOwnDataset\n    import mlir_data as data\nelse:\n    raise ValueError(f"Unsupported --dataset={FLAGS.dataset!r}; expected harp or mlir")\n\nSAVE_DIR = data.SAVE_DIR\n'''
patch_once(main, main_old, main_new, "Paired representation backend selection")

# 3) train_GNN.py: use the exact same selected backend for dataset type, split
# helpers and provenance constants.
train_old = '''# from data import MyOwnDataset, get_kernel_samples, split_dataset, split_dataset_resample, split_train_test_kernel\n# import data\nfrom mlir_data import (\n    MyOwnDataset,\n    get_kernel_samples,\n    split_dataset,\n    split_dataset_resample,\n    split_train_val_test_kernel,\n)\nimport mlir_data as data\nSAVE_DIR = data.SAVE_DIR\n'''
train_new = '''# Paired representation backend selection.  Keep every downstream optimizer,\n# sampler, loss, metric, and checkpoint rule shared.\nif FLAGS.dataset == "harp":\n    from harp_data import (\n        MyOwnDataset,\n        get_kernel_samples,\n        split_dataset,\n        split_dataset_resample,\n        split_train_val_test_kernel,\n    )\n    import harp_data as data\nelif FLAGS.dataset == "mlir":\n    from mlir_data import (\n        MyOwnDataset,\n        get_kernel_samples,\n        split_dataset,\n        split_dataset_resample,\n        split_train_val_test_kernel,\n    )\n    import mlir_data as data\nelse:\n    raise ValueError(f"Unsupported --dataset={FLAGS.dataset!r}; expected harp or mlir")\nSAVE_DIR = data.SAVE_DIR\n'''
patch_once(train, train_old, train_new, "Paired representation backend selection")

# Make provenance capture representation-aware.  This is a second guarded edit
# in train_GNN.py, separate from the import patch above.
text = train.read_text(encoding="utf-8")
prov_old = '''        'train_GNN.py': Path(__file__).resolve(),\n        'mlir_data.py': Path(__file__).with_name('mlir_data.py').resolve(),\n        'mlir_graph_gen.py': Path(__file__).with_name('mlir_graph_gen.py').resolve(),\n        'utils.py': Path(__file__).with_name('utils.py').resolve(),\n'''
prov_new = '''        'train_GNN.py': Path(__file__).resolve(),\n        f'{FLAGS.dataset}_data.py': Path(data.__file__).resolve(),\n        'utils.py': Path(__file__).with_name('utils.py').resolve(),\n'''
if "f'{FLAGS.dataset}_data.py'" not in text:
    if prov_old not in text:
        raise RuntimeError(
            "Refusing to patch train_GNN.py provenance: expected MLIR provenance "
            "anchor was not found."
        )
    text = text.replace(prov_old, prov_new, 1)
    train.write_text(text, encoding="utf-8")
    print(f"[PATCH] {train}: representation-aware provenance")
else:
    print(f"[SKIP] {train}: provenance already representation-aware")

print("[DONE] HARP reproduction source patch applied.")
print("Next: python -m py_compile GNN_branch/{config,main_GNN,train_GNN,harp_data}.py")
