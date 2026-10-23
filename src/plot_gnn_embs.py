import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

EMB_PATH   = "/home/elvouvali/GNN_embeddings/all_kernels_gnn_embeddings.pt"
JSONL_PATH = "/home/elvouvali/LLM_data/all_kernels_llm_data_filtered.jsonl"
OUT_DIR    = "/home/elvouvali/GNN_embeddings/plots"

import os
os.makedirs(OUT_DIR, exist_ok=True)

def get_many_colors(n_classes):
    # Get 20 colors from each qualitative colormap
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        for i in range(cmap.N):   # each has 20 entries
            colors.append(cmap(i))
    # Now colors has 60 entries (3 * 20)
    if n_classes > len(colors):
        raise ValueError(f"Requested {n_classes} colors, but only {len(colors)} are defined.")
    return colors[:n_classes]


# Load embeddings
embs = torch.load(EMB_PATH, map_location="cpu")  # [N, 64]
embs = embs.detach().cpu().numpy()
N, D = embs.shape
print(f"Loaded embeddings: {embs.shape}")


# Load JSONL kernel names
kernel_names = []
with open(JSONL_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        kernel_names.append(obj["kernel_name"])

assert len(kernel_names) == N, f"JSONL rows ({len(kernel_names)}) != embeddings ({N})"
print("Loaded JSONL rows:", len(kernel_names))

kernel_counts = Counter(kernel_names)
print("Num distinct kernels:", len(kernel_counts))

# Map kernel_name to integer label [0..53]
unique_labels = sorted(set(kernel_names))
label2id = {k: i for i, k in enumerate(unique_labels)}
y_kernel = np.array([label2id[k] for k in kernel_names], dtype=np.int64)

print("Kernel label mapping :")
for k, i in list(label2id.items()):
    print(f"  {i}: {k}")


# Dimensionality reduction: PCA (64 -> 20) + t-SNE (20 -> 2)
print("Running PCA...")
pca = PCA(n_components=min(20, D))
embs_pca = pca.fit_transform(embs)
print("PCA explained variance ratio (first 5):", pca.explained_variance_ratio_[:5])

print("Running t-SNE on PCA features...")
tsne = TSNE(
    n_components=2,
    init="pca",
    learning_rate="auto",
    perplexity=30,
    n_iter=1000,
    verbose=1,
)
embs_2d = tsne.fit_transform(embs_pca)  # [N, 2]


# Plot colored by kernel
plt.figure(figsize=(10, 8))

num_kernels = len(label2id) 
colors = get_many_colors(num_kernels)

for kernel_name, idx in label2id.items():
    mask = (y_kernel == idx)
    x = embs_2d[mask, 0]
    y = embs_2d[mask, 1]
    plt.scatter(
        x, y,
        s=5,
        alpha=0.7,
        color=colors[idx],
        label=kernel_name
    )

plt.title("GNN embeddings (t-SNE, colored by kernel)")
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.legend(markerscale=3, fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "embeddings_tsne_by_kernel.png")
plt.savefig(out_path, dpi=300)
print("Saved:", out_path)

