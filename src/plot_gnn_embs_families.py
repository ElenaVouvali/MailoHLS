import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Paths 
EMB_PATH   = "/home/elvouvali/GNN_embeddings/all_kernels_gnn_embeddings.pt"
JSONL_PATH = "/home/elvouvali/LLM_data/all_kernels_llm_data_filtered.jsonl"
OUT_DIR    = "/home/elvouvali/GNN_embeddings/plots_families"
os.makedirs(OUT_DIR, exist_ok=True)

# Special single-point embeddings (prediction & ground truth)
PRED_EMB_PATH = "/home/elvouvali/GNN_embeddings/rodinia-knn-2-pipeline-pred.pt"
GT_EMB_PATH   = "/home/elvouvali/GNN_embeddings/rodinia-knn-2-pipeline-gt.pt"

# ---------------------------------------------------
# Load base embeddings [N, D]
# ---------------------------------------------------
embs_base = torch.load(EMB_PATH, map_location="cpu").detach().cpu()
N, D = embs_base.shape
print("Base embeddings:", embs_base.shape)

# ---------------------------------------------------
# Load pred & gt embeddings and append
# ---------------------------------------------------
emb_pred = torch.load(PRED_EMB_PATH, map_location="cpu").detach().cpu()
emb_gt   = torch.load(GT_EMB_PATH,   map_location="cpu").detach().cpu()

# ensure shape [1, D]
emb_pred = emb_pred.view(1, -1)
emb_gt   = emb_gt.view(1, -1)

assert emb_pred.shape[1] == D and emb_gt.shape[1] == D, \
    f"Dim mismatch: pred {emb_pred.shape}, gt {emb_gt.shape}, base D={D}"

# Extended matrix [N+2, D]
embs_ext = torch.cat([embs_base, emb_pred, emb_gt], dim=0)
embs_ext_np = embs_ext.numpy()
N_ext = embs_ext_np.shape[0]
print("Extended embeddings:", embs_ext_np.shape)

# ---------------------------------------------------
# Load JSONL meta: kernel_name per row (for the N base points)
# ---------------------------------------------------
kernel_names = []
with open(JSONL_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        kernel_names.append(obj["kernel_name"])

kernel_names = np.array(kernel_names)
assert len(kernel_names) == N, "mismatch between base embeddings and JSONL"
print("JSONL rows:", len(kernel_names))

# ---------------------------------------------------
# PCA + t-SNE on the *extended* set
# ---------------------------------------------------
print("Running PCA + t-SNE for all points (base + pred + gt)...")
pca = PCA(n_components=min(20, D))
embs_pca_ext = pca.fit_transform(embs_ext_np)

tsne = TSNE(
    n_components=2,
    init="pca",
    learning_rate="auto",
    perplexity=30,
    n_iter=1000,
    verbose=1,
)
embs_2d_ext = tsne.fit_transform(embs_pca_ext)  # [N+2, 2]

# Split back:
embs_2d = embs_2d_ext[:N]        # dataset points
pred_2d = embs_2d_ext[N]         # prediction
gt_2d   = embs_2d_ext[N + 1]     # ground truth

# ---------------------------------------------------
# Define kernel families
# ---------------------------------------------------
FAMILIES = {
    "rodinia_knn":                  lambda k: "rodinia-knn" in k,
    "rodinia_kmeans":               lambda k: "rodinia-kmeans" in k,
    "rodinia_cfd_step":             lambda k: "rodinia_cfd_step_factor" in k,
    "rodinia_streamcluster":        lambda k: "rodinia_streamcluster" in k,
    "rodinia_lavaMD":               lambda k: "rodinia_lavaMD" in k,
    "rodinia-knn-2-pipeline":   lambda k: "rodinia-knn-2-pipeline" in k,
    "rodinia-hotspot":              lambda k: "rodinia-hotspot" in k,
}

def get_many_colors(n):
    """Get at least n distinct colors by chaining tab20 / tab20b / tab20c."""
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        for i in range(cmap.N):
            colors.append(cmap(i))
    if n > len(colors):
        raise ValueError(f"Requested {n} colors, but only {len(colors)} are available.")
    return colors[:n]

# ---------------------------------------------------
# Plot per family, with per-kernel colors inside family
# ---------------------------------------------------
for fam_name, predicate in FAMILIES.items():
    fam_mask = np.array([predicate(k) for k in kernel_names])
    fam_count = fam_mask.sum()
    if fam_count == 0:
        continue

    others_mask = ~fam_mask

    # Unique kernels inside this family
    fam_kernel_list = sorted(set(kernel_names[fam_mask]))
    n_kernels_in_family = len(fam_kernel_list)
    colors = get_many_colors(n_kernels_in_family)
    kernel_to_color = {k: colors[i] for i, k in enumerate(fam_kernel_list)}

    print(f"{fam_name}: {fam_count} points, {n_kernels_in_family} kernels")

    plt.figure(figsize=(9, 7))

    # Plot others in light grey
    plt.scatter(
        embs_2d[others_mask, 0],
        embs_2d[others_mask, 1],
        s=3,
        c="lightgrey",
        alpha=0.25,
        label="others",
    )

    # Plot each kernel in the family with its own color
    for k in fam_kernel_list:
        k_mask = fam_mask & (kernel_names == k)
        x = embs_2d[k_mask, 0]
        y = embs_2d[k_mask, 1]
        plt.scatter(
            x,
            y,
            s=10,
            color=kernel_to_color[k],
            alpha=0.9,
            label=f"{k} (n={k_mask.sum()})",
        )

    # ---------------------------------------------------
    # Overlay pred & gt ONLY on the hotspot-2-pipeline family plot
    # ---------------------------------------------------
    if fam_name == "rodinia-knn-2-pipeline":
        plt.scatter(
            pred_2d[0],
            pred_2d[1],
            s=120,
            marker="*",
            c="black",
            edgecolors="white",
            linewidths=0.8,
            label="LLM prediction",
        )
        plt.scatter(
            gt_2d[0],
            gt_2d[1],
            s=120,
            marker="X",
            c="red",
            edgecolors="white",
            linewidths=0.8,
            label="ground truth",
        )

    plt.title(f"GNN embeddings – family {fam_name}")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(
        markerscale=2,
        fontsize=7,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"tsne_{fam_name}_per_kernel.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved", out_path)

