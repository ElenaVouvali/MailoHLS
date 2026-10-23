import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------
# Paths 
# ---------------------------------------------------
EMB_PATH        = "/home/elvouvali/GNN_embeddings/all_kernels_gnn_embeddings.pt"
JSONL_PATH      = "/home/elvouvali/LLM_data/all_kernels_llm_data_filtered.jsonl"
PRED_EMB_PATH   = "/home/elvouvali/GNN_embeddings/rodinia-knn-2-pipeline-pred.pt"
GT_EMB_PATH     = "/home/elvouvali/GNN_embeddings/rodinia-knn-2-pipeline-gt.pt"
OUT_DIR         = "/home/elvouvali/GNN_embeddings/plots_families"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load base embeddings
# ---------------------------------------------------
embs_base = torch.load(EMB_PATH, map_location="cpu")  # [N, D] tensor
if isinstance(embs_base, np.ndarray):
    # In case you had saved as numpy; just in case
    embs_base = torch.from_numpy(embs_base)
N, D = embs_base.shape
print("Base Embeddings:", embs_base.shape)

# ---------------------------------------------------
# Load pred & gt embeddings and append (for TSNE geometry)
# ---------------------------------------------------
emb_pred = torch.load(PRED_EMB_PATH, map_location="cpu").detach().cpu()
emb_gt   = torch.load(GT_EMB_PATH,   map_location="cpu").detach().cpu()

# ensure shape [1, D]
emb_pred = emb_pred.view(1, -1)
emb_gt   = emb_gt.view(1, -1)

assert emb_pred.shape[1] == D and emb_gt.shape[1] == D, \
    f"Dim mismatch: pred {emb_pred.shape}, gt {emb_gt.shape}, base D={D}"

# Extended embeddings [N+2, D] for TSNE only
embs_ext = torch.cat([embs_base, emb_pred, emb_gt], dim=0)  # tensor
embs_ext_np = embs_ext.numpy()
N_ext = embs_ext_np.shape[0]
print("Extended embeddings for TSNE:", embs_ext_np.shape)

# ---------------------------------------------------
# Load JSONL meta: kernel_name + weight per row (only for base N points)
# ---------------------------------------------------
kernel_names = []
weights = []
with open(JSONL_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        kernel_names.append(obj["kernel_name"])
        # make sure to cast to float in case it's stored as string
        weights.append(float(obj["weight"]))

kernel_names = np.array(kernel_names)
weights = np.array(weights, dtype=float)

assert len(kernel_names) == N, "mismatch between base embeddings and JSONL"
assert len(weights) == N,      "mismatch between base embeddings and weights"

print("Loaded JSONL rows:", len(kernel_names))
print("Weight stats: min =", weights.min(), "max =", weights.max())

# ---------------------------------------------------
# PCA + t-SNE (computed once for all N+2 points)
# ---------------------------------------------------
print("Running PCA + t-SNE for all points (including pred & gt)...")
pca = PCA(n_components=min(20, D))
embs_pca = pca.fit_transform(embs_ext_np)   # [N_ext, k]

tsne = TSNE(
    n_components=2,
    init="pca",
    learning_rate="auto",
    perplexity=30,
    n_iter=1000,
    verbose=1,
)
embs_2d_ext = tsne.fit_transform(embs_pca)  # [N_ext, 2]

# Back out separate 2D coords
embs_2d      = embs_2d_ext[:N, :]      # base dataset
emb_pred_2d  = embs_2d_ext[N, :]       # predicted hotspot point
emb_gt_2d    = embs_2d_ext[N+1, :]     # ground truth hotspot point

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

# ---------------------------------------------------
# Plot per family, coloring by weight
# ---------------------------------------------------
# We keep a global color scale so "good vs bad" is comparable across families
w_min, w_max = weights.min(), weights.max()
norm = plt.Normalize(vmin=w_min, vmax=w_max)
cmap = plt.get_cmap("viridis")   # high = yellow-ish, low = dark blue

for fam_name, predicate in FAMILIES.items():
    fam_mask = np.array([predicate(k) for k in kernel_names])
    fam_count = fam_mask.sum()
    if fam_count == 0:
        print(f"{fam_name}: no points, skipping")
        continue

    others_mask = ~fam_mask
    w_family = weights[fam_mask]

    print(f"{fam_name}: {fam_count} points")

    plt.figure(figsize=(9, 7))

    # 1) Plot other points in light grey
    plt.scatter(
        embs_2d[others_mask, 0],
        embs_2d[others_mask, 1],
        s=3,
        c="lightgrey",
        alpha=0.2,
        label="others",
    )

    # 2) Plot family points colored by weight
    sc = plt.scatter(
        embs_2d[fam_mask, 0],
        embs_2d[fam_mask, 1],
        s=12,
        c=w_family,
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        label=f"{fam_name} ({fam_count})",
    )

    # 3) Highlight pred/gt points only on hotspot-2-pipeline family
    if fam_name == "rodinia-knn-2-pipeline":
        plt.scatter(
            emb_gt_2d[0], emb_gt_2d[1],
            s=80,
            marker="*",
            color="black",
            edgecolors="white",
            linewidths=1.0,
            label="GT embedding",
        )
        plt.scatter(
            emb_pred_2d[0], emb_pred_2d[1],
            s=80,
            marker="X",
            color="red",
            edgecolors="white",
            linewidths=1.0,
            label="Pred embedding",
        )

    plt.title(f"GNN embeddings – family {fam_name}\n(color = weight, higher = closer to Pareto)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # 4) Colorbar for weights
    cbar = plt.colorbar(sc)
    cbar.set_label("Pareto weight (higher = closer to frontier)")

    plt.legend(markerscale=1.8, fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"tsne_{fam_name}_by_weight.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved", out_path)

