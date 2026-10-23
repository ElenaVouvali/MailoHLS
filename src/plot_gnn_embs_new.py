import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------
EMB_PATH        = "/home/elvouvali/GNN_embeddings/all_kernels_gnn_embeddings.pt"
JSONL_PATH      = "/home/elvouvali/LLM_data/all_kernels_llm_data_filtered.jsonl"
# PRED_EMB_PATH   = "/home/elvouvali/GNN_embeddings/machsuite-gemm-blocked-pred.pt"
# GT_EMB_PATH     = "/home/elvouvali/GNN_embeddings/machsuite-gemm-blocked-gt.pt"
OUT_DIR         = "/home/elvouvali/GNN_embeddings/plots"

# Define your families here
FAMILIES = {
#    "rodinia_knn":              lambda k: "rodinia-knn" in k,
#    "rodinia_kmeans":           lambda k: "rodinia-kmeans" in k,
#    "rodinia_cfd_step":         lambda k: "rodinia_cfd_step_factor" in k,
#    "rodinia_streamcluster":    lambda k: "rodinia_streamcluster" in k,
#    "rodinia_lavaMD":           lambda k: "rodinia_lavaMD" in k,
    "rodinia-hotspot":          lambda k: "rodinia-hotspot" in k,
    "rodinia-hotspot-2-pipeline":       lambda k: "rodinia-hotspot-2-pipeline" in k,
    "rodinia-hotspot-0-baseline":       lambda k: "rodinia-hotspot-2-baseline" in k,
    "rodinia-hotspot-3-unroll":       lambda k: "rodinia-hotspot-3-unroll" in k,
    "rodinia-hotspot-1-tiling":       lambda k: "rodinia-hotspot-1-tiling" in k,
}

# The specific family where we want to highlight the Pred/GT points
TARGET_FAMILY = None    # "machsuite-gemm-blocked"


def get_many_colors(n):
    """Get at least n distinct colors by chaining tab20 / tab20b / tab20c."""
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        for i in range(cmap.N):
            colors.append(cmap(i))
    if n > len(colors):
        # Fallback: repeat colors if we run out (or raise error)
        print(f"[Warn] Requested {n} colors, reusing list.")
        return colors * (n // len(colors) + 1)
    return colors[:n]


def main():
    parser = argparse.ArgumentParser(description="Plot GNN embeddings with optional weight coloring.")
    parser.add_argument("--weight", action="store_true", help="If set, color points by 'weight' field in JSONL.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------------------------------------------
    # 1. Load Embeddings & Metadata
    # ---------------------------------------------------
    print(f"Loading base embeddings from {EMB_PATH}...")
    embs_base = torch.load(EMB_PATH, map_location="cpu")
    if isinstance(embs_base, np.ndarray):
        embs_base = torch.from_numpy(embs_base)
    
    # Detach/CPU in case they are tensors on GPU
    embs_base = embs_base.detach().cpu()
    N, D = embs_base.shape
    print(f"  Shape: {N} x {D}")

    # Load Pred & GT
#    print(f"Loading Pred/GT embeddings...")
#    emb_pred = torch.load(PRED_EMB_PATH, map_location="cpu").detach().cpu().view(1, -1)
#    emb_gt   = torch.load(GT_EMB_PATH,   map_location="cpu").detach().cpu().view(1, -1)

#    assert emb_pred.shape[1] == D, f"Pred dim {emb_pred.shape[1]} != {D}"
#    assert emb_gt.shape[1]   == D, f"GT dim {emb_gt.shape[1]} != {D}"

    # Concatenate for dimensionality reduction: [N + 2, D]
#    embs_ext = torch.cat([embs_base, emb_pred, emb_gt], dim=0)
#    embs_ext_np = embs_ext.numpy()
#    N_ext = embs_ext_np.shape[0]

    embs_ext = torch.cat([embs_base], dim=0)
    embs_ext_np = embs_ext.numpy()
    N_ext = embs_ext_np.shape[0]

    # Load JSONL metadata (Kernels & Weights)
    print(f"Loading metadata from {JSONL_PATH}...")
    kernel_names = []
    weights = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            kernel_names.append(obj["kernel_name"])
            weights.append(float(obj["weight"]))
    
    kernel_names = np.array(kernel_names)
    weights = np.array(weights, dtype=float)

    if len(kernel_names) != N:
        raise ValueError(f"JSONL row count ({len(kernel_names)}) does not match embedding count ({N})")

    # ---------------------------------------------------
    # 2. Dimensionality Reduction (PCA + t-SNE)
    # ---------------------------------------------------
    print("Running PCA (N -> 20)...")
    pca = PCA(n_components=min(20, D))
    embs_pca = pca.fit_transform(embs_ext_np)

    print("Running t-SNE (20 -> 2)...")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, n_iter=1000, verbose=1)
    embs_2d_ext = tsne.fit_transform(embs_pca)

    # Separate back into base, pred, gt
    embs_2d = embs_2d_ext[:N, :]
#    pred_2d = embs_2d_ext[N, :]
#    gt_2d   = embs_2d_ext[N+1, :]

    # ---------------------------------------------------
    # 3. Global Plot (All Kernels)
    # ---------------------------------------------------
    print("Generating Global Plot...")
    plt.figure(figsize=(12, 10))
    
    if args.weight:
        # Continuous coloring by weight
        sc = plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=weights, cmap='viridis', s=8, alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label("Pareto Weight")
        title_suffix = "(Colored by Weight)"
    else:
        # Discrete coloring by kernel name
        unique_kernels = sorted(list(set(kernel_names)))
        colors = get_many_colors(len(unique_kernels))
        k2c = {k: colors[i] for i, k in enumerate(unique_kernels)}
        
        for k in unique_kernels:
            mask = (kernel_names == k)
            plt.scatter(embs_2d[mask, 0], embs_2d[mask, 1], c=[k2c[k]], s=8, alpha=0.7, label=k)
        
        # Legend outside
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, markerscale=3)
        title_suffix = "(Colored by Kernel)"

    # Overlay Pred/GT on global plot (optional, but good for context)
#    plt.scatter(pred_2d[0], pred_2d[1], s=150, marker='*', c='red', edgecolors='black', label='Prediction')
#    plt.scatter(gt_2d[0],   gt_2d[1],   s=150, marker='X', c='black', edgecolors='white', label='Ground Truth')

    plt.title(f"Global GNN Embeddings {title_suffix}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    
    global_fname = "global_weight.png" if args.weight else "global_kernel.png"
    plt.savefig(os.path.join(OUT_DIR, global_fname), dpi=300)
    plt.close()
    print(f"Saved global plot to {global_fname}")

    # ---------------------------------------------------
    # 4. Family Plots
    # ---------------------------------------------------
    print("Generating Family Plots...")
    
    # Pre-calculate weight normalization for consistent coloring across plots
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())

    for fam_name, predicate in FAMILIES.items():
        fam_mask = np.array([predicate(k) for k in kernel_names])
        fam_count = fam_mask.sum()
        
        if fam_count == 0:
            print(f"  [Skip] Family '{fam_name}' has 0 points.")
            continue
            
        others_mask = ~fam_mask
        
        plt.figure(figsize=(10, 8))
        
        # Plot "Others" as background
        plt.scatter(embs_2d[others_mask, 0], embs_2d[others_mask, 1], c='lightgrey', s=5, alpha=0.2, label='Others')

        if args.weight:
            # Color family points by weight
            fam_weights = weights[fam_mask]
            sc = plt.scatter(embs_2d[fam_mask, 0], embs_2d[fam_mask, 1], 
                             c=fam_weights, cmap='viridis', norm=norm, s=20, alpha=0.9, label=fam_name)
            cbar = plt.colorbar(sc)
            cbar.set_label("Pareto Weight")
        else:
            # Color family points by kernel name (discrete)
            fam_unique_kernels = sorted(list(set(kernel_names[fam_mask])))
            fam_colors = get_many_colors(len(fam_unique_kernels))
            
            for i, k in enumerate(fam_unique_kernels):
                k_mask = fam_mask & (kernel_names == k)
                plt.scatter(embs_2d[k_mask, 0], embs_2d[k_mask, 1], 
                            c=[fam_colors[i]], s=20, alpha=0.9, label=k)
            
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        # Highlight Pred/GT ONLY if this is the target family
        if fam_name == TARGET_FAMILY:
            plt.scatter(pred_2d[0], pred_2d[1], s=200, marker='*', c='red', edgecolors='black', label='Prediction', zorder=10)
            plt.scatter(gt_2d[0],   gt_2d[1],   s=200, marker='X', c='black', edgecolors='white', label='Ground Truth', zorder=10)
            # Add small legend for these markers if not using discrete legend
            if args.weight:
                plt.legend(loc="upper right")

        plt.title(f"Family: {fam_name} {'(Weighted)' if args.weight else ''}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        
        fam_fname = f"{fam_name}_{'weight' if args.weight else 'kernel'}.png"
        plt.savefig(os.path.join(OUT_DIR, fam_fname), dpi=300)
        plt.close()
        print(f"  Saved {fam_fname}")


    # ---------------------------------------------------
    # 5. Combined Families Plot with Best Design & LLM Pred
    # ---------------------------------------------------
    print("Generating Combined Families Plot...")
    
    # 1. Load the new points
    best_design_emb = torch.load("/home/elvouvali/GNN_embeddings/best_hotspot.pt", map_location="cpu").detach().cpu().numpy().reshape(1, -1)
    llm_pred_emb = torch.load("/home/elvouvali/GNN_embeddings/llm_hotspot.pt", map_location="cpu").detach().cpu().numpy().reshape(1, -1)

    # 2. Project them into the SAME 2D space
    # IMPORTANT: We use pca.transform and tsne.fit_transform isn't possible for new points 
    # individually, so we usually transform them during the initial fit or use a trick.
    # Since we want them in the existing plot:
    
    # We must re-run dimensionality reduction including these points to ensure they align
    embs_combined_ext = np.concatenate([embs_ext_np, best_design_emb, llm_pred_emb], axis=0)
    
    print("Re-running projection to include highlights...")
    pca_ext = PCA(n_components=min(20, D))
    embs_pca_ext = pca_ext.fit_transform(embs_combined_ext)
    
    tsne_ext = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, n_iter=1000)
    embs_2d_all = tsne_ext.fit_transform(embs_pca_ext)
    
    # Slice the coordinates
    base_coords = embs_2d_all[:N, :]
    best_coords = embs_2d_all[N, :]
    llm_coords  = embs_2d_all[N+1, :]

    plt.figure(figsize=(12, 10))

    # 3. Plot Background (Gray) and Families
    any_fam_mask = np.zeros(N, dtype=bool)
    family_assignments = np.full(N, -1)
    fam_list = list(FAMILIES.keys())
    for i, (fam_name, predicate) in enumerate(FAMILIES.items()):
        mask = np.array([predicate(k) for k in kernel_names])
        any_fam_mask |= mask
        family_assignments[mask] = i

    plt.scatter(base_coords[~any_fam_mask, 0], base_coords[~any_fam_mask, 1], 
                c='lightgrey', s=5, alpha=0.15, label='Others')

    fam_colors = get_many_colors(len(fam_list))
    for i, fam_name in enumerate(fam_list):
        mask = (family_assignments == i)
        if mask.any():
            plt.scatter(base_coords[mask, 0], base_coords[mask, 1], 
                        c=[fam_colors[i]], s=25, alpha=0.8, label=fam_name)

    # 4. Add the special Highlight Points
    plt.scatter(best_coords[0], best_coords[1], s=250, marker='*', 
                c='gold', edgecolors='black', label='Best Design', zorder=15)
    
    plt.scatter(llm_coords[0], llm_coords[1], s=200, marker='P', 
                c='red', edgecolors='white', label='LLM Pred', zorder=15)

    plt.title("Combined Families: Comparison with Best Design & LLM Prediction")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=1.5)
    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, "combined_hotspot.png"), dpi=300)
    plt.close()
    print("Saved combined_with_highlights.png")

    print("Done.")

if __name__ == "__main__":
    main()
