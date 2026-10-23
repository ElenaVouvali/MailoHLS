#-----------------------------------------------------------
#                       Imports 
#-----------------------------------------------------------

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data

from config import FLAGS
from model import Net
from data import MyOwnDataset



def load_and_clean_graph(path):
    """Load a single .pt graph and remove any non-tensor attributes."""
    g = torch.load(path, weights_only=False)

    # Inspect the attributes of the Data object
    if isinstance(g, Data):
        keys = list(g.keys())
        # Optionally: drop any non-string keys (usually none in PyG)
        bad_keys = [k for k in keys if not isinstance(k, str)]
        for bk in bad_keys:
            del g[bk]

        # If you had edge_id_to_idx or similar, you can still drop it safely:
        if hasattr(g, "edge_id_to_idx"):
            del g.edge_id_to_idx

    return g



#-----------------------------------------------------------
#               .pt points to graph embeddings
#-----------------------------------------------------------

def extract_single_embedding(pt_point_path, checkpoint_path, device=FLAGS.device):
    """
    pt_point: a ProGraML graph from my dataset
    checkpoint_path: path to trained GNN state_dict (.pth)
    returns: Tensor [d] of frozen embedding (order matches dataset indexing)
    """

    pt_point = load_and_clean_graph(pt_point_path)

    num_features = pt_point.x.size(-1)
    edge_dim = pt_point.edge_attr.size(-1) if getattr(pt_point, "edge_attr", None) is not None else 0

    assert num_features == FLAGS.num_features, f"Feature dim mismatch: {num_features} vs {FLAGS.num_features}"
    assert edge_dim == FLAGS.edge_dim, f"Edge dim mismatch: {edge_dim} vs {FLAGS.edge_dim}"

    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    # freeze + eval
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    batch = Batch.from_data_list([pt_point]).to(device)
    with torch.no_grad():
        e = model.forward_embed(batch)   # shape: [batch_size, d]
    
    return e.cpu().squeeze(0)        # shape: [d]



def extract_embeddings(dataset, checkpoint_path, batch_size, device):
    """
    dataset: an instance of MyOwnDataset (pyg dataset)
    checkpoint_path: path to trained GNN state_dict (.pth)
    returns: Tensor [N_graphs, d] of frozen embeddings (order matches dataset indexing)
    """
    first = dataset[0]
    num_features = first.x.size(-1)
    edge_dim = first.edge_attr.size(-1) if getattr(first, "edge_attr", None) is not None else 0

    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    # freeze + eval
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,      
        pin_memory=False
    )

    embs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            e = model.forward_embed(batch)   # [batch_size, d]
            embs.append(e.cpu())
            # Optional light progress log:
            print(f"Processed {(i + 1) * batch_size} graphs")

    return torch.cat(embs, dim=0)


#-----------------------------------------------------------
#                   Main Function
#-----------------------------------------------------------

if __name__ == "__main__":
    # Choose only good graphs to extract embeddings from
    try:
        with open("good_files.txt") as f:
            good_files = [line.strip() for line in f if line.strip()]
        dataset = MyOwnDataset(data_files=good_files)
        print(f"Embedding filtered dataset with {len(dataset)} graphs")
    except FileNotFoundError:
        dataset = MyOwnDataset()
        print(f"Embedding full dataset with {len(dataset)} graphs")

    pt_point_path_1 = "/home/elvouvali/save/harp/rodinia-knn-2-pipeline/data_289.pt"
    pt_point_path_2 = "/home/elvouvali/LLM_predictions/rodinia-knn-2-pipeline-pred.pt"
    checkpoint_path = "/home/elvouvali/logs/all_kernels_GNN_train/run1/val_model_state_dict.pth"
    out_path = "/home/elvouvali/GNN_embeddings/all_kernels_gnn_embeddings.pt"

    embs = extract_embeddings(
        dataset,
        checkpoint_path,
        batch_size=FLAGS.batch_size,
        device=FLAGS.device
    )

    torch.save(embs, out_path)
    print("Saved embeddings to:", out_path)
    print("Embeddings shape:", embs.shape)

#    emb_1 = extract_single_embedding(
#        pt_point_path_1,
#        checkpoint_path,
#        device=FLAGS.device
#        )

#    torch.save(emb_1, "/home/elvouvali/GNN_embeddings/rodinia-knn-2-pipeline-gt.pt")

#    emb_2 = extract_single_embedding(
#        pt_point_path_2,
#        checkpoint_path,
#        device=FLAGS.device
#        )

#    torch.save(emb_2, "/home/elvouvali/GNN_embeddings/rodinia-knn-2-pipeline-pred.pt")

