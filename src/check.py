import torch

pt_path = "/home/ubuntu/save/harp/pragma-free_kernels/rodinia-knn-1-tiling_processed_result.pt"
mem_path = "/home/ubuntu/save/harp/memory_tokens/rodinia-knn-1-tiling.memory.pt"

pt = torch.load(pt_path, map_location="cpu", weights_only=False)
mem = torch.load(mem_path, map_location="cpu", weights_only=False)

print("Active slots inspection for rodinia-knn-1-tiling")
print("-" * 100)

active_slots = [i for i, m in enumerate(mem["node_embs_mask"].tolist()) if m]

if not active_slots:
    print("No active memory slots found.")
else:
    for slot in active_slots:
        lid = mem["labels"][slot]
        nid = mem["node_ids"][slot]
        scat = int(mem["slot_cats"][slot]) if "slot_cats" in mem else int(pt.X_llm_scopecat[nid].item())

        row = {
            "slot_idx": slot,
            "label_L": lid,
            "node_id": nid,
            "slot_cat": scat,  # 1 = pseudo-loop anchor, 2 = array-scope anchor
            "X_llm_scopeids": int(pt.X_llm_scopeids[nid].item()),
            "X_llm_scopecat": int(pt.X_llm_scopecat[nid].item()),
            "X_arrayscopenids": int(pt.X_arrayscopenids[nid].item()),
            "X_pseudonids": int(pt.X_pseudonids[nid].item()),
            "X_array_partition_scopeids": int(pt.X_array_partition_scopeids[nid].item()),
            "X_pipeline_scopeids": int(pt.X_pipeline_scopeids[nid].item()),
            "X_unroll_scopeids": int(pt.X_unroll_scopeids[nid].item()),
        }
        print(row)

print("\nArray-partition anchors only")
print("-" * 100)

array_slots = [s for s in active_slots if int(mem["slot_cats"][s]) == 2]

if not array_slots:
    print("No slot_cats==2 found for this kernel.")
    print("That means rodinia-knn-1-tiling may not actually contain ARRAY_PARTITION placeholders in your current graph/source mapping.")
else:
    ok = True
    for slot in array_slots:
        lid = mem["labels"][slot]
        nid = mem["node_ids"][slot]

        conds = {
            "llm_scope==1": bool(pt.X_llm_scopeids[nid].item() == 1),
            "llm_scopecat==2": bool(pt.X_llm_scopecat[nid].item() == 2),
            "arrayscopenid==1": bool(pt.X_arrayscopenids[nid].item() == 1),
            "pseudonid==0": bool(pt.X_pseudonids[nid].item() == 0),
        }

        print(f"L{lid} -> node {nid} -> {conds}")
        if not all(conds.values()):
            ok = False

    print("\nFINAL:", "PASS" if ok else "FAIL")
