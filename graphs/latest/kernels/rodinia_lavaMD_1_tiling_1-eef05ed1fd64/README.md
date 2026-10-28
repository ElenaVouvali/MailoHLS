# rodinia_lavaMD_1_tiling_1-eef05ed1fd64

[Back to all graphs](../../README.md)

RC3 snapshot.

| Representation | Nodes | Edges | File |
| --- | ---: | ---: | --- |
| MLIR canonical | 545 | 2,577 | [canonical JSON.gz](canonical.json.gz) |
| MLIR + hardware canonical | 550 | 2,788 | Select a target below |
| MLIR compact | 480 | 2,512 | [compact JSON.gz](compact.json.gz) |
| MLIR + hardware compact | 485 | 2,723 | Select a target below |

Actions: **14/14 compiler-matched/registered**. Unresolved: none.

Review [sealed source, headers and metadata](../../sources/eef05ed1fd64122571d825c1a36bb6d0eac4bfa9650b3f4d2f5b9ba711e2bee8/), [raw MLIR](module.mlir), [exact action registrations](action-points.json), [hardware nodes and every added edge, including evidence](hardware-review.json). The hardware excerpt uses ZCU104, 10 ns, full capacity. Actual mappings remain unknown.

A small, exact neighborhood (all node IDs and edges are copied from the stored graphs):

```mermaid
flowchart LR
  n0["action:L10"]
  n1["hardware:fpga"]
  n2["hardware:resource:BRAM_18K"]
  n3["hardware:resource:DSP"]
  n4["memory:_Z20lavaMD_tiling_paddedPfS_S_:op16:r0"]
  n5["op:_Z20lavaMD_tiling_paddedPfS_S_:op114"]
  n6["op:_Z20lavaMD_tiling_paddedPfS_S_:op127"]
  n5 -->|"target:arithmetic_compatible (possible)"| n3
  n4 -->|"target:storage_compatible (possible)"| n2
  n0 -->|"target:action_resource_relevance (possible)"| n2
  n0 -->|"target:requested_clock (not_applicable)"| n1
  n0 -->|"targets (proven)"| n6
  n2 -->|"hardware:belongs_to (not_applicable)"| n1
  n3 -->|"hardware:belongs_to (not_applicable)"| n1
```

[Exact nodes, edges and evidence behind this diagram](graph-neighborhood.json). `proven` describes a compiler relationship; `possible` describes hardware compatibility. The clock edge is a requested constraint, not a mapping or feasibility result.

| Device | Clock (ns) | Full canonical graph | Full compact graph |
| --- | ---: | --- | --- |
| u200 | 3.33 | [JSON.gz](hardware/u200/3p33ns_full/canonical.json.gz) | [JSON.gz](hardware/u200/3p33ns_full/compact.json.gz) |
| u200 | 5 | [JSON.gz](hardware/u200/5ns_full/canonical.json.gz) | [JSON.gz](hardware/u200/5ns_full/compact.json.gz) |
| u200 | 6.67 | [JSON.gz](hardware/u200/6p67ns_full/canonical.json.gz) | [JSON.gz](hardware/u200/6p67ns_full/compact.json.gz) |
| u200 | 10 | [JSON.gz](hardware/u200/10ns_full/canonical.json.gz) | [JSON.gz](hardware/u200/10ns_full/compact.json.gz) |
| u50 | 3.33 | [JSON.gz](hardware/u50/3p33ns_full/canonical.json.gz) | [JSON.gz](hardware/u50/3p33ns_full/compact.json.gz) |
| u50 | 5 | [JSON.gz](hardware/u50/5ns_full/canonical.json.gz) | [JSON.gz](hardware/u50/5ns_full/compact.json.gz) |
| u50 | 6.67 | [JSON.gz](hardware/u50/6p67ns_full/canonical.json.gz) | [JSON.gz](hardware/u50/6p67ns_full/compact.json.gz) |
| u50 | 10 | [JSON.gz](hardware/u50/10ns_full/canonical.json.gz) | [JSON.gz](hardware/u50/10ns_full/compact.json.gz) |
| zcu104 | 3.33 | [JSON.gz](hardware/zcu104/3p33ns_full/canonical.json.gz) | [JSON.gz](hardware/zcu104/3p33ns_full/compact.json.gz) |
| zcu104 | 5 | [JSON.gz](hardware/zcu104/5ns_full/canonical.json.gz) | [JSON.gz](hardware/zcu104/5ns_full/compact.json.gz) |
| zcu104 | 6.67 | [JSON.gz](hardware/zcu104/6p67ns_full/canonical.json.gz) | [JSON.gz](hardware/zcu104/6p67ns_full/compact.json.gz) |
| zcu104 | 10 | [JSON.gz](hardware/zcu104/10ns_full/canonical.json.gz) | [JSON.gz](hardware/zcu104/10ns_full/compact.json.gz) |

All contexts use full physical resources; platform reservations are unknown. Resource compatibility is possible, never a proven synthesis binding. Compiler facts, action IDs and uncertainty are preserved. Independent human semantic adjudication remains pending.

Decompress with `gzip -dk canonical.json.gz`, or use the catalog unpack command. See the [catalog guide](../../GUIDE.md) for schemas and the evidence contract.
