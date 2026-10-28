# spcl_example_03-fdea84258953

[Back to all graphs](../../README.md)

RC3 snapshot.

| Representation | Nodes | Edges | File |
| --- | ---: | ---: | --- |
| MLIR canonical | 95 | 162 | [canonical JSON.gz](canonical.json.gz) |
| MLIR + hardware canonical | 100 | 216 | Select a target below |
| MLIR compact | 71 | 138 | [compact JSON.gz](compact.json.gz) |
| MLIR + hardware compact | 76 | 192 | Select a target below |

Actions: **6/6 compiler-matched/registered**. Unresolved: none.

Review [sealed source, headers and metadata](../../sources/fdea84258953d00285a390dd854a09bf1d8f88be911b360eaac1b6aed8f95650/), [raw MLIR](module.mlir), [exact action registrations](action-points.json), [hardware nodes and every added edge, including evidence](hardware-review.json). The hardware excerpt uses ZCU104, 10 ns, full capacity. Actual mappings remain unknown.

A small, exact neighborhood (all node IDs and edges are copied from the stored graphs):

```mermaid
flowchart LR
  n0["action:L1"]
  n1["action:L3"]
  n2["hardware:fpga"]
  n3["hardware:resource:BRAM_18K"]
  n4["hardware:resource:DSP"]
  n5["memory:Stencil2D:op1:r0"]
  n6["memory:Stencil2D:op2:r0"]
  n7["op:Stencil2D:op16"]
  n8["op:Stencil2D:op3"]
  n7 -->|"target:arithmetic_compatible (possible)"| n4
  n5 -->|"target:storage_compatible (possible)"| n3
  n0 -->|"target:action_resource_relevance (possible)"| n3
  n1 -->|"target:requested_clock (not_applicable)"| n2
  n0 -->|"targets (proven)"| n6
  n1 -->|"targets (proven)"| n8
  n3 -->|"hardware:belongs_to (not_applicable)"| n2
  n4 -->|"hardware:belongs_to (not_applicable)"| n2
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
