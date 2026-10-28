# rodinia_lavaMD_0_baseline-37d49978e024

[Back to all graphs](../../README.md)

RC3 snapshot.

| Representation | Nodes | Edges | File |
| --- | ---: | ---: | --- |
| MLIR canonical | 415 | 2,562 | [canonical JSON.gz](canonical.json.gz) |
| MLIR + hardware canonical | 420 | 2,700 | Select a target below |
| MLIR compact | 375 | 2,522 | [compact JSON.gz](compact.json.gz) |
| MLIR + hardware compact | 380 | 2,660 | Select a target below |

Actions: **5/5 compiler-matched/registered**. Unresolved: none.

Review [sealed source, headers and metadata](../../sources/37d49978e02443f6532874c46772b3a96ee7ccb51ac6b66a5f69a40f4d9288a8/), [raw MLIR](module.mlir), [exact action registrations](action-points.json), [hardware nodes and every added edge, including evidence](hardware-review.json). The hardware excerpt uses ZCU104, 10 ns, full capacity. Actual mappings remain unknown.

A small, exact neighborhood (all node IDs and edges are copied from the stored graphs):

```mermaid
flowchart LR
  n0["action:L2"]
  n1["action:L3"]
  n2["hardware:fpga"]
  n3["hardware:resource:BRAM_18K"]
  n4["hardware:resource:DSP"]
  n5["memory:_Z22lavaMD_baseline_paddedPfS_S_:op14:r0"]
  n6["op:_Z22lavaMD_baseline_paddedPfS_S_:op102"]
  n7["op:_Z22lavaMD_baseline_paddedPfS_S_:op96"]
  n6 -->|"target:arithmetic_compatible (possible)"| n4
  n5 -->|"target:storage_compatible (possible)"| n3
  n0 -->|"target:action_resource_relevance (possible)"| n3
  n1 -->|"target:requested_clock (not_applicable)"| n2
  n0 -->|"targets (proven)"| n5
  n1 -->|"targets (proven)"| n7
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
