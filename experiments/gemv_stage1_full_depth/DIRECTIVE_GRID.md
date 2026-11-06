# Stage-1 directives on the complete GEMV target grid

The run used the V4 multi-reference latency Stage-1 checkpoint, the canonical prompt builder, the production constrained scorer (legacy admissibility, mean RHS token log-probability), deterministic greedy decoding, and the exact source-action schema. Bold values differ from R1. These are predictions, not QoR results.

| Request | Device | Clock | Budget | L1 A partition (type,factor,dim) | L2 x (type,factor,dim) | L3 y (type,factor,dim) | L4 outer (II,unroll) | L5 inner (II,unroll) |
|---|---|---:|---:|---|---|---|---|---|
| GRID_ZCU104_10P0NS_100PCT | ZCU104 | 10 ns | 100% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_ZCU104_10P0NS_50PCT | ZCU104 | 10 ns | 50% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_ZCU104_10P0NS_25PCT | ZCU104 | 10 ns | 25% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_ZCU104_5P0NS_100PCT | ZCU104 | 5 ns | 100% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_ZCU104_5P0NS_50PCT | ZCU104 | 5 ns | 50% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_ZCU104_5P0NS_25PCT | ZCU104 | 5 ns | 25% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_ZCU104_3P33NS_100PCT | ZCU104 | 3.33 ns | 100% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_ZCU104_3P33NS_50PCT | ZCU104 | 3.33 ns | 50% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (**1**, **0**) |
| GRID_ZCU104_3P33NS_25PCT | ZCU104 | 3.33 ns | 25% | (block, 2, 1) | (cyclic, **4**, 1) | (**complete**, **0**, 1) | (0, 2) | (0, **4**) |
| GRID_XCU200_10P0NS_100PCT | XCU200 | 10 ns | 100% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_XCU200_10P0NS_50PCT | XCU200 | 10 ns | 50% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (**1**, **0**) |
| GRID_XCU200_10P0NS_25PCT | XCU200 | 10 ns | 25% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_XCU200_5P0NS_100PCT | XCU200 | 5 ns | 100% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_XCU200_5P0NS_50PCT | XCU200 | 5 ns | 50% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_XCU200_5P0NS_25PCT | XCU200 | 5 ns | 25% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_XCU200_3P33NS_100PCT | XCU200 | 3.33 ns | 100% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_XCU200_3P33NS_50PCT | XCU200 | 3.33 ns | 50% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |
| GRID_XCU200_3P33NS_25PCT | XCU200 | 3.33 ns | 25% | (block, 2, 1) | (cyclic, 2, 1) | (cyclic, 4, 1) | (0, 2) | (0, 2) |

Historical diagnostic: `HIST_ZCU104_10NS_20PCT` used the prior frozen 20% request and is retained separately; its output is in `HIST_ZCU104_10NS_20PCT_official.json`. This 20% budget lies below the V4 training menu minimum of 25%.

All 43 complete assignments passed both the legacy source assignment check and the current V2 source/tool legality audit. No synthesis was run. The historical GEMV source has uninitialized local arrays and exposes only `y[0]`; it is retained for comparability and is not a functionally valid numerical QoR benchmark.
