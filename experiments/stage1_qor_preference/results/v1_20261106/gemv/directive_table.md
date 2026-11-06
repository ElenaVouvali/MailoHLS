| Action | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| L1 — A partition | block, 2, 1 | **block, 2, 2** | block, 2, 1 | block, 2, 1 |
| L2 — x partition | cyclic, 2, 1 | **cyclic, 4, 1** | cyclic, 2, 1 | cyclic, 2, 1 |
| L3 — y partition | complete, 0, 1 | complete, 0, 1 | complete, 0, 1 | complete, 0, 1 |
| L4 — outer (PIPE, UNROLL) | 0, 2 | 0, 2 | 0, 2 | 0, 2 |
| L5 — inner (PIPE, UNROLL) | 0, 2 | **1, 0** | 0, 2 | 0, 2 |
