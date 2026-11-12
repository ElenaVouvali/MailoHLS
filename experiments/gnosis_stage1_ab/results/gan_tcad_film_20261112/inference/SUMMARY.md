# GAN: selected Stage-1 FiLM checkpoint, 18 TCAD requests

All 18 requests completed on Liono V100S with arm B step 600, FiLM ON.

Objective: `PARETO_LATENCY`. No optimizer update or synthesis was performed.

Arrays L1–L3 show `type/factor/dimension`; loops L4–L11 show `pipeline/unroll`.
Array names: L1=`xbuf`, L2=`layer_1_out`, L3=`layer_2_out`.
Values use the checkpoint encoding: unroll 0 means disabled; complete partition uses factor 0.

|Device|Clock ns|Budget %|Config|L1|L2|L3|L4|L5|L6|L7|L8|L9|L10|L11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ZCU104|3.33|20|C1|complete/0/1|block/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|1/0|1/0|1/0|0/16|
|ZCU104|3.33|50|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|ZCU104|3.33|100|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|ZCU104|5.0|20|C3|complete/0/1|block/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|ZCU104|5.0|50|C3|complete/0/1|block/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|ZCU104|5.0|100|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|ZCU104|10.0|20|C3|complete/0/1|block/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|ZCU104|10.0|50|C4|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|1/0|1/0|1/0|0/16|
|ZCU104|10.0|100|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|3.33|20|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|3.33|50|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|3.33|100|C5|block/4/1|cyclic/4/1|complete/0/1|0/4|1/0|0/16|1/0|0/16|1/0|1/0|0/16|
|U200|5.0|20|C3|complete/0/1|block/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|5.0|50|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|5.0|100|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|10.0|20|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|10.0|50|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|
|U200|10.0|100|C2|complete/0/1|block/4/1|cyclic/4/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|

5 distinct complete assignments across 18 requests.

These are raw constrained-decoder predictions, checked against the sequential checkpoint proposal domains.
Resource feasibility, achieved clock, and hardware latency remain unmeasured.

Full directive text for every request: [DIRECTIVES.md](DIRECTIVES.md).
Every individual field: [directives.csv](directives.csv).
Candidate scores: `request_01.json` through `request_18.json`.

## Checkpoint identity

- Parent SHA256: `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`
- Residual LoRA SHA256: `6628fe1f268f93ce760428ca0c50b46dbcbfe55e00d45fba97a20142e89b2f80`
- FiLM SHA256: `d2697977c97c467c7f433b134dff3031c4e842f3b8f2cf9f0c57613c1ecf46ae`
