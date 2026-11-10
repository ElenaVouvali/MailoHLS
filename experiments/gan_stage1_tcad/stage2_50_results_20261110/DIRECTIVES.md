# GAN: Stage2 selected checkpoint50 directives for the TCAD requests

PARETO_LATENCY; frozen V4 parent, Stage2 selected checkpoint50. Raw native-checkpoint decoder outputs are preserved.

**No synthesis has been launched.** Offline source/tool checks are separate from raw predictions and do not establish resource feasibility or timing.

Array cells: `type/factor/dimension`. Loop cells: `pipeline/unroll`. Factor0 denotes the checkpoint encoding for disabled unroll/partial partition or the complete-partition sentinel, as applicable.

|Device|Clock(ns)|Budget|Config|L1 xbuf|L2 layer_1_out|L3 layer_2_out|L4 P/U|L5 P/U|L6 P/U|L7 P/U|L8 P/U|L9 P/U|L10 P/U|L11 P/U|Tool check|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ZCU104|3.33|20%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|3.33|50%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|3.33|100%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|5.0|20%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|5.0|50%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|5.0|100%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|10.0|20%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|10.0|50%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|ZCU104|10.0|100%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|3.33|20%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|3.33|50%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|3.33|100%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|5.0|20%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|5.0|50%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|5.0|100%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|10.0|20%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|
|U200|10.0|50%|C2|complete/0/1|complete/0/1|complete/0/1|0/2|1/0|1/0|1/0|1/0|1/0|1/0|1/0|PASS|
|U200|10.0|100%|C1|block/2/1|cyclic/4/1|cyclic/2/1|0/2|1/0|0/16|1/0|0/4|1/0|1/0|0/16|PASS|

## Full directives for every request

### GAN:ZCU104:3.33ns:20pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 125, "DSP": 346, "FF": 92160, "LUT": 46080}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:3.33ns:50pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 312, "DSP": 864, "FF": 230400, "LUT": 115200}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:3.33ns:100pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 624, "DSP": 1728, "FF": 460800, "LUT": 230400}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:5ns:20pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 125, "DSP": 346, "FF": 92160, "LUT": 46080}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:5ns:50pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 312, "DSP": 864, "FF": 230400, "LUT": 115200}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:5ns:100pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 624, "DSP": 1728, "FF": 460800, "LUT": 230400}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:10ns:20pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 125, "DSP": 346, "FF": 92160, "LUT": 46080}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:10ns:50pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 312, "DSP": 864, "FF": 230400, "LUT": 115200}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:ZCU104:10ns:100pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 624, "DSP": 1728, "FF": 460800, "LUT": 230400}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:3.33ns:20pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 864, "DSP": 1368, "FF": 472896, "LUT": 236448}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:3.33ns:50pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 2160, "DSP": 3420, "FF": 1182240, "LUT": 591120}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:3.33ns:100pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 4320, "DSP": 6840, "FF": 2364480, "LUT": 1182240}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:5ns:20pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 864, "DSP": 1368, "FF": 472896, "LUT": 236448}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:5ns:50pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 2160, "DSP": 3420, "FF": 1182240, "LUT": 591120}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:5ns:100pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 4320, "DSP": 6840, "FF": 2364480, "LUT": 1182240}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:10ns:20pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 864, "DSP": 1368, "FF": 472896, "LUT": 236448}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

### GAN:U200:10ns:50pct — C2

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 2160, "DSP": 3420, "FF": 1182240, "LUT": 591120}`.

```text
<L1>
auto{_ARRAY_T_L1} = complete
auto{_ARRAY_F_L1} = 0
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = complete
auto{_ARRAY_F_L2} = 0
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = complete
auto{_ARRAY_F_L3} = 0
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 1
auto{_UNROLL_L6} = 0
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 1
auto{_UNROLL_L8} = 0
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 1
auto{_UNROLL_L11} = 0
```

### GAN:U200:10ns:100pct — C1

Absolute BRAM18K/DSP/FF/LUT limits: `{"BRAM_18K": 4320, "DSP": 6840, "FF": 2364480, "LUT": 1182240}`.

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 4
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = cyclic
auto{_ARRAY_F_L3} = 2
auto{_ARRAY_D_L3} = 1
<L4>
auto{_PIPE_L4} = 0
auto{_UNROLL_L4} = 2
<L5>
auto{_PIPE_L5} = 1
auto{_UNROLL_L5} = 0
<L6>
auto{_PIPE_L6} = 0
auto{_UNROLL_L6} = 16
<L7>
auto{_PIPE_L7} = 1
auto{_UNROLL_L7} = 0
<L8>
auto{_PIPE_L8} = 0
auto{_UNROLL_L8} = 4
<L9>
auto{_PIPE_L9} = 1
auto{_UNROLL_L9} = 0
<L10>
auto{_PIPE_L10} = 1
auto{_UNROLL_L10} = 0
<L11>
auto{_PIPE_L11} = 0
auto{_UNROLL_L11} = 16
```

## Interpretation

2 distinct complete assignments across18 requests; 18/18 pass the separate pinned source/tool check.

Directive differences are descriptive, not measured hardware gains. The historical paper heatmap latency/utilization values belong to its own checkpoints/receipts and are not assigned to this run.
