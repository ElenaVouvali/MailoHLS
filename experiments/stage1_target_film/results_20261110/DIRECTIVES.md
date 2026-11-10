# Stage-1 target adaptation: GAN 18-request inference

**Complete inference; no synthesis.** Objective PARETO_LATENCY. These outputs do not establish feasible resources, timing, or measured QoR.

|Model|Completed|Distinct assignments|
|---|---:|---:|
|A_checkpoint32|18/18|4|
|B_checkpoint48|18/18|5|

|Request|A config SHA-256 prefix|B config SHA-256 prefix|Changed fields|
|---|---|---|---:|
|U200:10ns:100pct|fc75258d2098|fc75258d2098|0/25|
|U200:10ns:20pct|fc75258d2098|ad4bae730c48|4/25|
|U200:10ns:50pct|730f6bd66455|730f6bd66455|0/25|
|U200:3.33ns:100pct|fc75258d2098|fc75258d2098|0/25|
|U200:3.33ns:20pct|fc75258d2098|fc75258d2098|0/25|
|U200:3.33ns:50pct|fc75258d2098|fc75258d2098|0/25|
|U200:5ns:100pct|fc75258d2098|fc75258d2098|0/25|
|U200:5ns:20pct|fc75258d2098|ad4bae730c48|4/25|
|U200:5ns:50pct|fc75258d2098|edffd8bddf12|3/25|
|ZCU104:10ns:100pct|fc75258d2098|fc75258d2098|0/25|
|ZCU104:10ns:20pct|fc75258d2098|fc75258d2098|0/25|
|ZCU104:10ns:50pct|fc75258d2098|d1a2e04e20ec|1/25|
|ZCU104:3.33ns:100pct|fc75258d2098|d1a2e04e20ec|1/25|
|ZCU104:3.33ns:20pct|fc75258d2098|d1a2e04e20ec|1/25|
|ZCU104:3.33ns:50pct|3f04b3335840|d1a2e04e20ec|3/25|
|ZCU104:5ns:100pct|d1a2e04e20ec|d1a2e04e20ec|0/25|
|ZCU104:5ns:20pct|fc75258d2098|d1a2e04e20ec|1/25|
|ZCU104:5ns:50pct|fc75258d2098|d1a2e04e20ec|1/25|

## Configuration fc75258d2098

SHA-256: `fc75258d209883bdb38f71677b27459ac48551b1f6330e234d088387b8a46800`

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

## Configuration 3f04b3335840

SHA-256: `3f04b333584050d2fb1400e7a79d9de7d9d537825ae9cd86ef3072d924f265a6`

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
auto{_ARRAY_T_L3} = block
auto{_ARRAY_F_L3} = 4
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

## Configuration d1a2e04e20ec

SHA-256: `d1a2e04e20ec866e703efd90d99f689ce6b42a8b03276c57e9a6123aa741bcce`

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 2
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = block
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

## Configuration 730f6bd66455

SHA-256: `730f6bd66455cf894311039082a8b6cdefe86b6894ba23473311df2d1d5d09e7`

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

## Configuration ad4bae730c48

SHA-256: `ad4bae730c48198769f7692c4112a1d9bc32e1b41c9376ccd4279eeb1621fb7a`

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 4
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 2
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = block
auto{_ARRAY_F_L3} = 8
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

## Configuration edffd8bddf12

SHA-256: `edffd8bddf1233ad734e8d0739ab4ee1794638ab54026d48fbef9ed98c3ea1bf`

```text
<L1>
auto{_ARRAY_T_L1} = block
auto{_ARRAY_F_L1} = 4
auto{_ARRAY_D_L1} = 1
<L2>
auto{_ARRAY_T_L2} = cyclic
auto{_ARRAY_F_L2} = 2
auto{_ARRAY_D_L2} = 1
<L3>
auto{_ARRAY_T_L3} = block
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

## Comparison

A and B differ on 9/18 requests; mean changed fields 1.06/25, maximum 4/25. Conditional directive fields are included in this distance. No hardware quality is inferred.
