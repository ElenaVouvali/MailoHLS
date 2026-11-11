# GAN: Stage1 / Stage2 / Stage3 request separation

**Inference only; synthesis is stopped pending user inspection.**

Stage1 selected = final315. Stage2-50 and Stage2-150 are corrected-supervision checkpoints. Stage3 is DPO34 trained from Stage2-50, not Stage2-150.

A distance of n/25 counts changed raw directive fields. Conditional/forced fields are not independent decisions. Same D-ID means the complete25-field assignment is identical, across models as well as requests.

|Request|Stage1 final315|Stage2 selected50|Stage2 final150|
|---|---|---|---|
|GAN:ZCU104:3.33ns:20pct|D1|D1|D1|
|GAN:ZCU104:3.33ns:50pct|D1|D1|D1|
|GAN:ZCU104:3.33ns:100pct|D1|D1|D1|
|GAN:ZCU104:5ns:20pct|D1|D1|D1|
|GAN:ZCU104:5ns:50pct|D1|D1|D1|
|GAN:ZCU104:5ns:100pct|D1|D1|D1|
|GAN:ZCU104:10ns:20pct|D1|D1|D1|
|GAN:ZCU104:10ns:50pct|D1|D1|D1|
|GAN:ZCU104:10ns:100pct|D1|D1|D1|
|GAN:U200:3.33ns:20pct|D1|D1|D1|
|GAN:U200:3.33ns:50pct|D1|D1|D1|
|GAN:U200:3.33ns:100pct|D1|D1|D1|
|GAN:U200:5ns:20pct|D1|D1|D1|
|GAN:U200:5ns:50pct|D1|D1|D1|
|GAN:U200:5ns:100pct|D1|D1|D1|
|GAN:U200:10ns:20pct|D1|D1|D1|
|GAN:U200:10ns:50pct|D2|D2|D1|
|GAN:U200:10ns:100pct|D1|D1|D1|

## Separation within each checkpoint

|Model|Unique assignments|Identical pairs /153|Different pairs /153|Mean field distance|Max changed fields|Requests changed vsStage1|
|---|---:|---:|---:|---:|---:|---:|
|Stage1 final315|2|136|17|5.33%|12/25|0/18|
|Stage2 selected50|2|136|17|5.33%|12/25|0/18|
|Stage2 final150|1|153|0|0.00%|0/25|1/18|

## Every distinct full assignment

### D1

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

### D2

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

## Limits

These are one-program descriptive results, not independent generalization samples. Target-dependent directives or score changes do not establish improved measured latency, resource feasibility or timing. All models share the same source, requests, candidate policy and frozen Stage1 lineage. Memory C is target-independent; AP arithmetic in its compiler graph is analysis-only, not bit-true hardware evidence.
