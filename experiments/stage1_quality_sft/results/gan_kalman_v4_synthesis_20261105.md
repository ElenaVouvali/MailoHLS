# Stage-1 v4 GAN and Kalman synthesis results

R1: ZCU104 / 10 ns / 100%; R2: XCU200 / 10 ns / 25%; R3: ZCU104 / 3.33 ns / 100%.
Resource order: BRAM_18K / DSP / FF / LUT. HLS clock is an estimate, not post-route timing. Failed synthesis has unknown QoR.

| Kernel | Arm | Request | Status | Latency cycles | Resources | Budget feasible | HLS clock ns |
|---|---|---|---|---:|---|---|---:|
| Kalman | single_reference | R1 | failed | unknown | unknown | unknown | unknown |
| Kalman | single_reference | R2 | failed | unknown | unknown | unknown | unknown |
| Kalman | single_reference | R3 | failed | unknown | unknown | unknown | unknown |
| Kalman | multi_reference | R1 | failed | unknown | unknown | unknown | unknown |
| Kalman | multi_reference | R2 | failed | unknown | unknown | unknown | unknown |
| Kalman | multi_reference | R3 | failed | unknown | unknown | unknown | unknown |
| Gan | single_reference | R1 | success | 5604 | 83 / 52 / 3303 / 9357 | True | 6.598 |
| Gan | single_reference | R2 | success | 5572 | 83 / 54 / 3721 / 11314 | True | 6.688 |
| Gan | single_reference | R3 | failed | unknown | unknown | unknown | unknown |
| Gan | multi_reference | R1 | success | 8156 | 85 / 54 / 3593 / 10280 | True | 7.095 |
| Gan | multi_reference | R2 | success | 1281 | 81 / 80 / 8674 / 12556 | True | 6.688 |
| Gan | multi_reference | R3 | success | 8947 | 85 / 54 / 4974 / 10442 | True | 2.404 |

## Failure evidence

- kalman_single_reference_r1: ERROR: [XFORM 203-103] Array 'out_local.V' (kernel.cpp:46): partitioned elements number (262144) has exeeded the threshold (1024), which may cause long run-time.
- kalman_single_reference_r2: ERROR: [XFORM 203-103] Array 'in_local.V' (kernel.cpp:44): partitioned elements number (262144) has exeeded the threshold (1024), which may cause long run-time.
- kalman_single_reference_r3: ERROR: [XFORM 203-103] Array 'in_local.V' (kernel.cpp:44): partitioned elements number (262144) has exeeded the threshold (1024), which may cause long run-time.
- kalman_multi_reference_r1: ERROR: [XFORM 203-103] Array 'in_local.V' (kernel.cpp:44): partitioned elements number (262144) has exeeded the threshold (1024), which may cause long run-time.
- kalman_multi_reference_r2: ERROR: [XFORM 203-103] Array 'in_local.V' (kernel.cpp:44): partitioned elements number (262144) has exeeded the threshold (1024), which may cause long run-time.
- kalman_multi_reference_r3: ERROR: [XFORM 203-103] Array 'in_local.V' (kernel.cpp:44): partitioned elements number (262144) has exeeded the threshold (1024), which may cause long run-time.
- gan_single_reference_r3: ERROR: [HLS 200-908] Cannot partition array 'layer_2_out.V' (network.cpp:32): incorrect partition factor 64.

## Interpretation

- GAN single-reference R1 and R2 and multi-reference R1–R3 synthesized successfully. Every successful result met the requested resource budget and the requested clock according to the HLS estimate.
- GAN multi-reference used 45.54% more cycles than single-reference at R1 and 77.01% fewer at R2. At R3 the single-reference configuration was rejected, so there is no paired latency comparison.
- All six Kalman attempts were rejected by Vitis array-partition threshold checks. Their latency, resource feasibility and QoR are unknown.
- GAN single-reference R3 was rejected because its L3 array partition factor 64 exceeds the source extent 50. The prediction is preserved unchanged; no repair-and-retry was performed.
- These are C synthesis estimates only. No C simulation, RTL co-simulation, implementation or post-route timing was run. They do not establish model superiority or generalization.

## Provenance

GAN inference used the production constrained greedy decoder on NVIDIA A2; Kalman inference used V100S. No Kalman test measurement labels were loaded. Checkpoint, tokenizer, registry, request and source hashes are recorded in `gan_v4_20261105_provenance.json` and the historical Kalman result manifest.

[GAN predictions](gan_v4_20261105.md) · [Kalman predictions](kalman_v4_20261104.md) · [Machine-readable 12-attempt report](gan_kalman_v4_synthesis_20261105.json)
