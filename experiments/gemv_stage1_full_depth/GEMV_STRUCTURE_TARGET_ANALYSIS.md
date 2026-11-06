# GEMV compiler structure × target analysis

## RC3 authority and verified structural contrast

The original input bytes and action metadata match the read-only GNN branch’s frozen RC3 source manifest. That canonical graph records 66 nodes, 106 edges, and five matched actions; the original MLIR and graph are copied byte-for-byte into `inputs/` with the upstream manifest and checksums. The GNN authority is commit `172e05a69650a666c323955ea78cb501fd9364c2`.

The original source provides `A[32][32]`, `x[32]`, `y[32]`, accesses `A[i][j]` and `x[j]`, and nested 32-iteration loops. RC3 exposes affine indices, loop nesting, action-to-operation/memory relationships, and the scalarized compiler representation of `y`. Its L5 inner loop contains `%arg2` as `iter_args`, multiplication, addition and `affine.yield`; this is compiler evidence for a scalar carried recurrence, not a memory-independence statement.

The new valid diagnostic program uses initialized function inputs copied into local arrays and makes every partial/final result observable. `reduce32` is a scalar reduction. `independent32` writes each product independently and sets `y[i]=0`; it intentionally changes the algorithm and outputs. `reduce32_rename` consistently renames variables and retains the same recurrence. Each compiled with the authenticated `cgeist`/MLIR V2.2 RC3 pipeline and retained L1–L5. Deterministic numerical checks reported maximum absolute errors of 1.22e-4 for the reduction and rename control and 0 for the independent-product oracle; the reduction/rename check is within its float32 tolerance.

| Program | Source SHA-256 | Graph SHA-256 | RC3 scalar recurrence | Meaning relation |
|---|---|---|---|---|
| reduce32 | `0bf12183719b41fec7b50dd458848e991964454a89467577616f4460480caece` | `b6335e17be35bc028013b76b3f5cae49544fe6d3a635fefed503a69ede785954` | yes | initialized reduction |
| independent32 | `004692648f0ed934e1cb44d96118d49af9d207ff292782c6156a72232bae04ea` | `72d202b26b18b5883bdc1fa9749d165d4861a88718cf976e2b5646a2ad326f5b` | no explicit scalar carried result | different per-iteration program |
| reduce32_rename | `517281ed8f7082858dc9cf3e4df0f10b39f105a28195d36ad52f94778d9c2bde` | `31408f6364dd53c531bff0aa1dffe91cd6cfc54b43f2ad9ad1cadab536f621b7` | yes | rename control |

## Target and structure interaction

All three diagnostic sources were scored under eight matched contexts spanning both devices, 10/3.33 ns, and 100/25% requested budgets. Requests, prompts, source hashes, graph hashes, full legal candidate scores, and all 32-layer states are retained. Controlled comparisons use the original R1 assignment as the common preceding directive context. The official natural decoder outputs are separate and were not replaced by counterfactual argmaxes.

Across controlled positions, the independent-vs-reduction median relative vector movement was 0.0117; rename-vs-reduction movement was 0.0545. At L5 unroll these medians were 0.0676 and 0.1038, respectively. A source edit that changes computation therefore moves the states, but the meaning-preserving rename also moves them strongly. This is not recurrence-decoding evidence.

At ZCU104/10 ns/100% under the R1 controlled prefix, the L5 unroll scores rank `4` above `2` for reduction (margin 0.0609), and rank `2` above `4` for independent iterations (margin 0.0194). The actual natural top directive is the same (`UNROLL_L5=0`) for both. The rename control retains `UNROLL_L5=4` as the controlled argmax but flips `PIPE_L5` to a near-tie preference for 0; reduction prefers 1. L1 dimension ranks 1 for reduction and independent, but the rename source picks dimension 2. These observations motivate a syntax-robust compiler-memory ablation; they do not establish an optimal partition or loop pragma.

## Action-specific interpretation

- **L1 A:** C exposes array dimensions and `A[i][j]`; RC3 binds the memory operations and ordered indices to L4/L5. Across the original hardware grid L1 stays block/2/dim1. The rename diagnostic changes the L1 dimension preference despite preserving the recurrence program. Correct access-direction reasoning is not established.
- **L2 x:** C exposes `x[j]`; RC3 binds it to the inner-loop induction. Factor is stable on the original direct one-factor target contrasts, changing only in a combined tight-clock/tight-budget cell; no port or QoR inference follows.
- **L3 y:** Source says `y[32]` and returns `y[0]`; RC3 records scalarized compiler storage. L3 changes under combined constraints and lexical rename; no output-resource benefit is measured.
- **L4 outer:** RC3 verifies trip count 32, depth 0 and child loop L5. The original grid retains PIPE0/UNROLL2. Source contrasts change some conditional scores but not the selected value; useful structural use is unresolved.
- **L5 inner:** RC3 verifies the scalar SSA recurrence in the reduction source. Candidate preferences depend on the source variant under a fixed prefix, but the natural directive remains the same for reduction and independent. The controlled difference is confounded by different computation/output and token sequence; a syntax-only rename can also move preferences.

The structure×target score difference-in-differences for the fixed candidate pairs is small (median absolute margin interaction by target factor: device 0.0078, clock 0.0078, budget 0.0078). This supports only weak, small-sample structure×target score interactions under the tested alternatives. The model outputs no “correct” target-optimal label because QoR was not measured.
