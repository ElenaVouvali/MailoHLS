# SALT-HLS architecture findings (intermediate research report)

## Executive conclusion

The strongest current evidence points first to the **compiler-evidence-to-Memory-C boundary**, not a need to deepen the GNN or choose fusion layers. The RC3 corpus carries explicit loops, trip counts, access roles/maps, scalar recurrence witnesses, dependence statuses, and action-target provenance. The frozen two-layer V4b model has direct action-demand/semantic input paths and responds to controlled zeroing of compiler feature groups. But the existing C representation cannot be built for 37/55 measured graphs at cap 128, and action-route reconciliation fails on 12/55; exact test-family action routing is unavailable. Until these coverage contracts are reliable, architecture-level comparisons are confounded by missing or non-routable memory.

The Stage-1 full-depth GEMV work establishes that hardware prompts can change candidate scores and that selected hidden-state patches can causally alter rankings in some sampled fields. It does not establish cross-kernel accessibility of recurrence/access/dependence facts or identify a fusion layer. No trained Stage-2 C checkpoint was available for causal evidence-use analysis. Thus the central question—what compiler evidence adds beyond Stage-1 and where it should enter—remains unresolved experimentally.

## Finding / evidence / implication table

| Finding | Evidence level | Confidence | Architectural implication | Next experiment |
|---|---|---|---|---|
| RC3 compiler facts are explicitly available but unknown is common and meaningful | Compiler input/graph: 331/339 static trip counts; 61 proven recurrence positives; 1,447 proven vs 8,131 possible dependence edges | High for recorded graph values; independent raw-MLIR reparse still pending | Preserve status, provenance, and unknown masks; do not synthesize negative facts | Stratified source/MLIR revalidation |
| V4b action output depends on direct summaries and compiler tensor groups | Controlled in-memory interventions: all 534 action vectors change under direct action summary, trip/depth, edge-feature masking; shape group changes 481 | High that forward outputs respond; low about property meaning/use | Probe action outputs per property before modifying depth | Family-held-out action-aligned linear probes and matched irrelevant interventions |
| Existing Memory-C coverage is insufficient for broad cohort | Input/build execution: 37/55 evidence overflows (131–877); 12/55 relation builder action-label errors | High | Fix typed evidence capacity/selection and action mapping first, without silent truncation | RC3 graph-specific tests, full 55 graph rebuild, assert action ID/provenance invariants |
| Selected evidence may be beyond the message-passing radius | Topology: selected node distances 1–8, median 4; encoder has two layers; direct summaries bypass | High for path lengths, low for causal impact | Do not infer over-squashing; test witness-specific influence | Matched 1/2/>2-hop witness masking with direct summaries controlled |
| Stage-1 contains hardware-conditioned decision information in GEMV | Existing GEMV causal report: actual score changes and 4/189 patch preference flips, placebo zero | Moderate for GEMV; low generalization | Preserve target prompt in LLM; avoid claiming any best layer | Cross-kernel exact-position state/probe study when GPU available |
| Cross-attention implementation has a plausible hardware-conditioned query to static compiler K/V path | Static code and CPU mask/interface checks | High for path correctness, none for learned utility | Inspect K/Q and intervention results before architecture changes | Completed-checkpoint authentic/evidence-zero/permutation causal instrumentation |
| No fusion layer can yet be justified | Missing cross-kernel structural probes and trained C evidence causality | High | Keep layer selection unresolved | Single-layer matched pilots only after analytical shortlist |

## Where information can be lost or blocked

```text
Compiler analysis → RC3 graph → tensors → GNN → action/evidence memory → cross-attention → directive state
       facts           explicit       partial      unknown      1-vector + C      mask/routing      no trained C
       + status         + sidecar      known masks  per-property  cap/routes fail   consumes vectors  causal result
```

**Case A — Stage-1 representation:** hardware and some score sensitivity are demonstrated for GEMV. Structural property accessibility across kernels is not yet measured, so compiler complementarity is not quantified.

**Case B — tensorization:** raw graph/sidecar carries more detailed affine maps and witness provenance than the compact model inputs. Exact feature coverage is documented, but no evidence indicates a systematic schema bug in the numeric known masks. `trip_count_static==1` is a knownness flag; the value is separate.

**Case C — GNN encoding:** it responds to controlled input ablations, but no family-generalized property probe has established what survives. Depth is not the leading diagnosed issue.

**Case D — Memory C:** current coverage and routing fail for a large corpus fraction; metadata fields do not automatically enter vectors consumed by attention. This is the clearest actionable weakness.

**Case E — fusion:** implementation path is plausible and masked; learned C use is unmeasured. Do not modify Q/K/gate/layer placement without a completed checkpoint and evidence interventions.

**Case F — HLS outcome:** neither this audit nor the GEMV embedding observations establish physical feasibility or latency improvement from memory. Do not claim synthesis benefit.

## Prioritized roadmap

1. Correct and test the action-point reconciliation error and rebuild the complete RC3 evidence manifest; keep absolute slots, fail-closed mapping, source hashes, and evidence status.
2. Design versioned evidence selection/capacity based on measured type coverage. Avoid dropping evidence silently; evaluate hierarchical or segmented keys against an explicit overflow policy.
3. Run property-specific, family-disjoint probes on R0–R5 and Memory C, using positive-only/unknown-aware labels and family-level controls. Expand or redesign validation only in a new frozen split, never by moving the current test family after seeing results.
4. Complete cross-kernel Stage-1 exact decision-position traces and controlled probes when a GPU is available; do not rerun/occupy the active Stage-2 GPU.
5. Instrument an authenticated completed C checkpoint for evidence-zero and evidence-permutation causal effects, then shortlist fusion layers from complementary information plus causal score evidence.
6. Compare current two-layer V4b and historical four-layer + Jumping Knowledge only as a matched future encoder experiment after current representation diagnostics identify a specific depth/reachability weakness. Keep capacity-matched controls and the same RC3 split/budget.

No production code, model weights, or active Stage-2 work were modified. No training or synthesis experiment was executed.
