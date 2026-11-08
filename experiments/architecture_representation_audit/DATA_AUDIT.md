# Data audit — authenticated RC3 multi-family cohort

## Status

**Completed on CPU.** `audit_cohort.py` authenticates all canonical graph members against the RC3 release manifest and revalidates the measured subset against the frozen V4b `data-audit.json`. It writes the row-level provenance and label availability in [`results/cohort_manifest.json`](results/cohort_manifest.json). The frozen encoder run is recorded in [`results/gnn_extraction_manifest.json`](results/gnn_extraction_manifest.json).

| Cohort property | Result |
|---|---:|
| RC3 canonical graph snapshots authenticated | 63 |
| Exact graph/source matches in measured V4b cohort | 55 |
| Release snapshots without measured exact-source match | 8 |
| Family split | 18 train / 3 validation / 1 test |
| Measured graph variants | 50 train / 4 validation / 1 test |
| Nodes, min / median / max across 63 | 47 / 355 / 4,204 |
| Edges, min / median / max across 63 | 76 / 700 / 53,788 |
| Action counts among matched variants | 1–46; 19 distinct counts |
| Compiler loops / memory objects among matched variants | 1–40 / 3–61 |

Family boundaries are inherited from V4b and are not recomputed from variant names. The sole frozen test family is `serrano-kalman-filter`; test metrics based on it cannot support a broad family-generalization claim. The Stage-1 full-depth GEMV study has no compatible cross-kernel family split, so it is kept separate and not joined as though its observations were independent cohort examples.

## Compiler-label availability

Counts below cover the 55 matched graph/source snapshots, not independent families.

| Property | Available evidence / count | Label policy |
|---|---:|---|
| Loop actions | 339 | Action mapping is preserved; unknown action matches remain distinct |
| Static trip count | 331 loop actions have compiler-proven count and provenance | Numerical value with known mask; missing is unknown |
| Proven scalar loop-carried SSA witness | 61 of 333 loop actions with a mapped operation target are positive | Positive only with explicit proven edge on the target loop |
| Mapped loop actions without this proven witness | 272 | Unknown; not negative and includes lack of established witness |
| Loop actions without a mapped operation target | 6 of 339 | Unresolved target; no recurrence label |
| Proven dependence edges | 1,447 | Only explicit `depends_on` with `evidence_status=proven` |
| Possible dependence edges | 8,131 | Retained as possible, never promoted to proven |
| `not_established` dependence edges | 0 explicit edges | This is not proof of independence or complete alias analysis |
| Memory-access edges / operations with affine map | 2,205 / 1,949 | Read/write role and affine fields retained as available/known |

Array shape must be read from memory-object or action-source metadata, and the two sources are not collapsed into one assumed truth. Likewise, explicit scalar SSA recurrence is not a memory dependence label.

## Identity and reproducibility

The archive hash, per-member checksums, graph identities, source/metadata/MLIR hashes, semantic revision, frozen split digest, and frozen data-audit digest are in the machine-readable manifest. The archive SHA-256 is `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`; the family-split SHA-256 is `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`.

No program family was split across train/validation/test. No labels were derived from model predictions. The 63 release snapshots are the graph-coverage population; the 55 exact-source measured subset is the cohort for action probes and frozen encoder measurements.
