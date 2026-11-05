# GNN research branch artifact disposition

Reviewed on 5 November 2026 at `aa6ad1a`. This report narrows the active
research entry points without deleting research evidence. No deletion is
proposed: there is no file for which replacement, dependency closure, and
reproducibility are all demonstrated strongly enough to justify removal.

| Classification | Paths / evidence | Disposition |
|---|---|---|
| `ACTIVE` | `GNN_branch/mlir_graph_v2.py`, `mlir_semantics_v22.py`, `target_graph_v22.py`, `target_graph_tensor_v22.py`, `semantic_tensor_features_v22.py`, `action_demand_v22.py`; `experiments/v22_gnn_training/v4b/*.py` | Current graph, tensor and V4b experiment implementation. Keep. |
| `SELECTED_ARTIFACT` | `GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz` and its release manifest; `experiments/v22_gnn_training/v4b/receipts_v3/`; `paper_figures/gemv_rc3_20261105/manifest.json`, validation and canonical graph | Frozen RC3 remains unchanged (archive SHA-256 `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`). GEMV is a separate, out-of-cohort demonstration. Keep both. |
| `HISTORICAL_EVIDENCE` | V1 graph artifacts, V4/V4b prior receipts including `receipts_v2/`, older `RESULTS.md`/audit sections consolidated into root README, failed/invalid attempts recorded in receipts | Distinct production contract or historical evidence. In particular, V2 receipts establish the later zero-ADP correction; do not delete or rewrite them. |
| `REGENERABLE` | `gemv.rc3.overview.{dot,svg,png,pdf}`, `gemv.rc3.full.{dot,svg,png,pdf}`, `gemv_v4b_encoder.{dot,svg,png}`, and embedding heatmaps | Render outputs can be regenerated from committed DOT/CSV/NPZ plus documented renderer; formats serve different viewing and publication uses, so none is removed. Canonical graph and measured embedding arrays are not merely display files. |
| `REDUNDANT_VERIFIED` | No tracked files approved for removal. | Similar names and alternate formats do not prove semantic redundancy. |
| `UNKNOWN_DEPENDENCY` | No committed file proposed for removal. | Pre-existing local receipt edits and `update_gnn_readme_index.py` were not modified, staged, or included in this report's changes. |

## Checkpoint and cache storage (outside Git)

The selected V4b checkpoint weights are local under
`/home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37/`:

| Arm | Selected step | SHA-256 | File size |
|---|---:|---|---:|
| Compiler-only | 1500 | `ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162` | 600,337 B |
| Hardware-message | 1250 | `89db512c5536391d5bc24b35c784465a729dcaa7ac9fa268e7a1dd7cd5ef4f4a` | 600,529 B |
| Target-FiLM | 250 | `c4643f481515a60c414f92a5ffafea7b9c96239291ff414fef4238e062aca51f` | 858,910 B |
| Compiler + difference | 1500 | `14f1661a40d977ce6ba04672c020431fffe6f8be010f999afd6249042f5762c6` | 703,684 B |
| Independent no-graph | 1000 | `829a146568a1017fd4cddd4a7667313c404113314951edb2d4478d566f8f7cdf` | 577,637 B |

The full37 and matched14 run directories each occupy about 73 MiB; capacity
control is about 15 MiB. Keep these outside Git and retain their source hashes,
receipts and embedded-export identities. The standalone GEMV `*_active.csv` and
`*_raw.npz` are small committed embedding outputs; they do not include weights.

## Removal audit template and conclusion

There are no proposed removals, so no deletion path, hash/size record, or
regeneration claim is asserted. The optional rendered formats above are
regenerable but are retained for usability. The current researcher entry points
are this report, the [architecture and GEMV guide](README.md), and the dated
[corrected-v3 results](../../README.md#v4b-final-scientific-report--corrected-v3-2026-11-04).
