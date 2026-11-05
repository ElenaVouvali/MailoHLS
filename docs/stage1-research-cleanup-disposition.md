# Stage-1 research branch artifact disposition

Reviewed on 5 November 2026 at `ecb6e78`. The cleanup is documentation and
entry-point consolidation only. No removal is proposed: there is not enough
evidence that any tracked research artifact is both semantically redundant and
independent of reproduction or historical interpretation.

| Classification | Paths / evidence | Disposition |
|---|---|---|
| `ACTIVE` | `experiments/stage1_quality_sft/{build.py,core.py,training.py,inference.py,evaluate.py,select_checkpoints.py}`, `LLM_branch/common/{mailohls_contract.py,directive_domains.py}`, production constrained decoder in `LLM_branch/inference/eval_stage1_stage2_stage3.py` | Current prompt, source-domain, training, inference and evaluation code. Keep. |
| `SELECTED_ARTIFACT` | Local V4 root adapters under `/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/{single_reference,multi_reference}/run/`; tracked Kalman inference report | Adapter weights and exact hashes are summarized below; local weights are not Git artifacts. Kalman result records legal inference only, not QoR. |
| `HISTORICAL_EVIDENCE` | `experiments/stage1_quality_sft/results/kalman_v4_20261104.{md,json}`; consolidated recovery sections, receipts and the V1 production model/contract | Preserve. Recovery variants and V4 have different supervision, source coverage, or inference scope. |
| `REGENERABLE` | `docs/figures/gemv_stage1_gnn_boundary.{dot,svg,png}` | SVG/PNG can be rendered from committed DOT with Graphviz. Retain both formats for documentation and image preview. Source and prompt examples are tied to an exact GNN commit permalink. |
| `REDUNDANT_VERIFIED` | No tracked files approved for removal. | Similar names, old dates, adapter directories, or backup filenames do not prove files are unused. |
| `UNKNOWN_DEPENDENCY` | No committed file proposed for removal. | Pre-existing local `evaluate.py` edit, `README.txt`, manual handoff, backup copies, and `assessments/` remain untouched, unstaged, and excluded from this documentation commit. Their owners/dependencies were not assumed. |

## Selected adapter and storage inventory (outside Git)

The current report authenticates the local V4 root adapters. Both are 35,882,480
bytes and use the same saved tokenizer and source-domain registry identities.

| Arm | Local path | Adapter SHA-256 | Contract SHA-256 |
|---|---|---|---|
| Single-reference | `/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/single_reference/run` | `abef1af9e9e833b2a02173a803318b90a37b5015c8f597ebb02c70d0655bbd66` | `f4dcf440a98cf4cf75bf6ecd294193e8ba2362a825fe148b553907b9788d97b4` |
| Near-optimal multi-reference | `/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run` | `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164` | `1c7a2034dba45030e51525fd415367e6fcf276cceddd89eb1132edfd66840671` |

Each complete local run directory occupies about 291 MiB because it also retains
intermediate checkpoints and audits. Keep all of it outside Git; do not prune
intermediate checkpoints during this documentation task. The two adapters share
the pinned DeepSeek-Coder-6.7B-base revision and tokenizer, but have different
training targets. The final-root policy used for the V4 Kalman probe is distinct
from the optional `select_checkpoints.py` measured-quality selection utility.

## Removal audit and current entry points

No candidate for deletion passed a source/reference/dependency audit. Thus no
path/hash/size is presented as a proposed removal, and no measurement,
checkpoint, original recovery implementation, source action map, or result
receipt was deleted. The current reader entry points are the branch [README and
GEMV walkthrough](../README.md#current-research-status-5-november-2026), the
[quality-SFT package guide](../experiments/stage1_quality_sft/README.md), and the
[authenticated Kalman inference record](../experiments/stage1_quality_sft/results/kalman_v4_20261104.md).
