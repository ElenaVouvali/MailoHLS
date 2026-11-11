# Implementation decision and frozen experimental boundary

Audited from `origin/final-architecture` at `25c5ae3`; the supervisor visualization
was independently published at `d0fd9eb`. The original dirty detached checkout and
all other worktrees were preserved. Current code lives in an isolated research
worktree and is published only through fast-forward updates.

Reuse the authenticated standalone RC3 graph, source/domain registry, proven
action/evidence scopes, `PaperRC3Net` encoder, native Stage-1 parent, Stage-2
trainer, corrected multireference/action-normalized supervision, structural
checkpoint context, gated `post_self_attention_residual` fusion, and full causal
RHS/newline scorer. The selected preliminary checkpoint is retained exactly as
specified by the completed four-layer study; neither V1 nor V4b weights are
accepted. No production model registration is changed.

Extend memory with separately addressable node/edge records and a trainable typed
encoder. Edge input includes **ordered** trained endpoint states, compiler
relation/status features and knownness; it is not an endpoint proxy. All record
payloads and compiler IDs remain sidecars. Node, type, status and knownness
projections are learned through Stage 2. Original action states stay untouched.
Explicit identity normalization preserves the published raw first-layer/JK
vectors; it is not a fitted RMS normalization. Its hashed artifact is passed
through the existing trainer's normalization-contract checks.

Use two learned action-conditioned queries, 128D and four heads, to aggregate
each action's compiler-authorized evidence. Retain individual evidence alongside
these summaries and exact-slot action anchors. The same fresh memory is shared
at fusion layers 8/16/24/32. Keep query rows restricted to the first 64 slots;
evidence and summary keys cannot become action addresses. Empty summary routes
produce masked zeros. Overflow raises. A CPU cohort scan found 2,675 catalog
records in the largest sample; the explicit experimental capacity is 4,096.
The initial 1,024-cap export stopped on overflow without truncating; that partial
ignored runtime bank was preserved and excluded.

The official [GMT paper](https://arxiv.org/abs/2510.08966v2) and
[`gmt_model.py`](https://github.com/tongruiliu/GMT/blob/main/gmt_model.py) motivate
learned memory queries and deep memory-to-decoder alignment. GMT's
`SemanticGraphModule` uses knowledge-graph relation semantics, and its
`GraphMemoryTokenizer` aggregates encoded edges. Its `MemoryCrossAttention` and
`MemoryAugmentedLlamaDecoderLayer` integrate memory within the decoder.
[`pretrain_gmt.py`](https://github.com/tongruiliu/GMT/blob/main/pretrain_gmt.py)
trains entity/relation scores with negative triples; that objective is not used
here. [`finetune_gmt.py`](https://github.com/tongruiliu/GMT/blob/main/finetune_gmt.py)
aligns graph memory to frozen language modeling. Transfer aggregation/alignment,
not the LLaMA wrapper, KG relations, negative sampling or unrestricted routing.

One shared alignment module is registered under the first existing structural
attention module. This lets the maintained optimizer/freeze/save/resume code own
the extra parameters and leaves the Stage-1 adapter/backbone frozen. Scoped
adapters are restored when execution exits; historical inference defaults remain
unchanged. No new FiLM, DPO or refinement is included.

The first feasible GPU experiment is a **20-update matched pilot**, seed 123,
batch 1, accumulation 8, with the same sealed 1,164 training exposures from 17
families and 54 development exposures from two independent families. No GEMV
training or locked-test labels. This budget tests execution/alignment and does
not establish convergence or general superiority. Larger matched, converged runs
and independent-family qualification remain necessary.

Fixed-prefix interventions remove direct keys with summaries held fixed, or
remove tokenizer input records and recompute summaries. Both leave the frozen
GNN/action states unchanged. Additional content-changing controls use cross-route
donors matched by type/status/knownness. Key/value permutations within a route
are not used. Random deletions have matched route/cardinality and deterministic
seeds. Individual records are selected before measuring effects: the first
canonical eligible record of each type. All free-running configurations are
reported separately from conditional margin changes.
