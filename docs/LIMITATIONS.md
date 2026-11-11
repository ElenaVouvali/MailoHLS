# Qualification limits

The integrated latency policy is **unqualified**. The corrected Stage-2 and
DPO trials changed none of nine natural generated assignments versus Stage 1.
DPO remains an offline training requirement to qualify; its neutral pilot is
not the production default. Two validation families and few independent
target/clock winner reversals do not establish target adaptation.

The historical V1 encoder, RC3 two-layer executable encoder, and new four-layer
latency candidate have different graph, feature, checkpoint and memory contracts.
Equal 128D output width does not establish compatibility. New memory requires
matched fusion training before joint performance claims. The later local GNN
completion still says `FREEZE_NOT_YET_JUSTIFIED`; see the
[completed report](evidence/latency_gnn_completed_local.md).

The GNN study has only three development families, a post hoc ranking-selection
revision, an explicitly ignored Sort-radix L6, and percentage-derived resource
estimates. Timing qualification remains unknown. Restricted candidate-pool regret
is not global-optimum regret. Selection and regret coverage must be read together.

Compiler uncertainty remains explicit. An unresolved source action is not proof
of compiler elimination; an absent relation is not proof of independence.
Typed-witness embeddings and deep/query FiLM remain proposals. The neutral
final-hidden-state FiLM experiment does not test every possible FiLM placement.

Refinement is opt-in, charges baseline/cache/failure attempts under its contract,
and retains timeouts. The matched successful 1/4/8-query comparison is incomplete.
HLS-estimated timing is distinct from post-route timing and C/RTL equivalence.
No physical speedup follows from a successful alternative whose baseline timed out.

The current consolidation has not replayed a complete assignment from an
authenticated real Stage-1 checkpoint, tokenizer and source/action registry.
CPU fixtures cannot close that gate. Checkpoint access, clean environment
reproduction, authors/version for a citation record, top-level licensing and
dataset/third-party redistribution rights remain release review items. Existing
third-party notices are retained; no repository-wide license is inferred from them.

Reports copied from research branches are immutable snapshots, sometimes with
old relative links. [Provenance](evidence/SOURCE_MAP.tsv) identifies their exact
source commits. Local runtime completion receipts are authenticated separately;
their preservation here does not publish the original host weights or raw data.
