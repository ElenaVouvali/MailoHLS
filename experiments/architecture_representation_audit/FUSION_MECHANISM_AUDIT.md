# Cross-attention mechanism audit

## Status

**Interface and static flow audited; learned Stage-2 behavior not measured.** No completed, authenticated compatible Stage-2 C checkpoint was available in the isolated research base. The active Stage-2 multi-kernel worktree was not read for uncommitted results and was not modified or interrupted. No training, GPU inference, or production-decoder change was performed.

## Verified implementation path

The existing `MaskedCrossAttention` uses the LLM hidden state as query and compiler memory as K/V, applies an eligibility mask, then passes a gated structural residual through the existing integration boundary. The RC3 A/B/C adapter preserves absolute Lk slots. A is one exact action key; B adds proven compiler-relational action routes; C concatenates action and evidence keys and routes action-query rows to selected evidence. Memory is compiler-only and invariant to hardware requests under V4b because hardware edges are removed. Thus hardware context can condition Q through the LLM, while K/V remain fixed per source program.

The existing diagnostic report tests synthetic CPU cases: pre-anchor/no-key queries are finite and silent; unmatched slots and padded evidence are masked; empty legal routes fail; zero-gated path should reproduce the base Stage-1 output if applied at the defined residual boundary. These establish mask/interface behavior only. They do not demonstrate that a trained checkpoint uses authenticated evidence.

## Coverage constraints discovered in this audit

The multi-family extraction found C evidence overflow on 37/55 graphs and action-to-compiler mapping errors on 12/55. Before learned fusion analysis, the memory producer must be able to emit complete, typed, authenticated masks for the target corpus. Do not reduce evidence by silent truncation. The current evidence records carry type/knownness/provenance side metadata, but attention K/V use the deterministic 128D vector; sidecar compiler payload and relation type are not automatically made learnable inputs. A proven edge's vector is an endpoint-mean composite with frozen edge/relation projections, not an independently trained edge representation.

## Learned behavior questions left open

Without an authenticated trained C checkpoint, the audit cannot report query/key scales, masked attention entropy, gate values, structural residual norms, candidate-margin effects, or evidence intervention effects. It therefore cannot establish whether the bottleneck is Q–K compatibility, gate suppression, key redundancy, evidence count, residual placement, or supervision. Attention weights alone would not settle causal use even once logged.

## Required read-only checkpoint diagnostics

When an authenticated completed checkpoint exists and does not conflict with active training, instrument one held-out program/request at the exact Lk RHS position. Verify hardware-invariant memory hashes across requests, query hardware contrasts with matched prefixes, and save compact Q/K/V norm and logit statistics. Then compare authentic memory with evidence-zero, type-group-zero, and within-program evidence permutation while keeping action vectors and masks fixed. Record changes at attention output, gated residual, production candidate margins/rank, and directive winner. Include same-condition placebo and gate-zero reproduction. All interventions must preserve action IDs and label/prefix alignment.
