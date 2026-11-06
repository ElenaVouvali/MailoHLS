# Structural-memory interface contract

## Tensor contracts

| Item | A: exact action | B: relational action | C: typed evidence |
|---|---|---|---|
| Action vectors | `[B,64,128]` | Same bit-identical vectors as A | Same bit-identical vectors as A |
| Action validity | `[B,64]` | Same | Same |
| Relation mask | None; action slot k attends key k | `[B,64,64]` | Combined `[B,64+E,64+E]` |
| Extra evidence values | None | None | `[B,E,128]`, right padded |
| Evidence valid/known/type | None | None | `[B,E]`, `[B,E,9]`, `[B,E]` |
| Action-to-evidence routes | None | None | `[B,64,E]` before combination |

`Lk` remains absolute slot `k-1`; action IDs are never compacted. C uses a single combined key axis: action keys 0–63, then evidence keys 64 onward. Its relation-mask rows for directive/action queries can address routed action and evidence keys; evidence rows have no outgoing routes, so an evidence key cannot be mistaken for an action ID. `evidence_type_ids` and `evidence_known_mask` are side metadata for a future consumer; the current XAttn module consumes vectors and boolean masks only.

The build also compared the same compiler-only encoder under U200/10 ns and ZCU104/3.33 ns full-capacity overlays. Action memory and pooled program memory were **bit-identical** (maximum absolute action-memory difference 0.0). The V4b implementation removes hardware-node edges before encoding, so these A/B/C memories are static compiler context. Target awareness in a future Stage-2 comes from the LLM query and must not be described as target-conditioned GNN memory.

The `[B,128]` pooled program vector is retained in the raw diagnostic pack for provenance, but is deliberately not passed by A/B/C to cross-attention; all three arms stay action-aligned so routing and evidence visibility are the only differences.

## Relation derivation

`rc3_action_relations.py` requires schema `salt-hls-mlir-structural-v2.2`, semantic revision `rc3`, an authenticated graph identity, and a proven source-action target edge. It calls the existing RC3 relation queries, then projects proven self, loop hierarchy, loop-scoped array accesses, and supported inter-action dependence evidence into directed action eligibility. Each non-self route carries the compiler source relation and source edge/edge list, source/metadata/MLIR provenance, graph identity, evidence status, direction, and absolute slots. Local recurrences remain local evidence; a missing dependence witness means `not_established`, not independence.

The GEMV-derived matrix is:

```text
L1 -> L1, L4
L2 -> L2, L4
L3 -> L3, L4
L4 -> L1, L2, L3, L4, L5
L5 -> L1, L2, L4, L5
```

This matrix is a regression result, not an embedded lookup table.

## C memory representation

Node evidence uses the frozen GNN's existing 128D node output. Edge evidence is not represented as though an endpoint node fully encoded the edge. The current deterministic, untrained diagnostic vector is: mean endpoint vector; add the frozen relation embedding to dimensions 0–15 and frozen edge-feature projection to dimensions 16–31; remaining dimensions retain endpoint mean. The edge record also keeps the exact edge payload, endpoint IDs, relation, graph identity, and known/status fields in JSON. This composite has not been validated as a useful learned representation and should be compared with typed side channels or a trained edge encoder later.

Evidence ordering is deterministic: evidence type order, then canonical graph node/edge key. Capacity is 128 total records per program; overflow raises and does not truncate. Batch padding is right-sided with false valid/known masks. A caller must retain the returned lengths and masks.

## XAttn adapter and masking checks

`adapt_for_xattn(pack, 'A'|'B'|'C')` returns the existing memory, one-based placeholder-slot, and memory-mask inputs used by `LLM_branch.common.structural_xattn.MaskedCrossAttention`. A uses exact-slot routing (no explicit relation tensor), B uses the 64×64 relation tensor, C uses the combined key mask. The actual production attention module was exercised with synthetic hidden states on CPU:

- Tokens before any Lk marker route to no key and produce finite, zero structural output.
- A query at L1 sees only its own key in A; it sees L1/L4 in B; C exposes those action keys plus only L1's eligible evidence keys.
- Unmatched absolute slots and right-padding are masked out.
- An active action with an empty legal route fails loudly; all-invalid/no-anchor synthetic inputs remain finite and silent.
- No directive text or future assignment is part of the memory builder input.

Attention key count/entropy is not a measure of information quality. The adapter is a compatibility test, not a trained fusion arm. Test command and exact environment are in `README.md`.
