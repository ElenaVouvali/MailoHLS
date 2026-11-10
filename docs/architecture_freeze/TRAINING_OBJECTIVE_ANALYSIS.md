> Revision3 repairs the Stage2 uniform/top1 mismatch. Compatible Stage3 summed-value DPO(beta.1) and matched action-normalized chosen-only SFT completed34updates each; validation ranking ties25/54. The requested normal DPO default remains DPO_REQUIRED_BUT_NOT_QUALIFIED. The earlier text-only negative pilot is not the matched comparator.
>
> Current evidence: [Revision-3 results](RESULTS_V3.md); [technical architecture](FINAL_SALT_HLS_ARCHITECTURE.md).

# Training objective decision

The research parent remains V4 latency near-optimal multi-reference SFT. It uses rank8 attention Q/K/V/O LoRA and trained added-token embeddings; the saved adapters do not target FFN projections. Preserve action-normalized nonforced RHS supervision and complete correlated assignments. Do not select labels field by field from different measured designs.

This run did not compare matched continued SFT and switch-enriched SFT. Training-only latency contrasts are candidates for authenticated supervision; proxy resource boundaries and extreme spcl ratios need confirmation. Validation switch coverage must improve before architecture selection can claim adaptation quality.

Archived text DPO136 training pairs/17 families and54 validation pairs/2 families improved training ranking82.35% to84.56%, but validation fell35.19% to33.33%. It does not qualify a replacement. DPO necessity is INCONCLUSIVE, and it remains excluded from the selected latency endpoint. Standard summed nonforced completion log-probabilities differ from production mean RHS candidate scores and historical semantic-action DPO reductions. A new compatible Stage2 DPO arm requires matched reference memory, masks, padding, causal prefixes and a chosen-only SFT control.
