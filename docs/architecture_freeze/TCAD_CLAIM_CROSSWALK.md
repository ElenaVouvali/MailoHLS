# Active TCAD draft to evidence

The ZIP is now present and was read. Exact ZIP/member hashes are in results/paper_inventory.json. Historical report saying ZIP absent applies to the earlier run only. Manuscript files are extracted under ignored artifacts, unchanged. Active lines exclude commented alternatives.

| Active section / anchor | Claim | Code / artifact | Status and suggested correction |
|---|---|---|---|
|04 Design Overview; fig:design-overview | Three-stage hybrid for objectives | FINAL_ARTIFACTS.json, saved Stage1/Stage2/Stage3 contracts | Supported by older architecture only for compatible registered paths; describe latency research as Stage1 baseline until fusion qualifies |
|05 Detailed Design; subsubsec:peft, line665 | Attention and FFN LoRA | adapter_config.json and trained parameter receipts | Contradicted for authenticated adapters: list Q/K/V/O attention targets; do not claim FFN adaptation |
|05 gnn-backbone;06 training | Four-layer JK,128D first+JK export | V1 encoder and production memory lineage | Older architecture only; RC3 V4b uses two64D layers and128D projection |
|06 evaluation data |130310/13011/6705 raw rows, family split | split manifest; prior reconciliation | Historical full cohort; distinguish2524/54 V4 curated rows and1164/54 RC3 pilot from raw counts |
|05 target-conditioning; fig:target-conditioning | GAN target response | historical target-conditioning figure and synthesis identity | Locked external descriptive result; not architecture-selection evidence or RC3 proof; checkpoint attribution unresolved |
|05 structural-fusion; tab:memory-ablation | Fixed Kalman latency intervention | frozen historical experiment | Historical-only locked case; validity does not register a production latency Stage2 |
|06 Stage2 training | layers8/16/24/32;3epochs;8.68M | saved ADP contracts vs RC3 20-update receipt | Registered/older objective-specific contract; do not apply to RC3 convergence or all objectives |
|05 DPO equation;06 Stage3 | beta0.5, NLL0.02, completion likelihood | train_DPO_harp_xattn.py; saved reduction contract | Historical semantic-action reduction differs from standard summed completion DPO; state exact implementation and objective lineage |
|07 graph-representation-comparison; tab:representation-predictive | Matched graph superiority | original graph comparison receipt required | Not reproduced by cross-schema V1/RC3 comparisons; table attribution pending |
|07 target-generalization; fig:target-generalization |6.67ns/U50 transfer and post-route | historical external transfer receipts | Historical-only; no new transfer/post-route execution here |
|07 SoTA/cost figures;08 limitations | End-task gains and optimization cost | matching source/tool/checkpoint receipts required | Do not transfer figures to newly selected RC3 policy; new matched quality comparison pending |

Suggested wording: “Corrected RC3 experiments verify hardware-message transmission but do not demonstrate an incremental benefit under the historical ADP protocol. Latency-oriented Stage2 target adaptation remains under qualification.”

Source, checkpoint and objective attribution for every historical figure has not been recovered. Mark those pending rather than inventing identities. Preserve all original figures and measurements. The draft's constrained action space is local admissibility, not proof of synthesis, numerical correctness or timing closure.
