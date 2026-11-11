# SALT-HLS: FiLM, graph-memory retrieval, and historical encoder selection

**Architectural status (11 November 2026):** research design note. It distinguishes theoretical compatibility from hardware benefit and does not revise historical measurements.

## Two orthogonal conditioning operations

FiLM changes **features**: \(h_l' = h_l + a_l(c)\odot[\gamma_l(c)\odot\operatorname{Norm}(h_l)+\beta_l(c)]\), where \(c\) carries the FPGA part, clock, absolute resource budgets, and objective. A stable residual or zero/near-zero initialization retains the pretrained decoder; large uncalibrated modulation can interfere with its language/code representations. See [Perez et al., AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11671) and the adaptive-normalization design in [Peebles & Xie, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html).

GMT-style cross-attention changes **accessible/selected information**: \(A_l=\operatorname{softmax}(Q(h_l')K(M)^\top/\sqrt d+\mathcal R_k)V(M)\). Here \(M\) consists of compiler-grounded action/evidence representations; \(\mathcal R_k\) is the source-verified eligibility mask for directive \(L_k\). GMT motivates multi-layer cross-attention to graph memory [Liu et al., arXiv:2510.08966](https://arxiv.org/abs/2510.08966). SALT-HLS adds deterministic source-action alignment; learned attention is free to prioritize *eligible* witnesses, not to invent edges or override provenance.

**Compatibility:** FiLM can modify query geometry, changing the relative relevance of admitted compiler facts; it can also modulate later decoder hidden states. If exactly one key survives the routing mask, query FiLM cannot alter selection weight, only other parts of the computation may change. FiLM cannot repair source-action mapping errors or create physical target observations absent from training.

## What was actually tested

The `research/stage1-target-film-20261110` experiment introduced `experiments/stage1_target_film/model.py::ActionFiLM`, hooking the final hidden state immediately before the output head. It is **not deep residual-path FiLM**. The learned tanh gate had magnitude around 1.86e-6; in FP16 a median ~98% of the intended small correction norm was lost. ON/OFF/legacy produced the same 28/54 preference outcomes on two validation families. Eighteen generated GAN requests were not fully matched under the corrected masking policy. No physical latency benefit has been demonstrated. More aggressive *well-scaled* deep conditioning is therefore a legitimate new hypothesis, not an established conclusion.

The graph-side target-message/FiLM-style studies had weak incremental QoR evidence and insufficient independent device/clock winner reversals. This is not a theoretical failure of graph-level conditioning: compiler facts are stable across targets, but their performance relevance may change, provided the measured dataset contains counterfactual distinctions.

## Preferred historical representation (selected design, pending qualification)

Use the MailoHLS **four-layer, 64-dimensional edge-aware TransformerConv GNN with JK-max**, exporting **128-dimensional `[first-layer state | JK state]`** action representations. The local-to-multihop inductive bias is appropriate for loop nests, memory-access dependencies and directive interactions. The original MailoHLS implementation provides architectural continuity; [Jumping Knowledge Networks, ICML 2018](https://proceedings.mlr.press/v80/xu18c.html) offers the theoretical/locality motivation; [HARP, ICCAD 2023](https://github.com/UCLA-VAST/HARP) motivates hierarchical HLS program/action representations. The first latency-oriented training arm reported tau-b 0.13568 and measured-pool feasible regret@1 38.50% on ZCU104/10 ns, and worse all-context regret. **These are not superiority results over a matched two-layer encoder**, and the alternative/ranking-policy evaluation was pending at the inspected push. Retain the two-layer RC3 encoder as a reproducible control and never load its cross-attention checkpoint with four-layer outputs solely because both are 128-dimensional.

The **language decoder** is a different component: preserve the pinned DeepSeek-Coder-6.7B and source-valid autoregressive RHS candidate scoring, with conditional field domains and native mean causal continuation log-probabilities. Its actual behavior and pretrained/LoRA identities are part of the production contract; it is not synonymous with the four-layer graph encoder.

The later local Liono continuation has completed with **`FREEZE_NOT_YET_JUSTIFIED`** and no production promotion. Its preliminary selected checkpoint remains a research artifact. See the [completed local report](evidence/latency_gnn_completed_local.md) and [local receipt provenance](evidence/LOCAL_SOURCE_MAP.json); the pushed initial report above remains unchanged.

## Minimal decisive experiment

Hold graph, parent decoder, action domains, train/validation splits, measured candidate pool and physical budget fixed. Compare (1) current prompt-conditioned structural cross-attention, (2) same decoder with target-modulated **cross-attention queries**, and (3) selected-layer, bounded decoder-hidden-state FiLM plus cross-attention. Add (4) prompt-only/no-graph control. Measure gamma/beta/gates and effective residual perturbations **at deployed precision**, route-wise attention changes, complete assignment changes, finite timing-qualified feasible regret, and source/target heldout transfer. Include authentic physical **winner switches and stays** under device, clock and budget. A stronger parameterization alone, without switch supervision or synthesis ground truth, is not a defensible gain claim. Train DPO against each exact matching Stage-2 parent only after its decision policy is qualified.

**Scientific claim:** FiLM and graph memory answer different questions—*how should the decoder process this target context?* versus *which compiler-grounded evidence should its current directive decision consult?* Their combination is technically sensible but not yet experimentally established for SALT-HLS.
