# Architecture decisions and executable scope

This Revision2 package extends the prior [architecture report](../architecture-finalization/README.md). It corrects the interpretation of V4b and completes CPU cache/message/gradient and latency-common-pool audits. It does not claim a newly optimized model.

The policy remains complete constrained one-pass directive proposal. Source registration determines absolute L1-L64 slots; source-derived domains provide local legality. Target prompts carry objective, canonical part, fixed clock and four available counts/fractions. Production RHS candidates are ranked by mean continuation token log-probability, greedily serialized in field order. Local legality does not guarantee resource feasibility or timing.

The selected research parent is DeepSeek-Coder6.7B revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912, saved V4 latency tokenizer and rank8 attention LoRA with trained special-token deltas, adapter SHA28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164. Its choice is a retained research baseline, not a newly measured superiority claim. Reference request is PARETO_LATENCY / xczu7ev-ffvc1156-2-e /10ns /624BRAM18K,1728DSP,460800FF,230400LUT.

Compiler graphs preserve structured loop/access/SSA evidence and explicit missingness. Frozen V4b produces128D action/node features. A exposes exact action keys; B adds compiler relations; C exposes eligible typed witnesses with provenance and padding. Future compatible fusion uses a hardware-conditioned LLM query and static memory, injecting tanh(gate)*attention output after self-attention residual and before the native MLP. No fresh fusion weights have qualified here. Four-layer V1/ADP and RC3 pilot checkpoints remain independent.

See architecture_decisions.json for component choices, risks and conditions for reversal; V4B_IMPLEMENTATION_REAUDIT.md for controlled predictor evidence; TARGET_CONDITIONING_ANALYSIS.md for measured-bank coverage; REPRODUCIBILITY.md for commands. All new model architecture choices remain unresolved rather than rejected. Parameter cost, generated QoR, inference cost and held-out adaptation require GPU and physical follow-up.

Objective support: registered Stage1 supports latency/area/ADP; registered Stage2 is ADP, latency and area fall back to Stage1; ADP Stage3 is separately recorded with ablation lineage. This research package changes no registry/default. Automatic clock selection, U50 adaptation and optional synthesis-feedback refinement are outside this fixed-clock qualification.
