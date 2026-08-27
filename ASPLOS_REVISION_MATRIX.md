# ASPLOS'27 Revision Matrix

This file maps the MICRO reviews to concrete ASPLOS changes. Keep it as the paper's private control document.

| MICRO concern | ASPLOS response | Evidence/section | Priority |
|---|---|---|---|
| Single device/frequency; questionable generality | Explicit device + frequency conditioning; cross-device/frequency evaluation; optional few-shot adaptation; AUTO clock | Results C; Methodology target conditioning | MUST |
| Fine-tuned MailoHLS vs non-fine-tuned general LLMs is unfair | Stage-1-only task-specific control; same structured prompt/objectives; equal retry/pass@k; RAG/agent baselines when reproducible | Evaluation Baselines; Validity | MUST |
| Unclear novelty vs HARP/ProGraML / graph-language hybrids | MLIR compiler graph; explicit dependence semantics; action-aligned memory; one-pass role distinction; quantitative representation comparison | Intro pages 1-2; Results E | MUST |
| GNN role ambiguous; QoR predictions unused | State explicitly: QoR heads supervise representation; only structural embeddings enter Stage 2; no GNN QoR oracle at inference | Methodology; Discussion | MUST |
| Missing validity/synthesis/pass rate | Validity ladder: schema -> compile -> sim -> synth -> resource-feasible | Results B | MUST |
| Missing search/sample efficiency and runtime | One-pass inference cost + HLS-call count vs iterative DSE baselines | Results G | MUST |
| Resource metric undefined | Give exact equation; also report BRAM/DSP/FF/LUT separately | Evaluation Metrics | MUST |
| Weak baseline coverage | HARP, AutoDSE, LIFT; LLM-DSE/HLSPilot where apples-to-apples; HLS-Eval/HLStrans/ForgeHLS as benchmark context | Related Work; Results | MUST/HIGH |
| Limited applications/complexity | Add external Vitis kernels; action-space size and cyclomatic complexity; avoid cherry-picking | Evaluation Dataset; Results H | HIGH |
| No post-P&R / hardware evidence | Post-P&R on selected unseen kernels if feasible; otherwise clearly scope claims | Results H; Limitations | HIGH |
| Ablation confusing / attribution unclear | Stage 1 semantic-only; zero/aligned/shuffled Stage 2; Stage 3; backbone-size ablation | Results D/F | MUST |
| Backbone capacity may explain gains | Two-size factorial comparison (Stage1-only vs full) | Results F | HIGH |
| Unclear dataset split unit | Explicit family/kernel split table and locked test policy | Evaluation Dataset | MUST |
| Training/inference overhead missing | Report GNN training, LLM adaptation, graph build, inference, synthesis validation | Evaluation/Results G | MUST |
| Resource adapter sometimes worse | Case-study analysis and directive-level explanation | Results Explainability | HIGH |
| Fixed 64-slot memory scalability | Active-slot distribution, truncation rate, complexity correlation | Discussion | HIGH |
| Reproducibility | Public/anonymized artifact, exact contracts/manifests/seeds, commands, prompt appendix | Appendix/artifact | MUST |
