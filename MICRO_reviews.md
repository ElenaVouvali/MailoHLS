MICRO 2026 Paper #3267 Reviews and Comments
===========================================================================
Paper #3267 MailoHLS: Multi-Adapter Structure-Aware Learning for
Pareto-Driven HLS Pragma Optimization


Review #3267A
===========================================================================

Is the paper violating the submission format?
---------------------------------------------
1. The paper meets the submission format rules.

Reviewer expertise
------------------
3. Little or no familiarity.

Reviewer Confidence
-------------------
2. Medium confidence: I understand much of the paper but not all of it.

Paper summary
-------------
This paper proposes a framework that combines LLM-based semantic reasoning with GNN-based structural modeling for HLS pragma optimization on FPGA designs. With fine-tuning, the paper can automatically find high-quality HLS pragma configurations.

Strengths
---------
* This is a good engineering work that incorporates LLMs into high-performance FPGA/HLS optimization.
* The evaluation seems reasonable. It includes multiple axes, e.g., latency-oriented, resource-oriented, and balanced trade-offs.

Weaknesses
----------
* Generality is a key concern. The method is evaluated mainly on a specific FPGA platform. From my point of view, an LLM could easily overfit to a relatively small FPGA design space, but may not achieve similar gains on other FPGA devices.
* The comparison with other LLMs appears unfair. MailoHLS is fine-tuned, while most commercial LLM baselines are not fine-tuned. It is possible that some of the gains come from task-specific fine-tuning.

How easy is the paper easy to read and understand?
--------------------------------------------------
1. Very clear: I had no trouble understanding the work.

Novelty of Work
---------------
3. Limited novelty: small tweak or incremental improvement

Evaluation methodology
----------------------
2. Acceptable: Reasonable methodology and results to demonstrate value of
   idea

Pre-Rebuttal Overall merit
--------------------------
4. Major revision -- Fair work but with concerns. Satisfactory revision is
   critical for acceptance in the program

Comments for authors
--------------------
I think using LLMs to improve hardware and code efficiency is an active research direction. This paper follows this trend by leveraging an LLM and integrating a GNN to better capture code structure. The problem is important, and the proposed system seems reasonable.

However, I remain skeptical about the effectiveness and generality of this approach. In theory, a GNN can capture arbitrary code structure, so it is not clear why the method is specifically limited to FPGA/HLS optimization. The paper might need to explain that the GNN captures FPGA-specific constraints.

The paper also reports that switching from a 1.3B model to a 7B model significantly improves the overall speedup. This raises some concerns that the reported gains may be due only to model capacity and fine-tuning rather than the proposed structure-aware method.

Overall, I think this could be a good paper. However, I think the generality of the approach remains questionable without cross-platform evaluations.

Questions/Issues for the authors to address in the rebuttal/revision
--------------------------------------------------------------------
1. Can you evaluate the same trained model on different FPGA platforms, clock frequencies, or HLS tool versions, without more fine-tuning? 
2. What happens if the LLM backbone becomes larger? Would the major benefit come from model capacity rather than the proposed GNN-based structural fusion?



Review #3267B
===========================================================================

Is the paper violating the submission format?
---------------------------------------------
1. The paper meets the submission format rules.

Reviewer expertise
------------------
2. Knowledgeable: I used to work in this area and/or I try to keep up with
   the literature but might not know the latest developments

Reviewer Confidence
-------------------
2. Medium confidence: I understand much of the paper but not all of it.

Paper summary
-------------
This paper introduces a hybrid framework for HLS directive optimization for high-quality results (QoR). The framework, MailoHLS, integrates both semantic reasoning and structural modeling to enable objective-specific optimization. The core idea is to combine the GNN structural modeling and the LLM's semantic reasoning to provide guidance in HLS. The main contributions are the following:
1. It reformulates the HLS directive optimization problem as directive value prediction over the placeholders (template-based synthesis), and thus constrains the search space to make the synthesis feasible.
2. It unifies both structural (graph-based representation) and semantic modeling (LLM) to have joint semantic-structural awareness for HLS. 
3. It supports objective specialization (e.g. latency, balanced, and resource) via per-objective LoRA adapters trained in three stages.

Strengths
---------
1. The directive-level optimization formulation is sensible that it can constrain the search space of HLS. Also, fixing the pragma placement as a placeholder and querying the model to assign values only can also avoid syntactic failures from LLMs. 

2. The hybrid semantic structural design does address the important gap that LLMs alone cannot reliably handle HLS optimization problems due to their limitations. For example, general LLMs cannot explicitly capture structural dependencies or hardware-specific constraints. The following evaluation further strengthens such claims, showing LLM semantic reasoning alone is insufficient. This motivates the structural fusion via cross-attention, and it shows better performance than LLMs alone.

Weaknesses
----------
The evaluation should be expanded to demonstrate that MailoHLS can generate effective designs across a broader spectrum of applications with differing resource requirements, data sizes, and code complexities.  The goal is to show the wide applicability of the tool under different application and optimization scenarios.  It is also difficult to understand the baselines chosen for comparison.

How easy is the paper easy to read and understand?
--------------------------------------------------
1. Very clear: I had no trouble understanding the work.

Novelty of Work
---------------
2. Moderate novelty: meaningful extension or combination of ideas

Evaluation methodology
----------------------
3. Fair: Reasonable methodology but need more results

Pre-Rebuttal Overall merit
--------------------------
4. Major revision -- Fair work but with concerns. Satisfactory revision is
   critical for acceptance in the program

Comments for authors
--------------------
In general, I think this work is reasonable and well-motivated. For example, the templated-based approach (directive-placeholder formulation) can reduce the search space of HLS design space exploration. Additionally, the structural and semantics embedding using GNN and LLM and fusing them via cross-attention make the HLS feasible and effective. That being said, I think the main issue is about the evaluation. Specifically, MailoHLS does produce speedup against the baseline design but it does not show the convincing results against the high-end LLMs (e.g. GPT and Gemini). It loses to the LLM baseline in one kernel by a large amount. It would be more convincing if the authors could show more experimental results with a wider set of kernels that have differing memory, computation, and control characteristics.


Typos / minor comments

Line 414: "whichi" → "which"
Line 751: "n particular" → "In particular"

7.1.3. Ablation Study section is a little bit confusing and not easy to follow. About the speedup in each stage, do they represent individual or accumulative speedup? For example, in the Latency Adapter, the final speedup seems to be 8.6X not the accumulated speedup (2+6.3+8.6=16.9X)? Step 1 speedup is 2x and step 2 speedup is 6.3X. So this means step 2 has an addition 6.3-2 = 4.3X speedup? At first glance it is very confusing. 

In figure 10, in the left-subfigure, the colors overlap the same colors of FPGA resource limits legends.

Questions/Issues for the authors to address in the rebuttal/revision
--------------------------------------------------------------------
- What is the setup for 6.3. Baseline and State-of-the-Art? Particularly, for the general-purpose LLMs (Gemini, GPT), what are the prompts for the evaluation? Is it the same prompts as the structured prompts in section 5.3?  

- Also, is it possible to have LLMs baseline for objective constraints (latency, balanced, and resource)? Intuitively, users can have additional prompts to encode such information so LLMs can perform HIS based on those objectives. It would be great to have the results in Figure 11 if possible. 


- What is the compilation failure and validity rate for both LLM baseline and MailoHLS? MailoHLS seems to always choose some directive values for anchors. So, do they always produce the valid HLS design? On the other hand, this paper reports that the LLM baseline is not reliable that GPT-4o fails to generate any valid HLS. Does it always produce any syntax errors, or HLS syntax errors or HLS results are not satisfiable the constraints or even invalid? Even if LLM baseline fails to generate any design, can we use construct another prompt based on the error to generate the correct HLS results?



Review #3267C
===========================================================================

Is the paper violating the submission format?
---------------------------------------------
1. The paper meets the submission format rules.

Reviewer expertise
------------------
1. Expert: I have written one or more papers on this topic and/or I
   currently work in this area.

Reviewer Confidence
-------------------
1. High confidence: I understand the key aspects of the paper to a great
   extent.

Paper summary
-------------
This paper proposes MailoHLS, an HLS pragma optimization framework that combines GNN-based structural modeling with LLM-based semantic reasoning. The method reformulates pragma optimization as placeholder-value prediction, which helps avoid invalid pragma placement, and uses objective-specific LoRA adapters together with Pareto-guided DPO to target latency-, balanced-, and resource-oriented designs. The experimental results are promising across seen kernels, unseen MachSuite families, and two external applications. However, the evaluation remains limited: it relies on a single HLS tool, FPGA platform, and target frequency, lacks post-P&R and hardware validation, and does not clearly define or justify the aggregated resource metric.

Strengths
---------
+ The paper targets an important HLS problem: automatically selecting pragma configurations under complex interactions between program structure, memory behavior, latency, and resource utilization.

+ Combining directive-aware GNN embeddings, LLM semantic reasoning, selective cross-attention, and objective-specific LoRA adapters is well motivated for HLS, where pragma quality depends heavily on program structure and optimization goals.

Weaknesses
----------
- The technical novelty is incremental. Many components build directly on existing ideas, including ProGraML/HARP-style graph augmentation, GNN-based QoR modeling, LLM fine-tuning, LoRA adapters, cross-attention fusion, and Pareto/DPO-style preference optimization. The paper does not clearly explain the key architectural distinctions from prior graph-language hybrid frameworks.

- The evaluation scope is relatively narrow. The QoR evaluation mainly relies on Vitis HLS 2021.1, a single FPGA platform, and one target frequency, without post-P&R timing analysis, hardware validation, or functional simulation verification. In addition, the explored design space is heavily constrained by the predefined pragma placeholders and directive templates.

- The resource metric is insufficiently specified. “Resources (%)” is not clearly defined, and the paper does not explain how BRAM, DSP, FF, and LUT utilization are normalized or aggregated into a single metric. This makes the resource-efficiency claims difficult to interpret and reproduce. In addition, the visualization in Figures 8 and 9 is confusing because the speedup bars are colored using the resource metric.

- Several important practical metrics are missing, including synthesis/pass rate, invalid-generation rate, resource-budget violation rate, optimization runtime, and end-to-end turnaround time. Baseline coverage is also incomplete, as several recent HLS/DSE frameworks and hybrid LLM-surrogate methods (e.g., AutoDSE, ForgeHLS, HLStrans, and HLS-Eval) are not included in the comparison.

How easy is the paper easy to read and understand?
--------------------------------------------------
2. Could be better: some non-trivial bits are missing or difficult to
   understand; the writing is rough in some places.

Novelty of Work
---------------
2. Moderate novelty: meaningful extension or combination of ideas

Evaluation methodology
----------------------
3. Fair: Reasonable methodology but need more results

Pre-Rebuttal Overall merit
--------------------------
4. Major revision -- Fair work but with concerns. Satisfactory revision is
   critical for acceptance in the program

Comments for authors
--------------------
* The novelty of the proposed GNN/LLM architecture appears limited. The overall design closely resembles prior structure-aware code-learning frameworks that combine graph embeddings with Transformer/LLM representations through cross-attention or late fusion mechanisms, particularly Ref. [4]. The paper introduces objective-specific LoRA adapters and directive-level placeholders, but the core backbone design (e.g., LLVM/ProGraML graph construction, GNN embedding extraction, and cross-attention fusion into a decoder-only LLM) is conceptually similar to existing graph-language hybrid architectures. The paper should provide a clearer discussion of the key architectural distinctions and why the proposed fusion strategy is fundamentally better suited for HLS optimization.

* The paper motivates the GNN as a QoR-aware structural model, but the final optimization pipeline seems to use only its latent embeddings rather than its actual QoR predictions. This makes the practical role of the GNN in the DSE process somewhat ambiguous beyond serving as a structural feature extractor.

* The fully unseen application setting is not sufficiently convincing unless the applications are clearly outside the training distribution in terms of kernels, coding style, directive space, and problem size. More details are needed to show that this is not simply interpolation over similar HLS patterns.

* The paper motivates HLS pragma optimization as a large combinatorial search problem, but it does not clearly quantify the actual search-space size across benchmarks. More details on the number of action points, directive choices, and total configuration counts per kernel would help contextualize the difficulty of the optimization task and the practical significance of the reported results.

* The paper does not evaluate search/sample efficiency relative to prior HLS DSE methods. In particular, it is unclear how many synthesis evaluations or optimization iterations are required for MailoHLS to obtain high-quality designs compared with evolutionary search, Bayesian optimization, or prior learned DSE frameworks. Since reducing expensive HLS synthesis passes is a major motivation for learned optimization methods, such comparison would significantly strengthen the practical impact of the work.

* The comparison with LLM baselines may not be fully fair. General-purpose LLMs are prompted directly, while MailoHLS is trained on many synthesized design points and constrained by predefined placeholders. Stronger baselines with similar structured action spaces, retrieval, or fine-tuning would better isolate the benefit of the proposed architecture.

* The evaluation scope remains limited to a single toolchain and FPGA platform. Resource metrics are not clearly defined, synthesis success rates and end-to-end optimization time are not sufficiently reported, and comparisons against recent HLS datasets, AST-related approaches, and stronger DSE baselines are missing. In addition, the Pareto-related claims require stronger evidence, as Figure 8 does not clearly demonstrate the advantages of the three adapters for their intended optimization goals.

* Additional concerns include the limited discussion of GNN prediction accuracy, unclear optimization iteration counts, insufficient evaluation of general-purpose LLM baselines for HLS, and incomplete prompt descriptions that make some baseline comparisons difficult to assess fairly.

* The evaluated designs are relatively small and benchmark-oriented. It remains unclear whether the proposed framework scales effectively to larger real-world HLS applications with more complex control flow, deeper loop hierarchies, irregular memory behavior, and substantially larger pragma search spaces.

Questions/Issues for the authors to address in the rebuttal/revision
--------------------------------------------------------------------
* How is the reported “Resources (%)” metric computed from BRAM, DSP, FF, and LUT utilization, and what normalization or aggregation methodology is used?

* What are the synthesis success rates of MailoHLS across seen kernels, unseen kernels, and external applications, and how frequently do generated designs fail synthesis, violate resource constraints, or produce invalid pragma configurations?

* What is the end-to-end optimization time per kernel, including graph construction, GNN inference, and LLM inference, and how does this compare with iterative DSE approaches under comparable synthesis-evaluation budgets?

* How large is the effective design space for the evaluated kernels (e.g., number of placeholders and directive combinations per kernel)? In addition, does the fixed 64-slot structural memory limit scalability for larger or more complex applications?

* Has MailoHLS been evaluated on other FPGA platforms, HLS tool versions, frequencies, or alternative toolchains (e.g., Intel HLS) to assess cross-platform generalization and robustness?



Review #3267D
===========================================================================

Is the paper violating the submission format?
---------------------------------------------
1. The paper meets the submission format rules.

Reviewer expertise
------------------
2. Knowledgeable: I used to work in this area and/or I try to keep up with
   the literature but might not know the latest developments

Reviewer Confidence
-------------------
2. Medium confidence: I understand much of the paper but not all of it.

Paper summary
-------------
This work presents MailoHLS, a framework for automatically selecting HLS pragmas for FPGA accelerator design. It combines LLM-based code understanding with GNN-based structural program analysis, so the model can reason about both code semantics and hardware-relevant dependencies. Instead of generating code directly, MailoHLS predicts values for predefined directive placeholders, improving validity and robustness. It also uses objective-specific LoRA adapters to target latency, resource, or balanced optimization, achieving strong speedups on seen and unseen HLS kernels.

Strengths
---------
1. The paper effectively combines LLM semantic reasoning with GNN-based structural modeling, enabling the framework to capture both code semantics and hardware-level dependencies for HLS optimization.

2. The use of directive-level prediction with objective-specific LoRA adapters provides a clean and robust way to generate valid HLS pragmas while supporting different optimization goals such as latency and resource efficiency.

Weaknesses
----------
1. The framework combines a lot of machine learning techniques: GNNs, cross-attention, LoRA adapters, SFT, and DPO training stages, but some implementation details and design choices are only briefly described, which may make reproduction difficult.

2. Although the paper synthesizes generated designs with Vitis HLS, the evaluation does not include FPGA runtime measurement.

How easy is the paper easy to read and understand?
--------------------------------------------------
2. Could be better: some non-trivial bits are missing or difficult to
   understand; the writing is rough in some places.

Novelty of Work
---------------
2. Moderate novelty: meaningful extension or combination of ideas

Evaluation methodology
----------------------
2. Acceptable: Reasonable methodology and results to demonstrate value of
   idea

Pre-Rebuttal Overall merit
--------------------------
3. Minor revision -- Reasonable paper with incremental improvement or with
   some deficiencies which could be addressed by a revision.

Questions/Issues for the authors to address in the rebuttal/revision
--------------------------------------------------------------------
1. As mentioned in the weakness section, some ablation studies to investigate the effectiveness of each machine learning techniques will be helpful.

2. Also, please include some real FPGA measurement would be helpful.



Review #3267E
===========================================================================

Is the paper violating the submission format?
---------------------------------------------
1. The paper meets the submission format rules.

Reviewer expertise
------------------
2. Knowledgeable: I used to work in this area and/or I try to keep up with
   the literature but might not know the latest developments

Reviewer Confidence
-------------------
2. Medium confidence: I understand much of the paper but not all of it.

Paper summary
-------------
* The paper proposes MailoHLS, an LLM-aided strategy for HLS optimization that integrates a GNN-based structural encoder with an LLM backbone. 
* The paper demonstrates through empirical studies that general-purpose LLMs frequently generate invalid and resource-infeasible designs, while MailoHLS addresses this limitation by combining the graph-based structural representations with LLM reasoning. 
* MailoHLS achieves up to 10.2× speedup on fully unseen applications, outperforming advanced LLMs and prior approaches while narrowing the gap to the Pareto frontier.

Strengths
---------
* Very relevant system problem.
* Proposed framework achieves up to 10.2x speedup on fully unseen applications, outperforming high-end LLMs and prior approaches, which is promising.
* Experimental findings are insightful.

Weaknesses
----------
* Presentation in the Section 2 and 5 inconsistencies should be addressed.
* The computational overhead of the proposed framework, such as the training overhead of GNN and LLM backbone, is not sufficiently discussed.
* The unit of dataset partitioning is not clearly described.

How easy is the paper easy to read and understand?
--------------------------------------------------
2. Could be better: some non-trivial bits are missing or difficult to
   understand; the writing is rough in some places.

Novelty of Work
---------------
2. Moderate novelty: meaningful extension or combination of ideas

Evaluation methodology
----------------------
2. Acceptable: Reasonable methodology and results to demonstrate value of
   idea

Pre-Rebuttal Overall merit
--------------------------
4. Major revision -- Fair work but with concerns. Satisfactory revision is
   critical for acceptance in the program

Comments for authors
--------------------
This paper addresses a timely problem in the EDA field: automatically selecting high-quality HLS pragma configurations under latency/resource trade-offs. The experiments are insightful, and the experimental results are also encouraging, with the proposed framework achieving up to 10.2× speedup on fully unseen applications, outperforming several general-purpose LLMs and prior HLS-optimization baselines.

However, several aspects of the work would benefit from further clarification:

* The paper contains several presentation and methodological inconsistencies that should be addressed. For instance, in Figure 1 the background section introduces encoder–decoder architectures, while the work primarily focuses on decoder-only LLMs. This mismatch may confuse readers regarding the actual architecture used in MailoHLS. In addition, in Section 5.3.2 the authors state that "cross-attention is applied periodically every n layers (e.g., every 2–4 layers)" yet the subsequent discussion and experiments evaluate n = 8 and 16. A thorough clarification of the paper would significantly improve its overall readability.

* Although the proposed framework demonstrates promising efficiency improvements, the computational overhead of MailoHLS is not sufficiently discussed. In particular, the paper does not adequately analyze the training overhead of GNN and LLM backbone, and the execution time compared with prior HLS optimization methods. Reporting the end-to-end optimization overhead and inference cost would strengthen the practicality claims of the framework.

* In Figures 7 and 8, the resource-oriented setting appears to require more resources than the corresponding Pareto reference, which may weaken the claim that the adapter retains effective control over resource usage. The paper briefly attributes this behavior to the non-smooth nature of HLS resource metrics, but this explanation remains mostly high-level. A deeper analysis of concrete failure cases, or a discussion of why resource control is harder than latency optimization across three optimization objectives, would strengthen this paper.

* For Figure 10, although the paper explains that MailoHLS adopts a conservative feasibility-oriented strategy, CollectiveHLS and Claude Haiku 4.5 achieve much higher speedups while still remaining within FPGA resource limits. The paper would be stronger with a more comprehensive analysis of the compute-bound case, including directive-level comparisons among MailoHLS, CollectiveHLS, and Claude Haiku 4.5, and a clearer explanation of whether the performance gap comes from the optimization objectives, the training distribution, or the structural embedding.

Questions/Issues for the authors to address in the rebuttal/revision
--------------------------------------------------------------------
1. How did you partition the samples in the dataset, based on design point or kernel?

2. What is the training process of GNN models? Was the GNN trained on all kernel families, including MachSuite?

3. In Figure 8, the resource-optimized adapter achieves 0.992x speedup with higher resource usage on Viterbi. Can you elaborate on this behavior?
