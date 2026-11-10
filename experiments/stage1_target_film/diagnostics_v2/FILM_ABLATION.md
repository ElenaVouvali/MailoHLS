> Final snapshot: remaining diagnostics stopped by user request. Completed measurements and explicit limits are in FINAL_STATUS.md.

# B48 held fixed: completed development ablation

Same B48 residual LoRA and frozen parent for every condition.54 held-out pairs,2 independent families. Each complete assignment uses its own causal prefix. Action-normalized nonforced value scores exclude formatting; these are not the native per-candidate mean suffix scores.

|Condition|Correct pairs|Micro accuracy|Family-macro accuracy|
|---|---:|---:|---:|
|FiLM OFF|28/54|51.8519%|58.3333%|
|FiLM ON, corrected value mask|28/54|51.8519%|58.3333%|
|FiLM ON, historical suffix mask|28/54|51.8519%|58.3333%|

Viterbi14/18; Pathfinder14/36 for every condition. All recorded preferred and rejected scores and all pair margins are exactly unchanged across the three conditions. Zero pair ordering changes. Mean and median induced margin changes are0; preferred action-normalized NLL changes are0. This is a no-inference-benefit observation on this cohort, not a physical QoR result.

FiLM nevertheless changes some vocabulary logits: max absolute change0.0078125 over the selected RHS positions' vocabulary logits. That did not change the selected target log probabilities in this evaluation. Median intended structural ratio is2.0165e-7. FP16 retains only about1.8% of its norm at the median;968/2019 corrected-mask position observations are exactly zero after FP16 addition, no nonfinites.2019 includes auxiliary real-cache observations; it is not2019 independent actions/programs.

The native GAN evaluation was canceled at the user request after10/18 corrected ON requests. Full OFF and matched-legacy natural decodes were not run. They must finish before claiming whether the delimiter correction changes the historical18-request directives. Existing A/B diversity (4 vs5 configurations) is a training-path comparison and does not isolate FiLM. No synthesis was run.

## Preserved partial historical comparison

One of the10 completed corrected-ON requests differs from the original B48 result: ZCU104/10ns/full. All preserved same-prefix ON/OFF and value/legacy candidate scores are identical. The historical run used V100S while the corrected partial run used A2; the difference is not attributed to the mask correction or FiLM. Eight corrected requests and all full natural OFF/legacy outputs were not executed. See PARTIAL_GAN.json.
