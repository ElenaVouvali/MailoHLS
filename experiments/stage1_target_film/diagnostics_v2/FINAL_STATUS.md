# Final preserved status — stopped at user request

The user stopped further diagnostic time on2026-11-10. **No remaining evaluation or retraining is queued.** Historical weights, original18-request predictions and partial diagnostic artifacts are preserved. No new HLS calls or optimizer updates occurred in diagnostics v2.

## Completed

- Value-only causal FiLM mask correction;11 actual-tokenizer tests passed. Original mask had the right i-1 offset but also modulated newline prediction. Native suffix scoring still includes newline; domains/head/vocabulary remain unchanged.
- Real B48 adapter checks: both parent and residual active, parent tensor hash verified, model parameters frozen. Removing the residual changes logits (max0.21875 on the tested prefix).
-54-pair B48 OFF/value-ON/legacy-ON development ablation: all three28/54 correct (51.8519%), family macro58.3333%, no pair-order or recorded target-score changes. Viterbi14/18; Pathfinder14/36.
- Gate tanh=-1.8586086e-6. Median intended relative correction2.0165e-7; median actual relative hidden-state update4.6515e-9. Under the requested FP16 survival formula, median retained correction norm1.8169%, median unchanged elements99.9756%;968/2019 corrected-mask position observations vanish entirely. These observations include auxiliary cache probes. No nonfinites. Some vocabulary logits change (max0.0078125), while the scored target log-probabilities do not.
- Five-pair training-preparation precision control: max same-completion score difference from native0.0022230; FiLM's max score effect1.54972e-6; no pair-order flips. This is one small Viterbi control, not a convergence or end-task benefit result.
- Real cached-versus-full first-RHS logit check differs by max0.0336914. **Exact cached parity is not established.** The production constrained candidate decoder uses full-prefix scoring.
- Broader non-test data audit and source/domain-qualified clock/budget candidate banks. No locked test or GAN data enters training-bank construction.715 legal training clock-reversal views represent501 unique pairs across5 kernels/families;20 validation views/pairs occur in Viterbi. None is near-optimal in both contexts. Budget26 training switch views across6 families, no supported switch in the audited validation uniform-budget grid. Existing exact Coroni Viterbi exception: two correlated pairs at46 BRAM18K (7.37%), outside that grid. Resource proxies and assumed uncertainty intervals remain clearly labeled.
- Restored canonical measurements cannot silently replace the frozen bank: six restored source cohorts have3–4 canonical action fields absent from the frozen V4 registry. No registry/default was changed.

## Canceled / unavailable conclusions

Corrected natural GAN ON completed **10/18**; full OFF and matched-legacy natural runs completed **0/18**.1049 legal candidate traces were preserved, including a partial next request. In those traces, ON/OFF and value/legacy mean-score changes and native head-replay errors are exactly0. This **does not establish** unchanged decisions for all18 requests. `PARTIAL_GAN.json` contains the completed partial directives; `CANCELLATION.json` records the stopped processes.

Exact per-stage sampling-loss attribution beyond the completed near-optimal counterfactual remains incomplete. Do not relabel missing measurements as zero. The historical sampler replay is audit-only, not an optimization bank.

## Decision

**EXPLORATORY_RETRAIN_ONLY; no retraining performed.** Preserve V4 as the conservative parent; A32 and B48 are experimental. Original A training stopped after58 successful updates; B completed64. The existing A/B comparison is descriptive, not a final matched-budget superiority result. B-ON/OFF is the valid held-residual inference intervention, and its complete development result shows no preference benefit in native precision.

No measured improvement in generated hardware feasibility, latency, timing or generalization is established. FiLM is not rejected universally. Stronger labels/native-precision selection and independent applicable switch coverage would be needed to justify renewed work.

Original full GAN experiment remains complete and immutable: A32 four configurations, B48 five;9/18 request assignments differ. That is directive diversity, not measured QoR. See `../results_20261110/`.

## Artifacts and reproduction

`ARTIFACT_MANIFEST.json` records persistent local paths, byte sizes and SHA256 for original checkpoints, completed diagnostics, candidate banks and partial journals. Large weights/raw tensors remain outside Git. `development_completed.json`, `DEVELOPMENT_PAIRS.json`, `PRECISION_CONTROL.json`, `TRAINING_PRECISION_CONTROL_PAIRS.json`, `FILM_NUMERICS.json`, and data-bank manifests contain the compact results.

Current inference defaults to corrected `--mask-policy value`. Explicit `--mask-policy legacy` reproduces the previous masking policy; `--film-mode off` disables FiLM on the **same checkpoint**, not on Model A. Fresh output and trace paths are required. Counterfactual cache reuse checks checkpoint/request/GPU identities and records the trace hash.

No automatic launch is enabled by reading these reports. The stopped jobs require a deliberate new execution command and fresh outputs; there is no background continuation left running.

## Preserved partial historical comparison

One of the10 completed corrected-ON requests differs from the original B48 result: ZCU104/10ns/full. All preserved same-prefix ON/OFF and value/legacy candidate scores are identical. The historical run used V100S while the corrected partial run used A2; the difference is not attributed to the mask correction or FiLM. Eight corrected requests and all full natural OFF/legacy outputs were not executed. See PARTIAL_GAN.json.
