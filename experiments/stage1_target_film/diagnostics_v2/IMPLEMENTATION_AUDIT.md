# Implementation audit — completed and pending checks

Baseline code `e485c0a3f1038908dd53ea91593f8fcd00d581db`; original A32/B48 GAN files are preserved.

The original mask correctly starts at `base_len-1`, the causal predictor of the first RHS token. Its extent was wrong: it included delimiter-token predictions because it marked the whole `rhs + newline` suffix. The corrected `alignment.causal_kinds` uses the original tokenizer's value-span verifier and rejects merged value/delimiter tokens. It never assumes a one-token newline.11 real-tokenizer cases pass for all five directive kinds, single/multi-token numbers, modes and newline boundaries.

Training's `film_kinds` also marked delimiter positions, but those logits have no action-SFT or conditional-ranking loss. FiLM is head-only, so a delimiter's changed output hidden state cannot feed later Transformer states. The correction aligns inference with the positions that actually receive training loss; no weights or domains changed. The native mean suffix score still includes newline, intentionally retained.

Real B48 forward checks confirm active `mailohls_base` and `device_adapt`, no trainable model parameters, and the saved parent tensor hash `97686da3a3b1e74a9288d901a5b537c6061860fa0ce1d6d0ac9ebc01638e9c91`. The existing PEFT/token-wrapper mechanism remains. Disabling residual LoRA changed logits by up to0.21875 on the first tested causal prefix. Parent deltas were restored and rehashed after that intervention.

The real full-prefix versus one-position cached check had a vocabulary-wide max logit difference0.03369140625. This is **not exact cached parity**. It is a one-position diagnostic on NF4/FP16; target-specific score/margin sensitivity still needs qualification. The production candidate decoder uses full-prefix scoring, not this cached path. Tensor-only cache tests are not substituted for this measurement.

The trainer prepares k-bit training, upcasting nonquantized parameters, while inference uses native NF4/FP16 loading. A five-pair no-training preparation-precision control is running to isolate that difference. It does not change the deployed precision or retrain weights.

Full54-pair OFF/value/legacy development evaluation completed. Corrected versus historical GAN outputs and fixed-prefix native candidate effects remain pending their explicit run. No new optimizer update or synthesis occurred.
