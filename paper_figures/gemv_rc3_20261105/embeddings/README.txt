GEMV structural embedding inspection (CPU)

These are actual outputs of the trained corrected_v3_full37 compiler, hardware and target_film encoders on the separately generated 32x32 RC3 GEMV graph at 10 ns, for ZCU104 and U200. GEMV is outside the frozen training cohort. This is inference-only: no directive candidates, QoR measurements, training examples or performance claims were added.

The original script import error is fixed by setting PYTHONPATH=/tmp/MailoHLS-gnn-refinement when invoking a script located under the home directory.

The inference input cache is retained here as gemv_rc3_embedding_input.pt; its source/provenance is in standalone_inference_provenance.json. The original training cache has no GEMV entry. embedding_report.json records the exact checkpoint and original input cache hashes.

The film_effect images subtract two independently trained arms/checkpoints; they do not isolate the causal effect of the FiLM gate. Cross-arm latent-coordinate differences are descriptive. Target-difference images compare devices within the same trained model.
