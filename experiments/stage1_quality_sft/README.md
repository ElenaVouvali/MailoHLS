# Stage-1 quality-aware multi-reference SFT

This package defines a controlled latency-SFT comparison. Arm A selects one best feasible measured assignment; Arm B selects one complete assignment uniformly from that request's measured feasible set within 1% of best latency. The tolerance is an engineering rule, not a noise interval. No labels are combined across configurations. Both arms must use the same source-legal requests and action-normalized Stage-1 loss.

## Current status

The branch contains candidate/request preparation, observability checks, matched target-schedule export, production-trainer integration, GPU preflight, inference input export, measured-quality evaluation and checkpoint-selection utilities. This describes implemented capabilities; it does not establish that every scientific acceptance criterion has passed.

On 2026-11-05, the final-root single-reference and near-optimal multi-reference V4 latency adapters also completed matched, inference-only GEMV requests (ZCU104 and U200, both 10 ns/full budget). All four outputs passed source-domain and complete-field validation; each adapter emitted the same directive assignment for both requests. This is a source-grounded legality demonstration, not QoR evidence or evidence that either adapter adapts to targets. See the [GEMV inference record](results/gemv_v4_20261105.md) and its authenticated [machine-readable report](results/gemv_v4_20261105_report.json).

Final v4 single-reference and multi-reference latency adapters are available locally under `/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/{single_reference,multi_reference}/run`. On 2026-11-04, both adapters completed production constrained inference on the same three user-requested Kalman contexts. All six outputs passed complete-action and source-domain legality checks. The [result tables](results/kalman_v4_20261104.md) and [compact JSON record](results/kalman_v4_20261104.json) preserve exact predictions and source/checkpoint/tokenizer/domain hashes. Weights, full inference outputs and logs remain outside Git.

The canonical dataset uses the family split in `mailohls_runs/mailohls_final_family_split_s123.json`; Kalman is test-only. The reported requests used source metadata without test measurement labels and were not used for checkpoint selection. They are held-out inference probes, not a full matched validation study. No measured hardware-quality gain or model promotion is claimed from these outputs. No synthesis was performed for this inference run.

CUDA and the idle V100S were accessible outside the restricted sandbox. Earlier in-sandbox CUDA/NVML failures did not establish a host driver fault.

The primary cohort must exclude any directive action whose value is not directly measured or defensibly fixed by source/synthesis evidence. V4's unknown-action receipt is a warning signal, not proof that a Stage-1 target is wrong. The eligibility audit must join each action ID to original APL/CSV columns and source evidence before freezing requests.

## Intended frozen protocol

Authenticate raw dataset and split hashes; validate complete measured assignments against the source domain registry; collapse identical duplicate QoR receipts and reject conflicts; form per-source/device/clock candidate banks; apply the repository's four-resource fractional feasibility rule; construct identical requests for both arms. Record epsilon 0, 0.005 and 0.01 coverage, with 0.01 primary. Store complete accepted assignments and row identities in a checksummed manifest.

Arm A uses minimum latency with deterministic configuration-hash tie breaking. Arm B uses a precomputed exposure schedule derived from `(seed, request identity, exposure)` and uniform accepted-candidate sampling. A schedule avoids dependence on DataLoader worker RNG and can be regenerated after resume. It must be frozen before training and tested through the actual Stage-1 dataset/collator and action-normalized decision loss.

## Stage-2 boundary

The Stage-1 checkpoint must retain the existing objective-specific rank-8 LoRA, tokenizer/special-token map, source/action markers, prompt encodings, and directive-domain hash. Stage-2 may initialize from a selected Stage-1 adapter and add action-aligned V1 structural memory via trainable cross-attention. Device/clock may condition physical structural memory; objective and requested budget remain in Stage-1's prompt/decision path. Budget, objective, measured QoR, candidate labels and winners must not enter candidate-independent static memory. Experimental V4 memory is not presumed compatible with V1 cross-attention weights.

## GPU execution

Do not launch while the assigned V100S UUID is inaccessible or occupied. After CPU artifacts and tests pass, run a device preflight, then launch the same sealed protocol sequentially for the single-reference and multi-reference arms using fresh adapters from the same pinned pretrained backbone and matched seeds/settings. Never resume from a historical Stage-1 adapter. Record skipped AMP steps separately and select checkpoints using the frozen generated-decision and coverage-aware measured-quality rule, not CE alone.
