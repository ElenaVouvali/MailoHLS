# Stage-1 quality-aware multi-reference SFT

This package defines a controlled latency-SFT comparison. Arm A selects one best feasible measured assignment; Arm B selects one complete assignment uniformly from that request's measured feasible set within 1% of best latency. The tolerance is an engineering rule, not a noise interval. No labels are combined across configurations. Both arms must use the same source-legal requests and action-normalized Stage-1 loss.

## Current status

The branch contains candidate/request preparation, observability checks, matched target-schedule export, production-trainer integration, GPU preflight, inference input export, measured-quality evaluation and checkpoint-selection utilities. This describes implemented capabilities; it does not establish that every scientific acceptance criterion has passed.

On 2026-11-05, the final-root single-reference and near-optimal multi-reference V4 latency adapters also completed matched, inference-only GEMV requests (ZCU104 and U200, both 10 ns/full budget). All four outputs passed source-domain and complete-field validation; each adapter emitted the same directive assignment for both requests. This is a source-grounded legality demonstration, not QoR evidence or evidence that either adapter adapts to targets. See the [GEMV inference record](results/gemv_v4_20261105.md) and its authenticated [machine-readable report](results/gemv_v4_20261105_report.json).

Final v4 single-reference and multi-reference latency adapters are available locally under `/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/{single_reference,multi_reference}/run`. On 2026-11-04, both adapters completed production constrained inference on the same three user-requested Kalman contexts. All six outputs passed complete-action and source-domain legality checks. The [result tables](results/kalman_v4_20261104.md) and [compact JSON record](results/kalman_v4_20261104.json) preserve exact predictions and source/checkpoint/tokenizer/domain hashes. Weights, full inference outputs and logs remain outside Git.

The canonical dataset uses the family split in `mailohls_runs/mailohls_final_family_split_s123.json`; Kalman is test-only. The reported requests used source metadata without test measurement labels and were not used for checkpoint selection. They are held-out inference probes, not a full matched validation study. No measured hardware-quality gain or model promotion is claimed from these outputs. No synthesis was performed for this inference run.

On 2026-11-05, the frozen 54-request V4 single/multi inference passes completed on the V100S. A subsequent `nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used --format=csv,noheader` on the same execution host failed with `couldn't communicate with the NVIDIA driver`; the production-baseline and GAN/Kalman re-inference passes therefore remain pending. No driver repair, package installation, process signal, or alternate-GPU transfer was attempted.

## V4 paired inference and source/tool admissibility

The production inference CLI now has explicit `--admissibility_policy source_tool_v1` and `source_tool_v2` modes. The default `legacy` mode remains the historical saved-registry decoder. These modes layer a versioned policy over (and do not rewrite) each checkpoint's candidate registry. They filter complete array-action tuples before model scoring, with prefix-aware filtering across type, factor, and dimension fields. An empty admissible domain raises an error; no post-hoc clipping or fallback is performed. Per-output diagnostics retain checkpoint candidates, candidates added from source metadata, rejected values and reasons, policy/compiler identity, source action metadata, and final tuple validation. The v2-only `dim=0` allowance is an explicit inference option; default/training directive validation remains unchanged.

The Vitis HLS 2021.1 policy rejects partial block/cyclic factors greater than the verified selected source dimension. Factor values remain counts rather than a powers-of-two/divisor whitelist: valid saved values such as factor 3 are retained, and source proposal sets include the extent (so GAN's extent 50 may use 50). Complete partitioning is constrained to at most 1,024 elements on the selected dimension under the authenticated Vitis 2021.1 runner profile. Policy V1 is frozen for the completed paired outputs. Policy V2 explicitly models AMD's `dim=0` all-dimensions semantics: partial factor must fit each selected extent; complete partition counts the product of selected extents. An explicit nonzero dimension uses only that dimension's extent. This limit is a conservative compiler-policy guard, not a device resource limit. Kalman's 262,144-element arrays are rejected for complete partitioning; small arrays remain eligible. The exact multidimensional XFORM boundary remains subject to the prepared compiler probe. See both versioned files under `policies/`.

The new paired protocol authenticates the existing frozen 54-request V4 validation cohort, its seed-123 family boundary and validation candidate pools. It does not use the historical 383-case report or Kalman test labels. Run the two V4 adapters and compatible production Stage-1 baseline with the same prompt rows and greedy settings:

```bash
PYTHONPATH=. python -m experiments.stage1_quality_sft.paired_protocol
PYTHONPATH=. python -m experiments.stage1_quality_sft.freeze_inference_manifest \
  --gpu-uuid GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
GPU_UUID=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb bash experiments/stage1_quality_sft/run_paired_v4_inference.sh
```

This emits legacy and `source_tool_v1` predictions for single-reference, multi-reference, and production latency Stage-1. Evaluate each completed JSONL without replacing historical reports:

```bash
for model in single_reference multi_reference production_latency; do
  for policy in legacy source_tool_v1; do
    PYTHONPATH=. python -m experiments.stage1_quality_sft.evaluate_paired_v4 \
      --name "${model}_${policy}" \
      --predictions "artifacts/stage1_quality_sft/v4_paired_20261105/predictions/${model}_${policy}.jsonl"
  done
done
```

The GAN and Kalman compiler-failure diagnostics use the same V100S and run legacy, policy V1, and policy V2 for both V4 adapters. They are separate held-out probes and are excluded from checkpoint selection:

```bash
GPU_UUID=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb \
  bash experiments/stage1_quality_sft/run_gan_kalman_source_tool.sh
```

For the follow-up V2 all-dimension policy after GPU access is restored, the frozen 54-request outputs are written to new `*_source_tool_v2.jsonl` files; existing policy-V1 and legacy outputs are not overwritten:

```bash
bash experiments/stage1_quality_sft/run_source_tool_v2_inference.sh
PYTHONPATH=. python -m experiments.stage1_quality_sft.evaluate_paired_v4 \
  --name single_reference_source_tool_v2 \
  --predictions artifacts/stage1_quality_sft/v4_paired_20261105/predictions/single_reference_source_tool_v2.jsonl \
  --policy experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json
```

The QoR evaluator requires exact source/device/clock/full-assignment matches. Unmatched designs retain unknown QoR; request budgets are applied to measured resources, not included in physical measurement identity. `prepare_paired_synthesis.py` freezes predictions for the predeclared six-request synthesis subset and uses the existing sealed-input runner from the prior V4 synthesis bundle; equal configurations in the same physical context are shared and attributed to all applicable arms. Inference success, compiler admissibility, HLS success, resource feasibility, and post-route timing are reported as separate evidence.

The primary cohort must exclude any directive action whose value is not directly measured or defensibly fixed by source/synthesis evidence. V4's unknown-action receipt is a warning signal, not proof that a Stage-1 target is wrong. The eligibility audit must join each action ID to original APL/CSV columns and source evidence before freezing requests.

## Intended frozen protocol

Authenticate raw dataset and split hashes; validate complete measured assignments against the source domain registry; collapse identical duplicate QoR receipts and reject conflicts; form per-source/device/clock candidate banks; apply the repository's four-resource fractional feasibility rule; construct identical requests for both arms. Record epsilon 0, 0.005 and 0.01 coverage, with 0.01 primary. Store complete accepted assignments and row identities in a checksummed manifest.

Arm A uses minimum latency with deterministic configuration-hash tie breaking. Arm B uses a precomputed exposure schedule derived from `(seed, request identity, exposure)` and uniform accepted-candidate sampling. A schedule avoids dependence on DataLoader worker RNG and can be regenerated after resume. It must be frozen before training and tested through the actual Stage-1 dataset/collator and action-normalized decision loss.

## Stage-2 boundary

The Stage-1 checkpoint must retain the existing objective-specific rank-8 LoRA, tokenizer/special-token map, source/action markers, prompt encodings, and directive-domain hash. Stage-2 may initialize from a selected Stage-1 adapter and add action-aligned V1 structural memory via trainable cross-attention. Device/clock may condition physical structural memory; objective and requested budget remain in Stage-1's prompt/decision path. Budget, objective, measured QoR, candidate labels and winners must not enter candidate-independent static memory. Experimental V4 memory is not presumed compatible with V1 cross-attention weights.

## GPU execution

Do not launch while the assigned V100S UUID is inaccessible or occupied. After CPU artifacts and tests pass, run a device preflight, then launch the same sealed protocol sequentially for the single-reference and multi-reference arms using fresh adapters from the same pinned pretrained backbone and matched seeds/settings. Never resume from a historical Stage-1 adapter. Record skipped AMP steps separately and select checkpoints using the frozen generated-decision and coverage-aware measured-quality rule, not CE alone.
