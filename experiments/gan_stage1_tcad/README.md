# Selected Stage1 GAN inference: TCAD request grid

This isolated experiment runs the authenticated V4 multi-reference latency Stage1
adapter on the GAN requests in active `sections/05__Detailed_Design.tex`, lines518–549,
`fig:target-conditioning`, and its referenced heatmap. The grid contains18 requests:
ZCU104/U200 ×3.33/5/10ns ×20/50/100% of each resource. Reference: ZCU10410ns/full.

Objective is **PARETO_LATENCY**, because that is the selected adapter's contract.
The historical figure's exact objective/checkpoint attribution is not assumed.
Its published latency/utilization values are not assigned to these new predictions.
GAN remains an external descriptive evaluation; no training/checkpoint selection.

The backbone is the pinned DeepSeek-Coder6.7B revision and exact V4 tokenizer,
added-token mapping, prompt, objective and checkpoint-domain registry. The absent
GAN registry is extended only by the normal decoder's public source/action path,
with its unchanged checkpoint-compatible legacy proposal policy. Raw predictions
are not silently repaired. A separate post-inference source/tool-v2 validation
identifies any inadmissible tuples. Such checks do not establish HLS feasibility.

The source snapshot is `evaluation_set/GAN/code_to_memory_outputs/meta/GAN_placeholders.cpp`,
with its11-action metadata and original application headers. After removing
pragma lines/action annotations/comments/whitespace, its program tokens match
`evaluation_set/GAN/network.cpp`. There are25 directive fields. Sources/checkpoint/
paper/requests have SHA256 receipts. No backbone or adapter is duplicated in Git.

```bash
python -m experiments.gan_stage1_tcad.run \
  --draft /path/to/TCAD.zip --out /persistent/new/prepared_run
```

Preparation makes no GPU allocation. Add `--execute` to prepare and execute locally,
with an explicit idle `--gpu-uuid`. A busy preflight fails without interfering.
For an existing sealed bundle on another host:

```bash
python -m experiments.gan_stage1_tcad.execute_prepared \
  --prepared /persistent/prepared_run \
  --checkpoint /local/stage1_quality_sft_latency_v4/multi_reference/run \
  --gpu-uuid VERIFIED_IDLE_UUID --out /persistent/new/inference_run
```

The portable path verifies checkpoint, registry, contract, requests, metadata and
header hashes before loading. One greedy constrained sample/request, mean RHS
continuation scoring, all candidate scores (`debug_topk16`). Each completed case
is flushed to a durable journal before the final output is written. Fresh output
is required; partial files are preserved rather than silently overwritten.

```bash
python -m experiments.gan_stage1_tcad.inspect \
  --prepared /persistent/prepared_run --inference /persistent/inference_run \
  --policy-package /path/to/pinned/salt_qualification_contract \
  --out /persistent/new/inspection
```

Inspection writes `DIRECTIVES.md` with every complete assignment, `directives.csv`,
`inspect.html`, per-request raw text and materialized C++ for inspection, plus
request/configuration differences and source/tool findings in `results.json`.
These are proposed directives, not measured latency/resources/timing.

**User sequence: inspect the predictions first; synthesis remains unstarted.**
Any subsequent Vitis measurements must use the authorized Coroni workflow and
preserve source/header/action/part/clock/tool/settings identities. Budget-only
views may share exact physical measurements but are not independent programs.
