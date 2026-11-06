# Hardware input and embedding audit

## Prompt route

The canonical schema supplies learned device tokens, selected-clock tokens and the `<CLOCK>` response prefix, objective token, per-device numerical capacities/available-resource counts/percentages, source/action markers, then the autoregressive directive prefix. All 18 grid requests use the same supported clock menu `[3.33, 5, 10]` and vary the selected specified clock. The resource budget changes the numeric available counts and percentages. Full prompt strings and token IDs are frozen under `prompts/`; the 20% request is a separate historical row.

For example, the V4 prompt tokenizes `BRAM_18K`, `DSP`, `FF`, and `LUT` numeric counts as ordinary tokenizer pieces; the values are not dedicated learned hardware special tokens. `input_tokenization_audit.json` records the exact token IDs/spans for every numeric count and percentage. The device and clock markers are dedicated added tokens, and the fixed selected-clock response prefix is already part of the canonical inference path.

## Special-token rows

The V4 checkpoint has 140 trainable special-token input rows. PEFT’s `trainable_tokens_original` stores the exact resized-base initialization; `trainable_tokens_delta` in this PEFT version is the full learned row, not an additive delta tensor. The report therefore compares final rows with their preserved initialization. All 140 have trainable parameters, though many had zero observed update. These rows were added to the tokenizer after the pretrained vocabulary was loaded, so “initial” means resize-time initialization, not an original pretrained semantic vector.

| Token | Initial→trained cosine | Relative row update L2 | Interpretation |
|---|---:|---:|---|
| `<DEV=XCZU7EV_FFVC1156_2E>` | 0.8158 | 0.781 | learned input row; geometric relation is not physical device similarity |
| `<DEV=XCU200_FSGD2104_2E>` | 0.3137 | 2.899 | learned input row; geometric relation is not physical device similarity |
| `<CLK=10NS>` | 0.1639 | 6.274 | learned input row; geometric relation is not physical device similarity |
| `<CLK=5NS>` | 0.2088 | 4.273 | learned input row; geometric relation is not physical device similarity |
| `<CLK=3P33NS>` | 0.9920 | 0.127 | learned input row; geometric relation is not physical device similarity |
| `<OBJ=PARETO_LATENCY>` | 0.5641 | 1.454 | learned input row; geometric relation is not physical device similarity |

The ZCU104 device token has cosine `0.8158` to its resize-time row; XCU200 is `0.3137`. The 10 ns and 5 ns clock rows move substantially, while the 3.33 ns row changes little in this adapter. Those are observations about learned token weights only. The learned row vectors and pairwise cosine matrices are saved in `special_token_embeddings.npz` and `hardware_token_cosines.json`.

## Does target information reach choices?

Yes, as an observed representational effect: every matched device/clock/budget contrast changes the selected decision vectors, with median relative movement reported in `STAGE1_32_LAYER_ANALYSIS.md`. In the controlled score comparisons all legal RHS candidates remain identical. Several candidate margins change even where the winner remains stable. The 10 ns/100% device pair had zero assignment changes; the budget contrast produced two L5 winner changes in XCU200/50%. No result demonstrates a QoR improvement or that budget feasibility was binding.

The all-layer activation patch places the controlled R1 post-attention vector into a receiver at the corresponding directive-token position. Four sampled pairs changed sign, with exactly unchanged same-state placebo margins. This supports decision relevance of selected late/intermediate activations, while leaving mediation across all layers and physical optimization unproven.
