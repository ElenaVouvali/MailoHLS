# Kalman: selected Stage-1 checkpoints under four target contexts

All 16 configurations were computed: parent, selected A, selected B with FiLM on, and the B-off control. Each has 32 source-domain-valid directive fields. These are Stage-1 predictions; no GNN structural memory is used.

The common parent is the previously trained V4 SFT adapter from which both matched continuations began. A adds a residual LoRA; B adds a residual LoRA and intermediate-layer FiLM. Selected steps are A=60 and B=600.

## Observed behavior

parent: u200_10ns_100pct changes 5 generated fields; zcu104_3p33ns_100pct changes 0 generated fields; zcu104_10ns_20pct changes 12 generated fields.

A_selected: u200_10ns_100pct changes 2 generated fields; zcu104_3p33ns_100pct changes 4 generated fields; zcu104_10ns_20pct changes 0 generated fields.

B_selected_ON: u200_10ns_100pct changes 0 generated fields; zcu104_3p33ns_100pct changes 3 generated fields; zcu104_10ns_20pct changes 0 generated fields.

zcu104_10ns_100pct: turning B FiLM off changes AUTO{_UNROLL_L11}: 2 (on) → 4 (off).

The first difference is auto{_UNROLL_L11}, with identical preceding directive tokens. Its top-two margins are 0.008418 (on) and 0.005632 (off).

u200_10ns_100pct: B on/off generates the same complete assignment.

zcu104_3p33ns_100pct: B on/off generates the same complete assignment.

zcu104_10ns_20pct: B on/off generates the same complete assignment.

Under the separately controlled common-parent reference prefix, B on/off has 0 candidate-winner changes across four contexts. Candidate mean log-probabilities shift by up to 0.050164. This fixed-prefix result and the actual-generation first-divergence result answer different questions.

Kalman is absent from the authenticated parent training JSONL and from the A/B train/validation banks. Training-set identity checks are recorded in the companion provenance files. No measured Kalman labels were used for inference or checkpoint selection.

No quality ordering of parent/A/B follows from the number of changed fields. The same configuration can be appropriate for two resource budgets, and a changed configuration can be infeasible. These checkpoints were selected at different updates; B-on/off is the control that keeps B weights fixed.

## Changes relative to ZCU104, 10 ns, full budget

| Model | Changed context | Changed generated fields | Fixed-prefix candidate-winner changes |
| --- | --- | ---: | ---: |
| parent | u200_10ns_100pct | 5 | 1 |
| parent | zcu104_3p33ns_100pct | 0 | 0 |
| parent | zcu104_10ns_20pct | 12 | 1 |
| A_selected | u200_10ns_100pct | 2 | 1 |
| A_selected | zcu104_3p33ns_100pct | 4 | 1 |
| A_selected | zcu104_10ns_20pct | 0 | 0 |
| B_selected_ON | u200_10ns_100pct | 0 | 1 |
| B_selected_ON | zcu104_3p33ns_100pct | 3 | 1 |
| B_selected_ON | zcu104_10ns_20pct | 0 | 0 |
| B_selected_OFF | u200_10ns_100pct | 1 | 1 |
| B_selected_OFF | zcu104_3p33ns_100pct | 2 | 1 |
| B_selected_OFF | zcu104_10ns_20pct | 1 | 0 |

Fixed-prefix comparisons score the same legal continuations under the same earlier directives, taken from the parent baseline prediction. They do not use measured Kalman labels. Full scores and signed margin changes are in `observations.json` and the per-case files.

## Checkpoint and FiLM comparisons

| Comparison | Context | Changed generated fields | Fixed-prefix candidate-winner changes |
| --- | --- | ---: | ---: |
| parent → A_selected | zcu104_10ns_100pct | 19 | 6 |
| parent → A_selected | u200_10ns_100pct | 15 | 6 |
| parent → A_selected | zcu104_3p33ns_100pct | 23 | 5 |
| parent → A_selected | zcu104_10ns_20pct | 11 | 5 |
| parent → B_selected_ON | zcu104_10ns_100pct | 8 | 4 |
| parent → B_selected_ON | u200_10ns_100pct | 8 | 4 |
| parent → B_selected_ON | zcu104_3p33ns_100pct | 6 | 3 |
| parent → B_selected_ON | zcu104_10ns_20pct | 14 | 3 |
| A_selected → B_selected_ON | zcu104_10ns_100pct | 20 | 6 |
| A_selected → B_selected_ON | u200_10ns_100pct | 20 | 6 |
| A_selected → B_selected_ON | zcu104_3p33ns_100pct | 20 | 6 |
| A_selected → B_selected_ON | zcu104_10ns_20pct | 20 | 6 |
| B_selected_ON → B_selected_OFF | zcu104_10ns_100pct | 1 | 0 |
| B_selected_ON → B_selected_OFF | u200_10ns_100pct | 0 | 0 |
| B_selected_ON → B_selected_OFF | zcu104_3p33ns_100pct | 0 | 0 |
| B_selected_ON → B_selected_OFF | zcu104_10ns_20pct | 0 | 0 |

A changed field does not establish a better decision. An unchanged configuration can coexist with changed candidate scores. B-on/off keeps B weights fixed and toggles its FiLM contribution; free-running effects may propagate through earlier choices, while the fixed-prefix scores isolate local ranking changes. The report also compares actual B-on/off generation scores up to the first differing value: those earlier directive tokens are identical, providing an additional matched-prefix check at the first changed decision.

**Physical QoR improvement: NOT EVALUATED.** There are no matched synthesis results for these assignments, so feasibility under the 20% resource budget, achieved timing at 3.33 ns, latency and FPGA resource usage remain unverified.

The JSON manifest records checkpoint/source identities and complete target contexts. `predicted_directives.json` contains all assignments; `directive_comparison.csv` provides a field-by-field view.
