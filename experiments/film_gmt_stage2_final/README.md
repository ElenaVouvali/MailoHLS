# Final selected FiLM + historical GMT Stage-2

Read [the preparation report](../../docs/film_gmt_stage2_final.md) before launching.
The expanded 49+4 cohort is **NOT READY** under the current frozen V4 domains and
historical compiler action inventories. The explicitly named original FiLM 31+2
fallback is qualified; no missing-kernel intersection is accepted.

`bank_contract.py` audits all recovered sources and exposes strict pre-aggregation
complete-state joins. Missing frozen domains stop bank creation; they are not
filled with no-directive values. `prepare_memory.py` requires the exact original
bank, audits whole-cohort historical graphs, performs one 600-update train-only
GMT fit, and exports. The imported historical helpers are pinned from `da24e50`;
their old subset-selection CLI is never used by this final path.

`pin_cached_anchors.py` performs a **new** frozen export using the byte-exact
original historical cache vectors, verifies the pinned cache manifest and graph /
checkpoint / normalization correspondence, and retains reconstruction differences.
`assets.py` authenticates the complete memory/source chain and makes a fresh,
byte-identical four-file copy of the selected FiLM supervision bank.

`train.py` derives from the supplied v2 trainer. It authenticates selected Arm B
step 600 and its frozen parent independently of any newer bank contract. It keeps
four zero-gated attention layers, the gate-opening curriculum and action-normalized
loss. The schedule covers every bank context and both contexts of every selected
preference group within 450 updates; the old 112 / steps//3 caps cannot hide gaps.
CPU token checks reject source truncation. Native hardware gating enforces FP16 on
V100S. Backward diagnostics preserve checkpointing; ON/OFF probes use inference.

`commands.sh` defaults to audit and requires explicit `COHORT`. The qualified
fallback name is `original-film-31+2`; `expanded-recovery` cannot accept the legacy
bank. `ALLOW_EXPLICIT_SUBSET=1` is rejected. Model training starts only if the user
explicitly supplies `MODE=train`; preparation did not invoke it. New outputs must
be fresh, and resume requires the saved optimizer/protocol identity.

Binary datasets/checkpoints/memories remain in the recorded local artifact paths;
their SHA manifests and all validation outcomes are committed under `docs/provenance`.
