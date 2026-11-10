# Execution status

**FREEZE_NOT_YET_JUSTIFIED — final A is training on Liono; final B is pending.**

The active controller is configured to run A (corrected baseline) and B (the identical model plus delayed within-context latency-delta supervision) sequentially on Liono's V100S. A started when the V100S was idle. B will start after A completes and the device is available. It never terminates another user's process or shares an occupied GPU.

The earlier GPU-block snapshot is retained in `receipts/liono_gpu_status.json`. The later successful idle-at-launch receipt and actual optimizer progress are in `receipts/liono_execution_progress.json`. Availability is checked again immediately before CUDA model allocation. Other workloads can subsequently appear; no other process is stopped.

An earlier Davinci A attempt genuinely trained for 11 epochs / 3,256 optimizer steps, but failed the stronger full-source-action XAttn routing qualification. It remains preserved as an unqualified result. Additional Davinci v3 progress is also preserved, and is excluded from the final Liono comparison following the user's explicit host instruction. No final B metrics or trained-checkpoint claims are made.

The shared v3 repair represents the five missing registered array actions in each of two Hotspot kernels as isolated source-only nodes. No compiler binding or dependence is invented. The original RC3 archive and fixed four-layer GNN architecture remain unchanged. A real maintained XAttn forward/backward probe passed for all registered source actions; that probe uses preserved weights and is not a new training run.

The final validation cohort is frozen before both Liono arms: sort-radix (L6 explicitly ignored), Viterbi, Pathfinder baseline and Pathfinder doublebuffer. It contains 13,119 observed configurations across four kernels / three families. Sort-radix comparisons concern ten authenticated action assignments; its eleventh directive is not reconstructed.

Corrected training contains 90,248 eligible unique configurations from 37 kernels / 17 families. Raw-table recovery retains 3,215 over-capacity training rows and 118 development rows before aggregation and mapping checks. Original raw percentages, zeros, measured latency and provenance are preserved. Locked-test labels remain unused.

The final four-metric comparison, both qualified training receipts, checkpoint recommendation and selected memory bank remain pending actual execution. Existing models and production registrations are unchanged. Downstream LLM improvement remains unverified.
