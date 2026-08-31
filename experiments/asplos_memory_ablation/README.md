# ASPLOS fixed-checkpoint structural-memory ablation

This experiment compares the frozen ADP models on the same 18 post-freeze
Kalman contexts: Stage 1, Stage 2 with zero memory, Stage 2 with within-kernel
deranged action slots, and Stage 2 with aligned MLIR memory. It changes neither
the backbone, LoRA adapter, Stage-2 cross-attention weights, decoding policy,
nor target requests.

Run from the repository root:

```bash
GPU=0 bash experiments/asplos_memory_ablation/run_adp_kalman.sh
```

The context selector is QoR-blind: within every device/clock group it selects
three evenly spaced quantiles of mean requested budget fraction. QoR fields do
not participate in selection. `summary.csv` reports inference-level metrics;
`synthesis_queue.jsonl` deduplicates identical configurations for the required
end-to-end Vitis evaluation. Do not make a QoR claim from directive accuracy
alone.
