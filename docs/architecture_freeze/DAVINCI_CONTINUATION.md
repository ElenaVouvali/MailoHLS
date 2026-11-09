# Davinci continuation

SSH now works. The exact V4 adapter, prepared data, Memory C and pinned backbone shards were transferred and verified. Existing ML packages match Liono.29 attention/memory tests passed on Davinci. Other workloads were not interrupted.

Completed:6 production constrained V4 decodes; exact zero-gate assignment/candidate-score parity on the Viterbi reference;1 verified optimizer update with resumable checkpoint. Gate moved to -0.00019999168580397964. Frozen Stage1 runtime digest and original adapter hash remain unchanged. Teacher-forced loss0.7066142559→0.7065427303. Trainer runtime838.5422 seconds includes initial/post-update constrained selection. This is a smoke result, not convergence.

Viterbi changes PIPE/UNROLL at L6 and L8 for U200 versus ZCU104/10ns. Both kernels retain their assignments between10ns and5ns; Pathfinder also retains its assignment across devices. No physical quality improvement is established.

A fresh static-C baseline is RUNNING on Davinci V100 GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f: maximum500 updates/six hours wall, seed123, layers8/16/24/32, post-self-attention placement, frozen V4, attention LR1e-4/gate LR2e-4, batch1/accumulation8. Memory covers20/20 required kernels; constrained validation uses12 cases. No alternative arms were launched. The six-hour timeout applies only to this run's process group.

```bash
tail -f ~/MailoHLS-final-architecture-20261109/artifacts/final_architecture_20261109/static_train.log
ls ~/mailohls_runs/final_architecture_r2/static
```

Output root: /home/evouvali/mailohls_runs/final_architecture_r2/. The smoke checkpoint is smoke/checkpoint-1. Resume static training from a completed checkpoint using the same portable command and --resume, after verifying the GPU is idle. Do not launch a second training job while it runs.

Raw candidate evidence: experiments/final_architecture_20261109/results/davinci_baseline.jsonl. Machine status: davinci_execution.json. Earlier GPU-BLOCKED reports describe the pre-access snapshot; this continuation supersedes that blocker only. All architecture qualification limitations remain. No new HLS job ran and no final architecture freeze is claimed.
