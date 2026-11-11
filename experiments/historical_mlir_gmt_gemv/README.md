# Historical GEMV GMT experiment

Use the existing Liono research environment. The base Miniforge Python 3.13
does not contain `networkx`, PyTorch or the other experiment dependencies.

Run only the unit tests and interface self-test, without starting training:

```bash
cd /home/elvouvali/MailoHLS/worktrees/historical-mlir-gmt-gemv
bash run_historical_gmt.sh --test-only
```

The launcher defaults to `/home/elvouvali/.hls-llm/bin/python`
(Python 3.8, PyTorch 2.3.0+cu121, NetworkX 3.1). An explicitly supplied `PYTHON`
override is respected. No packages are installed or changed in base.

The direct unit-test command is:

```bash
/home/elvouvali/.hls-llm/bin/python -m unittest discover \
  -s experiments/historical_mlir_gmt_gemv \
  -p 'test_historical_gmt_slots.py' -v
```

The tests use small fixtures to check routing, masking, permutation invariance
and gradients. Passing them does not establish trained retrieval, downstream
LLM usefulness or physical QoR improvement. The full launcher without
`--test-only` also prepares data and starts the experiment; it is a separate
operation from this dependency/test fix.
