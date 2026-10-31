# V2.2 target-aware GNN training audit

## State and scope

This audit is for the isolated branch `research/v22-gnn-resumable-20261030`,
based on `research/v22-target-aware-gnn-training` commit
`43431de1e77efaf42692db3cfece7773541bb37c`. The resumability and preflight
repairs are committed as `53f293c8b1755252db061f8ba7077ae551048971`.
The original checkout and its staged Hotspot changes were not edited.

The audit covers the V2.2 training inputs, target/candidate pathways, loss and
selection behavior, restart safety, and A2 feasibility. It does not train
Stage 1/2, change production V1 models, regenerate compiler graphs, or run HLS.

## Authenticated data correspondence

The fresh audit was produced with `experiments.v22_gnn_training.audit` against
the extracted release at `/tmp/salt-v22-training-release`.

| Item | Verified result |
|---|---:|
| Authenticated V2.2 RC3 snapshots | 63 |
| Matched GNΩSIS kernels | 55 |
| Reconstructed GNΩSIS rows | 150,026 |
| Seed-123 split kernels | 50 train / 4 validation / 1 test |
| Eligible measured configurations | 104,147 |
| Eligible rows | 88,466 train / 8,976 validation / 6,705 test |
| Source/action/graph chains checked | 55 |
| Compiler-unresolved action rows retained with source values and masks | 8,258 |
| External target-context rows admitted to training | 0 |

The audit authenticated source, `kernel_info.txt`, local headers, APL mapping,
raw and preprocessed CSVs, graph identities, archive members, registered
actions, target part/clock, and complete source-legal action tuples. The exact
per-kernel identities, hashes, contexts, action inventories, unresolved
actions and exclusions are in the machine-readable audit:

`/tmp/v22-gnn-resumable-preflight-20261030/data_audit.json`

The derived per-kernel training disposition (all 55 kernels, measured/eligible/
excluded design-point counts and exclusion reasons) is:

`/tmp/v22-gnn-resumable-preflight-20261030/kernel_disposition.csv`

Its eligible-count column sums to 104,147. Rows from a kernel with an
unmeasured APL action are excluded in full; otherwise the per-kernel eligible
count is the audit's validated complete-assignment count. Invalid joint tuples
are separately recorded as source-aware TCAD exclusions.

The dataset SHA256 is
`afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`; the
seed-123 split SHA256 is
`d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`. The
authenticated graph archive SHA256 is
`feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.

### Exclusions and unresolved actions

The audit does not repair candidate labels:

| Exclusion | Count | Disposition |
|---|---:|---|
| Kernel has a registered source action with no measured APL/CSV column | 45,080 | Excluded. Preprocessing's disabled default is not a measured assignment. |
| Source-aware complete tuple violates TCAD at L3 | 300 | Excluded as serialized; not repaired. |
| Source-aware complete tuple violates TCAD at L4 | 339 | Excluded as serialized; not repaired. |
| Source-aware complete tuple violates TCAD at L6 | 160 | Excluded as serialized; not repaired. |

The 8,258 rows with a source-measured but compiler-unresolved registered action
remain eligible. Their legal measured candidate values remain inputs, while
the absolute action slot remains unresolved/masked and receives no fabricated
compiler embedding. The test partition is only Kalman (6,705 configurations),
so test results are a narrow single-kernel held-out-family evaluation.

## Input and leakage review

The static graph tensor is built from the authenticated V2.2 graph, the
hardware overlay, requested clock, and resource budget. The numerical arm
removes hardware-resource messages; both arms retain the same numerical
device/clock/budget context and the same compiler features. Candidate values
and `candidate_demand_tensor()` enter only the prediction head. The candidate
demand cache is keyed by kernel, profile, clock, and configuration hash.

The candidate-independent 128D action memory does not consume candidate
directives, candidate demand, measured QoR, selected winners, Pareto labels or
test membership. Dataset and graph files remain unchanged. Physical target
labels are derived from measured latency and resource utilization; changing a
budget changes feasibility context only and does not change those labels.

Both arms reset seed 123 before initialization and sampling. They use the same
candidate group sampling, objective sampling and budget sampling sequence.
Validation selection is performed only on the validation split; test
evaluation is deferred until both arm checkpoints have been selected.

## Objective and ranking audit

The measured resource fractions are converted using the authenticated physical
device capacities. QoR regression uses standardized `log1p` latency in ms and
physical BRAM_18K/DSP/FF/LUT counts, with normalization fitted on training rows
only. Predictions are inverted with `expm1` and clamped to nonnegative values;
they are not capped at device capacity.

Measured objectives follow the existing MailoHLS convention: latency in ms;
area as the arithmetic mean of the four physical-capacity utilization
fractions; ADP as latency-ms times that area proxy, ranked only within the same
kernel/device/clock context. Pairwise comparisons use measured-feasible
candidates only, omit relative differences of 5% or less, and never cross
device/clock contexts. Feasibility is derived from measured physical resource
counts, not predictions.

The initial one-update diagnostic on eight actual `machsuite-gemm-blocked`,
U200, 3.33 ns candidates exposed a scale issue: a single global standard
deviation of raw objective units is inappropriate for within-context ranking
and made ranking gradients vanishingly small. The trainer now uses the median
within-training-context standard deviation of `log1p(objective)` as a positive
score scale, fitted from training records only. The ranking coefficient stays
0.1; pair selection and architecture are unchanged. This monotonic scale
preserves the measured ordering and uses no validation/test statistics.

The follow-up diagnostic uses the same eight candidates, all three objectives
and both arms. Its per-objective pair counts, loss/gradient norms, candidate
and candidate-demand gradients, clipping flags, and hardware relation
embedding gradients are stored in:

`/tmp/v22-gnn-resumable-preflight-20261030/a2_memory_profile_context_rank.json`

All measured losses/gradients were finite; candidate and demand gradients were
nonzero in both arms. Hardware relation embedding gradients were nonzero only
in the hardware-message arm, as intended. The ranking-to-QoR gradient ratios
were small at this randomly initialized one-update checkpoint; they are
recorded as a limitation to inspect during training, not interpreted as
predictive evidence. No coefficient was further tuned from validation or test.

## Restart and failure safety

Training state is atomically written via temporary file, fsync, and rename.
It contains model/optimizer state, completed update and arm, best validation
state, early-stopping counter, training history, and Python/NumPy/CPU/CUDA RNG
states. Resume requires matching data, split, graph, schema, profile, source
hash, environment and training-protocol fingerprint. A detached launcher lock
prevents concurrent resume; its own direct child trainer is the only allowed
live-lock case. Selected arm checkpoints are written atomically and are never
silently replaced. Progress is appended and fsynced periodically; exceptions
write a traceback and failed status. SIGINT/SIGTERM are handled at optimizer
update boundaries. SIGKILL is not recoverable beyond the last atomic
checkpoint.

The deterministic interruption/resume regression test compares the resumed
optimizer/RNG sequence to an uninterrupted sequence. A paired GPU smoke also
tests actual process interruption and resume before the full run.

## Tests and validation

Using the isolated Python 3.10 environment with CUDA PyTorch 2.6.0+cu124, PyG
2.6.1 and SciPy 1.15.3:

- `tests/test_v22_gnn_resume.py` and `tests/test_v22_gnn_training_contracts.py`:
  15 passed after the rank-scale and launcher-resume-lock fixes.
- Final full relevant suite: **115 passed** in 181.12 s after the ranking,
  resume-lock, and busy-GPU-output handling changes.
- A first attempt with the Python 3.8 `.hls-llm` environment produced 21 setup
  errors because existing graph-release code uses `Path.is_relative_to()`;
  this environment mismatch was avoided by using an isolated Python 3.10 venv,
  not by modifying frozen graph code.
- The standalone authenticated audit passed with explicit documented
  exclusions.
- A2 small/medium/AES forward/backward profiles completed without OOM.
- A real measured-candidate update completed on both arms for all three
  objectives with finite losses and gradients.

Exact final test totals and smoke outcomes are updated after their execution.
