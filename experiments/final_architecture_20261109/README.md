# Revision2 architecture investigation

Completed CPU evidence:96 real cache comparisons;3664 reauthenticated selected measurements; corrected ADP replay;97442 non-test physical-record target audit;56 focused tests. GPU/convergence/physical qualification remain blocked or not run.

Start with [architecture decisions](../../docs/architecture_freeze/ARCHITECTURE_DECISIONS.md), [V4b re-audit](../../docs/architecture_freeze/V4B_IMPLEMENTATION_REAUDIT.md) and [Davinci reproduction](../../docs/architecture_freeze/REPRODUCIBILITY.md).

`python -m experiments.final_architecture_20261109.run --help` provides guarded CPU/baseline entry points; dry-run is default. New training/final synthesis are explicitly gated, not implemented as an automatic sweep.
