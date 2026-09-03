# SALT-HLS / MailoHLS ASPLOS evaluation-workload kit

## Frozen workload set

This kit records the proposed nine-workload application-level evaluation set:

1. Serrano Kalman — frozen held-out family
2. GAN — external application
3. Jacobi-2D — external PolyBench/HLSyn stencil
4. TRMM-opt — external PolyBench/HLSyn triangular linear algebra
5. CHStone AES — external security/bitwise workload
6. CHStone JPEG — external media/decode workload
7. Rosetta 3D Rendering — external realistic rendering application
8. SYR2K — external PolyBench/HLSyn dense linear algebra
9. Covariance — external PolyBench/HLSyn statistics/reduction

Do not replace a workload after observing SALT-HLS QoR. Report failures.

## What is already prepared

`prepared_polybench/` contains the four previously prepared PolyBench/HLSyn
MailoHLS inputs. For the locked headline set, use Jacobi-2D, TRMM-opt, SYR2K and
Covariance.

`existing_repo_workloads/` documents that Kalman and GAN should be used from the
current MailoHLS branch, avoiding a stale duplicate.

## CHStone AES/JPEG and Rosetta 3D Rendering

`new_external3/` is a deterministic preparation package pinned to exact upstream
commits and Git blob hashes. The upstream programs are multi-file (CHStone) and
Rosetta uses the Xilinx `ap_int.h`; therefore the package does not pretend a
hand-edited flattened fork is canonical. Run:

```bash
cd new_external3
python materialize_upstream.py
python validate_package.py
```

This creates `kernel.c`/`kernel.cpp`, `kernel_info.txt`, and
`kernel_placeholders.*` for each workload, using QoR-blind source-derived Lk
actions. Existing HLS optimization pragmas are stripped; interface pragmas are
retained. The source is verified against the pinned upstream Git blob hashes.

Then generate the MLIR graphs in the normal MailoHLS MLIR/Polygeist environment:

```bash
MAILOHLS_ROOT=/path/to/MailoHLS bash prepare_mlir_graphs.sh
```

For Stage 2/3, export structural memory with **the exact final GNN checkpoint,
feature schema, normalization, and structural-memory contract used by the frozen
Stage-2 model**. Never copy another kernel's `.memory.pt`.

## Aggregation rule for the paper

Report per-workload results, a kernel-macro aggregate, and a provenance/domain-macro
aggregate. This prevents the four PolyBench kernels from receiving four times the
weight of CHStone or Rosetta when making the heterogeneity/generalization claim.
