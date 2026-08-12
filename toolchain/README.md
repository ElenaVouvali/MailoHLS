# MailoHLS Polygeist toolchain lock

`lock.env` is the machine-readable source/configuration lock and
`SHA256SUMS` covers every patch byte-for-byte. The immutable upstream base is
Polygeist `77c04bb2a7a2406ca9480bcc9e729b07d2c8d077`; its recorded
`llvm-project` gitlink is `26eb4285b56edd8c897642078d91f16ff0fd3472`.

The patch order is part of the lock:

1. preserve printed source locations;
2. adapt compatible full-rank call operands;
3. share LLVM-as-MemRef registration with the Python bindings;
4. reject illegal MoveWhile conversions and add its regression;
5. preserve full-rank implicit-reference arrays;
6. use declaration source locations for allocas and test included types;
7. ingest exact MailoHLS actions and preserve marked array allocations;
8. prevent type-punned Mem2Reg forwarding;
9. build and test `_mailohls_analysis` as an MLIR Python extension.

The validated host tools were CMake 4.3.4, Ninja 1.13.0, Python 3.11.15,
pybind11 2.10.3, Git 2.25.1, and GCC 9.4.0. The semantic build requirements
are Python 3.11 with pybind11 2.10.3, Ninja, a C++17 compiler, a Release build
with `LLVM_ENABLE_ASSERTIONS=ON`, projects `clang;mlir`, host targets, and
`MLIR_ENABLE_BINDINGS_PYTHON=ON`. Optional CUDA, ROCm, and Polymer components
are disabled by the bootstrap script. Accordingly, the locked `LLVM_LIT_ARGS`
runs the complete CPU suites while filtering tests whose path contains
`CUDA/`; those require a CUDA SDK and hardware outside this toolchain contract.

Both scripts take explicit source/build paths and resolve their own repository
location, so no user home directory is embedded in active code. Bootstrap
refuses existing destinations to protect local work. Verification uses a
temporary checkout for applicability, but tests and smoke output only in the
caller-selected build/output directories.
