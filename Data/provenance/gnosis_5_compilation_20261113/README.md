# Five-kernel compiler-input investigation

This receipt extends, and never rewrites, the action-state recovery at commit
`3053f674f88a3562796e96d9b9cecd6be3ec09fd`. See
[`docs/gnosis_17_kernel_recovery.md`](../../../docs/gnosis_17_kernel_recovery.md).

`correspondence.json` binds the original pinned GNOSIS source/header blobs, the
unchanged canonical inputs, the original SQLite-state sidecars and the compiled
evidence. `kernels/*/measured_source/` retains the exact original bundles.
Licensing follows the pinned upstream materials; the parent recovery retains
the upstream licence. Installed vendor headers are not redistributed here.

`frontend_receipt.json` records the Vitis 2021.1 compiler/disassembler hashes,
installed header hashes, all 42 successful frontend commands, input/output hashes
and flags. The retained frontend log and extracted command are corroboration
from coroni, **not** original GNOSIS per-measurement logs. Those logs/complete
historical environments are unavailable. This is a bounded compiler-input
correspondence for the reconstructed Vitis HLS environment, not a claim of
byte-identical source, equivalent host executables, replayed QoR, or achieved timing.

Compressed LLVM modules preserve all compiler IR except for the two explicitly
permitted normalizations during comparison: file identity lines, and the exact
SPCL top-symbol correspondence. Preprocessed evidence retains repository source
bodies rather than installed proprietary headers. Every differing repository
source/header byte is retained in `source_changes.patch` and the original bundles.
No include, type, instruction, HLS annotation, or directive is removed from the
LLVM comparison. Canonical KNN fails this comparison and is never promoted.

All five original bundles have newly derived Polygeist v7 GEXF, MLIR and native
analysis reports. Graph receipts bind compiler, extension, generator, original
source, action inventory, flags and transitive header hashes. All source actions
match by exact compiler location, with no action fallback. KNN uses the genuine
AP **simulation** implementation with 8 x 64-bit storage for each 512-bit word;
it is not native Vitis i512 IR. Four KNN array scopes lack resolved memory accesses;
unknown memory effects remain conservative. These are new representations, not
replacements for RC3 graphs or historical GNN/GMT numeric memories.

The separate measured-source KNN dataset references the unchanged raw CSV and
sidecar by hash; it does not duplicate or alter QoR. Use only
`Preprocessing.gnosis_compilation_recovery.complete_original_rows`. Every row
names its source version, bundle, graph and `gnosis-measured-source-8888db17-sqlite-states-v1`
contract. Its unchanged family membership is `train`. The canonical join rejects it.

Reproduction requires the pinned upstream inputs and the licensed installed
compiler/headers. Local untracked working output is retained at
`artifacts/gnosis_5_compilation_20261113/`. Run
`experiments/gnosis_recovery/compare_five_compilation.py` on paired exact input
bundles; generate graphs using each recorded graph command/environment; then
seal a **fresh** output tree with `record_five_recovery.py`. The sealer refuses to
overwrite this receipt. `python -m Preprocessing.gnosis_compilation_recovery --verify`
rechecks the sealed evidence and every joined measurement, without the vendor tools.
