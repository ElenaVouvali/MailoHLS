# MailoHLS six-branch finalization

**SIX_BRANCH_CONSOLIDATION_COMPLETE**

## Final GitHub state

Default branch: `main`. Exactly six remote branches, zero open PRs and zero
Releases were verified after cutover. PR #3 merged through GitHub's normal
workflow; PR #2 was automatically marked merged because its commits were included.
No new PR, Release or tag was created for this finalization.

| Branch | Verified code/consolidation head | Role |
|---|---|---|
| `main` | `304d4c909206979aef3fd7f66770c46b71444cc7` | Finalized framework; final report commit follows the validated framework merge shown here |
| `mlir-target` | `bf82c648273a5cd5eb30363059a3809e7b3979db` | Immutable historical MLIR baseline |
| `llvm-ir` | `0b5581b2b22becf1d20f476e22e92ea3fd533a64` | Immutable historical LLVM baseline |
| `research/stage1-refinement` | `ef4bdea18342ee27df9b075e476f307b8cfcb8c7` | Stage-1 recovery, supervision and measured-quality work |
| `research/gnn-refinement` | `958c99b9fab2ee29b2b61f863b5654c5fc1fe409` | Versioned graph/tensor and resumable GNN research |
| `research/refinement-mechanism` | `49a32d07fa2630e570447aae08d0a22de5daa044` | Independent refinement/search development over stable main |

`304d4c9` is the exact validated framework merge. The final main HEAD also
includes this documentation/disposition commit; obtain its exact SHA with
`git log -1 --format=%H -- docs/REPOSITORY_CONSOLIDATION_REPORT.md` or the main
commit view. Final live refs are also recorded outside Git in the backup's
`final-remote-heads.json`. No runtime behavior changes in the report commit.

## Production and research boundaries

Main integrates the reviewed maintenance tree, tokenizer-ID correctness fix,
stable graph/refinement interfaces, deduplication and verified artifact recovery.
`FINAL_ARTIFACTS.json`, selected adapters, V1 GNN/memory contracts, raw Data,
source/APL mappings, family splits, frozen RC3 identities, FPGA capacities and
objective definitions are unchanged. V2.2 remains experimental; no checkpoint is
promoted and no QoR/generalization improvement is claimed.

Stage-1 consolidates original heads `72b42ae`, `f577a1a` and `0b592a9`: the
decision-loss module, full-data controls, recovery, measured-quality evaluation,
configuration/results and regression tests. Canonical Kalman data remains main's
version. Earlier changes are an explicit patch plus input/variant SHA256 manifest
under `experiments/stage1_recovery/dataset_variants/kalman-registration-v1`; they
are not silently applied or relabeled as verified measurements. Historical
incomplete runs remain incomplete and no new training was launched.

GNN consolidates overlay/readiness, action demand, semantic tensorization,
coverage repair, resumable training, Hotspot and complete catalog histories.
The current encoder has explicit V4 schema identity while retaining the two-layer
architecture, 128D action slots and static/candidate separation. Original V3
preflight/pilot reports remain historical; incompatible resume is rejected. No
V4 GPU training or target-generalization result is claimed. Frozen compiler
semantics are unchanged; quasi-affine/dependence uncertainty remains explicit.

Refinement keeps stable `refinement`, `refinement_search` and `v21_refinement`
functionality on main and starts independent development with a focused README
scope. No algorithm or duplicate HLS utility was added just to distinguish it.

Each research branch starts from finalized main `304d4c9`. Code was selected by
file/content review; provenance-only merges then retained original heads as
parents without changing the selected tree (tree identity was asserted). This
preserves earlier algorithms, configurations, graph catalogs and evidence in
public reachable history. [The disposition manifest](../configs/branch-preservation.json)
records every original branch/head and its destination. Seventeen redundant
remote refs were retired only after exact-head, ancestry, backup and PR checks.
Local branches/worktrees were not deleted or switched.

## Protected history and active local work

| Branch | Before | After |
|---|---|---|
| `mlir-target` | `bf82c648273a5cd5eb30363059a3809e7b3979db` | identical |
| `llvm-ir` | `0b5581b2b22becf1d20f476e22e92ea3fd533a64` | identical |

The original `/home/elvouvali/MailoHLS` stays on local `72b42ae`, with its 464
staged paths intact. Staged/unstaged patches and porcelain status were compared
byte-for-byte after work; no output directory, model weight or process was
modified. Process visibility was limited to the execution namespace; no claim
that external jobs were idle was made. Existing old worktrees may finish running
locally. When idle, preserve additional changes and transfer them deliberately
into fresh canonical research worktrees; do not push/recreate retired names.

## Artifact recovery without Releases

Both former Releases were retired only after all seven assets (five archives,
checksums and registry metadata) were preserved durably and their authoritative
SHA256 verified against the published digests. Git-only recovery then passed in
a separate clone: **1,514 original evidence paths**, 186 GEXF graphs, 179 CSVs
(689,286 rows), 431 JSON files, and **2,218 catalog checksums**. The ordinary
catalog unpack plus compiler/overlay consumer also passed.

Normal reproduction uses `scripts/artifacts.py restore-git`: source Git object
IDs, original sealed member SHA256 identities, sizes and executable modes are
checked; existing conflicts and unsafe links are refused. Original compressed
archives remain offline for exact container-byte verification. The catalog
generator is retained, but runtime/provenance receipts need not be byte-identical
when rebuilt; exact published payload recovery uses its original Git tree.
Raw measurements and frozen compiler archives remain tracked. Local weights and
HLS tools are still separately supplied, exactly as before. See
[reproducibility](reproducibility.md) for actual commands and limitations.

## Simplification measurements

| Metric | Initial live main / GitHub | Final |
|---|---:|---:|
| Remote branches | 20 | 6 |
| Open PRs | 2 | 0 |
| Releases | 2 | 0 |
| Tracked files | 3,696 | 2,278 |
| Tracked bytes | 565,985,079 | 262,318,574 |
| Exact duplicate src implementation bodies | 12 | 0 |
| Duplicate blob groups | 472 | 311 |
| Repeated paths | 1,140 | 584 |
| Top-level directories | 31 | 32 |

The five primary entry points are README, architecture, reproduction, experiments
and development. Specialized schema/evidence guides remain; the earlier
maintenance audit is now a short pointer to its immutable original snapshot.
Twelve legacy delegates identify one canonical implementation per exact copy.
Unique historical implementations remain recoverable. Earlier verified artifact
migration removed 304,472,276 indexed bytes; this is not a Git-history shrink or
a claim of deleting unique research capabilities.

## Executed validation

| Scope | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Main full CPU suite | 482 | 21 | 0 |
| Stage-1 branch LLM/contract suite | 151 | 1 | 0 |
| GNN branch overlay/tensor/demand/resume suite | 116 | 0 | 0 |
| Refinement branch suites | 105 | 5 | 0 |

Skipped checks require unavailable historical/external fixtures or CUDA. CPU
tests use the existing compatible Python/PyTorch/PyG environment and explicitly
hide CUDA. No dependency upgrades, model training or HLS synthesis were performed.
Main syntax and relevant relative documentation links were checked. All research
branches retain canonical Data, selected contracts, graph releases and profiles.
No unfinished Stage-1/GNN changes were merged into production.

## Backup and recovery

Durable location: `/home/elvouvali/MailoHLS-preservation/six-branches-20261031/`.
Bundle: `all-refs.bundle` (3.1 GB), SHA256:
`d889fa76567572c5fe238e4f8bc914063693fbbdbb2d88f28bd9258049fb841f`.

`git bundle verify` passed; a separate mirror clone recovered every original
remote head exactly. Another clone recovered all scientific payloads without
Releases or invisible runtime caches. Original staged patch application passed
`git apply --check` in a separate original-HEAD worktree. `SHA256.json`,
`worktree-SHA256.json`, `remote-heads.txt`, `tags.txt`, `prs.json`,
`retirement-plan.json`, `RECOVERY.txt`, per-worktree patches/untracked source,
original archives and Release metadata are preserved there. Ignored live model
weights are not in a Git bundle; they remain untouched at their original paths.

To recover, verify the bundle checksum, clone it into a **fresh** mirror directory
with `git clone --mirror /path/all-refs.bundle /fresh/recovery.git`, and inspect
its original refs. For uncommitted work, use the recorded original HEAD, apply
`staged.patch` with `git apply --index`, then a nonempty `unstaged.patch`, then
restore the saved untracked files. Do this in a new recovery worktree, never over
the original dirty checkout. `RECOVERY.txt` records the paths and distinctions.

## Final state

**SIX_BRANCH_CONSOLIDATION_COMPLETE**. Main is authoritative, the two historical
heads are unchanged, three independent research branches are published, and no
redundant remote branch, open PR or Release remains. There is no GitHub permission
blocker. Continuing experiments and integration of mature improvements use the
simple workflow in [development](development.md), without automatic merges of
unfinished research or a new branch/Release for each experiment.

The research heads above also include a normal merge of the finalized main
documentation/disposition update. A path-level check proved that only the five
report/layout/disposition files changed; tested research code and data are
identical. No additional algorithm, training or checkpoint change occurred.
