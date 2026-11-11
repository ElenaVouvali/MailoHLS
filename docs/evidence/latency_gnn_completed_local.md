# Final latency-GNN controlled study

FREEZE_NOT_YET_JUSTIFIED

Selected: **preliminary**. Candidate checkpoint SHA256: `79540c9f27e4807e0c3badbd876322bbde5badd9d4a04f601b316ee75e944ee8`. No model or production registration was promoted.

The user revised checkpoint selection after the original runs: family-macro Kendall tau-b first, log2 MAE second, feasible Regret@1 third. Feasible-selection rate is a diagnostic, with no selection veto. Both original runs are preserved; continuation restores each arm's own model, optimizer, scheduler and RNG states within the original 40-total-epoch cap and patience 8. This is an explicit post hoc revision, not the originally predeclared policy. Freeze upgrades additionally require ranking support in at least two independent families and no >10% MAE regression.

The two new arms use the same corrected data, fresh seed-123 initialization, architecture, optimizer and sampling schedule. B adds only delayed within-context delta supervision. Actual early-stopping totals are reported below. The fixed four-layer edge-aware TransformerConv, width 64, JK-max, one post-directive stage and 128-D [first-layer | JK] memory are preserved. No Stage-1/Stage-2 training or architecture search was performed.

## Data reconciliation

| Split | Raw rows | Successful measured under documented policy | Recovered >=100% raw rows | Legacy unique rows | Corrected unique rows | GNN eligible unique rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 189925 | 153869 | 3215 | 130310 | 133404 | 90248 |
| val | 19390 | 17634 | 118 | 13011 | 13119 | 13119 |

Successfully measured means finite positive latency within the established <=100000 ms range, nonnegative finite resource percentages and finite positive synthesis duration. Finite above-range values (including repeated 1,000,000 values) are ambiguous without outcome records; they are excluded rather than declared failures or successful latency labels. This conservative exclusion remains a data limitation. Original percentages, including zeros and overflow, are preserved in corrected tables and row-level raw provenance. Repeated physical designs are aggregated once using the historical median policy, with every raw measurement retained. Capacity-scaled percentages are fractional estimates, never exact resource counts.

The full TCAD split has 130,310 training, 13,011 development and 6,705 locked-test configurations across 55 source-resolvable kernels. Those figures are not the eligible GNN cohort. The original frozen training-time development cohort excluded Sort-radix because L6 has no authentic APL column. The user explicitly requested final metrics on all four validation kernels, ignoring only L6 in Sort-radix; this was frozen before both final qualified arms. Final evaluation therefore covers four kernels in three independent families; the corrected training measurements remain unchanged and both final arms use the same four-kernel checkpoint-selection pools. L6 is unobserved and its candidate coordinates are masked, with no invented directive assignment. The original 9,070-candidate cohort and the amended broader evaluation are separately manifested. No locked-test labels were parsed. Exact exclusions, domains, coverage, hashes and per-context counts are in cohort_reconciliation.json and candidate_pool_manifest.json.

## Matched evaluation

ZCU104/10 ns is reported separately from all-measured-target contexts. The new models train on all measured contexts; evaluating their ZCU104/10 ns slice does not reproduce historical single-target training. All rows in each comparison table use the identical complete candidate pool, measured physical context, budgets and shared evaluation code. V1 is evaluated only on its authenticated single-target candidate intersection.

| Protocol | Checkpoint | Unique physical candidates | Tau-b | Feasible Regret@1 (ratio) | Log2 latency MAE | Feasible selection |
|---|---|---:|---:|---:|---:|
| corrected_all_contexts | preliminary | 13119 | 0.123986 | 0.054309 | 3.039891 | 0.916667 |
| corrected_all_contexts | A | 13119 | 0.194155 | 0.805151 | 2.601820 | 0.944444 |
| corrected_all_contexts | B | 13119 | 0.178470 | 0.598498 | 2.488328 | 0.944444 |
| corrected_zcu104_10ns | preliminary | 2362 | 0.081774 | 0.062556 | 2.731744 | 0.833333 |
| corrected_zcu104_10ns | A | 2362 | 0.177480 | 0.439950 | 2.030962 | 0.833333 |
| corrected_zcu104_10ns | B | 2362 | 0.147520 | 1.497039 | 1.964016 | 1.000000 |
| original_legacy_cohort_intersection | preliminary | 8976 | 0.148034 | 1.174334 | 2.862332 | 1.000000 |
| original_legacy_cohort_intersection | A | 8976 | 0.187796 | 1.218663 | 2.617302 | 0.979167 |
| original_legacy_cohort_intersection | B | 8976 | 0.172414 | 0.738854 | 2.417136 | 0.958333 |
| v1_matched_zcu104_10ns_intersection | preliminary | 2341 | 0.081774 | 1.119591 | 2.732373 | 1.000000 |
| v1_matched_zcu104_10ns_intersection | A | 2341 | 0.177480 | 1.537928 | 2.030913 | 1.000000 |
| v1_matched_zcu104_10ns_intersection | B | 2341 | 0.147520 | 1.497039 | 1.963181 | 1.000000 |
| v1_matched_zcu104_10ns_intersection | V1 | 2341 | 0.305557 | 0.046093 | 1.375131 | 1.000000 |

Regret is selected measured latency / best measured-feasible latency - 1 within the measured pool; multiply the ratio by 100 for percent regret. It is conditional on successful measured-feasible selection, with coverage reported in each JSON. Selection filters by predicted resource requirements only; measured feasibility is checked afterward. Rounded percentage equality at a capacity limit is unknown. No measured-feasible-candidate pools and undefined tau are reported separately. Timing qualification is unknown. Nominal rounded utilization does not prove exact post-route resource or timing closure. Tau-b follows the inherited measured-feasible budget-pool convention; MAE uses all measured candidates. Budget views average within contexts, contexts within kernels, kernels within families, then independent families equally.

### Family results on complete corrected pools

| Checkpoint | Family | Tau-b | Regret ratio | Log2 MAE | Feasible selection |
|---|---|---:|---:|---:|---:|
| preliminary | machsuite_sort_radix | 0.075891 | 0.139410 | 3.397057 | 1.000000 |
| preliminary | machsuite_viterbi | 0.366254 | 0.011843 | 4.430760 | 1.000000 |
| preliminary | rodinia_pathfinder | -0.070187 | 0.011674 | 1.291855 | 0.750000 |
| A | machsuite_sort_radix | 0.206873 | 0.059551 | 2.566222 | 1.000000 |
| A | machsuite_viterbi | 0.371693 | 0.935760 | 3.871175 | 1.000000 |
| A | rodinia_pathfinder | 0.003899 | 1.420144 | 1.368062 | 0.833333 |
| B | machsuite_sort_radix | 0.190583 | 0.052285 | 2.625231 | 1.000000 |
| B | machsuite_viterbi | 0.285828 | 0.936036 | 3.631690 | 1.000000 |
| B | rodinia_pathfinder | 0.059000 | 0.807174 | 1.208062 | 0.833333 |

Family bootstrap intervals in the evaluation JSONs are exploratory; three independent eligible families cannot provide reliable narrow confidence intervals. Per-context and per-kernel results are retained, so a Pathfinder outlier cannot be hidden by the macro-average.

### Published TCAD reference (not a newly replayed result)

| Representation | Published latency tau-b | Published latency Regret@1 | Log2 MAE | Feasible selection |
|---|---:|---:|---|---|
| HARP representation | 0.201 | 0.404 | Not reported | Not reported |
| Structured MLIR | 0.200 | 0.355 | Not reported | Not reported |

These publication values have their historical candidate protocol; they are not directly ranked against the new matched tables. The V1 replay authenticates the saved implementation/schema, checkpoint, cached directive vectors, scope injection and measured labels, and converts log2(ms+epsilon) predictions to log2(ms). It does not reconstruct missing Sort-radix directives.

## Execution and scientific decision

| Arm | Epochs run | Optimizer steps | Selected epoch (1-based) | Selected checkpoint SHA256 |
|---|---:|---:|---:|---|
| A | 38 | 11248 | 30 | `598857c7944d90971abf96be5f05dcfd65a54d4b177d5089ddf8a2073c69c552` |
| B | 26 | 7696 | 18 | `f1c6de1deb8ebeddffefd9053cd0acb9c600142b438526285ae6d0046703a73e` |

Earlier Davinci attempts are preserved separately. Both final matched arms ran sequentially on the Liono V100S using the same prepared environment and corrected data. GPU receipts identify process ownership. All executed code and checkpoint states, optimizer/scheduler, sampler/Torch/CUDA RNG states, selected/last checkpoints, pair counts and component gradients are retained. Kernel exposure is equal at eight context updates per eligible kernel per epoch. Device/clock contexts rotate deterministically. Complete distinct candidates are sampled once per physical batch; regression is never replicated by budget. Ranking/delta near-ties within 5% are excluded. Nonzero ranking gradients are required, and B requires nonzero delta gradients after activation. B delta coefficients are 0,0,0,0.025,0.05 for epochs 1-5; the final coefficient remains 0.05. The loss uses unstandardized log2 latency differences.

Decision checks:

```json
{
  "B_vs_A": {
    "ranking_first_priority_gain": false,
    "mae_nonsevere": true,
    "independent_family_support": {
      "machsuite_sort_radix": false,
      "machsuite_viterbi": false,
      "rodinia_pathfinder": true
    },
    "at_least_two_families_improve": false,
    "feasible_selection_rate_is_diagnostic_only": true
  },
  "candidate_vs_preliminary_all_contexts": {
    "ranking_first_priority_gain": true,
    "mae_nonsevere": true,
    "independent_family_support": {
      "machsuite_sort_radix": true,
      "machsuite_viterbi": true,
      "rodinia_pathfinder": true
    },
    "at_least_two_families_improve": true,
    "feasible_selection_rate_is_diagnostic_only": true
  },
  "candidate_vs_preliminary_zcu104_10ns": {
    "ranking_first_priority_gain": true,
    "mae_nonsevere": true,
    "independent_family_support": {
      "machsuite_sort_radix": true,
      "machsuite_viterbi": false,
      "rodinia_pathfinder": true
    },
    "at_least_two_families_improve": true,
    "feasible_selection_rate_is_diagnostic_only": true
  },
  "candidate_vs_V1_matched": {
    "ranking_first_priority_gain": false,
    "mae_nonsevere": false,
    "independent_family_support": {
      "machsuite_sort_radix": false,
      "machsuite_viterbi": true,
      "rodinia_pathfinder": false
    },
    "at_least_two_families_improve": false,
    "feasible_selection_rate_is_diagnostic_only": true
  }
}
```

## Structural representation and Stage-2 compatibility

The RC3 adapter repairs wrong bound/use-count field names and transmits authenticated SSA/type/action/root/feature categories and static memory volume. Tensor schema v3 is versioned; both final arms receive the same correction, including isolated static nodes for registered source actions absent from the RC3 graph. These source-only nodes contain authenticated action metadata and have no invented compiler bindings or dependence edges. Their memory has identity visibility only. The earlier trained v2 A attempt is preserved and excluded from final checkpoint qualification because it failed full-source XAttn routing. The authenticated compiler graph topology, compiler relations, registered source actions and GNN architecture are unchanged; only the tensor adapter adds isolated source-only slots. Structural audit distinguishes archive availability from encoded features. Compiler dependence/evidence relation types enter edge-aware messages; detailed relation metadata also accompanies memory export. Static memory contains neither candidate values, device/clock context, budgets nor measured QoR. All source action identities, active/unbound masks, exact [conv1 | JK] layout, padding and training-only kernel-balanced RMS normalization are checked. The selected bank passes the maintained loader and real XAttn forward/backward with fresh fusion weights. Historical XAttn weights are not reused.

QoR prediction improvement and feasible-ranking improvement are distinct in the four-metric tables. Valid memory export and XAttn compatibility establish representation integration only. **Downstream LLM improvement remains unverified until objective-matched Stage-2 training.** No complete Stage-2 run was launched.
