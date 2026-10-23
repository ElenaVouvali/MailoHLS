# Measured evaluation results

This directory collects the existing experiment JSONL files moved from the
repository root. Their contents are preserved. `SALT_experiments.jsonl` is the
canonical measured input for the objective-response plotter.

From the repository root:

```bash
python plots/objective_response/objective_response.py
python scripts/select_nondominated_salt.py --output /tmp/salt_candidates.jsonl
```

Other tables contain implementation, target/generalization, comparator, and
cross-attention experiments. Historical reports remain in `asplos_results/`.
`src/objective_response.py` contains mock example values; use the measured-data
plotter above for figures shown to reviewers or supervisors.
