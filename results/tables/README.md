# Appendix and results tables

Each CSV contains exact estimates, interval bounds, task counts, cluster counts, and bootstrap settings. Matching LaTeX files are standalone `booktabs` tabular fragments, not manuscript edits.

- `main`: first/final success and chosen-action prediction quality.
- `feedback`: paired full-feedback minus retry contrasts after a shared failed first action; novel probes are compared after one rollout.
- `feedback_decomposition`: canonical visual, frames-only, status-only, retry, and JSON conditions, with paired task-level contrasts.
- `json_controls`: primary and broad populations are explicitly separated; JSON uses an independent initial action.
- `correspondence`: Figure 4A/C are adjusted PQ differences; B is a partial Spearman correlation, with basic and richer controls listed separately.
- `transfer`: fixed novel-placement pre/post quality, gain, and gain beyond retry.
- `pq_scale_sensitivity`: the same finalized populations under Gaussian scales 75, 100, and 150 pixels. This supplementary check preserves its historical bootstrap settings (10,000 draws for mean contrasts); primary Figure 3 uses 2,000 draws. Point estimates agree at scale 150, while interval endpoints can differ slightly from Monte Carlo sampling.

Success rates are fractions; success differences are probability differences, not relative percentages. PQ is graded spatial similarity, not percentage accuracy. All intervals are 95% whole-layout bootstrap intervals. Robotics-ER 2 uses its own canonical population; task counts distinguish it from the three main models. See `../populations.json` and `../method.json` for exact membership and estimators.
