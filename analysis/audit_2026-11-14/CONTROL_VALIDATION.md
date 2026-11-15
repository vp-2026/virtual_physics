# Control audit

## Coordinate robustness

| Model | Condition | Episodes | Episode rounding agreement | Neighbor action agreement | Isolated successes | Invalid rounded | Invalid neighbors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini 3.1 Pro | full | 1657 | 1632/1657 = 98.49% | 14011/15688 = 89.31% | 13/664 = 1.96% | 30 | 714 |
| Gemini 3.1 Pro | neither | 1657 | 1638/1657 = 98.85% | 15763/17960 = 87.77% | 24/903 = 2.66% | 14 | 782 |
| Gemini 3.1 Pro | trace_status | 1657 | 1586/1657 = 95.72% | 13979/15824 = 88.34% | 27/689 = 3.92% | 142 | 815 |
| GPT-5 | full | 1531 | 1504/1531 = 98.24% | 13502/15080 = 89.54% | 24/551 = 4.36% | 16 | 1067 |
| GPT-5 | neither | 1531 | 1513/1531 = 98.82% | 15046/17128 = 87.84% | 21/776 = 2.71% | 15 | 1394 |
| GPT-5 | trace_status | 1531 | 1460/1531 = 95.36% | 13526/15408 = 87.79% | 25/591 = 4.23% | 318 | 2632 |
| Qwen 3.6 Plus | full | 268 | 265/268 = 98.88% | 2457/2824 = 87.00% | 6/165 = 3.64% | 2 | 126 |
| Qwen 3.6 Plus | neither | 268 | 267/268 = 99.63% | 3181/3616 = 87.97% | 2/259 = 0.77% | 1 | 157 |
| Qwen 3.6 Plus | trace_status | 268 | 266/268 = 99.25% | 2372/2616 = 90.67% | 3/122 = 2.46% | 1 | 17 |


The main three-arm ranges reproduce: **95.4–99.6%** rounding agreement, **87.0–90.7%** neighbor agreement and **0.8–4.4%** isolated successes. These ranges exclude the much smaller status-only/frames-only samples; including every condition gives 95.4–100%, 85.4–90.7% and 0–4.4%. Full 15-condition counts are in `results/coordinate_reproduction/results/coordinate/coordinate_summary.csv`.

The archive contains 60,775 recorded actions and 10,902 model-condition episodes across 132 layout/direction cells. It is a July snapshot: primary arms have 1,531 GPT, 1,657 Gemini and only 268 Qwen episodes each. These are not the final 1,641-task common analysis. Every saved action in this replay snapshot has a mechanical 10-pixel rounding replay; exact half-grid ties round downward. There is no feasibility projection and no choice of the best neighboring coordinate.

Jitter applies to **attempt 1 and originally successful actions**, not every subsequent failed action. It tests eight fixed offsets: (±5,0), (0,±5), (±5,±5); diagonals are 7.07 pixels in Euclidean distance. Invalid neighbors count as failures. Episode rounding preserves the saved sequence and treats invalid rounded placements as failures. There is no new model replanning after perturbation.

The runner scores canonical replay success with a two-second dwell and noncanonical success with its saved endpoint/event signatures. **720/60,775 logged original-action outcomes differ from replay-scored originals** (717 noncanonical, three canonical). The reported robustness comparison is replayed original versus replayed perturbed coordinates, not logged outcome versus perturbation. Different saved predicates/simulator versions may contribute; these disagreements were not silently relabeled. This audit re-scores cached simulations and verifies their input hashes and rounding maps. It does not establish fresh simulator repeatability or attribute every disagreement to stochasticity.

Accurate compact statement: “On an archived fixed-sequence replay sample, mechanical coordinate rounding preserved 95.4–99.6% of episode classifications; eight fixed local perturbations preserved 87.0–90.7% of tested action outcomes.”

## PQ scoring sensitivity

`results/sensitivity/results/scoring_sensitivity.csv` contains all 36 estimates and layout-bootstrap CIs for sigma 75, 100 and 150. The score is mean object-level exp(−error²/(2 sigma²)); state/exit-side mismatches get zero and matching exits get one. The primary continuous score is spatial similarity, not percent correct or confidence. Objects are averaged within an attempt/probe, then episodes receive the specified weight. It is not a role-balanced or pixel-perfect accuracy score.

Broad feedback patterns persist, but significance is not invariant. GPT's Figure 4A coefficient is 0.031 [−0.004, 0.065] at sigma75, 0.038 [0.0002, 0.074] at100, and 0.040 [0.003, 0.077] at150. Gemini's early novel benefit is 0.076 [0.004, 0.153] at75 versus 0.067 [−0.008, 0.144] at150. Do not write “all conclusions/significance unchanged.” Chosen-PQ feedback benefits for GPT and Qwen, and terminal novel gains beyond retry for GPT and Qwen on the declared 118-task set, remain positive across these scales.

## Feedback decomposition

All entries below use the same 132 canonical tasks per model; values are percentages with 95% whole-layout intervals:

| Model | Visual + status | Status only | Frames only | Retry only | JSON + status |
| --- | --- | --- | --- | --- | --- |
| GPT-5 | 72.73 [63.64, 81.06] | 75.00 [66.67, 82.58] | 71.21 [62.12, 79.55] | 68.18 [59.09, 77.27] | 78.03 [70.45, 85.61] |
| Gemini 3.1 Pro | 81.06 [73.48, 87.88] | 78.03 [69.70, 86.36] | 78.03 [69.70, 85.61] | 76.52 [68.94, 84.09] | 79.55 [72.73, 86.36] |
| Qwen 3.6 Plus | 72.73 [64.39, 81.06] | 71.97 [63.64, 80.30] | 73.48 [64.39, 82.58] | 69.70 [59.85, 78.79] | 54.55 [44.70, 63.64] |


Full versus status-only differences are −2.27, +3.03, +0.76 percentage points (GPT/Gemini/Qwen); all their intervals include zero. The full decompositions and pairwise CIs are in the reconciliation CSV. Broad JSON solve rates are 46.94%, 48.76% and 27.44% on 1,651 tasks. This is model-dependent; it does not support saying JSON uniformly helps or uniformly hurts.

## Prompt wording: missing record recovered and old discrepancy explained

The exact Qwen branch record for `canonical_0116_Prevention_10_upward` was
recovered from iCloud. Its original SHA-256 is
`78de23ae4cd96fce2823d1acc1d78b95e619dc3e87bfc96c7ef6a70656930a86`.
Accepted coordinates[300,400] match the response; object PQ values0.07485518,
0.36203800 and0 give episode PQ0.14563106. Nothing was imputed.

The complete132-pair Qwen estimate is **full−compact PQ−0.00725443
[−0.04227812,0.02776680]**; means0.45388517 and0.46113960.
Success is52/132 versus59/132; difference−0.0530303
[−0.121212,0.015152]. Intervals use10,000 whole-layout percentile resamples,
seed20261114. The prior131-pair result is superseded only in this new audit;
its frozen source remains unchanged.

The July28 scorer was recovered from the original Codex chat. It used
shared-state predictions directly. Exactly36/132 compact Qwen records predict
a different action from the executed action; their PQ scores differ. This
reproduces its compact mean0.46690147 and old difference−0.01301630 exactly.
The discrepancy is a stale prediction/action join, not merely the missing
observation. Use accepted responses and the corrected complete-panel estimate.

GPT/Gemini historical matched comparisons remain131 pairs. Their PQ
full−compact differences are+0.00923341[−0.02657592,0.04496615] and
−0.00267357[−0.04742894,0.04270027]. These full values retain archived derived-CSV
provenance. Qwen full now has125 raw re-scored records plus7 archived derived
values; all previously compared raw and derived values agree. The compact
panel contains395 accepted raw trials across models. The historical control
uses a three-second goal; the final protocol uses a two-second goal and up to
eight placements. Qwen's dedicated full/compact comparison is one-shot;
GPT/Gemini references also differ in budget. Do not generalize these to an
isolated wording effect in the final protocol, or treat a null CI as equivalence.

Sources: `results/prompt_recovery/completed_prompt_control.json`,
`results/prompt_recovery/historical_prompt_pairs_complete.csv`,
`scripts/complete_prompt_recovery.py`,
`scripts/recovered_july28_prompt_scorer_original.py`,
`reports/qwen_historical_shared_vs_accepted.json`.

## Reproduction and remaining limits

All60,775 cached action rows,10,902 episode rows and15 coordinate summary rows
reproduce exactly. All36 PQ-scale estimates/intervals,100,000-draw feedback
checks, main tables and ER2 results match the November14 audit. No new
simulation or paid prompt-control call was performed. Current-scene VQA is the
only new inference experiment; see `CURRENT_SCENE_VQA_RESULTS.md`.

Correct local output locations are `results/coordinate_reproduction/`,
`results/sensitivity/results/`, and `results/model_tables/` in this directory.
Public commands are in `repo/analysis/REPRODUCIBILITY.md` and their verification
record is `reports/public_command_validation.json`.

The release now includes the missing VQA generator and final binary Figure4
estimators. License provenance was traced to the original upstream tree and
Codex discussion; no explicit simulator/task license grant was recovered.
`repo/LICENSE_STATUS.md` records that unresolved issue. Seed2026 is supported;
no completed final-protocol three-seed experiment is supported. The separate
GitHub PR/cache cleanup remains unresolved and was not pursued, as instructed.
No manuscript, participant data or frozen experiment result was edited.
