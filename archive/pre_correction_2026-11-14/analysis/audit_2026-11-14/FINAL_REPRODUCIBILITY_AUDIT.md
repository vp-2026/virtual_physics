# Final scientific and reproducibility audit — November 14, 2026

Scope: model archives, experimental controls, current-scene VQA, Figure4 and
public reproducibility. No Overleaf source, participant data or frozen
experiment output was modified. This report does not validate human sample
counts or the manuscript's claims about human prediction.

## Reconciled populations and solve rates

| Model | Primary1641 tasks,95% CI | Completed feedback1651 tasks,95% CI |
| --- | --- | --- |
| GPT-5 | 688/1641 = 41.93% [38.63, 45.34] | 689/1651 = 41.73% [38.44, 45.09] |
| Gemini 3.1 Pro | 764/1641 = 46.56% [43.48, 49.73] | 766/1651 = 46.40% [43.36, 49.57] |
| Qwen 3.6 Plus | 681/1641 = 41.50% [38.38, 44.67] | 683/1651 = 41.37% [38.20, 44.59] |


The populations differ by10 tasks whose prediction histories do not reconcile
across all three models. Rates must retain their own denominators. These are
episode-ever-success estimates with whole-layout bootstrap intervals, not
single-attempt accuracy. Experiment seed2026 is supported; three completed
final-protocol seeds are not. The task bank contains1560 noncanonical plus132
canonical goals=1692; the completed/common analysis populations are smaller.

## Figure4 verification

| Model | A: success–chosen PQ | B: richer partial Spearman | C: early success–novel PQ |
| --- | --- | --- | --- |
| GPT-5 | +0.040 [+0.003, +0.077] | -0.053 [-0.302, +0.202] | -0.030 [-0.136, +0.072] |
| Gemini 3.1 Pro | +0.100 [+0.075, +0.127] | +0.038 [-0.235, +0.329] | -0.130 [-0.296, +0.045] |
| Qwen 3.6 Plus | +0.047 [+0.020, +0.072] | +0.061 [-0.211, +0.327] | -0.109 [-0.296, +0.086] |


A uses1641 tasks and exact-goal/attempt controls with equal episode weights.
B/C use118 canonical tasks,65 layouts and53 direction pairs. Richer B controls
layout, direction, initial chosen/novel prediction and feedback exposure.
C compares novel prediction at the same one-rollout checkpoint, adjusting
baseline novel PQ. All model observations, means, point estimates, intervals
and sample sizes match the independent audit; see
`reports/figure4_verification.json`. The PDF is pure vector, one page, with zero
raster image objects. The latest source already contained the requested B
values; this is verified regeneration, not an invented numerical change.

**The unchanged human prediction layer is illustrative, not measured human
data.** Preserving the requested existing design does not validate that layer
as an empirical result. Do not report its scenario intervals as participant CIs.

## Feedback controls

| Model | Later success | Next chosen PQ | Early novel PQ |
| --- | --- | --- | --- |
| GPT-5 | +0.064 [+0.036, +0.091]; n=1336 | +0.073 [+0.049, +0.100]; n=1336 | +0.174 [+0.108, +0.242]; n=61 |
| Gemini 3.1 Pro | +0.040 [+0.019, +0.062]; n=1178 | +0.011 [-0.005, +0.027]; n=1178 | +0.067 [-0.009, +0.145]; n=52 |
| Qwen 3.6 Plus | +0.018 [+0.000, +0.036]; n=1305 | +0.047 [+0.029, +0.067]; n=1305 | +0.099 [+0.031, +0.169]; n=63 |


These are full-feedback minus retry contrasts after the same first failure;
early novel probes use a separate matched early checkpoint. Intervals here
use100,000 whole-layout resamples, seed20261114. Historical2000-draw intervals
are retained with their provenance; small endpoint differences are Monte Carlo
precision, not changed observations. Qwen's later-success CI reaches zero.
Full/status-only/frames-only/retry/JSON canonical comparisons use132 tasks;
broad JSON tables use1651. JSON supplies32 observable coordinate/state
snapshots plus outcome and has a separately generated first action. Earlier
action/prediction text stays in history; old rollout media do not. Success
stopping means terminal comparisons have unequal feedback exposure.

Feedback benefits are supported for several components, with model/component
variation. The evidence does not support blanket claims of absent replanning
or absent transfer. Raw prediction correlations do not establish shared
capacity beyond difficulty/exposure; adjusted novel-related associations remain
uncertain rather than proven absent or negative.

## Historical controls and Robotics-ER2

The missing Qwen historical prompt record was recovered from iCloud and its
original checksum verified. Complete132-pair full−compact PQ is−0.00725443
[−0.04227812,0.02776680]. The old−0.01301630 is reproduced only by the stale
shared-state scorer:36 compact predictions refer to the wrong executed action.
This historical three-second/one-shot comparison is not a final-protocol
wording ablation. See `CONTROL_VALIDATION.md` and `results/prompt_recovery/`.

Coordinate replay re-scoring reproduces60,775 action rows and10,902 episodes.
Primary-arm episode counts are1531GPT,1657Gemini and268Qwen per arm; it is an
earlier cache, not the final common bank. Rounding agreement95.4–99.6%; fixed
neighbor agreement87.0–90.7%. Diagonal offsets are7.07px; jitter covers initial
and successful actions.720 logged/replayed original outcomes disagree and
remain documented. All36 sigma75/100/150 analyses reproduce; some CIs cross
zero as scale changes, so no blanket significance-robustness claim is warranted.

Robotics-ER2 solves99/132. Exact-goal/attempt-controlled success–chosen-PQ
associations reproduce: visual +0.182 [+0.119, +0.242];
JSON +0.249 [+0.175, +0.323]. These are distinct from novel
transfer. Historical125-task pre/post PQ is0.544419→0.582078, raw gain
+0.038 [-0.020, +0.096]; the separate127-task
own-clean result is0.546520→0.582074, raw gain
+0.036 [-0.021, +0.093]. Both gain intervals span zero.
Population membership and all alternative net-control estimates are saved in
`results/model_tables/results/er2_analysis.json`.

## Release repairs and remaining limitations

Tested repairs include the historical VQA generation dependency/original66
worlds; final Figure4 exact-goal FE and partial-rank implementations; main,
ER2, PQ-scale, prompt and cached-coordinate reproduction; current-VQA execution,
scoring and summarization. Commands are in `repo/analysis/REPRODUCIBILITY.md`;
verification is in `reports/public_command_validation.json`. Tests compare
independently reconstructed model data, not merely hard-coded expected values.

Remaining limitations are explicit: upstream simulator/task licensing is
unresolved; two current screenshot/world goal locations disagree; the Figure4
human layer is illustrative; historical GPT/Gemini prompt reference values and
seven Qwen full values retain derived-CSV provenance; cached replays are not
fresh simulator repeatability tests. Current VQA format/truncation outcomes and
protocol differences are reported separately in `CURRENT_SCENE_VQA_RESULTS.md`.
GitHub PR/cache cleanup remains unresolved and was not pursued, as instructed.
Do not claim complete historical erasure. No participant-level file is added.

Release commit and file-manifest checks are recorded separately after publication.
