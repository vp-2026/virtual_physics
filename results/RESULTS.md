# Finalized model results

The benchmark contains 1,692 goals on 66 layouts with two tool directions. The reported common populations contain **1641 main tasks**, **1651 broad-control tasks**, and **116 matched transfer tasks**. Exact task IDs are in `populations.json`. Robotics-ER 2 is a separate canonical extension. All intervals below are 95% layout-cluster bootstrap intervals. They describe variation across layouts for the recorded runs, not variation across repeated model seeds.

## Interaction

| Model | First action success (%) | Final episode success (%) | Chosen PQ, first | Chosen PQ, final | Novel PQ, pre | Novel PQ, post |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5 | 18.83 [15.79, 22.16] | 41.80 [38.53, 45.16] | 0.462 [0.426, 0.497] | 0.525 [0.499, 0.551] | 0.478 [0.420, 0.534] | 0.594 [0.538, 0.649] |
| Gemini 3.1 Pro | 28.28 [25.52, 31.09] | 46.50 [43.33, 49.75] | 0.435 [0.389, 0.479] | 0.453 [0.410, 0.496] | 0.452 [0.384, 0.518] | 0.497 [0.433, 0.556] |
| Qwen 3.6 Plus | 20.29 [17.36, 23.44] | 41.62 [38.54, 44.71] | 0.435 [0.399, 0.470] | 0.487 [0.458, 0.517] | 0.546 [0.492, 0.598] | 0.618 [0.566, 0.668] |
| Gemini Robotics-ER 2 | 47.73 [37.88, 57.58] | 75.76 [68.18, 83.33] | 0.513 [0.467, 0.557] | 0.557 [0.515, 0.598] | 0.549 [0.490, 0.605] | 0.582 [0.529, 0.633] |

Matched transfer eligibility also requires that frozen probes remain untried and that all required queries are available. The following tasks are omitted from the common transfer population only:

- `canonical_0013_BackUp_8_downward`: Qwen 3.6 Plus: full:provider_error_terminal_post, full:missing_score.
- `canonical_0063_Falling_1_downward`: Qwen 3.6 Plus: neither:provider_error_terminal_post, neither:missing_score.

Initial/final slopes are descriptive: actions can change, retries occur, and visible-success branches stop. Novel pre/post predictions concern the same fixed intervention.

## Feedback beyond retry

Only episodes sharing the same failed first action enter these comparisons. Early novel probes are compared after exactly one rollout.

| Model | Later success difference | Next chosen-PQ difference | Early novel-PQ difference |
|---|---:|---:|---:|
| GPT-5 | 0.060 [0.032, 0.087]; n=1332 | 0.073 [0.049, 0.100]; n=1332 | 0.177 [0.114, 0.251]; n=60 |
| Gemini 3.1 Pro | 0.040 [0.020, 0.062]; n=1177 | 0.010 [-0.006, 0.026]; n=1177 | 0.060 [-0.016, 0.139]; n=52 |
| Qwen 3.6 Plus | 0.021 [0.005, 0.039]; n=1308 | 0.045 [0.028, 0.064]; n=1308 | 0.099 [0.032, 0.168]; n=63 |
| Gemini Robotics-ER 2 | 0.072 [-0.029, 0.172]; n=69 | -0.015 [-0.055, 0.024]; n=69 | 0.031 [-0.038, 0.103]; n=66 |

## Correspondence

| Model | Action success / chosen PQ (adjusted difference) | Chosen / novel PQ (richer partial Spearman) | First-action success / early novel PQ (adjusted difference) |
|---|---:|---:|---:|
| GPT-5 | 0.044 [0.008, 0.082] | -0.125 [-0.353, 0.107] | -0.034 [-0.130, 0.067] |
| Gemini 3.1 Pro | 0.106 [0.077, 0.137] | -0.003 [-0.282, 0.293] | -0.116 [-0.275, 0.049] |
| Qwen 3.6 Plus | 0.044 [0.018, 0.070] | 0.083 [-0.183, 0.345] | -0.109 [-0.309, 0.089] |

Robotics-ER 2: 100/132 solved; Figure 4A estimator 0.180 [0.118, 0.237]. Its own valid novel-probe population contains 127 tasks.

A controls exact goal and attempt with equal episode weights. B controls layout, direction, initial chosen prediction, baseline novel prediction and feedback exposure. C controls baseline probe quality, layout and direction at the common one-rollout checkpoint. These are conditional associations, not causal effects of success.

## Corrected-scene VQA

| Model | Correct / eligible | Accuracy | 95% layout-bootstrap CI |
|---|---:|---:|---:|
| GPT-5 | 629/670 | 93.88% | [91.84, 95.79]% |
| Gemini 3.1 Pro | 644/670 | 96.12% | [94.49, 97.60]% |
| Qwen 3.6 Plus | 630/670 | 94.03% | [89.37, 97.47]% |

The same frozen compatible-format control is used: JSON-object output, low reasoning effort, 16,384-token cap, seed 2026, and temperature 0.2 where supported. Nine calls replace the three corrected scenes; 189 previously returned calls are preserved. Missing/invalid answers receive no credit; the original 60-pixel divider rule controls quadrant eligibility.

## Tables and figures

Vector Figures 3 and 4 are in `figures/`. Every plotted estimate and interval is recorded in `figures/plotted_values.json`. Machine-readable appendix tables and standalone LaTeX tabular fragments are in `tables/`; they require the `booktabs` package. Structured-state JSON controls supply observable object positions/states rather than rollout images and have an independent initial action.

## Reporting statement

“We corrected inconsistent initial visual states in three layouts and reran their affected model episodes and fixed prediction probes. Reported model results combine these corrected observations with unchanged records for all other tasks.”

Human participant-level data remain private. Human estimates and hypothetical human curves are not included in this model-only release.
