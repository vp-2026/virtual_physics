# Current-scene VQA — November 14, 2026

The final compatible-format condition evaluates all three models on the exact
66 released screenshots. There are726 questions/model:671 eligible and55
quadrant questions excluded by the pre-existing strict60-pixel divider rule.
Every planned eligible item remains in the denominator. Missing or invalid
answers receive no credit; format compliance is reported separately.

## Full-panel compatible-format results

| Model | Correct / eligible | Accuracy | 95% layout-bootstrap CI | Valid response calls | Historical image VQA |
| --- | --- | --- | --- | --- | --- |
| Gemini 3.1 Pro | 645/671 | 96.13% | [94.49, 97.61]% | 65/66 | 659/677 = 97.34% |
| GPT-5 | 629/671 | 93.74% | [91.70, 95.67]% | 66/66 | 643/677 = 94.98% |
| Qwen 3.6 Plus | 630/671 | 93.89% | [89.21, 97.33]% | 64/66 | 645/677 = 95.27% |


These are end-to-end scores, including unanswered items. A lower score caused
by truncation is not evidence that the model visually misidentified those
objects. The selected valid-response-only summaries are descriptive, not an
unbiased estimate of all-scene perception. See `sensitivity.csv`.
Format validity is a strict call-level diagnostic: an extra unrequested key
flags the call but does not discard valid answers to the scored questions.
Exact unanswered-item and entirely unanswered-call counts are recorded in
`completion_summary.json`; transport-missing calls are zero in this condition.
GPT-5 and Gemini answered all 671 eligible items. Gemini had one extra-key
format violation, with its valid scored answers retained. Qwen had two
truncated calls: 20 eligible items were unanswered and counted wrong. On its
64 format-valid scenes alone Qwen scored 630/651 = 96.77%; that selected-sample
number must not replace the full-panel 630/671 = 93.89% result.

All models used JSON-object output, low reasoning effort, maximum16384 output
tokens and seed2026; temperature0.2 for Gemini/Qwen and omitted for GPT-5.
Exact requested IDs: `openai/gpt-5`, `google/gemini-3.1-pro-preview`,
`qwen/qwen3.6-plus`. Returned IDs/providers and usage are retained per response.
This condition was frozen after interface diagnostics and explicitly approved
with a$4.90 cap. All198 calls were planned; earlier answers were not substituted.
It is a separately reported protocol condition, not a post-hoc correction of
the original historical-settings run.

## Historical-settings replication and diagnostics

The first complete run matched the actual historical records: medium reasoning,
8192 output tokens and strict JSON schema. Its198 requests completed:

| Model | Correct / eligible | End-to-end accuracy | Valid response calls |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 643/671 | 95.83% | 66/66 |
| GPT-5 | 628/671 | 93.59% | 66/66 |
| Qwen 3.6 Plus | 0/671 | 0.00% | 0/66 |


All66 Qwen responses violated the required output format. Thirty-three stopped
with a schema-shaped answer wrapper;32 exhausted the token limit with null
content and one exhausted it with partial content. The resulting0/671 is a
format-constrained end-to-end score, **not a meaningful0% perception estimate**.
A separate66-layout Qwen JSON-object/medium8192 diagnostic is preserved in
the data and summary CSVs. It changes only the output interface and does not
replace any original answer. An earlier incomplete low/2048 pilot is excluded
from all reported accuracy results and retained privately for provenance.

Local disk exhaustion interrupted the diagnostic and compatible conditions.
Saved raw API responses were recovered without answer selection. Only calls
without a saved response were retried; all returned malformed answers were
kept. The original logs, recovery manifests, request bodies and raw JSON
remain in the local audit. Public copies redact local-path error metadata only;
raw model answers and scoring inputs are unchanged.

## Interpretation and asset checks

Historical/current screenshots differ on38/66 layouts;49 question records have
scene-dependent key or option changes. Historical eligibility was677 rather
than671. Thus old/new accuracy differences are descriptive: scene versions,
eligibility, provider versions and (for the compatible condition) output/
reasoning settings differ. `unchanged_question_comparison.csv` additionally
restricts to identically worded/keyed items eligible in both versions.

Two released screenshots disagree with their world files about green-goal
location: `BackUp_8` and `Balance_8`. Screenshot-derived VQA-only annotations
were frozen before inference. No image, simulator world or historical model
output was changed. The64-layout exclusion sensitivity is in `sensitivity.csv`.
The other64 layouts pass the red/green center audit within3 pixels. This checks
those landmarks, not every object's complete geometry.
Excluding the two image/world-mismatch layouts gives GPT-5 610/651 = 93.70%,
Gemini 625/651 = 96.01%, and Qwen 611/651 = 93.86%. These are different exclusions
from Qwen's two truncated calls, despite the coincidentally equal denominators.

High coarse static-VQA accuracy supports a narrower claim: substantial endpoint
prediction errors can coexist with successful recognition of many scene
properties. It does not rule out coordinate-localization, representation,
interface, or task-specific perception errors. PQ is a graded spatial score,
not multiple-choice accuracy; these percentages must not be subtracted from PQ.

Historical Qwen JSON remains **565/677=83.46%**. The rebuttal's93.5% is unsupported.

## Reproduction, uncertainty and cost

`current_scene_vqa/summary/` contains overall, category, historical-comparison,
unchanged-item, exclusion-sensitivity and completion CSV/JSON files. Intervals
use10,000 whole-layout percentile bootstrap draws, seed20261114. Each layout
contains its questions; individual questions are not treated as independent
sampling units. No inferential test isolates a screenshot-version effect.

`repo/vqa/scripts/score_current_scene_vqa.py` verifies frozen hashes and selects
the first returned answer. `summarize_current_scene_vqa.py` independently checks
every question's credit against its saved letter/key before aggregation.
Reproduction commands and the public frozen-run archive are documented in
`repo/vqa/current_scene_2026-11-14/README.md`.

Reported API charges including the excluded pilot: **$6.5966**.
Conservative total including reservations for missing usage: **$7.3451**,
below the$10 combined ceiling. No paid inference beyond current-scene VQA was
performed. Costs do not imply a second scientific observation for retries.
