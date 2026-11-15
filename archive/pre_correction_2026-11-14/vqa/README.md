# VQA protocol

## Static scene

In the historical battery, the model receives its archived initial scene image or observable scene
JSON. No action, motion, goal outcome, or hidden physical parameter is shown.
Each layout has 11 authored questions covering shape, colored-object count,
movability, red/green spatial relations, coarse location, containment, and an
adaptive relation between clearly named movable objects.

There are 66 unique initial layouts. Static calls are deduplicated across
upward and downward tool mechanisms because the initial scene is identical.
The primary score excludes quadrant items whose target center is not strictly
more than 60 pixels from both screen dividers. This leaves 677 eligible
questions per model and input condition.

The raw static log has 397 rows for 396 planned calls because one Qwen request
ended with an `IncompleteRead` transport error and was retried successfully.
Both records are preserved for provenance; the scorer uses the completed valid
response and does not count the transport failure as a substantive answer.

## Coordinate localization

The model receives the exact 600-by-600 initial image and the full convention:
origin at bottom-left, x increasing rightward, y increasing upward, valid
coordinates 0–599. It estimates the centers of the red target ball and green
goal container. The prompt explicitly says that reasonable visual estimates,
not pixel-perfect values, are expected.

## Temporal rollout

The final qualitative battery uses all 132 layout-by-mechanism cells. The image
condition supplies 32 ordered full-resolution frames; the symbolic condition
supplies the matched 32 observable JSON states. Questions ask only about clear
qualitative motion:

- whether the orange tool moves upward or downward;
- whether the red ball finishes clearly left or right of its start;
- whether the red ball visibly changes position.

Numerical thresholds are never shown to the model. Simulator thresholds are
used only to include unambiguous items: at least 60 pixels for visible motion
and at most 15 pixels for stationary examples, with intermediate cases
excluded. Answer meanings and option letters are exactly balanced within each
retained question family under seed 2026. Selection never uses model outputs,
goal success, or benchmark performance.

The final temporal battery contains 204 questions per model per representation:
132 tool-direction, 32 red-endpoint, and 40 red-motion questions.

One of Qwen's 132 image calls was rejected by the provider's image-content
filter, including after a lossless re-encoding retry. The released primary
score conservatively counts its two questions wrong (98.0%); the valid-response
score is 99.0%. All 132 Qwen JSON calls completed. Provider account identifiers
are redacted from the released error record.

## Files within each control

- `questions.jsonl` or `ground_truth.jsonl`: frozen answer keys;
- `call_manifest.jsonl`: one row per model call;
- `payloads/`: exact system/user prompt material;
- `results/raw_responses_seed2026.jsonl`: unedited response records;
- `results/scored/`: question-level rows, summaries, and score report.

## Historical regeneration and current-scene follow-up

The historical static assets differ from the current release on 38/66 layout
screenshots. The historical Qwen JSON result remains **565/677 = 83.5%**;
93.5% is not supported by archived answers. See
[asset-version details](../analysis/ASSET_PROVENANCE.md).

```bash
python vqa/scripts/generate_static_vqa.py --output outputs/historical_vqa
python vqa/scripts/generate_current_scene_vqa.py --output outputs/current_vqa_battery
# Free dry run; pricing.json is a frozen subset of https://openrouter.ai/api/v1/models
python vqa/scripts/run_current_scene_vqa.py --battery outputs/current_vqa_battery --output outputs/current_vqa_run --pricing pricing.json
# Add --execute only to authorize paid calls; API key is read from OPENROUTER_API_KEY.
python vqa/scripts/score_current_scene_vqa.py --battery outputs/current_vqa_battery --responses outputs/current_vqa_run/responses.jsonl --output outputs/current_vqa_scored
```

Freeze hashes before inference. Current-scene scoring includes every planned
eligible item in its denominator; missing/invalid answers cannot silently
shrink it. The first returned answer is retained, including malformed answers;
only transport or local-storage failures can be retried. The generator makes no model calls.
The primary replication matches archived historical settings: medium reasoning,
8192 output-token limit, seed2026; temperature0.2 for Gemini/Qwen, omitted for
GPT-5 where unsupported. Raw responses, full requests, actual model IDs, usage,
and image hashes are retained. Do not mix the superseded low/2048 pilot into
final results.


The separately frozen compatible-format condition uses JSON-object output, low
reasoning and16384 tokens for all three models. It follows recorded Qwen schema
and truncation failures; it does not replace primary or historical answers.
See [current-scene frozen runs, results and tested commands](current_scene_2026-11-14/README.md).
