# Current-scene VQA, November 14, 2026

The frozen archive retains exact screenshots, keys, prompts, requests, returned
model IDs, raw API JSON, parsed response logs, usage and infrastructure-recovery
records. Original historical VQA files are unchanged. No participant data.
`archive_manifest.json` records the archive and individual-file SHA-256 hashes.
Only local home paths in error metadata are redacted; answers, predictions,
provider responses and usage are preserved. Private originals remain archived.

Three conditions are separate throughout:

| Directory | Calls planned | Settings | Purpose |
| --- | --- | --- | --- |
| primary | 198 | Strict JSON schema, medium reasoning, 8192 tokens | Match actual historical settings on current scenes |
| qwen_interface | 66 | JSON object, medium reasoning, 8192 tokens | Qwen interface diagnostic after schema failures |
| compatible | 198 | JSON object, low reasoning, 16384 tokens | Separately authorized all-model compatible-format condition |

All use seed2026 and exact released images; Gemini/Qwen temperature0.2 and
GPT-5 temperature omitted because unsupported. All retain671 eligible questions
per model and55 excluded quadrant items. No answer is substituted across runs.
The compatible condition was designed after diagnostics; it is not an
independent preregistered historical replication or a controlled asset-only
comparison. Historical/current providers, keys and eligibility can also differ.

An incomplete low/2048 pilot is excluded. Local disk exhaustion interrupted two
conditions. Complete saved API responses were recovered, missing/infrastructure
calls alone were retried, and malformed returned answers were retained. Request
logs can include failed/partial local writes; these are documented rather than
silently removed. Record counts are not independent experimental sample sizes.
The first available returned answer, not the most accurate retry, is scored.

## Reproduce without paid calls

Run from the repository root in the pinned analysis environment, using new
output directories. These commands were executed and the results compared to
the released summary CSVs.

```bash
python -m zipfile -e vqa/current_scene_2026-11-14/frozen_runs.zip outputs/current_vqa_archive
python vqa/scripts/score_current_scene_vqa.py --battery outputs/current_vqa_archive/primary/battery --responses outputs/current_vqa_archive/primary/run/responses.jsonl --output outputs/current_vqa_primary_scores
python vqa/scripts/score_current_scene_vqa.py --battery outputs/current_vqa_archive/qwen_interface/battery --responses outputs/current_vqa_archive/qwen_interface/run/responses.jsonl --output outputs/current_vqa_interface_scores
python vqa/scripts/score_current_scene_vqa.py --battery outputs/current_vqa_archive/compatible/battery --responses outputs/current_vqa_archive/compatible/run/responses.jsonl --output outputs/current_vqa_compatible_scores
python vqa/scripts/summarize_current_scene_vqa.py --battery outputs/current_vqa_archive/primary/battery --primary-scores outputs/current_vqa_primary_scores --interface-scores outputs/current_vqa_interface_scores --compatible-scores outputs/current_vqa_compatible_scores --output outputs/current_vqa_summary
```

The scorer verifies every frozen battery hash and retains missing answers in
fixed denominators. Extra unrequested keys flag format validity but do not
discard valid answers to scored questions. The summarizer independently checks
letter-level credit and computes10,000 whole-layout percentile bootstrap
intervals, seed20261114. Valid-response-only sensitivities are selected samples.

`generate_current_scene_vqa.py` regenerates the primary questions/keys, prompts,
images and eligibility from released assets. The sorted diagnostic change-log
field order may differ from the original frozen file, without changing any
question or scoring input. To inspect costs without inference:

```bash
python vqa/scripts/run_current_scene_vqa.py --battery outputs/current_vqa_archive/compatible/battery --output outputs/current_vqa_dryrun --pricing outputs/current_vqa_archive/pricing.json --budget-usd 4.9
```

Paid inference requires `--execute` and a privately supplied
`OPENROUTER_API_KEY`. No key is in this archive. The runner reserves pending
request costs, limits infrastructure retries and stops queuing below512MiB free
local disk space. Do not rerun paid inference merely to reproduce the tables.

## Results and interpretation

See `CURRENT_SCENE_VQA_RESULTS.md` and `summary/` for exact counts, category
scores, historical comparisons, unanswered items, format diagnostics,
unchanged-item comparisons and the64-layout image/world sensitivity.
Two green-goal image/world mismatches are documented in
`../../analysis/ASSET_PROVENANCE.md`; VQA annotations were frozen before calls.
Historical Qwen JSON remains565/677=83.5%, not93.5%.

Licensing is not implied by public availability; see `../../LICENSE_STATUS.md`.
