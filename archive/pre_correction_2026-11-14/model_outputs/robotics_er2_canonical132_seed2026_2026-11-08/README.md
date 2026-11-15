# Gemini Robotics ER2: canonical 132-task results

This archive contains **132 completed episodes and 132 complete fixed-probe
bundles** for `gemini-robotics-er-2-preview`. The task-selection seed is 2026;
the provider does not expose a sampling seed. Records include the requested
and returned model identifiers and per-call request parameters.

Each episode evaluates the same canonical task in three conditions:

- `full`: visual rollout frames and success/failure feedback;
- `neither`: retry without outcome feedback, retaining all eight attempts;
- `trace_status`: structured observable states and success/failure feedback.

The visual conditions share their initial action. The structured-state
condition has its own initial action. Fixed novel-action probes are recorded
before interaction, after the early checkpoint, and at the terminal checkpoint.
The sidecar records identify the exact probe and checkpoint; complete collection
does not imply eligibility for every clean-probe analysis.

## Contents

- `data/robotics_er2/`: deterministic compressed JSONL shards, one episode per
  line, including prompts, raw responses, parsed actions and predictions,
  simulator endpoint truth, and complete novel-action probe records.
- `snapshot_manifest.json`: model identity, collection settings, task indices,
  shard SHA-256 hashes, and coverage.
- `coverage.csv`: completed episodes, solve counts, and attempt counts by condition.
- `protocol/`: collection source snapshots and their hashes. The runner matches
  its recorded collection hash. The final shared provider adapter differs from
  the earlier recorded hash; this distinction is retained in the source manifest.
  Per-call model identities, parameters, and responses remain in the data.
- Each record's `export_source_files` lists the relative source paths, byte
  lengths, and SHA-256 hashes used during export.

Collection began through the Google developer endpoint and continued through
the Google Cloud endpoint under the same recorded model identifier. The
per-call provider field preserves this distinction. Exporting and validating
this archive makes no model calls.

## Validation

From the repository root:

```bash
python3 scripts/release/validate_robotics_er2.py
```

The validator checks every shard and record, all 132 tasks and probe bundles,
terminal-state rules, outcome counts, and exact task/asset matching against
GPT-5's canonical population. It also scans decompressed records for private
identifiers and excluded fields.

ER2 is a separate canonical extension. It is not the earlier ER1.6 control or
a broad 1,692-goal run. Different collection settings and budgets mean the two
robotics versions should not be treated as a randomized version comparison.
The export omits credentials, personal paths, provider response identifiers,
and regenerable media/full traces; endpoint truth and rendering metadata remain.
