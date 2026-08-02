# Goal-A Recovery Protocol Exception — 2026-08-01

Authorization: the user explicitly authorized an exception to cost and retry limits for Gemini 3.1 Pro and Qwen 3.6 Plus on 2026-08-01 in order to wrap up the incomplete Goal-A collection.

Scope is limited to units lacking a valid canonical `unit_summary.json` and the one incomplete Qwen held-out-prediction sidecar. Existing valid completed units must not be rerun or replaced.

Unchanged experimental fields:

- model identifiers and temperatures;
- seed 2026;
- canonical-2s main and transfer manifests;
- goal ordering, gravity assignments, and condition assignments;
- rendered prompts and prompt variants;
- fixed attempt budget of 8;
- maximum response tokens of 8192;
- action parsing, simulation, scorer, hashes, and completion semantics.

Authorized recovery ceilings:

- Gemini main index 1379 only: maximum accumulated provider cost USD 24.00, maximum format repairs 3, maximum blocked/procedural repairs 24.
- Qwen incomplete general-main shards only: maximum accumulated provider cost USD 149.34 per frozen shard (2× the original 74.67 ceiling), maximum format repairs 3, maximum blocked/procedural repairs 24, with the established 15 shards per gravity.
- Qwen held-out sidecar for canonical index 1690 only: maximum accumulated provider cost USD 20.00, maximum format repairs 3, maximum blocked/procedural repairs 24, and at most 24 outer runner attempts.

All outputs produced under this exception must be identified in the release audit. Counts remain incomplete until the normal atomic completion artifacts and validation gates pass. Provider rejections and in-progress units are not benchmark failures.

## Authorized continuation — 2026-08-02

The user explicitly directed Qwen recovery to continue through all 1,692 main units and authorized publication of a checkpoint version once Qwen reaches 1,560 strict valid completions. Qwen workers must continue after the checkpoint publication.

Because the 2× tier drained with 153 partial units remaining, the bounded Qwen general-main cost ceiling is raised to USD 298.68 per frozen shard (4× the original USD 74.67 ceiling). The established 15 shards per gravity, maximum format repairs 3, and maximum blocked/procedural repairs 24 remain unchanged. Existing valid completions remain immutable and are skipped idempotently.

Checkpoint publication policy:

- require a fresh strict audit showing Qwen at least 1,560 valid completed main units;
- disclose Gemini index 1379 as incomplete if it remains partial;
- disclose this recovery exception and exact model/unit counts;
- run the complete de-identification, credential/privacy, manifest, hash, scorer, and Goal-B-exclusion scans before any external submission;
- continue Qwen recovery toward 1,692 after the checkpoint is published.
