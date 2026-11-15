# Finalized model observations

`manifest.json` identifies the authoritative shards, SHA-256 checksums, completed units, missing replacements, and record-level replacement lineage. Read each `*.jsonl.gz` shard with Python's `gzip.open(path, "rt")`; each line is one model-task episode across its feedback conditions.

- `summary`: model, seed, task definition, branch success and valid-attempt counts.
- `attempt_records`: executed placements, goal outcomes and simulated object endpoints.
- `provider_calls`: prompts, image hashes, raw responses and normalized response payloads.
- `sidecar_records`: fixed early/terminal novel interventions, predictions and scoring truth.
- `collection_provenance` on replacements: correction date and original-record fingerprint.

Only BackUp_8, Balance_8 and Falling_1 are replaced. Unaffected observations retain their original fingerprints; historical source files remain byte-for-byte in the archive. Interrupted responses and recovery calls are retained rather than converted into failures.

Completed model-unit counts are not interchangeable with analysis sample sizes. Use `results/populations.json` for the exact main, broad-control and matched-probe memberships, and `results/method.json` for estimators. Coordinates use the original 600-pixel Cartesian scene convention. Human participant-level records are not included.
