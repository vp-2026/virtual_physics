# Model outputs

`snapshot_seed2026_2026-07-27/` is a de-identified snapshot of every completed
main-benchmark unit available when the release was encoded. It contains 3,353
completed model × goal × seed episodes:

| Model | Completed units | Coverage of 1,692 goals |
|---|---:|---:|
| Gemini 3.1 Pro Preview | 1,653 | 97.695% |
| GPT-5 | 1,483 | 87.648% |
| Qwen 3.6 Plus | 217 | 12.825% |

Coverage is reported explicitly because collection was still running at the
snapshot boundary. Only directories containing a finalized
`unit_summary.json` are included. Partial recovery directories are excluded
rather than scored as failures. Exact counts by model, feedback condition,
canonical status, and completed transfer sidecars are in `coverage.csv`.

## Format

Records are sharded by model as deterministic `jsonl.gz` files. Each line is
one completed model × goal × seed unit containing:

- model identifier, provider, seed, prompt variant, task and asset hashes;
- exact rendered prompts and raw response text;
- parsed action and prospective prediction at every readable valid attempt;
- geometry-only format or placement repairs;
- condition-specific feedback provenance;
- simulator truth, endpoint coordinates, and success result;
- completed early and terminal held-out prediction sidecars;
- completed exploratory same-environment Goal-A → Goal-B sidecars;
- schema-validity and completion flags; and
- token usage, latency, and cost fields when returned by the provider.

`snapshot_manifest.json` records every shard's unit count, byte size, and
SHA-256 digest. Validate the downloaded package with:

```bash
python scripts/release/export_model_outputs.py \
  --check \
  --output-root model_outputs/snapshot_seed2026_2026-07-27
```

## De-identification and media policy

The release excludes credentials, provider response identifiers, personal
filesystem paths, incomplete sidecars, rollout PNGs, videos, and regenerable
full simulator traces. Visual and JSON conditions both use 32 uniformly
sampled states at run time. Rollout media can be regenerated from the released
world, action, simulator, seed, sampling indices, and recorded hashes rather
than duplicated for every attempt in Git.
