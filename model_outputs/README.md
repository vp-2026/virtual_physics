# Model outputs

`snapshot_goalA_2026-08-02/` is the current de-identified Goal-A checkpoint.
It contains 4,923 completed model × goal × seed episodes and excludes every
partial unit and all Goal-A → Goal-B material:

| Model | Completed units | Coverage of 1,692 goals | Canonical sidecars |
|---|---:|---:|---:|
| GPT-5 | 1,692 | 100.000% | 132/132 |
| Gemini 3.1 Pro Preview | 1,691 | 99.941% | 132/132 |
| Qwen 3.6 Plus | 1,540 | 91.017% | 131/132 |

Gemini index 1379 and 152 Qwen main units were still partial at this snapshot
boundary and are not serialized as completed observations. Qwen canonical
sidecar 1690 was also incomplete and is excluded. `PROTOCOL_EXCEPTION.md`
records the bounded recovery exception used after the original collection
ceilings were exhausted.

The checkpoint was validated across 50 deterministic compressed shards. A
recursive scan found no credentials, personal filesystem paths, email
addresses, Goal-B fields, provider-account identifiers, or provider response
identifiers.

`snapshot_seed2026_2026-07-27/` is a de-identified snapshot of every completed
main-benchmark unit available when the release was encoded. It contains 4,270
completed model × goal × seed episodes:

| Model | Completed units | Coverage of 1,692 goals |
|---|---:|---:|
| Gemini 3.1 Pro Preview | 1,679 | 99.232% |
| GPT-5 | 1,682 | 99.409% |
| Qwen 3.6 Plus | 909 | 53.723% |

`robotics_er_canonical132_seed2026_2026-07-27/` contains the complete
132-environment Gemini Robotics-ER 1.6 Preview control. All 132 source
episodes and all 132 held-out-prediction sidecar bundles are included.

`analysis_seed2026_2026-07-27/` contains the corresponding frozen planning,
action-quality, prospective-prediction, held-out-prediction, and coupling
summary tables. Each table records its own complete-case or matched-case
coverage.

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
- schema-validity and completion flags; and
- token usage, latency, and cost fields when returned by the provider.

`snapshot_manifest.json` records every shard's unit count, byte size, and
SHA-256 digest. Validate the downloaded package with:

```bash
python scripts/release/export_model_outputs.py \
  --check \
  --output-root model_outputs/snapshot_seed2026_2026-07-27

python scripts/release/export_model_outputs.py \
  --check \
  --output-root \
  model_outputs/robotics_er_canonical132_seed2026_2026-07-27
```

## De-identification and media policy

The release excludes credentials, provider response identifiers, personal
filesystem paths, incomplete sidecars, rollout PNGs, videos, and regenerable
full simulator traces. Visual and JSON conditions both use 32 uniformly
sampled states at run time. Rollout media can be regenerated from the released
world, action, simulator, seed, sampling indices, and recorded hashes rather
than duplicated for every attempt in Git.
