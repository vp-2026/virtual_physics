# Task and VQA asset provenance

## Follow-up: scene mismatch origin established

The [scene provenance and sensitivity audit](scene_mismatch_2026-11-14/README.md)
establishes that the `BackUp_8` and `Balance_8` mismatches below occurred in
the actual visual model experiments, before release. A full RGB comparison
also identifies `Falling_1`'s blue-body geometry. Historical assets and VQA
records remain untouched. A separate prospective scene index supplies three
native-rendered corrections, and all 66 layouts in that index now match the
native world renderings pixel-for-pixel. The previous two-object check was
insufficient to detect the blue-body discrepancy.

MIT code/simulator publication was separately authorized by the maintainer;
upstream asset scope and retained attribution are described in
[LICENSE_STATUS.md](../LICENSE_STATUS.md) and [NOTICE.md](../NOTICE.md).

`task_configs/asset_index_132.json` maps 66 layouts × two tool directions to
132 released cells. The builder's SHA checks pass for all cells and for the
1,560 noncanonical / 1,692 expanded task manifests. Those frozen assets are
unchanged by the November 14 reproducibility repair.

The historical static VQA and coordinate tests used an earlier scene version.
A pixel comparison finds 28/66 historical screenshots identical to the current
release and 38/66 different. Observable world geometry is identical for 30/66
and different for 36/66. A hash check only establishes which file was used; it
does not establish image/world semantic alignment. Per-layout comparisons are
in `asset_version_comparison.json`.

## Restored historical generation inputs

`vqa/static_scene/assets/source_worlds/` adds the 66 original worlds recovered
from the research archive. Each matches its existing historical
`source_world_sha256` in `vqa/coordinate_localization/ground_truth.jsonl`.
No screenshot, archived response, question, answer key, or historical result
was replaced. The restored `vqa/scripts/generate_vqa.py` preserves the original
question-helper bodies (source SHA-256
`023125f4bc3359ab520671c6c53ffc3914c88084ffa8ed65de47a6a0c7c4407c`).
Legacy private default paths and unrelated showcase-image generation were
omitted. `generate_static_vqa.py` uses portable public assets and reproduces
all 726 historical questions/keys and all 66 observable scene JSON objects.

## Current screenshot/world discrepancies

A separate image-mask audit found two green-container location discrepancies:

| Layout | Current world center (x,y) | Visible green center (x,y) |
| --- | --- | --- |
| BackUp_8 | (505,307) | (96.5,570.5) |
| Balance_8 | (548,271) | (401.5,277.5) |

The red target centers match within one pixel in both. The other 64 layouts'
red/green center checks pass within three pixels. The exact released screenshots
are retained for current-scene VQA. Two explicit **VQA-only** green rectangle
annotations determine visible-scene answer keys before inference; they do not
change simulator worlds or historical outputs. Report a sensitivity excluding
these two layouts. This discrepancy warrants a caveat in interpretation of
visual physics prediction; file hashes alone did not detect it.

Current questions retain the historical templates, option balancing, and
strict >60-pixel divider exclusion. Scene-dependent keys/options are updated
before inference: 49 question records change; 671/726 questions per model are
eligible and 55 quadrant items excluded. Historical eligibility was 677/726.
No post-inference key correction or answer selection is allowed. Provider
versions can drift, so the old/new comparison is not an asset-only experiment.

See `../LICENSE_STATUS.md` for attribution and unresolved permission scope.
