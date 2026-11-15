# Numbered VTools environments

The numbered `cells/` are **historical experiment inputs**, preserved byte for
byte. Three initial images disagree with the executed worlds: `BackUp_8`,
`Balance_8`, and `Falling_1` (both tool directions). New, native-rendered images
are under `corrections/2026-11-14/`; use the
[versioned corrected index](../task_configs/asset_index_132_scene_corrected_2026-11-14.json)
for new work. See the [correction record](../analysis/scene_mismatch_2026-11-14/README.md)
for source hashes, all affected tasks, and exclusion analyses. Re-rendering does
not retroactively repair predictions or actions collected with a different image.

Code licensing and upstream-derived asset scope are distinguished in
[LICENSE_STATUS.md](../LICENSE_STATUS.md) and [NOTICE.md](../NOTICE.md).

`cells/001` through `cells/132` are the release-facing environment IDs.

Each directory contains:

- `environment.json`: exact simulator world definition;
- `tool.json`: exact orange-tool geometry, physical properties, placement
  convention, and upward/downward mechanism;
- `initial_scene.png`: exact static observation used by the model run;
- `metadata.json`: hashes, paired cell, and provenance.

The same visual layout appears twice—once per tool mechanism—so paired cells
share an `environment.json` hash but have different `tool.json` mechanisms.
The package therefore contains 132 cells, 66 unique world hashes, 66 upward
tools, and 66 downward tools.

Use `index.csv` for analysis and `index.jsonl` for programmatic loading.
Historical family names are provenance fields only; they are not release
filenames.
