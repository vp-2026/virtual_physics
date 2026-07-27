# Numbered VTools environments

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
