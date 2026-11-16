# Benchmark data

[`tasks.json`](tasks.json) contains **1,692 tasks**: 1,560 diverse physical goals and 132 canonical red-ball-in-container goals. The benchmark uses 66 visual layouts, each with upward- and downward-accelerating tools.

- [`asset_index.json`](asset_index.json) maps each layout and direction to its simulator world, initial image, and tool configuration. Paths are relative to this directory.
- [`assets/`](assets/) stores the layout assets. Both tool directions share the same world and initial image, with separate tool configurations.
- [`checksums.json`](checksums.json) contains file hashes for verification.
- [`evaluation_task_ids.json`](evaluation_task_ids.json) supplies task selections to the offline reproduction scripts.

The scene coordinate system is 600 × 600 pixels, with its origin at the bottom left. See [`simulator/README.md`](../simulator/README.md) for execution conventions.
