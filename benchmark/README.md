# Finalized benchmark

`tasks.json` is the only default task bank: **1,692 goals**, comprising 1,560 diverse physical goals and 132 canonical red-ball-in-container goals. There are **66 scene layouts**, each paired with upward- and downward-accelerating tools.

`asset_index.json` maps every layout-direction cell to its world, initial image, and tool configuration. Paths are relative to this directory. Layout assets are stored once in `assets/`; both directions share the same world and initial image. All 66 images match a native simulator rendering pixel-for-pixel.

The corrected images for BackUp_8, Balance_8, and Falling_1 are the default. Task IDs, goals, worlds and tool dynamics are unchanged. Model episodes involving these layouts were recollected for the finalized analysis; historical records remain in `archive/`.

Benchmark size and analysis sample size differ: completed common tasks, prediction-reconciled tasks, and eligible untried probes have distinct denominators. Use `results/populations.json` for the exact reported memberships.
