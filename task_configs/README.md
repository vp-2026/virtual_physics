# Task configurations

## Frozen manifests

| File | Contents |
|---|---|
| `paper_goals_1560.json` | The 1,560 noncanonical goals retained from the paper benchmark. |
| `benchmark_1692_seed2026.json` | The same 1,560 goals plus 132 canonical container goals. |
| `solution_density_summary.csv` | Per-goal simulator success numerator, valid-placement denominator, and solution density. |
| `asset_index_132.json` | Historical task-key to numbered-cell mapping with SHA-256 world and screenshot verification. |
| `manifest_summary.json` | Compact count and range audit. |

## Population

- 1,560 paper-saved noncanonical goals: 312 each in contact, containment,
  movement, orientation, and position categories.
- 132 canonical goals: the red ball remains continuously inside the green
  container for at least two seconds, one per environment × tool-mechanism
  cell.
- 1,692 goals total.
- 757 upward-tool and 935 downward-tool goals. Noncanonical goals need not have
  a solvable counterpart under both mechanisms; paired upward/downward
  inference is reserved for the 132 canonical cells.

## Selection rule

The manifest is deterministic with seed 2026 and does not read model outputs.
For the 1,560 noncanonical goals, solution density is:

```text
successful valid placements / all valid placements
```

Goals with zero coverage are excluded because they are not demonstrably
solvable. Goals with coverage above `1 - 1/sqrt(2) =
0.2928932188134524` are excluded because a single random action would make the
task too easy. The released noncanonical range is approximately
`0.00015035–0.28868661`.

The CSV is the compact audit artifact for the frozen selection. It contains
the sufficient per-goal counts used to recompute each density. Candidate-level
raw sweep trajectories are substantially larger and are not required to run
or score the benchmark; the simulator and sweep-generation code needed to
reproduce them are released under `simulator/goal_semantics/`.
