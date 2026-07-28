# Acting without Knowing: Virtual Physics release

This repository contains the task assets and evaluation code for studying
whether a model can act successfully in a deterministic 2-D rigid-body
environment while accurately predicting and transferring the consequences of
its actions.

The release is organized around three distinct quantities:

1. **Action quality:** does a proposed tool placement solve the goal, and how
   close is it to a known solution?
2. **Prediction quality:** before seeing the outcome, how accurately does the
   model predict the terminal locations of the tool and relevant movable
   objects?
3. **Transfer quality:** after interacting with one goal, does the model
   improve on unexecuted placements or a different goal in the same unchanged
   environment?

## Release contents

```text
132_base_environments/     132 numbered environment × tool-mechanism cells
task_configs/              frozen 1,560- and 1,692-goal manifests and densities
prompts/                   exact full and compact prompt protocols
simulator/                 Pygame/Pymunk runtime and goal evaluators
scripts/experiment/        main feedback-arm and isolated-transfer runners
scripts/evaluation/        action, prediction, transfer, and sensitivity analyses
scripts/release/           release-building and integrity checks
vqa/                       static, coordinate, and temporal VQA packages
model_outputs/             completed-unit output snapshot and coverage manifest
examples/                  small runnable illustration retained from the initial release
```

The 132 numbered cells comprise 66 visual layouts under two tool mechanisms:
an orange tool that accelerates downward and the same tool accelerated upward.
Every cell includes its exact world JSON, tool configuration, initial scene,
metadata, and SHA-256 hashes.

## Task manifests

- `task_configs/paper_goals_1560.json` freezes the 1,560 noncanonical goals
  selected using the paper’s simulator-derived solution-density rule.
- `task_configs/benchmark_1692_seed2026.json` adds one canonical
  red-ball-into-green-container goal for every cell, for 1,692 total goals.
- `task_configs/solution_density_summary.csv` records the simulator success
  count, valid-placement count, and solution density for every released goal.
- `task_configs/asset_index_132.json` maps historical task keys to the numbered
  release cells and verifies the exact scene hashes.

Selection is deterministic with seed 2026 and never reads model outputs.
Noncanonical goals with zero solution coverage or coverage above
`1 - 1/sqrt(2)` are excluded. See
[`task_configs/README.md`](task_configs/README.md) for the exact population and
provenance.

## Feedback protocol

The four image-initialized branches share the same attempt-1 response for a
model × goal × seed unit. Later turns differ only in the evidence made
available:

- `full`: latest 32 visual rollout frames plus success/failure;
- `trace_status`: latest 32 observable JSON states plus success/failure;
- `frames_only`: latest 32 visual frames without success/failure;
- `status_only`: success/failure without rollout media;
- `neither`: retry alone, without rollout media or success/failure.

Because `trace_status` replaces visual input with a symbolic scene
representation, it has its own attempt 1; it is a representation control, not
part of the visual-arm shared-attempt randomization.

Earlier action coordinates and prospective predictions remain in text.
Previously supplied rollout images or JSON traces do not persist: only the
latest rollout is supplied. Held-out predictive-transfer queries use fixed,
unexecuted placements in isolated branches that never return to the solver
conversation.

The full and compact prompts are a controlled wording check. They differ only
in the first system-role sentence; all task rules, dynamics, coordinate
conventions, schemas, and interaction context are identical. See
[`prompts/README.md`](prompts/README.md).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Headless rendering uses `SDL_VIDEODRIVER=dummy`. API credentials are read only
from environment variables and are never stored in task manifests or outputs.

## Validate the release

```bash
python scripts/release/build_task_configs.py --check
python scripts/release/attach_initial_scenes.py --check
python -m compileall scripts simulator
```

## Run an experiment

The main runner supports a deterministic mock provider for a local protocol
smoke test:

```bash
python scripts/experiment/forked_feedback_runner.py \
  --provider mock \
  --model gpt5 \
  --seed 2026 \
  --prompt-variant full \
  --limit 1
```

For a paid run, export the relevant provider credential and select the desired
provider/model arguments. Run `--help` before launching; outputs are resumable
and written beneath `outputs/`, which is gitignored.

The evaluation entry points and expected input files are documented in
[`scripts/evaluation/README.md`](scripts/evaluation/README.md).

## Model outputs and VQA

The published snapshot contains only protocol-finalized units. Partial
checkpoints are retained for recovery but are neither scored as failures nor
represented as completed outputs. Exact coverage is reported in
[`model_outputs/README.md`](model_outputs/README.md).

The VQA release contains static scene recognition, free-response coordinate
localization, and qualitative temporal recognition from both 32 visual frames
and 32 observable JSON states. Exact questions, prompts, raw model responses,
answer keys, scorers, and result tables are under [`vqa/`](vqa/).

## Historical minimal example

The original one-environment example remains under `data/` and `examples/`.
The exact benchmark simulator and semantics used by the expanded release are
under `simulator/`.
