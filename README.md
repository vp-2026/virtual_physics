# Virtual Physics

A benchmark of action selection, outcome prediction, and prediction for new actions: **1,692 tasks across 66 layouts with upward and downward tools**.

## Data

| Material | Location |
|---|---|
| All 1,692 task definitions and goals | [`benchmark/tasks.json`](benchmark/tasks.json) |
| Scene images, simulator worlds, and tool configurations | [`benchmark/assets/`](benchmark/assets/), mapped by [`asset_index.json`](benchmark/asset_index.json) |
| GPT-5 model outputs | [`data/finalized/gpt5/`](data/finalized/gpt5/) |
| Gemini 3.1 Pro model outputs | [`data/finalized/gemini31pro/`](data/finalized/gemini31pro/) |
| Qwen 3.6 Plus model outputs | [`data/finalized/qwen36plus/`](data/finalized/qwen36plus/) |
| Gemini Robotics-ER 2 outputs, on the 132 canonical tasks | [`data/finalized/robotics_er2/`](data/finalized/robotics_er2/) |
| Model-record format, file index, and checksums | [`data/finalized/README.md`](data/finalized/README.md), [`manifest.json`](data/finalized/manifest.json) |
| Static scene-reading control: questions, prompts, and responses | [`data/vqa/`](data/vqa/) |

Model outputs are compressed JSONL files: each line contains one model–task episode, its actions, predictions, simulator outcomes, feedback conditions, and recorded model calls.

## Prompts and experiment setup

| Setup | Location |
|---|---|
| Prompt templates, response schema, conversation history, and stopping rules | [`prompts/README.md`](prompts/README.md) |
| Full and compact system prompts | [`system_full.txt`](prompts/system_full.txt), [`system_compact.txt`](prompts/system_compact.txt) |
| Worked two-attempt interaction, including model responses | [`docs/worked_interaction_example.md`](docs/worked_interaction_example.md) |
| Action selection and feedback controls | [`scripts/experiment/forked_feedback_runner.py`](scripts/experiment/forked_feedback_runner.py) |
| New-action prediction before and after interaction | [`scripts/experiment/transfer_sidecar_runner.py`](scripts/experiment/transfer_sidecar_runner.py) |
| Shared provider and simulator helpers | [`scripts/experiment/goal_probe_runner.py`](scripts/experiment/goal_probe_runner.py) |
| Simulator and goal-generation code | [`simulator/`](simulator/) |

The feedback runner selects conditions through `--conditions`:

| Condition | Information supplied after an action |
|---|---|
| `full` | Visual rollout frames and success/failure status |
| `neither` | Retry without rollout frames or success/failure status |
| `trace_status` | Structured JSON object coordinates/states and success/failure status |
| `frames_only` | Visual rollout frames without success/failure status |
| `status_only` | Success/failure status without rollout frames |

Prompt framing is selected with `--prompt-variant full` or `--prompt-variant compact`. Exact model settings are stored with the model records.

Offline scoring and reproduction: [`scripts/README.md`](scripts/README.md). Licensing and attribution: [`LICENSE`](LICENSE), [`NOTICE.md`](NOTICE.md), [`LICENSE_STATUS.md`](LICENSE_STATUS.md).
