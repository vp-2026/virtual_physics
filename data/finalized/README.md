# Model outputs

| Model | Files |
|---|---|
| GPT-5 | [`gpt5/`](gpt5/) |
| Gemini 3.1 Pro | [`gemini31pro/`](gemini31pro/) |
| Qwen 3.6 Plus | [`qwen36plus/`](qwen36plus/) |
| Gemini Robotics-ER 2 | [`robotics_er2/`](robotics_er2/) |

[`manifest.json`](manifest.json) indexes the compressed JSONL shards and their checksums. Each line is one model–task episode containing:

- `summary`: model, seed, task, and outcomes for each feedback condition.
- `attempt_records`: tool placements, action success, and simulated object endpoints.
- `provider_calls`: rendered prompts, image hashes, raw responses, and parsed predictions.
- `sidecar_records`: novel-action probes, predictions, and simulator truth.

Read a shard from the repository root:

```python
import gzip
import json

with gzip.open("data/finalized/gpt5/units-0000.jsonl.gz", "rt") as stream:
    episode = json.loads(next(stream))
print(episode["summary"]["goal"]["balanced_goal_id"])
```

Task IDs correspond to [`benchmark/tasks.json`](../../benchmark/tasks.json). Coordinates use the 600-pixel Cartesian scene convention described in [`simulator/README.md`](../../simulator/README.md).
