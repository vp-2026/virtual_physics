# Static scene-reading control

- [`battery/`](battery/) contains scene images, questions, answer keys, prompt payloads, and protocol settings.
- [`responses.jsonl`](responses.jsonl) contains the response log for GPT-5, Gemini 3.1 Pro, and Qwen 3.6 Plus.
- [`source_map.json`](source_map.json) maps responses to their exact request and response files in [`recorded_calls.zip`](recorded_calls.zip) and [`corrected_calls/`](corrected_calls/).
- [`scripts/evaluation/reproduce_vqa.py`](../../scripts/evaluation/reproduce_vqa.py) re-scores these records offline.

See [`scripts/README.md`](../../scripts/README.md) for the reproduction command.
