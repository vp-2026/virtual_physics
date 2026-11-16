# Current-scene static VQA

`responses.jsonl` retains 198 returned responses (66 layouts per model) and six historical storage-interruption events. Nine returned responses use corrected scene inputs; the other 189 are unchanged. The frozen scorer selects the first returned, non-error response per call, retaining malformed responses rather than choosing a better answer.

`battery/` contains frozen questions, eligibility rules and corrected answer keys. `source_map.json` resolves retained raw requests and responses, including the unchanged calls in `recorded_calls.zip`. `corrected_calls/` preserves the nine new requests and responses. No other layout was recollected.

Invalid or missing answers receive no credit. The original quadrant-boundary rule yields 670 eligible items per model after the scene corrections. Run `python scripts/reproduce.py` from the repository root to reproduce scores and layout-bootstrap intervals without inference. The primary results reference is `results/RESULTS.md`.
