# Collection source snapshots

These sources document the corrected collection, including cross-process budget accounting, checkpoint resumption, collection-specific Robotics-ER 2 stopping rules, and the historical Qwen empty-length-only recovery policy. They are provenance snapshots, not the offline reproduction entry point. Machine-specific deployment paths are replaced by named placeholders; hashes before and after relocation are recorded in `source_manifest.json`.

The finalized wrapper versions include exclusive unit locks and immediate recovery-response logging added during collection. Per-call records retain actual prompts, image hashes, requested and returned model identifiers, usage, and response text. No new response is selected based on its task success or prediction accuracy.

The first Robotics-ER 2 requests used the original Gemini Developer API. After its daily quota was exhausted, unfinished checkpoints continued through the user-authorized Vertex AI endpoint with the same model ID and generation settings. Completed calls were retained. The direct-provider source includes adapters for other historical models, but those adapters were not used in this correction run.

For tested offline reproduction, run `python scripts/reproduce.py` from the repository root. Future paid recollection requires separately supplied credentials and an explicit budget; these snapshots do not launch calls automatically.
