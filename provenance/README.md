# Collection provenance

The finalized dataset combines unchanged original records with corrected-input reruns for BackUp_8, Balance_8, and Falling_1: 59 goal tasks for each of GPT-5, Gemini 3.1 Pro, and Qwen 3.6 Plus, plus six canonical Robotics-ER 2 tasks. VQA replaces only the three corrected layouts for each main model. The historical archive is retained byte-for-byte.

`corrected_run_scope.json` freezes task IDs, conditions and original probe placements. Each final record is linked to its original record by the dataset manifest. The replacement is determined by input validity, not performance.

Collection retains each model's original response schema, medium reasoning effort, seed where supported, temperature, eight-valid-attempt budget, shared visual first action, separate JSON first action, and collection-specific stopping rules. The main models' legacy protocol absorbs immediate first-attempt success in all visual branches; after an initial failure, hidden-status branches continue to eight attempts. Robotics-ER 2 always continues hidden-status branches to eight.

The base output cap is 8,192 tokens; structured-state JSON continuation queries use 16,384, as in the original runner. Qwen uses the already documented empty-length-only recovery rule: increase the output cap only when final text is empty and the provider reports length truncation, at most twice and at most 32,768 tokens. The first nonempty response is retained regardless of correctness. Expansion calls and costs are preserved; inference budgets are not represented as equal across models.

Explicit provider-error interruptions receive at most one reviewed checkpoint restart with the original query; ordinary malformed or nonempty token-limited responses are not resampled. Original error records and all calls are preserved. Recovered probe queries must leave completed action records unchanged. Any unresolved provider errors are treated as missing, not as zero prediction quality.

The same requested hosted model IDs remain available. Returned IDs and provider metadata are recorded per call. Hosted preview endpoints do not guarantee immutable weights or routing across collection dates, so corrected runs are protocol-matched repetitions, not claimed to be bit-identical replications.

`runtime_validation.json` documents local numerical replay differences. All 183 original first-action success labels reproduced. Newly collected action outcomes and probe scores use recorded simulator truth from their own run.

Human participant-level data remain private and unmodified. This release does not construct or modify human estimates.
