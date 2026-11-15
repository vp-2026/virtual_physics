# Historical cached coordinate robustness

`frozen_replay_cache.zip` contains 134 original model-only cache files:
60,775 accepted-action records, the replay task mapping, and 132 simulated
layout/direction cache files. Decompressed file bytes are checked against
`source_sha256.json`. Original simulation outputs are preserved unchanged.

Run `python scripts/evaluation/reproduce_coordinate_cache.py --output-dir
outputs/coordinate_verified` from the repository root. This re-scores cached
replays, not fresh physics simulations or new model actions.

The primary-arm episode counts are GPT-5 1,531, Gemini 1,657, and Qwen 268 per
arm. This earlier collection is not the final 1,641 common-task population.
Mechanical 10-pixel rounding uses downward half-grid ties, with invalid
placements counted as failures. Jitter covers first attempts and originally
successful actions only: eight offsets (±5,0), (0,±5), and (±5,±5). Diagonal
shifts are 7.07 Euclidean pixels. There is no feasibility projection or
best-neighbor selection. The saved action sequence is kept fixed.

720 logged outcomes differ from re-scored cached originals. Robustness compares
replayed original to replayed perturbed outcomes, and retains these mismatches;
it does not establish equivalence to every final logged goal predicate.
