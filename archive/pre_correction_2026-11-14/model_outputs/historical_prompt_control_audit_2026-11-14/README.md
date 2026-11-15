# Historical prompt audit, recovered November 14

Model-only derived paired scores and a selected-field projection of the recovered
Qwen record. Original private archives remain unchanged. The source SHA-256
identifies the original branch state, not this projection. No participant data.

The missing Qwen trial was recovered from the encrypted iCloud backup and
verified against its original checksum. All132 paired Qwen PQ values are now
available: full0.45388517, compact0.46113960, difference−0.00725443
[−0.04227812,0.02776680],10,000 whole-layout resamples, seed20261114.

The old July28 scorer used shared-state predictions without joining to the
accepted action. For36/132 compact Qwen records those actions differ; its
compact mean0.46690147 reproduces the old−0.01301630 contrast. Use the corrected
accepted-action comparison. This is a historical three-second goal/one-shot
panel, not the final two-second/eight-placement protocol. GPT/Gemini comparisons
retain archived derived-CSV provenance and131 matched pairs; their full and
compact prompt budgets differ. Do not call these a clean final-protocol ablation.

Run `python scripts/evaluation/reproduce_prompt_audit.py --output-dir
outputs/prompt_audit` from the repository root. The script recomputes paired
means and intervals from these released model-only scores; it does not claim
to rerun inference or independently rescore all original GPT/Gemini raw trials.
