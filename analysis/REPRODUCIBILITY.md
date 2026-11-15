# Verified numerical reproduction — November 14, 2026

All commands below were executed on the release candidate. They make no paid
calls unless the explicitly marked VQA `--execute` flag is supplied. Use new
output directories: scripts reject nonempty destinations and do not overwrite
frozen experiment files.

The subsequent [scene discrepancy audit](scene_mismatch_2026-11-14/README.md)
adds native-render geometry checks and model-only exclusion analyses. It does
not overwrite the numerical results documented below. Both the requested
two-layout exclusion and the expanded three-layout exclusion are retained.

## Environment

The numerical audit used Python 3.14, NumPy 2.5.4, SciPy 1.18.1, and Pillow
13.0.0. Install `requirements-analysis.txt` in a clean environment. Matplotlib
is needed only for figure rendering. Simulator replay needs the separate
Pygame/Pymunk dependencies in the root requirements; cached replay rescoring
does not run a new simulation. No paid inference is needed for model tables.

```bash
python scripts/evaluation/analyze_figure4_correspondence.py --output-dir outputs/figure4_verified
python scripts/evaluation/reproduce_model_tables.py --figure4-results outputs/figure4_verified --output-dir outputs/tables_verified
python scripts/evaluation/reproduce_pq_sensitivity.py --figure4-results outputs/figure4_verified --output-dir outputs/sensitivity_verified
python scripts/evaluation/reproduce_coordinate_cache.py --output-dir outputs/coordinate_verified
python scripts/evaluation/reproduce_prompt_audit.py --output-dir outputs/prompt_verified
python vqa/scripts/generate_static_vqa.py --output outputs/historical_vqa_regenerated
```

Figure 4 coefficients and CI endpoints are checked automatically against
`figure4/verified_estimates_2026-11-14.json` (absolute tolerance 1e-10). The
reference is a verification target; it is not the analysis input. Model
observations are reconstructed from raw public provider calls and simulator
truth. Join accepted branch calls to executed coordinates rather than relying
on occasionally stale shared-state response records. The historical static
VQA command reproduces all 726 questions/keys and 66 observable scenes; release
wrapper fields such as `environment_ids` are not part of question generation.

## Populations, seed, and estimands

| Analysis | Population | Estimator |
| --- | --- | --- |
| Main action results / Figure 4A | 1,641 common tasks, 66 layouts | Prediction histories reconcile in all three models; 10 of 1,651 completed common tasks excluded from this population |
| Broad feedback tables | 1,651 completed common tasks | Episode success under each arm, and paired full-minus-retry contrasts; missing runs not counted as failures |
| Figure 4B/C | Frozen 118 canonical tasks, 65 layouts, 53 paired directions | Task-ID membership in `figure4/task_populations.json`; not the full bank |
| Feedback decomposition | 132 canonical tasks per model | Full, status-only, frames-only, retry, and JSON arms on the same canonical population |
| Robotics ER2 correspondence | 132 canonical tasks | Binary action success versus chosen-action PQ, separately under visual and JSON feedback |
| Robotics ER2 novel prediction | Historical matched 125 and own-clean 127, separately | Fixed-probe pre/post prediction; the 125 intersection is not ER2-only eligibility |

The experiment seed is **2026**. Bootstrap RNG seeds below are not additional
experimental seeds. No final-protocol three-seed claim is supported.

Figure 4A uses PQ ~ binary success + categorical attempt + exact-goal fixed
effects, inverse episode-length weights, and 2,000 whole-layout resamples
(seed 6112702). It is a within-goal association identified by recovering
episodes, not a causal success effect. B is residual-rank correlation, with
layout and direction controls; its richer specification additionally controls
initial chosen PQ, baseline novel PQ, and rollout count. Ranks and nuisance
fits are recalculated in each of 2,000 layout resamples (6112703). C compares
first-action success groups at the common one-rollout novel checkpoint,
controlling baseline probe PQ, layout, and direction (2,000 resamples,
6112702). Raw top-panel least-squares lines are descriptive, not partial rank
estimates. Both uncertainty levels and sample sizes are saved in outputs.

The model table script preserves historical CI seeds/draw counts. Its output
records each seed explicitly. PQ sensitivity additionally uses 100,000 draws
for feedback checks and 10,000 for scale contrasts (seed 20261114); small CI
endpoint changes from 2,000-draw historical intervals reflect Monte Carlo
precision, not changed observations.

## Feedback and stopping

Visual arms share a first response; JSON uses a separate initial symbolic
representation/first response. Full feedback contains the latest 32 frames
plus outcome; JSON supplies 32 observable coordinate/state snapshots plus
outcome. Earlier action/prediction text remains, while older media do not.
Success-stopping and retry histories differ, so a terminal slope is not by
itself evidence of feedback learning. The released scripts distinguish
all-task solve rates from shared-first-failure contrasts, and early fixed
exposure from terminal novel probes. Probe branches never inform the solver.

## Known limitations

Historical prompt comparisons use a three-second goal; some reference arms
also differ in attempt budget. They are not clean final-protocol wording
ablations. The missing Qwen trial was recovered and checksum-verified from
iCloud: full132-pair PQ difference−0.00725443 [−0.04227812,0.02776680].
The historical−0.01301630 came from36 stale compact prediction/action joins.
See `../model_outputs/historical_prompt_control_audit_2026-11-14/README.md`.
Current-scene VQA is separate and preserves historical Qwen JSON565/677=83.5%.

The full existing Figure 4 contains an illustrative human prediction layer.
Only model observations/estimates are reproduced by these public scripts;
no participant-level human file or illustrative human values are added here.
The separate PDF supplied to the author preserves that existing layer.

See `ASSET_PROVENANCE.md` and `../LICENSE_STATUS.md` for asset mismatches and
unresolved licensing. This release does not claim those issues are closed.


## Current-scene VQA

The completed current-scene conditions, immutable inputs, raw responses, exact
counts and tested no-inference reproduction commands are in
[`vqa/current_scene_2026-11-14`](../vqa/current_scene_2026-11-14/README.md).
Primary historical-settings replication, Qwen interface diagnostic and the
separately authorized all-model compatible-format condition remain separate.
Aggregate audit tables are also available in `audit_2026-11-14/`.
