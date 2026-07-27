# Evaluation scripts

The scripts in this folder operate on finalized experiment outputs.

| Script | Purpose |
|---|---|
| `analyze_action_quality_trajectories.py` | Attempt-indexed success and continuous action quality, including matched feedback-vs-retry contrasts. |
| `analyze_action_predictions.py` | Prospective per-attempt coordinate prediction scoring. |
| `analyze_action_prediction_coupling.py` | Within-goal relation between action quality and prediction quality. |
| `analyze_transfer_pilot.py` | Early/terminal held-out prediction and same-scene goal-transfer scoring. |
| `audit_transfer_cleanliness.py` | Sidecar completion, schema, target, and contamination checks. |
| `run_coordinate_replay.py` | Geometry-valid 10-pixel snap and fixed ±5-pixel perturbation replay without new model calls. |
| `analyze_coordinate_replay.py` | Sensitivity, clearance, isolated-success, and fixed-sequence summaries. |
| `analyze_prompt_seed_robustness.py` | Paired compact/full prompt and seed robustness panels. |
| `analyze_confirmatory_planning.py` | Frozen confirmatory summary tables. |
| `validate_canonical_april_solutions.py` | Canonical saved-solution verification. |

Every CLI exposes its required inputs:

```bash
python scripts/evaluation/analyze_action_quality_trajectories.py --help
python scripts/evaluation/analyze_transfer_pilot.py --help
python scripts/evaluation/run_coordinate_replay.py --help
```

Primary summaries exclude incomplete conversations rather than treating them
as failures, while reporting their count. Resampling is clustered at the
visual-layout/environment level. Upward and downward mechanisms are reported
separately; paired upward/downward comparisons are reserved for canonical
goals that exist in both mechanisms.

Coordinate replay preserves each model’s original eight-action sequence. It
never searches neighboring points and selects the best perturbation after the
fact.
