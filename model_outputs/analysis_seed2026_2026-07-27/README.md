# Seed-2026 analysis snapshot

This directory contains deterministic summaries computed from completed
episodes available at the snapshot boundary. Partial episodes are excluded,
not scored as failures. Collection was still active, so every table reports
its own completed-unit or matched-unit coverage.

## Conditions

- `full`: latest 32-frame visual rollout plus success/failure.
- `trace_status`: latest 32-state JSON rollout plus success/failure.
- `neither`: retry alone, with neither rollout nor status.

`full` and `neither` share the exact attempt-1 response and simulation, so
their later differences form the matched causal feedback comparison.
`trace_status` uses a JSON initial scene and therefore has an independently
sampled attempt 1. It is a representation control, not a matched causal
contrast against the visual `neither` arm.

## Measures

- Planning tables report solve-by-eight and restricted mean attempts to
  success or 9.
- Action quality is
  `exp(-distance_to_nearest_solution^2 / (2 * 72^2))`, using the 72-pixel tool
  diameter as the primary scale. Best-so-far trajectories carry solved
  branches forward so the matched population remains fixed.
- Prediction quality is the role-averaged Gaussian endpoint score at the
  150-pixel primary scale.
- Held-out prediction measures the change in prediction quality for a fixed,
  unexecuted placement selected before feedback.
- Action-prediction coupling relates action quality to the prospective
  prediction made before executing that same action.
- Action-transfer coupling relates feedback-minus-retry source-action
  improvement to feedback-minus-retry held-out prediction improvement.

The `coupling/` directory also includes:

- `action_prediction_success_contrast_cluster_bootstrap.csv`, the primary
  successful-minus-failed prospective-PQ contrast with 20,000
  base-layout-clustered bootstrap replicates;
- `action_prediction_arm_slopes.csv`, arm-specific slopes from the adjusted
  goal-fixed-effect regression of continuous AQ on prospective PQ; and
- `symbolic_state_transfer_coupling.csv`, JSON-state Spearman associations
  between source action or prospective-prediction gains and terminal
  held-out-prediction gains, with base-layout-clustered intervals; and
- matched versions of both tables for the complete Robotics-ER control.

The `prediction/` directory includes
`symbolic_state_first_final_cluster_bootstrap.csv`, which reports the
first-to-final prospective-PQ change under JSON-state feedback with
base-layout-clustered intervals.

The regression outcome is continuous AQ at the 72-pixel scale. Predictors are
prospective PQ at the 150-pixel scale, attempt, feedback-arm indicators, and
PQ-by-arm interactions. Goal fixed effects absorb fixed goal category and
solution-density differences; CR1 uncertainty is clustered by the 66 base
layouts.

The noncanonical action-quality snapshot covers only goals with a recoverable
simulator-derived solution set. Canonical action quality uses the saved April
solution spaces. The paper-defined linear normalization is implemented in
`scripts/evaluation/score_paper_action_quality.py`; the Gaussian score is used
here because it remains defined for all covered goals and directly tests
tolerance at the tool-diameter scale.

All confidence intervals and fixed-effect regressions are produced by the
released evaluation scripts with seed 2026.
