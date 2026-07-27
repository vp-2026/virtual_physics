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

The noncanonical action-quality snapshot covers only goals with a recoverable
simulator-derived solution set. Canonical action quality uses the saved April
solution spaces. The paper-defined linear normalization is implemented in
`scripts/evaluation/score_paper_action_quality.py`; the Gaussian score is used
here because it remains defined for all covered goals and directly tests
tolerance at the tool-diameter scale.

All confidence intervals and fixed-effect regressions are produced by the
released evaluation scripts with seed 2026.
