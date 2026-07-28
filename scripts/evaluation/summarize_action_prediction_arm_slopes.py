#!/usr/bin/env python3
"""Estimate arm-specific AQ-on-PQ slopes with goal fixed effects.

This is the arm-slope presentation of the joint model used by
``analyze_action_prediction_coupling.py``. The dependent variable is
continuous action quality at the 72-pixel scale. The model includes
prospective prediction quality, attempt, feedback-arm indicators, and
prediction-quality-by-arm interactions. All variables are demeaned by goal,
and CR1 uncertainty is clustered by base layout.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ARM_ORDER = ("full", "frames_only", "status_only", "neither", "trace_status")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def demean_by_goal(
    values: np.ndarray,
    goal_ids: list[str],
) -> np.ndarray:
    output = values.astype(float, copy=True)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, goal_id in enumerate(goal_ids):
        groups[goal_id].append(index)
    for indices in groups.values():
        output[indices] -= output[indices].mean(axis=0)
    return output


def estimate_model(
    rows: list[dict[str, str]],
    model_key: str,
) -> list[dict[str, object]]:
    group = [row for row in rows if row["model_key"] == model_key]
    arm_levels = [
        arm
        for arm in ARM_ORDER[1:]
        if any(row["condition"] == arm for row in group)
    ]
    names = ["pq_sigma150", "attempt"]
    names.extend(f"arm_{arm}" for arm in arm_levels)
    names.extend(f"pq_x_{arm}" for arm in arm_levels)

    x_rows = []
    for row in group:
        pq = float(row["pq_sigma150"])
        arm = row["condition"]
        indicators = [float(arm == level) for level in arm_levels]
        x_rows.append(
            [
                pq,
                float(row["attempt"]),
                *indicators,
                *[pq * indicator for indicator in indicators],
            ]
        )
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray([float(row["aq_sigma72"]) for row in group])
    goals = [row["goal_id"] for row in group]
    x = demean_by_goal(x, goals)
    y = demean_by_goal(y[:, None], goals).ravel()

    keep = np.asarray(
        [
            not np.allclose(x[:, column], 0.0)
            for column in range(x.shape[1])
        ]
    )
    x = x[:, keep]
    kept_names = [name for name, retained in zip(names, keep) if retained]

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(group):
        clusters[row["base_layout"]].append(index)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for indices in clusters.values():
        score = x[indices].T @ residual[indices]
        meat += np.outer(score, score)
    n, k = x.shape
    cluster_n = len(clusters)
    correction = (
        (cluster_n / (cluster_n - 1)) * ((n - 1) / (n - k))
        if cluster_n > 1 and n > k
        else 1.0
    )
    covariance = correction * bread @ meat @ bread

    name_to_index = {name: index for index, name in enumerate(kept_names)}
    output = []
    observed_arms = [
        arm for arm in ARM_ORDER if any(row["condition"] == arm for row in group)
    ]
    for arm in observed_arms:
        contrast = np.zeros(len(kept_names), dtype=float)
        contrast[name_to_index["pq_sigma150"]] = 1.0
        interaction = f"pq_x_{arm}"
        if arm != "full" and interaction in name_to_index:
            contrast[name_to_index[interaction]] = 1.0
        estimate = float(contrast @ beta)
        se = float(np.sqrt(max(0.0, contrast @ covariance @ contrast)))
        output.append(
            {
                "model_key": model_key,
                "condition": arm,
                "dependent_variable": "aq_sigma72",
                "predictor": "pq_sigma150",
                "estimate": estimate,
                "cluster_se": se,
                "ci95_low": estimate - 1.96 * se,
                "ci95_high": estimate + 1.96 * se,
                "attempt_rows_joint_model": n,
                "goal_fixed_effects": len(set(goals)),
                "base_layout_clusters": cluster_n,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.attempt_rows)
    output = []
    for model_key in sorted({row["model_key"] for row in rows}):
        output.extend(estimate_model(rows, model_key))
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
