#!/usr/bin/env python3
"""Relate source-task action improvement to held-out prediction transfer.

The primary matched estimand is computed only for visual feedback versus its
shared-attempt-1 retry branch. For each model and canonical source goal, it
relates:

1. feedback-minus-retry improvement in best-so-far source action quality; and
2. feedback-minus-retry change in terminal held-out prediction quality.

The held-out placement is fixed before feedback and never executed in the
solver conversation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def bootstrap_spearman(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> tuple[float, float]:
    if len(rows) < 3:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [rng.choice(rows) for _ in rows]
        result = spearmanr(
            [row["feedback_minus_retry_action_gain"] for row in sample],
            [
                row["feedback_minus_retry_terminal_transfer"]
                for row in sample
            ],
        )
        statistic = float(result.statistic)
        if np.isfinite(statistic):
            values.append(statistic)
    if not values:
        return float("nan"), float("nan")
    return quantile(values, 0.025), quantile(values, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_action_rows", type=Path)
    parser.add_argument("heldout_unit_rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    action_rows = read_csv(args.canonical_action_rows)
    heldout_rows = read_csv(args.heldout_unit_rows)
    actions: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(
        list
    )
    for row in action_rows:
        actions[
            (row["model_key"], row["goal_id"], row["condition"])
        ].append((int(row["attempt"]), float(row["aq_sigma72"])))

    action_gains = {}
    for key, attempts in actions.items():
        ordered = sorted(attempts)
        first = ordered[0][1]
        action_gains[key] = max(value for _, value in ordered) - first

    observations = []
    for row in heldout_rows:
        if row["condition"] != "full":
            continue
        model = row["model_key"]
        goal = row["source_goal_id"]
        full_key = (model, goal, "full")
        retry_key = (model, goal, "neither")
        if full_key not in action_gains or retry_key not in action_gains:
            continue
        if (
            row.get("early_delta_minus_neither") in (None, "")
            or row.get("terminal_delta_minus_neither") in (None, "")
        ):
            continue
        observations.append(
            {
                "model_key": model,
                "source_goal_id": goal,
                "layout_id": row["layout_id"],
                "gravity": row["gravity"],
                "feedback_minus_retry_action_gain": (
                    action_gains[full_key] - action_gains[retry_key]
                ),
                "feedback_minus_retry_early_transfer": float(
                    row["early_delta_minus_neither"]
                ),
                "feedback_minus_retry_terminal_transfer": float(
                    row["terminal_delta_minus_neither"]
                ),
            }
        )

    summaries = []
    for model in sorted({row["model_key"] for row in observations}):
        for gravity in ("all", "downward", "upward"):
            group = [
                row
                for row in observations
                if row["model_key"] == model
                and (gravity == "all" or row["gravity"] == gravity)
            ]
            if len(group) < 3:
                continue
            result = spearmanr(
                [row["feedback_minus_retry_action_gain"] for row in group],
                [
                    row["feedback_minus_retry_terminal_transfer"]
                    for row in group
                ],
            )
            low, high = bootstrap_spearman(
                group,
                draws=args.bootstrap_draws,
                seed=args.seed
                + sum(ord(char) for char in f"{model}:{gravity}"),
            )
            summaries.append(
                {
                    "model_key": model,
                    "gravity": gravity,
                    "matched_goals": len(group),
                    "spearman_action_gain_terminal_transfer": float(
                        result.statistic
                    ),
                    "spearman_p_value": float(result.pvalue),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "action_transfer_observations.csv", observations)
    write_csv(args.output_dir / "action_transfer_summary.csv", summaries)
    report = {
        "schema_version": 1,
        "matched_observations": len(observations),
        "models": sorted({row["model_key"] for row in observations}),
        "action_quality": "best-so-far Gaussian AQ, sigma=72 pixels",
        "transfer_quality": (
            "change in role-averaged PQ for the same fixed unexecuted "
            "terminal held-out placement, sigma=150 pixels"
        ),
        "primary_coupling": (
            "Spearman relation between feedback-minus-retry source AQ gain "
            "and feedback-minus-retry terminal held-out PQ change"
        ),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
