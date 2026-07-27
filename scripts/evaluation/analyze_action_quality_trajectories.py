#!/usr/bin/env python3
"""Analyze matched feedback-arm action-quality trajectories.

Primary curves use best-so-far AQ through each attempt and carry that value
forward after a branch terminates. This keeps the matched goal population fixed
across attempts and avoids conditioning later-attempt comparisons on continued
failure. Raw action AQ at attempt t is emitted separately as a risk-set
description.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ARMS = ("full", "trace_status")
REFERENCE = "neither"
ATTEMPTS = tuple(range(1, 9))
GRAVITY_SUFFIX = re.compile(r"_(upward|downward)$")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def layout_id(puzzle_key: str) -> str:
    return GRAVITY_SUFFIX.sub("", puzzle_key)


def summarize_curve_rows(
    units: dict[tuple[str, int, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, int], list[dict[str, float]]
    ] = defaultdict(list)
    for (model, _seed, _goal, condition), attempts in units.items():
        ordered = sorted(attempts, key=lambda row: int(row["attempt"]))
        gravity = str(ordered[0]["gravity"])
        by_attempt = {int(row["attempt"]): row for row in ordered}
        running_best = -math.inf
        running_min_distance = math.inf
        for attempt in ATTEMPTS:
            current = by_attempt.get(attempt)
            if current is not None:
                running_best = max(
                    running_best, float(current["original_aq_sigma72"])
                )
                running_min_distance = min(
                    running_min_distance,
                    float(current["original_nearest_solution_distance_px"]),
                )
            if running_best == -math.inf:
                continue
            values = {
                "best_aq": running_best,
                "best_distance": running_min_distance,
                "current_aq": (
                    float(current["original_aq_sigma72"])
                    if current is not None
                    else math.nan
                ),
                "current_distance": (
                    float(current["original_nearest_solution_distance_px"])
                    if current is not None
                    else math.nan
                ),
            }
            for gravity_level in (gravity, "all"):
                grouped[(model, condition, gravity_level, attempt)].append(
                    values
                )

    output = []
    for (model, condition, gravity, attempt), values in sorted(
        grouped.items()
    ):
        current_aq = [
            row["current_aq"]
            for row in values
            if not math.isnan(row["current_aq"])
        ]
        current_distance = [
            row["current_distance"]
            for row in values
            if not math.isnan(row["current_distance"])
        ]
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "gravity": gravity,
                "attempt": attempt,
                "fixed_population_units": len(values),
                "risk_set_units": len(current_aq),
                "mean_best_so_far_aq_sigma72": mean(
                    [row["best_aq"] for row in values]
                ),
                "mean_best_so_far_distance_px": mean(
                    [row["best_distance"] for row in values]
                ),
                "mean_current_action_aq_sigma72_risk_set": mean(current_aq),
                "mean_current_action_distance_px_risk_set": mean(
                    current_distance
                ),
            }
        )
    return output


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> tuple[float | None, float | None, float | None, int]:
    by_layout: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_layout[str(row["layout_id"])].append(float(row["delta"]))
    layout_means = {
        key: sum(values) / len(values) for key, values in by_layout.items()
    }
    observed = mean(list(layout_means.values()))
    if observed is None:
        return None, None, None, 0
    keys = sorted(layout_means)
    if len(keys) == 1:
        return observed, observed, observed, 1
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        sampled = [layout_means[rng.choice(keys)] for _ in keys]
        boot.append(sum(sampled) / len(sampled))
    return (
        observed,
        quantile(boot, 0.025),
        quantile(boot, 0.975),
        len(keys),
    )


def matched_contrasts(
    units: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, int, str], dict[str, list[dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for (model, unit_seed, goal, condition), attempts in units.items():
        indexed[(model, unit_seed, goal)][condition] = attempts

    observation_rows: list[dict[str, Any]] = []
    for (model, unit_seed, goal), branches in sorted(indexed.items()):
        reference = branches.get(REFERENCE)
        if not reference:
            continue
        reference_by_attempt = {
            int(row["attempt"]): row for row in reference
        }
        reference_first = reference_by_attempt.get(1)
        if reference_first is None or bool(reference_first["original_success"]):
            continue
        for arm in ARMS:
            treatment = branches.get(arm)
            if not treatment:
                continue
            treatment_by_attempt = {
                int(row["attempt"]): row for row in treatment
            }
            treatment_first = treatment_by_attempt.get(1)
            if treatment_first is None:
                continue
            if treatment_first["original_coords"] != reference_first[
                "original_coords"
            ]:
                continue
            treatment_best = -math.inf
            reference_best = -math.inf
            treatment_min_distance = math.inf
            reference_min_distance = math.inf
            gravity = str(reference_first["gravity"])
            base_layout = layout_id(str(reference_first["puzzle_key"]))
            for attempt in ATTEMPTS:
                treatment_current = treatment_by_attempt.get(attempt)
                reference_current = reference_by_attempt.get(attempt)
                if treatment_current is not None:
                    treatment_best = max(
                        treatment_best,
                        float(treatment_current["original_aq_sigma72"]),
                    )
                    treatment_min_distance = min(
                        treatment_min_distance,
                        float(
                            treatment_current[
                                "original_nearest_solution_distance_px"
                            ]
                        ),
                    )
                if reference_current is not None:
                    reference_best = max(
                        reference_best,
                        float(reference_current["original_aq_sigma72"]),
                    )
                    reference_min_distance = min(
                        reference_min_distance,
                        float(
                            reference_current[
                                "original_nearest_solution_distance_px"
                            ]
                        ),
                    )
                if treatment_best == -math.inf or reference_best == -math.inf:
                    continue
                for gravity_level in (gravity, "all"):
                    observation_rows.append(
                        {
                            "model_key": model,
                            "seed": unit_seed,
                            "goal_id": goal,
                            "layout_id": base_layout,
                            "gravity": gravity_level,
                            "arm": arm,
                            "attempt": attempt,
                            "outcome": "best_so_far_aq_sigma72",
                            "treatment": treatment_best,
                            "reference": reference_best,
                            "delta": treatment_best - reference_best,
                        }
                    )
                    observation_rows.append(
                        {
                            "model_key": model,
                            "seed": unit_seed,
                            "goal_id": goal,
                            "layout_id": base_layout,
                            "gravity": gravity_level,
                            "arm": arm,
                            "attempt": attempt,
                            "outcome": "best_so_far_distance_px",
                            "treatment": treatment_min_distance,
                            "reference": reference_min_distance,
                            "delta": (
                                treatment_min_distance
                                - reference_min_distance
                            ),
                        }
                    )

    grouped: dict[
        tuple[str, str, str, int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in observation_rows:
        grouped[
            (
                str(row["model_key"]),
                str(row["arm"]),
                str(row["gravity"]),
                int(row["attempt"]),
                str(row["outcome"]),
            )
        ].append(row)
    summaries = []
    for key, rows in sorted(grouped.items()):
        model, arm, gravity, attempt, outcome = key
        effect, low, high, layouts = cluster_bootstrap(
            rows,
            draws=draws,
            seed=seed
            + attempt
            + sum(ord(char) for char in f"{model}:{arm}:{gravity}:{outcome}"),
        )
        summaries.append(
            {
                "model_key": model,
                "arm": arm,
                "reference": REFERENCE,
                "gravity": gravity,
                "attempt": attempt,
                "outcome": outcome,
                "matched_units": len(rows),
                "base_layouts": layouts,
                "mean_arm_minus_retry": effect,
                "ci95_low": low,
                "ci95_high": high,
                "eligibility": "shared_attempt_1_failed",
            }
        )
    return observation_rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinate_action_rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = list(read_jsonl(args.coordinate_action_rows))
    units: dict[
        tuple[str, int, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        units[
            (
                str(row["model_key"]),
                int(row["seed"]),
                str(row["goal_id"]),
                str(row["condition"]),
            )
        ].append(row)

    curves = summarize_curve_rows(units)
    observations, contrasts = matched_contrasts(
        units,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "action_quality_curves.csv", curves)
    write_csv(
        args.output_dir / "paired_action_quality_observations.csv",
        observations,
    )
    write_csv(
        args.output_dir / "paired_action_quality_contrasts.csv",
        contrasts,
    )
    report = {
        "schema_version": 1,
        "coordinate_action_rows": len(rows),
        "model_goal_condition_units": len(units),
        "matched_observations": len(observations),
        "contrast_rows": len(contrasts),
        "primary_estimand": (
            "layout-equal-weighted feedback-arm minus retry-only difference "
            "in best-so-far AQ through attempt t among shared-attempt-1 "
            "failures; solved branches are carried forward"
        ),
        "secondary_estimand": (
            "current proposed-action AQ among branches still at risk at "
            "attempt t; descriptive because the risk set changes"
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
