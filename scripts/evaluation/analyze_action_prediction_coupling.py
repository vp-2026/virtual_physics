#!/usr/bin/env python3
"""Join offline action quality with prospective prediction quality."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


ARM_ORDER = ("full", "frames_only", "status_only", "neither", "trace_status")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def first_numeric(row: dict[str, Any], *fields: str) -> float:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of the required fields are present: {fields}")


def gaussian_action_quality(row: dict[str, Any], sigma: float) -> float:
    field = f"aq_sigma{int(sigma)}"
    original_field = f"original_{field}"
    for candidate in (original_field, field):
        value = row.get(candidate)
        if value not in (None, ""):
            return float(value)
    distance = first_numeric(
        row,
        "original_nearest_solution_distance_px",
        "nearest_solution_distance_px",
    )
    return math.exp(-(distance**2) / (2.0 * sigma**2))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def normalize_action_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize noncanonical replay and canonical April-solution schemas."""
    normalized = dict(row)
    normalized.setdefault("seed", 2026)
    if not normalized.get("puzzle_key"):
        normalized["puzzle_key"] = re.sub(
            r"^canonical_\d+_", "", str(normalized.get("goal_id", ""))
        )
    if normalized.get("original_success") in (None, ""):
        normalized["original_success"] = normalized.get(
            "goal_succeeded", False
        )
    if normalized.get("nearest_solution_distance_px") in (None, ""):
        normalized["nearest_solution_distance_px"] = normalized.get(
            "nearest_april_solution_distance_px"
        )
    normalized.setdefault("category", "canonical")
    return normalized


def join_rows(
    coordinate_rows: Iterable[dict[str, Any]],
    prediction_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prediction_by_key = {
        (
            row["model_key"],
            int(row.get("seed") or 2026),
            row["condition"],
            row["balanced_goal_id"],
            int(row["attempt"]),
        ): row
        for row in prediction_rows
    }
    joined = []
    missing = 0
    for action in coordinate_rows:
        key = (
            str(action["model_key"]),
            int(action.get("seed") or 2026),
            str(action["condition"]),
            str(action["goal_id"]),
            int(action["attempt"]),
        )
        prediction = prediction_by_key.get(key)
        if prediction is None:
            missing += 1
            continue
        puzzle_key = str(action["puzzle_key"])
        joined.append(
            {
                "model_key": key[0],
                "seed": key[1],
                "condition": key[2],
                "goal_id": key[3],
                "attempt": key[4],
                "puzzle_key": puzzle_key,
                "base_layout": puzzle_key.rsplit("_", 1)[0],
                "gravity": str(action["gravity"]),
                "category": str(action["category"]),
                "success": int(parse_bool(action["original_success"])),
                "aq_sigma36": gaussian_action_quality(action, 36.0),
                "aq_sigma72": gaussian_action_quality(action, 72.0),
                "aq_sigma144": gaussian_action_quality(action, 144.0),
                "nearest_solution_distance_px": first_numeric(
                    action,
                    "original_nearest_solution_distance_px",
                    "nearest_solution_distance_px",
                ),
                "pq_sigma75": float(
                    prediction["prediction_quality_sigma75"]
                ),
                "pq_sigma100": float(
                    prediction["prediction_quality_sigma100"]
                ),
                "pq_sigma150": float(
                    prediction["prediction_quality_sigma150"]
                ),
                "prediction_coverage": float(
                    prediction["prediction_coverage"]
                ),
            }
        )
    return joined, {
        "coordinate_action_rows": len(list(coordinate_rows))
        if isinstance(coordinate_rows, list)
        else len(joined) + missing,
        "prediction_attempt_rows": len(prediction_rows),
        "joined_rows": len(joined),
        "coordinate_rows_without_prediction_row": missing,
    }


def correlation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_key"], row["condition"], row["gravity"])].append(
            row
        )
        groups[(row["model_key"], row["condition"], "all")].append(row)
    output = []
    for (model, condition, gravity), group in sorted(groups.items()):
        aq = [float(row["aq_sigma72"]) for row in group]
        pq = [float(row["pq_sigma150"]) for row in group]
        result = spearmanr(aq, pq)
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "gravity": gravity,
                "attempts": len(group),
                "spearman_aq72_pq150": (
                    float(result.statistic)
                    if math.isfinite(float(result.statistic))
                    else ""
                ),
                "spearman_p_value": (
                    float(result.pvalue)
                    if math.isfinite(float(result.pvalue))
                    else ""
                ),
                "mean_aq72": mean(aq),
                "mean_pq150": mean(pq),
                "mean_pq150_success": mean(
                    [
                        float(row["pq_sigma150"])
                        for row in group
                        if row["success"]
                    ]
                ),
                "mean_pq150_failure": mean(
                    [
                        float(row["pq_sigma150"])
                        for row in group
                        if not row["success"]
                    ]
                ),
            }
        )
    return output


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


def clustered_goal_fe_model(
    rows: list[dict[str, Any]],
    *,
    model_key: str,
) -> list[dict[str, Any]]:
    group = [row for row in rows if row["model_key"] == model_key]
    arm_levels = [arm for arm in ARM_ORDER[1:] if any(
        row["condition"] == arm for row in group
    )]
    names = ["pq_sigma150", "attempt"]
    names.extend(f"arm_{arm}" for arm in arm_levels)
    names.extend(f"pq_x_{arm}" for arm in arm_levels)
    x_rows = []
    for row in group:
        pq = float(row["pq_sigma150"])
        arm = str(row["condition"])
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
    goals = [str(row["goal_id"]) for row in group]
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
        clusters[str(row["base_layout"])].append(index)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for indices in clusters.values():
        score = x[indices].T @ residual[indices]
        meat += np.outer(score, score)
    n, k = x.shape
    g = len(clusters)
    correction = (
        (g / (g - 1)) * ((n - 1) / (n - k))
        if g > 1 and n > k
        else 1.0
    )
    covariance = correction * bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return [
        {
            "model_key": model_key,
            "coefficient": name,
            "estimate": float(estimate),
            "cluster_se": float(se),
            "ci95_low": float(estimate - 1.96 * se),
            "ci95_high": float(estimate + 1.96 * se),
            "attempt_rows": n,
            "goal_fixed_effects": len(set(goals)),
            "layout_clusters": g,
            "reference_arm": "full",
        }
        for name, estimate, se in zip(kept_names, beta, standard_errors)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinate_action_rows", type=Path)
    parser.add_argument("prediction_attempt_rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    coordinate_rows = [
        normalize_action_row(row)
        for row in (
            read_csv(args.coordinate_action_rows)
            if args.coordinate_action_rows.suffix.lower() == ".csv"
            else list(read_jsonl(args.coordinate_action_rows))
        )
    ]
    prediction_rows = read_csv(args.prediction_attempt_rows)
    joined, report = join_rows(coordinate_rows, prediction_rows)
    correlations = correlation_summary(joined)
    coefficients = []
    for model_key in sorted({row["model_key"] for row in joined}):
        coefficients.extend(
            clustered_goal_fe_model(joined, model_key=model_key)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aq_pq_attempt_rows.csv", joined)
    write_csv(args.output_dir / "aq_pq_correlations.csv", correlations)
    write_csv(
        args.output_dir / "aq_pq_goal_fe_clustered_coefficients.csv",
        coefficients,
    )
    report.update(
        {
            "models": sorted({row["model_key"] for row in joined}),
            "correlation_rows": len(correlations),
            "regression_coefficient_rows": len(coefficients),
            "aq_definition": "exp(-distance_to_solution_set^2/(2*72^2))",
            "pq_definition": "mean object PQ at sigma=150 pixels",
            "regression": (
                "per-model goal-fixed-effect OLS of AQ on PQ, arm, attempt, "
                "and PQ-by-arm interactions; CR1 uncertainty clustered by "
                "the 66 base layouts"
            ),
        }
    )
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
