#!/usr/bin/env python3
"""Score action-conditioned endpoint predictions in a VTools result tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PRIMARY = ("full", "frames_only", "status_only", "neither")
SIGMA_PX = 150.0
SIGMA_SENSITIVITIES = (75.0, 100.0, 150.0)
TRACE_CACHE: dict[
    str,
    tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, tuple[str, float, float, str | None]],
    ],
] = {}
WORLD_TYPE_CACHE: dict[str, dict[str, str]] = {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def actual_endpoint(
    trace: dict[str, Any], object_id: str
) -> tuple[str, float, float, str | None] | None:
    samples = (trace.get("pose_samples") or {}).get(object_id) or []
    if not samples:
        return None
    final = samples[-1]
    x, y = float(final["x"]), float(final["y"])
    state = "in_scene" if 0 <= x <= 599 and 0 <= y <= 599 else "exited"
    violations = (
        (max(0.0, -x), "left"),
        (max(0.0, x - 599.0), "right"),
        (max(0.0, -y), "bottom"),
        (max(0.0, y - 599.0), "top"),
    )
    exit_side = (
        max(violations, key=lambda item: item[0])[1]
        if state == "exited"
        else None
    )
    return state, x, y, exit_side


def trace_endpoints(
    trace_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, tuple[str, float, float, str | None]],
]:
    key = str(trace_path)
    if key in TRACE_CACHE:
        return TRACE_CACHE[key]
    trace = read_json(trace_path)
    objects = trace.get("objects") or {}
    endpoints: dict[str, tuple[str, float, float, str | None]] = {}
    for object_id, spec in objects.items():
        if not bool((spec or {}).get("is_dynamic")):
            continue
        endpoint = actual_endpoint(trace, str(object_id))
        if endpoint is not None:
            endpoints[str(object_id)] = endpoint
    TRACE_CACHE[key] = (trace, objects, endpoints)
    return trace, objects, endpoints


def prediction_map(attempt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    response = attempt.get("model_response") or {}
    predictions = response.get("coordinate_predictions") or []
    return {
        str(item["id"]): item
        for item in predictions
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def resolve_trace_path(trace_value: str, unit_dir: Path) -> Path | None:
    """Resolve both live absolute paths and paths in a relocated result tree."""
    recorded = Path(trace_value)
    if recorded.exists():
        return recorded
    parts = recorded.parts
    matching = [index for index, part in enumerate(parts) if part == unit_dir.name]
    if matching:
        relocated = unit_dir.joinpath(*parts[matching[-1] + 1 :])
        if relocated.exists():
            return relocated
    candidates = list(unit_dir.rglob(recorded.name))
    return candidates[0] if len(candidates) == 1 else None


def world_object_types(unit_dir: Path) -> dict[str, str]:
    key = str(unit_dir)
    if key in WORLD_TYPE_CACHE:
        return WORLD_TYPE_CACHE[key]
    world_path = unit_dir / "assets" / "simulation_world.json"
    types: dict[str, str] = {"PLACED": "Ball"}
    if world_path.exists():
        world = (read_json(world_path).get("world") or {}).get("objects") or {}
        types.update(
            {
                str(object_id): str((spec or {}).get("type") or "")
                for object_id, spec in world.items()
            }
        )
    WORLD_TYPE_CACHE[key] = types
    return types


def goal_mentions_object(
    goal: dict[str, Any],
    object_id: str,
    object_label: str,
    object_role: str,
) -> bool:
    text = " ".join(
        str(goal.get(key) or "")
        for key in ("goal_text", "goal_text_cleaned", "signature")
    ).lower()
    aliases = {object_label.lower(), object_id.lower()}
    if object_role == "target":
        aliases.add("red ball")
    elif object_role == "goal":
        aliases.update(("green container", "green goal container"))
    elif object_role == "tool":
        aliases.update(
            ("orange ball tool", "dropped tool", "big orange ball")
        )
    return any(alias and alias in text for alias in aliases)


def wrapped_orientation_error(predicted: float, actual: float) -> float:
    return abs((predicted - actual + 180.0) % 360.0 - 180.0)


def score_attempt(
    *,
    summary: dict[str, Any],
    unit_dir: Path,
    condition: str,
    attempt: dict[str, Any],
) -> list[dict[str, Any]]:
    predictions = prediction_map(attempt)
    truth = attempt.get("truth") or {}
    compact = truth.get("prediction_endpoints") or {}
    compact_objects = compact.get("objects") or {}
    if compact_objects:
        objects = {
            str(object_id): {
                **spec,
                "is_dynamic": True,
            }
            for object_id, spec in compact_objects.items()
        }
        trace = {
            "pose_samples": {
                str(object_id): [
                    {
                        "x": float(spec["final_x"]),
                        "y": float(spec["final_y"]),
                        "angle_rad": float(
                            spec.get("final_angle_rad") or 0.0
                        ),
                    }
                ]
                for object_id, spec in compact_objects.items()
            }
        }
        endpoints = {
            str(object_id): actual_endpoint(trace, str(object_id))
            for object_id in compact_objects
        }
    else:
        trace_value = truth.get("structured_trace")
        if not trace_value:
            return []
        trace_path = resolve_trace_path(str(trace_value), unit_dir)
        if trace_path is None:
            return []
        trace, objects, endpoints = trace_endpoints(trace_path)
    object_types = world_object_types(unit_dir)
    rows: list[dict[str, Any]] = []
    for object_id, spec in objects.items():
        if not bool((spec or {}).get("is_dynamic")):
            continue
        actual = endpoints.get(str(object_id))
        if actual is None:
            continue
        actual_state, true_x, true_y, actual_exit_side = actual
        object_label = str((spec or {}).get("label") or object_id)
        object_role = str((spec or {}).get("role") or "")
        object_type = str(
            (spec or {}).get("shape")
            or object_types.get(str(object_id), "")
        )
        is_circular = object_type.lower() == "ball"
        displacement = float(
            (spec or {}).get("total_displacement_px") or 0.0
        )
        actual_affected = int(
            str(object_id) != "PLACED" and displacement > 30.0
        )
        goal_mentioned = int(
            goal_mentions_object(
                summary["goal"],
                str(object_id),
                object_label,
                object_role,
            )
        )
        prediction = predictions.get(str(object_id))
        predicted_state = str((prediction or {}).get("state") or "missing")
        point = (prediction or {}).get("point")
        has_point = (
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) for value in point)
        )
        state_correct = predicted_state == actual_state
        predicted_exit_side = str(
            (prediction or {}).get("exit_side") or ""
        ) or None
        exit_side_correct = (
            predicted_exit_side == actual_exit_side
            if predicted_state == "exited" and actual_state == "exited"
            else None
        )
        actual_orientation = ""
        predicted_orientation = ""
        orientation_error = ""
        pose_samples = (trace.get("pose_samples") or {}).get(
            str(object_id)
        ) or []
        raw_prediction_orientation = (prediction or {}).get(
            "orientation_deg"
        )
        if not is_circular and pose_samples:
            actual_orientation = math.degrees(
                float(pose_samples[-1].get("angle_rad") or 0.0)
            )
            if isinstance(raw_prediction_orientation, (int, float)):
                predicted_orientation = float(raw_prediction_orientation)
                orientation_error = wrapped_orientation_error(
                    predicted_orientation, actual_orientation
                )
        error = math.nan
        quality_by_sigma = {
            sigma: 0.0 for sigma in SIGMA_SENSITIVITIES
        }
        x_half_correct = 0
        y_half_correct = 0
        quadrant_correct = 0
        if predicted_state == "in_scene" and actual_state == "in_scene" and has_point:
            error = math.hypot(float(point[0]) - true_x, float(point[1]) - true_y)
            quality_by_sigma = {
                sigma: math.exp(-(error**2) / (2 * sigma**2))
                for sigma in SIGMA_SENSITIVITIES
            }
            x_half_correct = int(
                (float(point[0]) < 300) == (true_x < 300)
            )
            y_half_correct = int(
                (float(point[1]) < 300) == (true_y < 300)
            )
            quadrant_correct = int(
                x_half_correct and y_half_correct
            )
        elif predicted_state == "exited" and actual_state == "exited":
            matched = 1.0 if exit_side_correct else 0.0
            quality_by_sigma = {
                sigma: matched for sigma in SIGMA_SENSITIVITIES
            }
        rows.append(
            {
                "model_key": summary["model_key"],
                "seed": int(summary.get("seed") or 2026),
                "balanced_goal_id": summary["goal"]["balanced_goal_id"],
                "gravity": summary["goal"]["condition"],
                "condition": condition,
                "attempt": int(attempt["attempt"]),
                "goal_succeeded": int(bool(attempt["goal_succeeded"])),
                "object_id": str(object_id),
                "object_label": object_label,
                "object_role": object_role,
                "object_type": object_type,
                "goal_mentioned": goal_mentioned,
                "total_displacement_px": displacement,
                "actual_affected_gt30px": actual_affected,
                "prediction_present": int(prediction is not None),
                "predicted_state": predicted_state,
                "actual_state": actual_state,
                "state_correct": int(state_correct),
                "predicted_exit_side": predicted_exit_side or "",
                "actual_exit_side": actual_exit_side or "",
                "exit_side_correct": (
                    int(exit_side_correct)
                    if exit_side_correct is not None
                    else ""
                ),
                "true_x": true_x,
                "true_y": true_y,
                "pred_x": float(point[0]) if has_point else "",
                "pred_y": float(point[1]) if has_point else "",
                "pred_orientation_deg": predicted_orientation,
                "actual_orientation_deg": actual_orientation,
                "orientation_error_deg": orientation_error,
                "endpoint_error_px": error if math.isfinite(error) else "",
                "prediction_quality_sigma75": quality_by_sigma[75.0],
                "prediction_quality_sigma100": quality_by_sigma[100.0],
                "prediction_quality_sigma150": quality_by_sigma[150.0],
                "within_50px": int(math.isfinite(error) and error <= 50),
                "within_100px": int(math.isfinite(error) and error <= 100),
                "x_half_correct": x_half_correct,
                "y_half_correct": y_half_correct,
                "quadrant_correct": quadrant_correct,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["condition"], row["gravity"])].append(row)
        grouped[(row["model_key"], row["condition"], "all")].append(row)
    output: list[dict[str, Any]] = []
    for (model, condition, gravity), group in sorted(grouped.items()):
        finite_errors = [
            float(row["endpoint_error_px"])
            for row in group
            if row["endpoint_error_px"] != ""
        ]
        attempts = {
            (row["balanced_goal_id"], int(row["attempt"])) for row in group
        }
        units = {row["balanced_goal_id"] for row in group}
        exited_pairs = [
            row
            for row in group
            if row["predicted_state"] == "exited"
            and row["actual_state"] == "exited"
        ]
        orientation_expected = [
            row for row in group if row["actual_orientation_deg"] != ""
        ]
        orientation_errors = [
            float(row["orientation_error_deg"])
            for row in orientation_expected
            if row["orientation_error_deg"] != ""
        ]
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "gravity": gravity,
                "units": len(units),
                "attempts": len(attempts),
                "object_predictions_scored": len(group),
                "prediction_coverage": mean(
                    [float(row["prediction_present"]) for row in group]
                ),
                "state_accuracy": mean([float(row["state_correct"]) for row in group]),
                "mean_endpoint_error_px_when_both_in_scene": mean(finite_errors),
                "median_endpoint_error_px_when_both_in_scene": median(
                    finite_errors
                ),
                "exit_side_accuracy_when_both_exited": mean(
                    [
                        float(row["exit_side_correct"])
                        for row in exited_pairs
                    ]
                ),
                "orientation_prediction_coverage_non_circular": mean(
                    [
                        float(row["orientation_error_deg"] != "")
                        for row in orientation_expected
                    ]
                ),
                "mean_orientation_error_deg_when_present": mean(
                    orientation_errors
                ),
                "mean_prediction_quality_sigma75": mean(
                    [float(row["prediction_quality_sigma75"]) for row in group]
                ),
                "mean_prediction_quality_sigma100": mean(
                    [float(row["prediction_quality_sigma100"]) for row in group]
                ),
                "mean_prediction_quality_sigma150": mean(
                    [float(row["prediction_quality_sigma150"]) for row in group]
                ),
                "within_50px_rate": mean([float(row["within_50px"]) for row in group]),
                "within_100px_rate": mean([float(row["within_100px"]) for row in group]),
                "x_half_accuracy": mean(
                    [float(row["x_half_correct"]) for row in group]
                ),
                "y_half_accuracy": mean(
                    [float(row["y_half_correct"]) for row in group]
                ),
                "quadrant_accuracy": mean(
                    [float(row["quadrant_correct"]) for row in group]
                ),
            }
        )
    return output


def summarize_by_role(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        role = str(row["object_role"] or "unknown")
        grouped[
            (
                row["model_key"],
                row["condition"],
                row["gravity"],
                role,
            )
        ].append(row)
        grouped[
            (
                row["model_key"],
                row["condition"],
                "all",
                role,
            )
        ].append(row)
    output = []
    for (model, condition, gravity, role), group in sorted(
        grouped.items()
    ):
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "gravity": gravity,
                "object_role": role,
                "object_predictions_scored": len(group),
                "prediction_coverage": mean(
                    [float(row["prediction_present"]) for row in group]
                ),
                "state_accuracy": mean(
                    [float(row["state_correct"]) for row in group]
                ),
                "mean_prediction_quality_sigma150": mean(
                    [
                        float(row["prediction_quality_sigma150"])
                        for row in group
                    ]
                ),
            }
        )
    return output


def summarize_by_dimension(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        dimensions = (
            ("object_role", str(row["object_role"] or "unknown")),
            (
                "goal_relevance",
                "mentioned" if row["goal_mentioned"] else "not_mentioned",
            ),
            (
                "rollout_effect",
                "tool"
                if row["object_id"] == "PLACED"
                else (
                    "affected_gt30px"
                    if row["actual_affected_gt30px"]
                    else "unaffected_le30px"
                ),
            ),
        )
        for dimension, level in dimensions:
            grouped[
                (
                    row["model_key"],
                    row["condition"],
                    row["gravity"],
                    dimension,
                    level,
                )
            ].append(row)
            grouped[
                (
                    row["model_key"],
                    row["condition"],
                    "all",
                    dimension,
                    level,
                )
            ].append(row)
    output = []
    for (
        model,
        condition,
        gravity,
        dimension,
        level,
    ), group in sorted(grouped.items()):
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "gravity": gravity,
                "dimension": dimension,
                "level": level,
                "object_predictions_scored": len(group),
                "prediction_coverage": mean(
                    [float(row["prediction_present"]) for row in group]
                ),
                "state_accuracy": mean(
                    [float(row["state_correct"]) for row in group]
                ),
                "mean_prediction_quality_sigma150": mean(
                    [
                        float(row["prediction_quality_sigma150"])
                        for row in group
                    ]
                ),
            }
        )
    return output


def attempt_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, int, str, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model_key"],
                int(row["seed"]),
                row["condition"],
                row["balanced_goal_id"],
                int(row["attempt"]),
            )
        ].append(row)
    output = []
    for (model, seed, condition, goal_id, attempt), group in sorted(
        grouped.items()
    ):
        output.append(
            {
                "model_key": model,
                "seed": seed,
                "condition": condition,
                "balanced_goal_id": goal_id,
                "gravity": group[0]["gravity"],
                "attempt": attempt,
                "goal_succeeded": group[0]["goal_succeeded"],
                "objects": len(group),
                "prediction_coverage": mean(
                    [float(row["prediction_present"]) for row in group]
                ),
                "prediction_quality_sigma75": mean(
                    [
                        float(row["prediction_quality_sigma75"])
                        for row in group
                    ]
                ),
                "prediction_quality_sigma100": mean(
                    [
                        float(row["prediction_quality_sigma100"])
                        for row in group
                    ]
                ),
                "prediction_quality_sigma150": mean(
                    [
                        float(row["prediction_quality_sigma150"])
                        for row in group
                    ]
                ),
            }
        )
    return output


def first_last_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_attempt: dict[
        tuple[str, int, str, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            row["model_key"],
            int(row["seed"]),
            row["condition"],
            row["balanced_goal_id"],
            int(row["attempt"]),
        )
        by_attempt[key].append(row)
    attempt_scores = {
        key: mean(
            [float(row["prediction_quality_sigma150"]) for row in object_rows]
        )
        for key, object_rows in by_attempt.items()
    }
    by_unit: dict[
        tuple[str, int, str, str], list[tuple[int, float]]
    ] = defaultdict(list)
    for (
        model,
        seed,
        condition,
        goal_id,
        attempt,
    ), value in attempt_scores.items():
        by_unit[(model, seed, condition, goal_id)].append((attempt, value))
    grouped: dict[
        tuple[str, int, str], list[tuple[float, float]]
    ] = defaultdict(list)
    for (model, seed, condition, _goal_id), values in by_unit.items():
        ordered = sorted(values)
        grouped[(model, seed, condition)].append(
            (ordered[0][1], ordered[-1][1])
        )
    output: list[dict[str, Any]] = []
    for (model, seed, condition), values in sorted(grouped.items()):
        first = [item[0] for item in values]
        last = [item[1] for item in values]
        output.append(
            {
                "model_key": model,
                "seed": seed,
                "condition": condition,
                "units": len(values),
                "mean_first_attempt_prediction_quality": mean(first),
                "mean_last_attempt_prediction_quality": mean(last),
                "mean_last_minus_first": mean(
                    [last_value - first_value for first_value, last_value in values]
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-last-only", action="store_true")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(args.result_root.rglob("unit_summary.json")):
        summary = read_json(summary_path)
        unit_dir = summary_path.parent
        shared = read_json(unit_dir / "shared_attempt_1" / "state.json")
        shared_attempt = shared["attempt"]
        for condition in PRIMARY:
            state_path = unit_dir / "branches" / condition / "state.json"
            if not state_path.exists():
                continue
            state = read_json(state_path)
            attempts = state.get("attempts") or []
            if args.first_last_only and attempts:
                attempts = (
                    [attempts[0]]
                    if len(attempts) == 1
                    else [attempts[0], attempts[-1]]
                )
            for attempt in attempts:
                rows.extend(
                    score_attempt(
                        summary=summary,
                        unit_dir=unit_dir,
                        condition=condition,
                        attempt=attempt,
                    )
                )
        trace_state_path = unit_dir / "branches" / "trace_status" / "state.json"
        if trace_state_path.exists():
            trace_state = read_json(trace_state_path)
            attempts = trace_state.get("attempts") or []
            if args.first_last_only and attempts:
                attempts = (
                    [attempts[0]]
                    if len(attempts) == 1
                    else [attempts[0], attempts[-1]]
                )
            for attempt in attempts:
                rows.extend(
                    score_attempt(
                        summary=summary,
                        unit_dir=unit_dir,
                        condition="trace_status",
                        attempt=attempt,
                    )
                )
    write_csv(args.output_dir / "action_prediction_object_rows.csv", rows)
    write_csv(args.output_dir / "action_prediction_summary.csv", summarize(rows))
    write_csv(
        args.output_dir / "action_prediction_role_summary.csv",
        summarize_by_role(rows),
    )
    write_csv(
        args.output_dir / "action_prediction_group_summary.csv",
        summarize_by_dimension(rows),
    )
    write_csv(
        args.output_dir / "action_prediction_attempt_rows.csv",
        attempt_summary(rows),
    )
    write_csv(
        args.output_dir / "action_prediction_first_last.csv",
        first_last_summary(rows),
    )
    print(
        json.dumps(
            {
                "object_rows": len(rows),
                "summaries": summarize(rows),
                "first_last": first_last_summary(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
