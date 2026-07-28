#!/usr/bin/env python3
"""Analyze matched held-out counterfactual prediction transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CONDITIONS = ("full", "frames_only", "status_only", "neither", "trace_status")
SIGMA_PX = 150.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_recorded_path(value: str, unit_dir: Path) -> Path | None:
    recorded = Path(value)
    if recorded.exists():
        return recorded
    matching = [
        index
        for index, part in enumerate(recorded.parts)
        if part == unit_dir.name
    ]
    if matching:
        relocated = unit_dir.joinpath(
            *recorded.parts[matching[-1] + 1 :]
        )
        if relocated.exists():
            return relocated
    candidates = list(unit_dir.rglob(recorded.name))
    return candidates[0] if len(candidates) == 1 else None


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_row_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "object_count": float(len(rows)),
        "prediction_coverage": mean(
            float(bool(row["prediction_present"])) for row in rows
        ),
        "state_accuracy": mean(
            float(bool(row["state_correct"])) for row in rows
        ),
        "prediction_quality_sigma150": mean(
            float(row["prediction_quality_sigma150"]) for row in rows
        ),
        "within_50px_rate_all_objects": mean(
            float(bool(row["within_50px"])) for row in rows
        ),
        "within_100px_rate_all_objects": mean(
            float(bool(row["within_100px"])) for row in rows
        ),
    }


def attempt_prediction_score(
    attempt: dict[str, Any],
    *,
    unit_dir: Path,
) -> dict[str, float]:
    response = attempt.get("model_response") or {}
    predictions = response.get("coordinate_predictions") or []
    prediction_map = {
        str(item["id"]): item
        for item in predictions
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    truth = attempt.get("truth") or {}
    compact_objects = (
        (truth.get("prediction_endpoints") or {}).get("objects") or {}
    )
    trace_value = truth.get("structured_trace")
    trace_path = None
    if not compact_objects and trace_value:
        trace_path = resolve_recorded_path(str(trace_value), unit_dir)
    if not compact_objects and trace_path is None:
        return {
            "prediction_coverage": math.nan,
            "state_accuracy": math.nan,
            "prediction_quality_sigma150": math.nan,
        }
    if compact_objects:
        trace = {
            "objects": {
                str(object_id): {
                    **spec,
                    "is_dynamic": True,
                }
                for object_id, spec in compact_objects.items()
            },
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
            },
        }
    else:
        trace = read_json(trace_path)
    rows = []
    for object_id, spec in (trace.get("objects") or {}).items():
        if not bool((spec or {}).get("is_dynamic")):
            continue
        samples = (trace.get("pose_samples") or {}).get(object_id) or []
        if not samples:
            continue
        final = samples[-1]
        x, y = float(final["x"]), float(final["y"])
        actual_state = (
            "in_scene" if 0 <= x <= 599 and 0 <= y <= 599 else "exited"
        )
        violations = (
            (max(0.0, -x), "left"),
            (max(0.0, x - 599.0), "right"),
            (max(0.0, -y), "bottom"),
            (max(0.0, y - 599.0), "top"),
        )
        actual_exit_side = (
            max(violations, key=lambda item: item[0])[1]
            if actual_state == "exited"
            else None
        )
        prediction = prediction_map.get(str(object_id))
        predicted_state = str((prediction or {}).get("state") or "missing")
        point = (prediction or {}).get("point")
        has_point = (
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) for value in point)
        )
        quality = 0.0
        if (
            actual_state == "in_scene"
            and predicted_state == "in_scene"
            and has_point
        ):
            error = math.dist(
                [x, y], [float(point[0]), float(point[1])]
            )
            quality = math.exp(-(error**2) / (2 * SIGMA_PX**2))
        elif actual_state == "exited" and predicted_state == "exited":
            quality = float(
                str((prediction or {}).get("exit_side") or "")
                == str(actual_exit_side)
            )
        rows.append(
            {
                "present": prediction is not None,
                "state_correct": predicted_state == actual_state,
                "quality": quality,
            }
        )
    return {
        "prediction_coverage": mean(
            float(bool(row["present"])) for row in rows
        ),
        "state_accuracy": mean(
            float(bool(row["state_correct"])) for row in rows
        ),
        "prediction_quality_sigma150": mean(
            float(row["quality"]) for row in rows
        ),
    }


def grouped_summary(
    rows: list[dict[str, Any]],
    *,
    value_fields: tuple[str, ...],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups[key].append(row)
        if "gravity" in group_fields:
            all_key = tuple(
                "all" if field == "gravity" else row[field]
                for field in group_fields
            )
            groups[all_key].append(row)
    output = []
    for key, group in sorted(groups.items()):
        record = {
            field: value for field, value in zip(group_fields, key)
        }
        record["units"] = len(group)
        for field in value_fields:
            values = [
                float(row[field])
                for row in group
                if row.get(field) is not None
                and math.isfinite(float(row[field]))
            ]
            record[f"mean_{field}"] = mean(values)
        output.append(record)
    return output


def clustered_summary(
    rows: list[dict[str, Any]],
    *,
    value_fields: tuple[str, ...],
    group_fields: tuple[str, ...],
    bootstrap_draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Equal-layout means and layout-cluster percentile intervals."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        keys = [()]
        for field in group_fields:
            values = [row[field]]
            if field in {"model_key", "gravity"} and row[field] != "all":
                values.append("all")
            keys = [
                (*prefix, value)
                for prefix in keys
                for value in values
            ]
        for key in keys:
            groups[key].append(row)

    output = []
    for key, group in sorted(groups.items()):
        record = {
            field: value for field, value in zip(group_fields, key)
        }
        record["units"] = len(group)
        record["layouts"] = len(
            {str(row["layout_id"]) for row in group}
        )
        for field in value_fields:
            layout_values = []
            for layout_id in sorted(
                {str(row["layout_id"]) for row in group}
            ):
                layout_rows = [
                    row
                    for row in group
                    if str(row["layout_id"]) == layout_id
                    and row.get(field) is not None
                    and math.isfinite(float(row[field]))
                ]
                if not layout_rows:
                    continue
                if record.get("model_key") == "all":
                    model_values = []
                    for model_key in sorted(
                        {str(row["model_key"]) for row in layout_rows}
                    ):
                        model_values.append(
                            mean(
                                float(row[field])
                                for row in layout_rows
                                if str(row["model_key"]) == model_key
                            )
                        )
                    layout_values.append(mean(model_values))
                else:
                    layout_values.append(
                        mean(float(row[field]) for row in layout_rows)
                    )
            record[f"mean_{field}"] = mean(layout_values)
            record[f"layouts_{field}"] = len(layout_values)
            if not layout_values or bootstrap_draws <= 0:
                record[f"ci95_low_{field}"] = math.nan
                record[f"ci95_high_{field}"] = math.nan
                continue
            seed_material = (
                f"{seed}|{key}|{field}|{len(layout_values)}"
            ).encode("utf-8")
            group_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8],
                "big",
            )
            generator = np.random.default_rng(group_seed)
            values = np.asarray(layout_values, dtype=float)
            indices = generator.integers(
                0,
                len(values),
                size=(bootstrap_draws, len(values)),
            )
            draws = values[indices].mean(axis=1)
            record[f"ci95_low_{field}"] = float(
                np.quantile(draws, 0.025)
            )
            record[f"ci95_high_{field}"] = float(
                np.quantile(draws, 0.975)
            )
        output.append(record)
    return output


def collect_units(result_root: Path) -> tuple[
    list[tuple[Path, dict[str, Any], dict[str, Any]]],
    dict[tuple[str, str], tuple[Path, dict[str, Any]]],
]:
    by_model_goal = {}
    for summary_path in sorted(result_root.rglob("unit_summary.json")):
        summary = read_json(summary_path)
        by_model_goal[
            (
                str(summary["model_key"]),
                str(summary["goal"]["balanced_goal_id"]),
            )
        ] = (summary_path.parent, summary)
    units = []
    for transfer_path in sorted(
        result_root.rglob("transfer_sidecars/summary.json")
    ):
        unit_dir = transfer_path.parent.parent
        main_summary = read_json(unit_dir / "unit_summary.json")
        transfer = read_json(transfer_path)
        units.append((unit_dir, main_summary, transfer))
    return units, by_model_goal


def prediction_result(
    compact: dict[str, Any],
    *,
    unit_dir: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    result_path = resolve_recorded_path(
        str(compact["result_path"]), unit_dir
    )
    if result_path is None:
        raise FileNotFoundError(compact["result_path"])
    raw = read_json(result_path)
    return raw, score_row_summary(raw.get("score_rows") or [])


def usage_summary(result_root: Path) -> dict[str, Any]:
    cost = 0.0
    calls = 0
    returned_models: dict[str, int] = defaultdict(int)
    finish_reasons: dict[str, int] = defaultdict(int)
    for path in result_root.rglob("provider_calls.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("provider_metadata") or {}
            usage = metadata.get("usage") or {}
            cost += float(usage.get("cost") or 0.0)
            calls += 1
            returned_models[str(metadata.get("returned_model"))] += 1
            finish_reasons[str(metadata.get("finish_reason"))] += 1
    return {
        "provider_call_records": calls,
        "reported_cost_usd": cost,
        "returned_models": dict(sorted(returned_models.items())),
        "finish_reasons": dict(sorted(finish_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.result_root = args.result_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    units, by_model_goal = collect_units(args.result_root)
    heldout_rows = []
    clear_path_rows = []
    format_rows = []

    for unit_dir, main_summary, transfer in units:
        model = str(main_summary["model_key"])
        goal = main_summary["goal"]
        gravity = str(goal["condition"])
        layout_id = str(goal["puzzle_key"]).rsplit("_", 1)[0]
        source_goal_id = str(goal["balanced_goal_id"])
        pre = transfer["results"]["pre_feedback"]
        visual_early_raw, visual_early = prediction_result(
            pre["visual_early_probe"],
            unit_dir=unit_dir,
        )
        visual_terminal_raw, visual_terminal = prediction_result(
            pre["visual_terminal_probe"],
            unit_dir=unit_dir,
        )
        trace_early_raw, trace_early = prediction_result(
            pre["trace_early_probe"],
            unit_dir=unit_dir,
        )
        trace_terminal_raw, trace_terminal = prediction_result(
            pre["trace_terminal_probe"],
            unit_dir=unit_dir,
        )
        for label, raw in (
            ("pre_visual_early", visual_early_raw),
            ("pre_visual_terminal", visual_terminal_raw),
            ("pre_trace_early", trace_early_raw),
            ("pre_trace_terminal", trace_terminal_raw),
        ):
            format_rows.append(
                {
                    "model_key": model,
                    "gravity": gravity,
                    "layout_id": layout_id,
                    "source_goal_id": source_goal_id,
                    "condition": label,
                    "schema_valid": int(bool(raw.get("schema_valid"))),
                    "provider_call_count": int(
                        raw.get("provider_call_count") or 0
                    ),
                }
            )

        clear_compact = pre.get("visual_clear_path_probe") or {
            "available": False
        }
        clear_selection = (
            transfer.get("probe_selection") or {}
        ).get("clear_path_probe") or {}
        clear_coords = clear_selection.get("coords") or [None, None]
        if clear_compact.get("available") is False:
            clear_path_rows.append(
                {
                    "model_key": model,
                    "gravity": gravity,
                    "layout_id": layout_id,
                    "source_goal_id": source_goal_id,
                    "available": 0,
                    "expected_tool_exit_side": clear_selection.get(
                        "expected_tool_exit_side"
                    ),
                    "placement_x": clear_coords[0],
                    "placement_y": clear_coords[1],
                    "schema_valid": None,
                    "tool_prediction_present": None,
                    "tool_state_accuracy": None,
                    "tool_exit_side_accuracy": None,
                    "tool_prediction_quality_sigma150": None,
                    "all_object_prediction_quality_sigma150": None,
                }
            )
        else:
            clear_raw, clear_all = prediction_result(
                clear_compact,
                unit_dir=unit_dir,
            )
            clear_tool_rows = [
                row
                for row in (clear_raw.get("score_rows") or [])
                if str(row.get("object_id")) == "PLACED"
            ]
            clear_tool = score_row_summary(clear_tool_rows)
            clear_exit_rows = [
                row
                for row in clear_tool_rows
                if row.get("exit_side_correct") is not None
            ]
            clear_exit_accuracy = mean(
                float(bool(row["exit_side_correct"]))
                for row in clear_exit_rows
            )
            clear_path_rows.append(
                {
                    "model_key": model,
                    "gravity": gravity,
                    "layout_id": layout_id,
                    "source_goal_id": source_goal_id,
                    "available": 1,
                    "expected_tool_exit_side": clear_selection.get(
                        "expected_tool_exit_side"
                    ),
                    "placement_x": clear_coords[0],
                    "placement_y": clear_coords[1],
                    "schema_valid": int(
                        bool(clear_compact.get("schema_valid"))
                    ),
                    "tool_prediction_present": clear_tool[
                        "prediction_coverage"
                    ],
                    "tool_state_accuracy": clear_tool["state_accuracy"],
                    "tool_exit_side_accuracy": clear_exit_accuracy,
                    "tool_prediction_quality_sigma150": clear_tool[
                        "prediction_quality_sigma150"
                    ],
                    "all_object_prediction_quality_sigma150": clear_all[
                        "prediction_quality_sigma150"
                    ],
                }
            )
            format_rows.append(
                {
                    "model_key": model,
                    "gravity": gravity,
                    "layout_id": layout_id,
                    "source_goal_id": source_goal_id,
                    "condition": "pre_visual_clear_path",
                    "schema_valid": int(
                        bool(clear_compact.get("schema_valid"))
                    ),
                    "provider_call_count": int(
                        clear_compact.get("provider_call_count") or 0
                    ),
                }
            )

        active_conditions = transfer["results"]["conditions"]
        for condition in CONDITIONS:
            if condition not in active_conditions:
                continue
            branch = active_conditions[condition]
            early_raw, early_post = prediction_result(
                branch["early_prediction"],
                unit_dir=unit_dir,
            )
            terminal_raw, terminal_post = prediction_result(
                branch["terminal_prediction"],
                unit_dir=unit_dir,
            )
            if condition == "trace_status":
                early_pre = trace_early
                terminal_pre = trace_terminal
            else:
                early_pre = visual_early
                terminal_pre = visual_terminal
            heldout_rows.append(
                {
                    "model_key": model,
                    "gravity": gravity,
                    "layout_id": layout_id,
                    "source_goal_id": source_goal_id,
                    "condition": condition,
                    "source_attempt_count": int(
                        branch["source_attempt_count"]
                    ),
                    "source_first_success_attempt": (
                        branch["source_first_success_attempt"]
                    ),
                    "terminal_probe_contaminated": int(
                        bool(branch["terminal_probe_contaminated"])
                    ),
                    "early_pre_quality": early_pre[
                        "prediction_quality_sigma150"
                    ],
                    "early_post_quality": early_post[
                        "prediction_quality_sigma150"
                    ],
                    "early_delta_quality": early_post[
                        "prediction_quality_sigma150"
                    ]
                    - early_pre["prediction_quality_sigma150"],
                    "early_pre_state_accuracy": early_pre["state_accuracy"],
                    "early_post_state_accuracy": early_post[
                        "state_accuracy"
                    ],
                    "terminal_pre_quality": terminal_pre[
                        "prediction_quality_sigma150"
                    ],
                    "terminal_post_quality": terminal_post[
                        "prediction_quality_sigma150"
                    ],
                    "terminal_delta_quality": (
                        None
                        if bool(branch["terminal_probe_contaminated"])
                        else terminal_post["prediction_quality_sigma150"]
                        - terminal_pre["prediction_quality_sigma150"]
                    ),
                    "terminal_pre_state_accuracy": terminal_pre[
                        "state_accuracy"
                    ],
                    "terminal_post_state_accuracy": terminal_post[
                        "state_accuracy"
                    ],
                }
            )
            format_rows.extend(
                [
                    {
                        "model_key": model,
                        "gravity": gravity,
                        "source_goal_id": source_goal_id,
                        "condition": f"{condition}_early",
                        "schema_valid": int(
                            bool(early_raw.get("schema_valid"))
                        ),
                        "provider_call_count": int(
                            early_raw.get("provider_call_count") or 0
                        ),
                    },
                    {
                        "model_key": model,
                        "gravity": gravity,
                        "source_goal_id": source_goal_id,
                        "condition": f"{condition}_terminal",
                        "schema_valid": int(
                            bool(terminal_raw.get("schema_valid"))
                        ),
                        "provider_call_count": int(
                            terminal_raw.get("provider_call_count") or 0
                        ),
                    },
                ]
            )


    # Pair each feedback-arm change with the same unit's neither-arm change.
    neither = {
        (row["model_key"], row["source_goal_id"]): row
        for row in heldout_rows
        if row["condition"] == "neither"
    }
    for row in heldout_rows:
        control = neither.get((row["model_key"], row["source_goal_id"]))
        row["early_delta_minus_neither"] = (
            row["early_delta_quality"] - control["early_delta_quality"]
            if control is not None
            else None
        )
        row["terminal_delta_minus_neither"] = (
            row["terminal_delta_quality"]
            - control["terminal_delta_quality"]
            if control is not None
            and row["terminal_delta_quality"] is not None
            and control["terminal_delta_quality"] is not None
            else None
        )

    heldout_summary = clustered_summary(
        heldout_rows,
        value_fields=(
            "early_pre_quality",
            "early_post_quality",
            "early_delta_quality",
            "early_delta_minus_neither",
            "terminal_pre_quality",
            "terminal_post_quality",
            "terminal_delta_quality",
            "terminal_delta_minus_neither",
        ),
        group_fields=("model_key", "condition", "gravity"),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    clear_path_summary = clustered_summary(
        clear_path_rows,
        value_fields=(
            "available",
            "schema_valid",
            "tool_prediction_present",
            "tool_state_accuracy",
            "tool_exit_side_accuracy",
            "tool_prediction_quality_sigma150",
            "all_object_prediction_quality_sigma150",
        ),
        group_fields=("model_key", "gravity"),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    format_summary = grouped_summary(
        format_rows,
        value_fields=("schema_valid", "provider_call_count"),
        group_fields=("model_key", "condition", "gravity"),
    )
    usage = usage_summary(args.result_root)

    write_csv(args.output_dir / "heldout_unit_rows.csv", heldout_rows)
    write_csv(args.output_dir / "heldout_summary.csv", heldout_summary)
    write_csv(args.output_dir / "clear_path_rows.csv", clear_path_rows)
    write_csv(
        args.output_dir / "clear_path_summary.csv",
        clear_path_summary,
    )
    write_csv(args.output_dir / "format_rows.csv", format_rows)
    write_csv(args.output_dir / "format_summary.csv", format_summary)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "completed_transfer_units": len(units),
                "heldout_unit_rows": len(heldout_rows),
                "clear_path_rows": len(clear_path_rows),
                "usage": usage,
                "heldout_summary": heldout_summary,
                "clear_path_summary": clear_path_summary,
                "format_summary": format_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "completed_transfer_units": len(units),
                "heldout_unit_rows": len(heldout_rows),
                "clear_path_rows": len(clear_path_rows),
                "usage": usage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
