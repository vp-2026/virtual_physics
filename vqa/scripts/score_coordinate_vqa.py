#!/usr/bin/env python3
"""Score free-response VQA coordinates and clean half-location accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=ROOT / "coordinate_localization_v2" / "ground_truth.jsonl",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.responses.parent / "scored_coordinates"
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = {
        row["layout_id"]: row for row in read_jsonl(args.ground_truth)
    }
    rows: list[dict[str, Any]] = []
    response_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    responses_by_call: dict[str, dict[str, Any]] = {}
    for response in read_jsonl(args.responses):
        call_id = str(response["call_id"])
        previous = responses_by_call.get(call_id)
        if previous is None or bool(response.get("format_valid")):
            responses_by_call[call_id] = response
    for response in responses_by_call.values():
        model = response["model_key"]
        response_counts[model][1] += 1
        response_counts[model][0] += int(
            not response.get("error") and bool(response.get("format_valid"))
        )
        parsed = response.get("parsed_response") or {}
        for target_id, target in truth[response["layout_id"]]["targets"].items():
            point = parsed.get(target_id)
            valid = (
                isinstance(point, list)
                and len(point) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 0 <= float(value) <= 599
                    for value in point
                )
            )
            true_x, true_y = map(float, target["center"])
            pred_x = float(point[0]) if valid else math.nan
            pred_y = float(point[1]) if valid else math.nan
            error = (
                math.hypot(pred_x - true_x, pred_y - true_y)
                if valid
                else math.nan
            )
            y_flip_error = (
                math.hypot(pred_x - true_x, pred_y - (600 - true_y))
                if valid
                else math.nan
            )
            x_scored = bool(target["x_half_scored"])
            y_scored = bool(target["y_half_scored"])
            both_scored = bool(target["combined_halves_scored"])
            x_correct = valid and ((pred_x < 300) == (true_x < 300))
            y_correct = valid and ((pred_y < 300) == (true_y < 300))
            rows.append(
                {
                    "call_id": response["call_id"],
                    "layout_id": response["layout_id"],
                    "model_key": model,
                    "model_id": response.get("model_id_returned")
                    or response.get("model_id_requested"),
                    "target_id": target_id,
                    "target_label": target["label"],
                    "valid_prediction": int(valid),
                    "true_x": true_x,
                    "true_y": true_y,
                    "pred_x": pred_x if valid else "",
                    "pred_y": pred_y if valid else "",
                    "absolute_x_error_px": abs(pred_x - true_x) if valid else "",
                    "absolute_y_error_px": abs(pred_y - true_y) if valid else "",
                    "euclidean_error_px": error if valid else "",
                    "y_flipped_euclidean_error_px": y_flip_error if valid else "",
                    "within_60px": int(valid and error <= 60),
                    "within_100px": int(valid and error <= 100),
                    "within_150px": int(valid and error <= 150),
                    "x_half_scored": int(x_scored),
                    "x_half_correct": int(x_scored and x_correct),
                    "y_half_scored": int(y_scored),
                    "y_half_correct": int(y_scored and y_correct),
                    "combined_halves_scored": int(both_scored),
                    "combined_halves_correct": int(
                        both_scored and x_correct and y_correct
                    ),
                }
            )
    write_csv(output_dir / "coordinate_rows.csv", rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_key"], "__overall__")].append(row)
        groups[(row["model_key"], row["target_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, target), group in sorted(groups.items()):
        valid_errors = [
            float(row["euclidean_error_px"])
            for row in group
            if row["euclidean_error_px"] != ""
        ]
        y_flip_errors = [
            float(row["y_flipped_euclidean_error_px"])
            for row in group
            if row["y_flipped_euclidean_error_px"] != ""
        ]
        x_scored = [row for row in group if row["x_half_scored"]]
        y_scored = [row for row in group if row["y_half_scored"]]
        both_scored = [row for row in group if row["combined_halves_scored"]]
        summaries.append(
            {
                "model_key": model,
                "target": target,
                "targets": len(group),
                "valid_coordinate_rate": mean(
                    [float(row["valid_prediction"]) for row in group]
                ),
                "mean_euclidean_error_px": mean(valid_errors),
                "median_euclidean_error_px": (
                    statistics.median(valid_errors)
                    if valid_errors
                    else math.nan
                ),
                "within_60px_rate": mean(
                    [float(row["within_60px"]) for row in group]
                ),
                "within_100px_rate": mean(
                    [float(row["within_100px"]) for row in group]
                ),
                "within_150px_rate": mean(
                    [float(row["within_150px"]) for row in group]
                ),
                "mean_y_flipped_error_px": mean(y_flip_errors),
                "x_half_eligible_targets": len(x_scored),
                "x_half_accuracy_60px_exclusion": mean(
                    [float(row["x_half_correct"]) for row in x_scored]
                ),
                "y_half_eligible_targets": len(y_scored),
                "y_half_accuracy_60px_exclusion": mean(
                    [float(row["y_half_correct"]) for row in y_scored]
                ),
                "combined_eligible_targets": len(both_scored),
                "combined_halves_accuracy_60px_exclusion": mean(
                    [
                        float(row["combined_halves_correct"])
                        for row in both_scored
                    ]
                ),
            }
        )
    write_csv(output_dir / "coordinate_summary.csv", summaries)
    report = {
        "responses": str(args.responses),
        "models": {
            model: {
                "valid_calls": values[0],
                "calls": values[1],
                "format_rate": values[0] / values[1],
            }
            for model, values in sorted(response_counts.items())
        },
        "coordinate_rows": str(output_dir / "coordinate_rows.csv"),
        "coordinate_summary": str(output_dir / "coordinate_summary.csv"),
        "notes": [
            "Threshold accuracy counts invalid or missing coordinates wrong.",
            "Half-location accuracy is scored only beyond a strict 60-pixel exclusion band.",
            "Original-versus-y-flipped error diagnoses coordinate-origin misunderstanding.",
        ],
    }
    (output_dir / "score_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
