#!/usr/bin/env python3
"""Summarize the preregistered coordinate-interface replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def action_summary(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        for gravity in (str(row["gravity"]), "all"):
            groups[
                (str(row["model_key"]), str(row["condition"]), gravity)
            ].append(row)
    output = []
    for (model, condition, gravity), group in sorted(groups.items()):
        jitter = [row for row in group if row["jitter_evaluated"]]
        displacement = [
            float(row["snap_displacement_px"]) for row in group
        ]
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "gravity": gravity,
                "actions": len(group),
                "original_success_rate": mean(
                    [float(row["original_success"]) for row in group]
                ),
                "snapped_success_rate": mean(
                    [float(row["snapped_success"]) for row in group]
                ),
                "snapped_minus_original_success_rate": mean(
                    [
                        float(row["snapped_success"])
                        - float(row["original_success"])
                        for row in group
                    ]
                ),
                "success_agreement_rate": mean(
                    [
                        float(row["original_snapped_success_agreement"])
                        for row in group
                    ]
                ),
                "median_snap_displacement_px": quantile(displacement, 0.5),
                "p95_snap_displacement_px": quantile(displacement, 0.95),
                "snap_displacement_gt_7_1px_rate": mean(
                    [
                        float(value > math.sqrt(50.0))
                        for value in displacement
                    ]
                ),
                "mean_original_aq72": mean(
                    [float(row["original_aq_sigma72"]) for row in group]
                ),
                "mean_snapped_aq72": mean(
                    [float(row["snapped_aq_sigma72"]) for row in group]
                ),
                "jitter_actions": len(jitter),
                "mean_jitter_success_fraction": mean(
                    [
                        float(row["jitter_success_fraction"])
                        for row in jitter
                        if row["jitter_success_fraction"] is not None
                    ]
                ),
                "jitter_all_valid_agreement_rate": mean(
                    [
                        float(row["jitter_all_valid_agree_with_original"])
                        for row in jitter
                        if row[
                            "jitter_all_valid_agree_with_original"
                        ]
                        is not None
                    ]
                ),
                "isolated_success_rate_among_original_successes": mean(
                    [
                        float(row["original_isolated_success"])
                        for row in jitter
                        if row["original_success"]
                    ]
                ),
            }
        )
    return output


def clearance_summary(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        level = str(row["initial_geometry_clearance_class"] or "")
        if not level:
            continue
        groups[
            (str(row["model_key"]), str(row["condition"]), level)
        ].append(row)
    output = []
    for (model, condition, level), group in sorted(groups.items()):
        successful = [row for row in group if row["original_success"]]
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "clearance_class": level,
                "jitter_actions": len(group),
                "original_success_rate": mean(
                    [float(row["original_success"]) for row in group]
                ),
                "mean_jitter_success_fraction": mean(
                    [
                        float(row["jitter_success_fraction"])
                        for row in group
                        if row["jitter_success_fraction"] is not None
                    ]
                ),
                "isolated_success_rate_among_original_successes": mean(
                    [
                        float(row["original_isolated_success"])
                        for row in successful
                    ]
                ),
            }
        )
    return output


def sequence_summary(sequences: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in sequences:
        for gravity in (row["gravity"], "all"):
            groups[(row["model_key"], row["condition"], gravity)].append(
                row
            )
    output = []
    for (model, condition, gravity), group in sorted(groups.items()):
        original = [
            float(row["original_solve_by_8"].lower() == "true")
            for row in group
        ]
        snapped = [
            float(row["snapped_solve_by_8"].lower() == "true")
            for row in group
        ]
        output.append(
            {
                "model_key": model,
                "condition": condition,
                "gravity": gravity,
                "sequences": len(group),
                "original_solve_by_8": mean(original),
                "snapped_solve_by_8": mean(snapped),
                "snapped_minus_original_solve_by_8": mean(
                    [
                        snapped_value - original_value
                        for original_value, snapped_value in zip(
                            original, snapped
                        )
                    ]
                ),
                "solve_by_8_agreement": mean(
                    [
                        float(row["solve_by_8_agreement"].lower() == "true")
                        for row in group
                    ]
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action_rows", type=Path)
    parser.add_argument("sequence_rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    actions = list(read_jsonl(args.action_rows))
    sequences = read_csv(args.sequence_rows)
    action_rows = action_summary(actions)
    clearance_rows = clearance_summary(actions)
    sequence_rows = sequence_summary(sequences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "coordinate_action_summary.csv", action_rows)
    write_csv(
        args.output_dir / "coordinate_clearance_summary.csv",
        clearance_rows,
    )
    write_csv(
        args.output_dir / "coordinate_sequence_summary.csv",
        sequence_rows,
    )
    report = {
        "actions": len(actions),
        "sequences": len(sequences),
        "action_summary_rows": len(action_rows),
        "clearance_summary_rows": len(clearance_rows),
        "sequence_summary_rows": len(sequence_rows),
        "interpretation": (
            "offline fixed-sequence coordinate sensitivity; model "
            "conversations and feedback were not regenerated"
        ),
    }
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
