#!/usr/bin/env python3
"""Score the paper-defined normalized action quality from saved sweeps.

For goal ``g``, the submitted metric is

    AQ(a, g) = max(0, 1 - d_success(a, g) / D_g)

where ``d_success`` is Euclidean distance to the nearest successful sweep
placement and ``D_g`` is the largest such nearest-solution distance among all
valid sweep placements for that goal. This script computes ``D_g`` directly
from the frozen sweep and appends ``paper_action_quality`` to an existing
action-row CSV that already contains nearest-solution distances.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_goals(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["goals"] if isinstance(payload, dict) else payload
    return {str(row["balanced_goal_id"]): row for row in records}


def compute_normalizers(
    *,
    goal_ids: set[str],
    goals: dict[str, dict[str, Any]],
    sweep_root: Path,
    goal_builder: Path,
    coordinate_replay: Path,
) -> dict[str, dict[str, Any]]:
    builder = load_module(goal_builder, "paper_aq_goal_builder")
    replay = load_module(coordinate_replay, "paper_aq_coordinate_replay")
    goals_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for goal_id in sorted(goal_ids):
        if goal_id not in goals:
            raise KeyError(f"Goal missing from manifest: {goal_id}")
        goals_by_cell[str(goals[goal_id]["run_name"])].append(goals[goal_id])

    output: dict[str, dict[str, Any]] = {}
    for cell_index, (puzzle_key, cell_goals) in enumerate(
        sorted(goals_by_cell.items()), 1
    ):
        signature_to_ids: dict[str, list[str]] = defaultdict(list)
        for goal in cell_goals:
            signature_to_ids[str(goal["signature"])].append(
                str(goal["balanced_goal_id"])
            )
        valid_points: list[tuple[int, int]] = []
        solution_points: dict[str, list[tuple[int, int]]] = {
            str(goal["balanced_goal_id"]): [] for goal in cell_goals
        }
        placements_path = sweep_root / puzzle_key / "placements.jsonl"
        if not placements_path.exists():
            raise FileNotFoundError(placements_path)
        for row in read_jsonl(placements_path):
            if not row.get("valid"):
                continue
            coords = tuple(int(value) for value in row["placement_xy"])
            valid_points.append(coords)
            row_signatures = replay.primary_row_signatures(builder, row)
            for signature in row_signatures.intersection(signature_to_ids):
                for goal_id in signature_to_ids[signature]:
                    solution_points[goal_id].append(coords)
        if not valid_points:
            raise RuntimeError(f"No valid placements for {puzzle_key}")
        valid_array = np.asarray(valid_points, dtype=float)
        for goal in cell_goals:
            goal_id = str(goal["balanced_goal_id"])
            points = solution_points[goal_id]
            if not points:
                raise RuntimeError(
                    f"No successful placements for {goal_id} in {puzzle_key}"
                )
            distances = cKDTree(np.asarray(points, dtype=float)).query(
                valid_array
            )[0]
            normalizer = float(np.max(distances))
            if not math.isfinite(normalizer) or normalizer <= 0:
                raise RuntimeError(
                    f"Invalid action-quality normalizer for {goal_id}: "
                    f"{normalizer}"
                )
            output[goal_id] = {
                "goal_id": goal_id,
                "puzzle_key": puzzle_key,
                "valid_placement_count": len(valid_points),
                "solution_placement_count": len(points),
                "max_nearest_solution_distance_px": normalizer,
            }
        print(
            f"[{cell_index}/{len(goals_by_cell)}] {puzzle_key}: "
            f"{len(valid_points)} valid placements, {len(cell_goals)} goals",
            flush=True,
        )
    return output


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action_rows", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--goal-builder", type=Path, required=True)
    parser.add_argument("--coordinate-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--distance-field",
        default="nearest_solution_distance_px",
    )
    args = parser.parse_args()

    with args.action_rows.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No action rows in {args.action_rows}")
    goal_ids = {str(row["goal_id"]) for row in rows}
    normalizers = compute_normalizers(
        goal_ids=goal_ids,
        goals=load_goals(args.manifest),
        sweep_root=args.sweep_root,
        goal_builder=args.goal_builder,
        coordinate_replay=args.coordinate_replay,
    )

    scored = []
    for row in rows:
        goal_id = str(row["goal_id"])
        distance = float(row[args.distance_field])
        normalizer = float(
            normalizers[goal_id]["max_nearest_solution_distance_px"]
        )
        action_quality = max(0.0, 1.0 - distance / normalizer)
        scored.append(
            {
                **row,
                "original_nearest_solution_distance_px": distance,
                "paper_action_quality": action_quality,
                "paper_action_quality_normalizer_px": normalizer,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_output = args.output_dir / "paper_action_quality_rows.csv"
    with action_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    normalizer_output = args.output_dir / "paper_action_quality_normalizers.csv"
    with normalizer_output.open("w", newline="", encoding="utf-8") as handle:
        records = [normalizers[key] for key in sorted(normalizers)]
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "schema_version": 1,
        "action_rows": len(scored),
        "goals": len(normalizers),
        "metric": (
            "max(0, 1 - nearest_solution_distance / "
            "max_valid_nearest_solution_distance)"
        ),
        "success_rows_with_nonzero_distance": sum(
            truthy(row.get("original_success"))
            and float(row[args.distance_field]) > 0
            for row in rows
        ),
    }
    (args.output_dir / "paper_action_quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
