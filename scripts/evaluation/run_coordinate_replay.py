#!/usr/bin/env python3
"""Offline coordinate-interface replay for completed VTools model runs.

The replay never changes a model conversation. It replaces each accepted
action with the nearest geometry-valid point whose coordinates are multiples
of 10 and simulates the fixed proposed sequence from the initial state. It
also evaluates the frozen +/-5-pixel neighborhood for attempt-1 and originally
successful actions. One simulation is shared across every goal and model that
uses the same layout, gravity, and coordinate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from scipy.spatial import cKDTree


CONDITIONS = (
    "full",
    "frames_only",
    "status_only",
    "neither",
    "trace_status",
)
JITTER_OFFSETS = (
    (-5, 0),
    (5, 0),
    (0, -5),
    (0, 5),
    (-5, -5),
    (-5, 5),
    (5, -5),
    (5, 5),
)
TOOL_RADIUS_PX = 36.0
AQ_SCALES = (36.0, 72.0, 144.0)
CLEARANCE_CLASS_LIMIT_PX = 20


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def coordinate_key(coords: Sequence[int]) -> str:
    return f"{int(coords[0])},{int(coords[1])}"


def is_canonical(goal: dict[str, Any]) -> bool:
    return str(goal.get("source") or "") == "canonical_world_gcond"


def goal_success_from_simulation(
    goal: dict[str, Any],
    simulation: dict[str, Any],
) -> bool:
    if not simulation.get("valid"):
        return False
    if is_canonical(goal):
        return bool(simulation.get("canonical_in_goal_dwell_2s"))
    return str(goal["signature"]) in set(simulation.get("signatures") or [])


def collect_actions(result_root: Path) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    actions: list[dict[str, Any]] = []
    goals: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(result_root.rglob("unit_summary.json")):
        summary = read_json(summary_path)
        goal = dict(summary["goal"])
        goal_id = str(goal["balanced_goal_id"])
        goals[goal_id] = goal
        unit_dir = summary_path.parent
        for condition in CONDITIONS:
            state_path = unit_dir / "branches" / condition / "state.json"
            if not state_path.exists():
                continue
            state = read_json(state_path)
            for attempt in state.get("attempts") or []:
                coords = [int(value) for value in attempt["coords"]]
                attempt_number = int(attempt["attempt"])
                actions.append(
                    {
                        "action_id": sha256_json(
                            {
                                "model": summary["model_key"],
                                "seed": summary["seed"],
                                "goal": goal_id,
                                "condition": condition,
                                "attempt": attempt_number,
                                "coords": coords,
                            }
                        )[:24],
                        "model_key": str(summary["model_key"]),
                        "seed": int(summary["seed"]),
                        "goal_id": goal_id,
                        "puzzle_key": str(goal["puzzle_key"]),
                        "gravity": str(goal["condition"]),
                        "category": str(
                            goal.get("category_5")
                            or goal.get("category")
                            or ""
                        ),
                        "condition": condition,
                        "attempt": attempt_number,
                        "original_coords": coords,
                        "original_success": bool(
                            attempt.get("goal_succeeded")
                        ),
                        "jitter_scope": bool(
                            attempt_number == 1
                            or attempt.get("goal_succeeded")
                        ),
                        "environment_json": str(
                            (
                                Path(goal["environment_json"])
                                if Path(
                                    str(goal.get("environment_json") or "")
                                ).exists()
                                else unit_dir
                                / "assets"
                                / "simulation_world.json"
                            ).resolve()
                        ),
                    }
                )
    if not actions:
        raise RuntimeError(f"No completed model actions under {result_root}")
    return actions, goals


def geometry_valid_10px_points(
    *,
    pred: Any,
    world_data: dict[str, Any],
) -> list[tuple[int, int]]:
    points = []
    for x in range(40, 561, 10):
        for y in range(40, 561, 10):
            world = pred.loadFromDict(world_data["world"]).copy()
            if pred.place_ball_tool(
                world,
                (float(x), float(y)),
                TOOL_RADIUS_PX,
                "downward",
                "orange",
            ):
                points.append((x, y))
    if not points:
        raise RuntimeError("Scene has no geometry-valid 10-pixel placement")
    return points


def placement_is_geometry_valid(
    *,
    pred: Any,
    world: Any,
    coords: Sequence[int],
    tool_polygons: Sequence[Sequence[tuple[float, float]]],
) -> bool:
    position = (float(coords[0]), float(coords[1]))
    return not any(
        world.checkCollision(position, polygon)
        for polygon in tool_polygons
    )


def integer_geometry_clearance(
    *,
    pred: Any,
    world: Any,
    coords: Sequence[int],
    tool_polygons: Sequence[Sequence[tuple[float, float]]],
    validity_cache: dict[tuple[int, int], bool],
    limit_px: int = CLEARANCE_CLASS_LIMIT_PX,
) -> float | None:
    """Nearest invalid integer-center displacement, censored above limit."""
    offsets = sorted(
        (
            (math.hypot(dx, dy), dx, dy)
            for dx in range(-limit_px, limit_px + 1)
            for dy in range(-limit_px, limit_px + 1)
            if (dx or dy) and math.hypot(dx, dy) <= limit_px
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    x, y = int(coords[0]), int(coords[1])
    for distance, dx, dy in offsets:
        candidate = (x + dx, y + dy)
        valid = validity_cache.get(candidate)
        if valid is None:
            valid = placement_is_geometry_valid(
                pred=pred,
                world=world,
                coords=candidate,
                tool_polygons=tool_polygons,
            )
            validity_cache[candidate] = valid
        if not valid:
            return float(distance)
    return None


def clearance_class(clearance_px: float | None) -> str:
    if clearance_px is None:
        return "greater_than_20px"
    if clearance_px < 10.0:
        return "less_than_10px"
    return "10_to_20px"


def nearest_point(
    coords: Sequence[int],
    candidates: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    x, y = int(coords[0]), int(coords[1])
    return min(
        candidates,
        key=lambda point: (
            (point[0] - x) ** 2 + (point[1] - y) ** 2,
            point[0],
            point[1],
        ),
    )


def compact_simulation(
    builder: Any,
    row: dict[str, Any],
    requested_signatures: set[str],
) -> dict[str, Any]:
    signatures = {
        builder.event_signature(event)
        for event in (row.get("event_graph") or [])
    }
    return {
        "valid": bool(row.get("valid")),
        "signatures": sorted(signatures.intersection(requested_signatures)),
        "canonical_in_goal_dwell_2s": bool(
            row.get("canonical_in_goal_dwell_2s")
        ),
    }


def primary_row_signatures(
    builder: Any,
    row: dict[str, Any],
) -> set[str]:
    signatures = {
        builder.event_signature(event)
        for event in (row.get("event_graph") or [])
    }
    signatures.update(
        str(signature)
        for signature in (
            row.get("endpoint_final_state_signatures") or []
        )
    )
    return signatures


def simulate_cell(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    builder = load_module(
        Path(task["goal_builder"]),
        f"coordinate_replay_builder_{os.getpid()}",
    )
    pred = builder.get_predicate_module()
    environment_path = Path(task["environment_json"])
    world_data = pred.load_world_file(environment_path)
    world_data["_source_name"] = environment_path.stem
    world_data["_env_set"] = environment_path.parent.name
    world_data["_env_id"] = environment_path.stem
    label_map, role_map, dynamic_map = builder.build_label_maps(
        pred, world_data
    )
    valid_10 = geometry_valid_10px_points(
        pred=pred,
        world_data=world_data,
    )
    geometry_world = pred.loadFromDict(world_data["world"]).copy()
    tool_polygons = pred.regular_ngon(TOOL_RADIUS_PX, 32)
    validity_cache: dict[tuple[int, int], bool] = {}
    clearance_by_original = {}
    for coords in task["jitter_source_coordinates"]:
        clearance = integer_geometry_clearance(
            pred=pred,
            world=geometry_world,
            coords=coords,
            tool_polygons=tool_polygons,
            validity_cache=validity_cache,
        )
        clearance_by_original[coordinate_key(coords)] = {
            "nearest_invalid_integer_displacement_px": clearance,
            "clearance_class": clearance_class(clearance),
            "right_censored_above_px": (
                CLEARANCE_CLASS_LIMIT_PX if clearance is None else None
            ),
        }
    requested_signatures = set(task["requested_signatures"])
    snapped_by_original = {
        coordinate_key(coords): list(nearest_point(coords, valid_10))
        for coords in task["original_coordinates"]
    }
    simulation_coordinates = {
        tuple(coords) for coords in snapped_by_original.values()
    }
    for coords in task["jitter_source_coordinates"]:
        x, y = int(coords[0]), int(coords[1])
        simulation_coordinates.update(
            (x + dx, y + dy) for dx, dy in JITTER_OFFSETS
        )

    simulations: dict[str, Any] = {}
    for x, y in sorted(simulation_coordinates):
        row = builder.simulate_valid_placement(
            pred=pred,
            world_data=world_data,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            condition=task["gravity"],
            coords=(int(x), int(y)),
            movement_threshold_px=50.0,
            rotation_threshold_deg=15.0,
            contact_min_duration_s=0.5,
            include_tool_events=False,
            save_trace_path=None,
            # Coordinate replay must match the primary run: submitted endpoint
            # semantics for original final-state goals. The simulator still
            # emits canonical_in_goal_dwell_2s independently for the 132
            # canonical goals.
            terminal_persistence_s=0.0,
            persistence_candidate_signatures=None,
        )
        simulations[coordinate_key((x, y))] = compact_simulation(
            builder,
            row,
            requested_signatures,
        )
    return {
        "schema_version": 1,
        "puzzle_key": task["puzzle_key"],
        "gravity": task["gravity"],
        "input_sha256": task["input_sha256"],
        "environment_json": str(environment_path.resolve()),
        "valid_10px_point_count": len(valid_10),
        "snapped_by_original": snapped_by_original,
        "clearance_by_original": clearance_by_original,
        "simulations": simulations,
    }


def task_for_cell(
    *,
    puzzle_key: str,
    actions: Sequence[dict[str, Any]],
    goals: dict[str, dict[str, Any]],
    goal_builder: Path,
) -> dict[str, Any]:
    goal_ids = sorted({str(action["goal_id"]) for action in actions})
    requested_signatures = sorted(
        {
            str(goals[goal_id]["signature"])
            for goal_id in goal_ids
            if not is_canonical(goals[goal_id])
        }
    )
    original_coordinates = sorted(
        {
            tuple(int(value) for value in action["original_coords"])
            for action in actions
        }
    )
    jitter_source_coordinates = sorted(
        {
            tuple(int(value) for value in action["original_coords"])
            for action in actions
            if action["jitter_scope"]
        }
    )
    environment_paths = sorted(
        {str(action["environment_json"]) for action in actions}
    )
    if len(environment_paths) != 1:
        hashes = {
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in environment_paths
        }
        if len(hashes) != 1:
            raise RuntimeError(
                f"Conflicting environment assets for {puzzle_key}: "
                f"{environment_paths}"
            )
    task = {
        "puzzle_key": puzzle_key,
        "gravity": str(actions[0]["gravity"]),
        "environment_json": environment_paths[0],
        "goal_builder": str(goal_builder.resolve()),
        "goal_ids": goal_ids,
        "requested_signatures": requested_signatures,
        "original_coordinates": [list(coords) for coords in original_coordinates],
        "jitter_source_coordinates": [
            list(coords) for coords in jitter_source_coordinates
        ],
        "jitter_offsets": [list(offset) for offset in JITTER_OFFSETS],
        "terminal_persistence_s": 0.0,
        "canonical_success_field": "canonical_in_goal_dwell_2s",
    }
    task["input_sha256"] = sha256_json(task)
    return task


def load_solution_sets(
    *,
    goals: dict[str, dict[str, Any]],
    sweep_root: Path,
    goal_builder: Path,
) -> dict[str, list[tuple[int, int]]]:
    builder = load_module(goal_builder, "coordinate_replay_signature_builder")
    goals_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for goal in goals.values():
        goals_by_cell[str(goal["puzzle_key"])].append(goal)
    output: dict[str, list[tuple[int, int]]] = {
        goal_id: [] for goal_id in goals
    }
    for puzzle_key, cell_goals in sorted(goals_by_cell.items()):
        signatures = {
            str(goal["signature"]): str(goal["balanced_goal_id"])
            for goal in cell_goals
            if not is_canonical(goal)
        }
        canonical_ids = [
            str(goal["balanced_goal_id"])
            for goal in cell_goals
            if is_canonical(goal)
        ]
        placements_path = sweep_root / puzzle_key / "placements.jsonl"
        if not placements_path.exists():
            raise FileNotFoundError(placements_path)
        for row in read_jsonl(placements_path):
            if not row.get("valid"):
                continue
            coords = tuple(int(value) for value in row["placement_xy"])
            row_signatures = primary_row_signatures(builder, row)
            # The persistence audit filters nonpersistent final-state events
            # out of event_graph, but preserves their submitted endpoint truth
            # in this explicit field. Reintroduce only that endpoint set when
            # constructing primary AQ solution sets.
            for signature in row_signatures.intersection(signatures):
                output[signatures[signature]].append(coords)
            if row.get("canonical_in_goal_dwell_2s"):
                for goal_id in canonical_ids:
                    output[goal_id].append(coords)
    missing = [goal_id for goal_id, points in output.items() if not points]
    if missing:
        raise RuntimeError(
            f"{len(missing)} retained goals have no replay solution set: "
            f"{missing[:10]}"
        )
    return output


def aq_values(
    tree: cKDTree,
    coords: Sequence[int],
) -> dict[str, float]:
    distance = float(tree.query([float(coords[0]), float(coords[1])])[0])
    output = {"nearest_solution_distance_px": distance}
    for scale in AQ_SCALES:
        output[f"aq_sigma{int(scale)}"] = math.exp(
            -(distance**2) / (2.0 * scale**2)
        )
    return output


def merge_results(
    *,
    actions: Sequence[dict[str, Any]],
    goals: dict[str, dict[str, Any]],
    solution_sets: dict[str, list[tuple[int, int]]],
    cell_results: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    trees = {
        goal_id: cKDTree(points)
        for goal_id, points in solution_sets.items()
    }
    action_rows: list[dict[str, Any]] = []
    for action in actions:
        goal = goals[str(action["goal_id"])]
        cell = cell_results[str(action["puzzle_key"])]
        original = action["original_coords"]
        snapped = cell["snapped_by_original"][coordinate_key(original)]
        snapped_sim = cell["simulations"][coordinate_key(snapped)]
        row = {
            **{
                key: value
                for key, value in action.items()
                if key not in {"environment_json", "jitter_scope"}
            },
            "snapped_coords": snapped,
            "snap_displacement_px": math.dist(original, snapped),
            "snapped_valid": bool(snapped_sim["valid"]),
            "snapped_success": goal_success_from_simulation(
                goal, snapped_sim
            ),
            "original_snapped_success_agreement": (
                bool(action["original_success"])
                == goal_success_from_simulation(goal, snapped_sim)
            ),
            **{
                f"original_{key}": value
                for key, value in aq_values(
                    trees[str(action["goal_id"])],
                    original,
                ).items()
            },
            **{
                f"snapped_{key}": value
                for key, value in aq_values(
                    trees[str(action["goal_id"])],
                    snapped,
                ).items()
            },
            "jitter_evaluated": bool(action["jitter_scope"]),
            "jitter_valid_count": None,
            "jitter_success_count": None,
            "jitter_success_fraction": None,
            "jitter_all_valid_agree_with_original": None,
            "jitter_all_valid_success": None,
            "jitter_any_success": None,
            "original_isolated_success": None,
            "minimum_distance_to_successful_jitter_px": None,
            "initial_geometry_clearance_px": None,
            "initial_geometry_clearance_class": "",
        }
        if action["jitter_scope"]:
            jitter_sims: list[bool] = []
            successful_distances: list[float] = []
            for dx, dy in JITTER_OFFSETS:
                coords = [original[0] + dx, original[1] + dy]
                simulation = cell["simulations"][coordinate_key(coords)]
                if simulation["valid"]:
                    succeeded = goal_success_from_simulation(
                        goal, simulation
                    )
                    jitter_sims.append(succeeded)
                    if succeeded:
                        successful_distances.append(math.hypot(dx, dy))
            row["jitter_valid_count"] = len(jitter_sims)
            row["jitter_success_count"] = sum(jitter_sims)
            row["jitter_success_fraction"] = (
                sum(jitter_sims) / len(jitter_sims)
                if jitter_sims
                else None
            )
            row["jitter_all_valid_agree_with_original"] = (
                all(
                    value == bool(action["original_success"])
                    for value in jitter_sims
                )
                if jitter_sims
                else None
            )
            row["jitter_all_valid_success"] = (
                all(jitter_sims) if jitter_sims else None
            )
            row["jitter_any_success"] = (
                any(jitter_sims) if jitter_sims else None
            )
            row["original_isolated_success"] = bool(
                action["original_success"] and not any(jitter_sims)
            )
            row["minimum_distance_to_successful_jitter_px"] = (
                min(successful_distances)
                if successful_distances
                else None
            )
            clearance = cell["clearance_by_original"][
                coordinate_key(original)
            ]
            row["initial_geometry_clearance_px"] = clearance[
                "nearest_invalid_integer_displacement_px"
            ]
            row["initial_geometry_clearance_class"] = clearance[
                "clearance_class"
            ]
        action_rows.append(row)

    sequences: dict[
        tuple[str, int, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in action_rows:
        sequences[
            (
                str(row["model_key"]),
                int(row["seed"]),
                str(row["goal_id"]),
                str(row["condition"]),
            )
        ].append(row)
    sequence_rows = []
    for (model, seed, goal_id, condition), rows in sorted(
        sequences.items()
    ):
        ordered = sorted(rows, key=lambda row: int(row["attempt"]))
        original_first = next(
            (
                int(row["attempt"])
                for row in ordered
                if row["original_success"]
            ),
            None,
        )
        snapped_first = next(
            (
                int(row["attempt"])
                for row in ordered
                if row["snapped_success"]
            ),
            None,
        )
        sequence_rows.append(
            {
                "model_key": model,
                "seed": seed,
                "goal_id": goal_id,
                "puzzle_key": ordered[0]["puzzle_key"],
                "gravity": ordered[0]["gravity"],
                "category": ordered[0]["category"],
                "condition": condition,
                "attempt_count": len(ordered),
                "original_first_success": original_first,
                "snapped_first_success": snapped_first,
                "original_solve_by_8": original_first is not None,
                "snapped_solve_by_8": snapped_first is not None,
                "solve_by_8_agreement": (
                    (original_first is not None) == (snapped_first is not None)
                ),
                "mean_original_aq_sigma72": sum(
                    float(row["original_aq_sigma72"]) for row in ordered
                )
                / len(ordered),
                "mean_snapped_aq_sigma72": sum(
                    float(row["snapped_aq_sigma72"]) for row in ordered
                )
                / len(ordered),
            }
        )

    action_path = output_root / "coordinate_action_rows.jsonl"
    sequence_path = output_root / "coordinate_sequence_rows.csv"
    write_jsonl(action_path, action_rows)
    write_csv(sequence_path, sequence_rows)
    report = {
        "schema_version": 1,
        "actions": len(action_rows),
        "sequences": len(sequence_rows),
        "cells": len(cell_results),
        "jitter_action_count": sum(
            int(row["jitter_evaluated"]) for row in action_rows
        ),
        "action_rows": str(action_path.resolve()),
        "sequence_rows": str(sequence_path.resolve()),
        "nearest_10px_rule": (
            "nearest initial-geometry-valid point with x and y divisible by "
            "10; Euclidean distance then lexicographic x,y tie-break"
        ),
        "jitter_offsets": [list(offset) for offset in JITTER_OFFSETS],
        "jitter_scope": (
            "every attempt-1 action and every originally successful action"
        ),
        "clearance_definition": (
            "nearest geometry-invalid integer tool-center displacement, "
            "right-censored above 20 pixels, for every jitter-scoped action"
        ),
        "counterfactual_interpretation": (
            "fixed saved action sequence only; feedback conversations are "
            "not regenerated"
        ),
    }
    atomic_write_json(output_root / "coordinate_replay_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--goal-builder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "result_root",
        "manifest",
        "sweep_root",
        "goal_builder",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    actions, observed_goals = collect_actions(args.result_root)
    manifest_goals = {
        str(goal["balanced_goal_id"]): goal
        for goal in (read_json(args.manifest).get("goals") or [])
    }
    goals = {
        goal_id: {
            **manifest_goals.get(goal_id, {}),
            **goal,
        }
        for goal_id, goal in observed_goals.items()
    }
    actions_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        actions_by_cell[str(action["puzzle_key"])].append(action)
    tasks = [
        task_for_cell(
            puzzle_key=puzzle_key,
            actions=cell_actions,
            goals=goals,
            goal_builder=args.goal_builder,
        )
        for puzzle_key, cell_actions in sorted(actions_by_cell.items())
    ]
    write_jsonl(args.output_root / "replay_tasks.jsonl", tasks)
    write_jsonl(args.output_root / "accepted_actions.jsonl", actions)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "actions": len(actions),
                    "cells": len(tasks),
                    "prepared_only": True,
                },
                indent=2,
            )
        )
        return

    pending = []
    cell_results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        output_path = (
            args.output_root
            / "cells"
            / f"{task['puzzle_key']}.json"
        )
        if output_path.exists():
            existing = read_json(output_path)
            if existing.get("input_sha256") == task["input_sha256"]:
                cell_results[task["puzzle_key"]] = existing
                continue
        pending.append((task, output_path))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(simulate_cell, task): (task, output_path)
            for task, output_path in pending
        }
        for future in concurrent.futures.as_completed(futures):
            task, output_path = futures[future]
            result = future.result()
            atomic_write_json(output_path, result)
            cell_results[task["puzzle_key"]] = result
            print(
                f"[coordinate replay] {len(cell_results)}/{len(tasks)} "
                f"{task['puzzle_key']}",
                flush=True,
            )

    solution_sets = load_solution_sets(
        goals=goals,
        sweep_root=args.sweep_root,
        goal_builder=args.goal_builder,
    )
    report = merge_results(
        actions=actions,
        goals=goals,
        solution_sets=solution_sets,
        cell_results=cell_results,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
