#!/usr/bin/env python3
"""Validate one April-saved solution in every canonical cell."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

import numpy as np


def validate_one(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import forked_feedback_runner as runner

    goal = task["goal"]
    candidate_coords = task["candidate_coords"]
    builder = pathlib.Path(task["goal_builder"])
    goal_bank = runner.base.GoalBank(
        pathlib.Path(__file__).resolve().parents[2] / "task_configs",
        builder,
    )
    invalid = 0
    errors = []
    for candidate_index, coords in enumerate(candidate_coords, start=1):
        try:
            with tempfile.TemporaryDirectory() as temporary:
                result = runner.simulate_attempt(
                    goal=goal,
                    goal_bank=goal_bank,
                    world_path=pathlib.Path(goal["environment_json"]),
                    coords=coords,
                    attempt_dir=pathlib.Path(temporary),
                    attempt_number=1,
                    frame_count=32,
                    attempt_budget=8,
                    frames_condition=False,
                    stop_on_success=True,
                    trace_condition=False,
                    trace_state_count=32,
                )
        except RuntimeError as exc:
            invalid += 1
            errors.append(str(exc))
            continue
        if result.get("goal_succeeded"):
            return {
                "goal_id": goal["balanced_goal_id"],
                "gravity": goal["condition"],
                "coords": coords,
                "goal_succeeded": True,
                "candidate_index": candidate_index,
                "candidates_tried": candidate_index,
                "invalid_candidates": invalid,
                "intervals": (
                    (result.get("truth") or {}).get(
                        "canonical_goal_intervals"
                    )
                    or []
                ),
            }
    return {
        "goal_id": goal["balanced_goal_id"],
        "gravity": goal["condition"],
        "coords": None,
        "goal_succeeded": False,
        "candidate_index": None,
        "candidates_tried": len(candidate_coords),
        "invalid_candidates": invalid,
        "errors": errors[:3],
        "intervals": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--asset-index", type=pathlib.Path, required=True)
    parser.add_argument("--solution-spaces", type=pathlib.Path, required=True)
    parser.add_argument("--goal-builder", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--candidates-per-cell", type=int, default=64)
    parser.add_argument(
        "--coordinate-mode",
        choices=("raw", "flip_y"),
        default="raw",
    )
    parser.add_argument(
        "--retry-failures-from",
        type=pathlib.Path,
        default=None,
        help="Restrict validation to goal IDs listed in a prior report's failures.",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import forked_feedback_runner as runner

    raw_manifest = json.loads(args.manifest.read_text())
    raw_by_id = {
        str(goal["balanced_goal_id"]): goal
        for goal in raw_manifest["goals"]
    }
    raw_asset_index = json.loads(args.asset_index.read_text())
    asset_index = raw_asset_index.get("puzzles") or raw_asset_index
    goals = [
        goal
        for goal in runner.load_balanced_goals(
            args.manifest,
            asset_index=asset_index,
            budget_mode="fixed",
            fixed_budget=8,
        )
        if str(goal.get("source") or "") == "canonical_world_gcond"
    ]
    if args.retry_failures_from is not None:
        previous = json.loads(args.retry_failures_from.read_text())
        retry_ids = {
            str(result["goal_id"])
            for result in previous.get("failures") or []
        }
        goals = [
            goal
            for goal in goals
            if str(goal["balanced_goal_id"]) in retry_ids
        ]
    spaces = np.load(args.solution_spaces, allow_pickle=True).item()
    tasks = []
    for goal in goals:
        gravity = str(goal["condition"])
        expected_size = int(
            raw_by_id[str(goal["balanced_goal_id"])][
                "canonical_solution_size"
            ]
        )
        matching_keys = [
            key
            for key, points in spaces.items()
            if str(key[1]) == gravity and len(points) == expected_size
        ]
        if len(matching_keys) != 1:
            raise RuntimeError(
                f"Expected one April solution set for "
                f"{goal['balanced_goal_id']} with size {expected_size}; "
                f"found {matching_keys}"
            )
        points = spaces[matching_keys[0]]
        if not points:
            raise RuntimeError(
                f"Empty April solution set: {goal['balanced_goal_id']}"
            )
        ordered_points = sorted(
            ([int(point[0]), int(point[1])] for point in points),
            key=lambda point: (point[0], point[1]),
        )
        seed_bytes = hashlib.sha256(
            str(goal["balanced_goal_id"]).encode("utf-8")
        ).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
        permutation = rng.permutation(len(ordered_points))
        chosen = [
            ordered_points[int(index)]
            for index in permutation[: args.candidates_per_cell]
        ]
        if args.coordinate_mode == "flip_y":
            chosen = [[point[0], 600 - point[1]] for point in chosen]
        tasks.append(
            {
                "goal": goal,
                "candidate_coords": chosen,
                "goal_builder": str(args.goal_builder),
            }
        )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        results = list(executor.map(validate_one, tasks))
    failures = [
        result for result in results if not result["goal_succeeded"]
    ]
    report = {
        "cells": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "coordinate_mode": args.coordinate_mode,
        "candidates_per_cell": args.candidates_per_cell,
        "failures": failures,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("cells", "passed", "failed")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
