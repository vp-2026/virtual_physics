#!/usr/bin/env python3
"""Generate a retrospective 32-frame rollout-perception control.

Stimulus actions are selected from scene geometry only: for each unique layout,
choose a geometry-valid 10-pixel lattice point with a long initially clear
vertical path in both directions, using stable geometry-only tie breaks. The
same point is used under upward and downward tool acceleration. No model output,
task goal, or task success enters stimulus selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
STATIC_VQA_ROOT = WORKSPACE / "outputs" / "perception_vqa"
RUNNER_PATH = WORKSPACE / "outputs" / "forked_feedback_baseline" / "forked_feedback_runner.py"
PACKAGED_RUNTIME = (
    WORKSPACE
    / "outputs"
    / "robotics_er_subset"
    / "packaged_runtime"
    / "runtime_home"
)
DEFAULT_GOAL_BUILDER = PACKAGED_RUNTIME / "Documents" / "Playground" / "build_goal_bank_from_placement_sweep.py"
DEFAULT_ENVIRONMENT_ROOT = (
    PACKAGED_RUNTIME
    / "Documents"
    / "Playground"
    / "virtual-tools-web-experiment-nonoverlap22"
    / "simulation"
    / "runtime"
    / "environment_sets"
)
DEFAULT_MODELS = ("gpt", "gemini", "qwen")
FRAME_COUNT = 32
CANVAS_SIZE = 600
TOOL_RADIUS_PX = 36.0


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_cells(scene_manifest: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(scene_manifest)
    cells = sorted(rows, key=lambda row: (row["layout_id"], row["gravity_mode"]))
    if len(cells) != 132:
        raise ValueError(f"Expected 132 layout-by-gravity cells, found {len(cells)}")
    if len({row["layout_id"] for row in cells}) != 66:
        raise ValueError("Scene manifest does not contain 66 unique layouts")
    return cells


def geometry_only_clear_path_point(
    *,
    goal_bank: Any,
    representative_goal: dict[str, Any],
) -> tuple[int, int]:
    """Choose a shared upward/downward placement without simulating dynamics.

    Evaluate initial overlap validity on a 10-pixel lattice, then maximize the
    smaller of the unobstructed vertical distances above and below the point.
    This creates a readable motion calibration under both gravity directions.
    """
    context = goal_bank.context_for(representative_goal)
    pred = context["pred"]
    world_data = context["world_data"]
    xs = list(range(40, 561, 10))
    ys = list(range(40, 561, 10))
    valid_points: set[tuple[int, int]] = set()
    for x in xs:
        for y in ys:
            world = pred.loadFromDict(world_data["world"]).copy()
            if pred.place_ball_tool(
                world,
                (float(x), float(y)),
                TOOL_RADIUS_PX,
                "downward",
                "orange",
            ):
                valid_points.add((x, y))
    if not valid_points:
        raise RuntimeError(
            f"No geometry-valid diagnostic point for {representative_goal['puzzle_key']}"
        )

    scored: list[tuple[tuple[float, ...], tuple[int, int]]] = []
    for x, y in valid_points:
        up = y
        while (x, up + 10) in valid_points:
            up += 10
        down = y
        while (x, down - 10) in valid_points:
            down -= 10
        up_clearance = up - y
        down_clearance = y - down
        score = (
            -float(min(up_clearance, down_clearance)),
            -float(up_clearance + down_clearance),
            float((x - 300) ** 2 + (y - 300) ** 2),
            float(x),
            float(y),
        )
        scored.append((score, (x, y)))
    return min(scored, key=lambda item: item[0])[1]


def cached_geometry_point(
    *,
    output_root: Path,
    layout_id: str,
) -> tuple[int, int] | None:
    """Recover the deterministic point from a prior interrupted generation."""
    points: set[tuple[int, int]] = set()
    for gravity in ("downward", "upward"):
        trace_dir = (
            output_root
            / "cells"
            / f"{layout_id}_{gravity}"
            / "truth_trace"
        )
        for trace_path in trace_dir.glob("trace_*_*.json"):
            parts = trace_path.stem.split("_")
            if len(parts) != 3:
                continue
            points.add((int(parts[1]), int(parts[2])))
    if len(points) > 1:
        raise RuntimeError(
            f"Interrupted output has conflicting diagnostic points for "
            f"{layout_id}: {sorted(points)}"
        )
    return next(iter(points)) if points else None


def sampled_object_paths(
    trace: dict[str, Any],
    sample_indices: Sequence[int],
) -> dict[str, list[tuple[float, float]]]:
    output: dict[str, list[tuple[float, float]]] = {}
    for object_id, samples in (trace.get("pose_samples") or {}).items():
        if not isinstance(samples, list) or not samples:
            continue
        output[str(object_id)] = [
            (
                float(samples[min(index, len(samples) - 1)]["x"]),
                float(samples[min(index, len(samples) - 1)]["y"]),
            )
            for index in sample_indices
        ]
    object_metadata = trace.get("objects") or {}
    for object_id, metadata in object_metadata.items():
        role = str((metadata or {}).get("role") or "").lower()
        object_path = output.get(str(object_id))
        if object_path is None:
            continue
        if role == "tool":
            output["__tool__"] = object_path
        elif role == "target":
            output["__target__"] = object_path
    return output


def terminal_class(point: tuple[float, float]) -> str:
    x, y = point
    if 0 <= x <= 599 and 0 <= y <= 599:
        return "remained in the scene"
    violations = [
        (max(0.0, -x), "exited through the left"),
        (max(0.0, x - 599.0), "exited through the right"),
        (max(0.0, -y), "exited through the bottom"),
        (max(0.0, y - 599.0), "exited through the top"),
    ]
    return max(violations, key=lambda item: item[0])[1]


def vertical_relation(start: tuple[float, float], end: tuple[float, float]) -> str:
    delta = end[1] - start[1]
    if delta > 30:
        return "more than 30 pixels above"
    if delta < -30:
        return "more than 30 pixels below"
    return "within 30 pixels vertically"


def initial_vertical_direction(path: Sequence[tuple[float, float]]) -> str:
    initial_y = path[0][1]
    for _, y in path[1:]:
        delta = y - initial_y
        if delta >= 30:
            return "upward"
        if delta <= -30:
            return "downward"
    return "no vertical displacement of 30 pixels"


def has_large_vertical_reversal(path: Sequence[tuple[float, float]]) -> bool:
    ys = [point[1] for point in path]
    for pivot in range(1, len(ys) - 1):
        before = ys[: pivot + 1]
        after = ys[pivot:]
        rose_then_fell = ys[pivot] - min(before) >= 60 and ys[pivot] - min(after) >= 60
        fell_then_rose = max(before) - ys[pivot] >= 60 and max(after) - ys[pivot] >= 60
        if rose_then_fell or fell_then_rose:
            return True
    return False


def path_length(path: Sequence[tuple[float, float]]) -> float:
    return sum(math.dist(first, second) for first, second in zip(path, path[1:]))


def displacement_bucket(path: Sequence[tuple[float, float]]) -> str:
    distance = math.dist(path[0], path[-1])
    if distance < 30:
        return "less than 30 pixels"
    if distance < 100:
        return "30 to 99 pixels"
    if distance < 200:
        return "100 to 199 pixels"
    return "at least 200 pixels"


def place_answer(
    *,
    question_id: str,
    category: str,
    question: str,
    options: Sequence[str],
    answer: str,
    answer_slot: int,
) -> dict[str, Any]:
    if answer not in options:
        raise ValueError(f"{answer!r} not in {options!r}")
    distractors = [value for value in options if value != answer]
    rotate = answer_slot % max(len(distractors), 1)
    distractors = distractors[rotate:] + distractors[:rotate]
    ordered = list(distractors)
    ordered.insert(answer_slot % len(options), answer)
    letters = "ABCDEFGHI"
    return {
        "question_id": question_id,
        "category": category,
        "question": question,
        "options": {
            letters[index]: value for index, value in enumerate(ordered)
        },
        "answer": letters[ordered.index(answer)],
        "answer_value": answer,
    }


def questions_for_trace(
    *,
    cell_id: str,
    cell_index: int,
    paths: dict[str, list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    tool = paths.get("__tool__") or paths.get("PLACED")
    red = (
        paths.get("__target__")
        or paths.get("Ball")
        or paths.get("ball")
    )
    if tool is None or red is None:
        raise RuntimeError(f"Trace for {cell_id} lacks tool or red-ball samples")
    tool_length = path_length(tool)
    red_length = path_length(red)
    if tool_length > red_length + 60:
        farther = "orange tool"
    elif red_length > tool_length + 60:
        farther = "red ball"
    else:
        farther = "approximately the same total path length"

    specs = [
        (
            "tool_initial_direction",
            "initial_motion",
            "From the frames, what was the orange tool's first vertical displacement of at least 30 pixels after release?",
            ["upward", "downward", "no vertical displacement of 30 pixels"],
            initial_vertical_direction(tool),
        ),
        (
            "tool_terminal_state",
            "terminal_state",
            "By the end of the shown rollout, what happened to the orange tool's center?",
            [
                "remained in the scene",
                "exited through the left",
                "exited through the right",
                "exited through the bottom",
                "exited through the top",
            ],
            terminal_class(tool[-1]),
        ),
        (
            "red_vertical_change",
            "endpoint_change",
            "Where was the red ball's center vertically at the end relative to its center in frame 1?",
            [
                "more than 30 pixels above",
                "more than 30 pixels below",
                "within 30 pixels vertically",
            ],
            vertical_relation(red[0], red[-1]),
        ),
        (
            "tool_vertical_reversal",
            "path_shape",
            "Did the orange tool's shown vertical path reverse by at least 60 pixels after moving at least 60 pixels in the other vertical direction?",
            ["yes", "no"],
            "yes" if has_large_vertical_reversal(tool) else "no",
        ),
        (
            "tool_vs_red_path",
            "trajectory_comparison",
            "Across all 32 frames, which object traveled the larger total path? Treat path lengths within 60 pixels as approximately the same.",
            [
                "orange tool",
                "red ball",
                "approximately the same total path length",
            ],
            farther,
        ),
        (
            "red_net_displacement",
            "trajectory_magnitude",
            "Approximately how far was the red ball's final center from its center in frame 1?",
            [
                "less than 30 pixels",
                "30 to 99 pixels",
                "100 to 199 pixels",
                "at least 200 pixels",
            ],
            displacement_bucket(red),
        ),
    ]
    return [
        place_answer(
            question_id=f"{cell_id}:{suffix}",
            category=category,
            question=question,
            options=options,
            answer=answer,
            answer_slot=(cell_index * len(specs) + question_index) % len(options),
        )
        for question_index, (suffix, category, question, options, answer) in enumerate(specs)
    ]


def system_prompt() -> str:
    return (
        "You are answering retrospective perception questions about one observed "
        "2D rollout. Do not predict unshown physics and do not choose an action. "
        "Use the 32 attached full-resolution frames as visual evidence and return "
        "one JSON object mapping every question_id to exactly one option letter."
    )


def user_prompt(questions: Sequence[dict[str, Any]]) -> str:
    lines = [
        "The 32 attached images are consecutive sampled observations labeled FRAME 01 through FRAME 32 in chronological order.",
        "The scene is 600 by 600 pixels. World coordinate (0,0) is at bottom-left; x increases right and y increases up.",
        "The orange circle is the placed tool, the red circle is the target ball, blue filled objects can move, and black objects are fixed.",
        "This is a retrospective reading test: answer only what happened in the shown frames. No goal or success/failure label is supplied.",
        "",
    ]
    for question in questions:
        lines.append(f"{question['question_id']}: {question['question']}")
        lines.extend(
            f"  {letter}. {value}"
            for letter, value in question["options"].items()
        )
    return "\n".join(lines)


def goal_for_cell(
    *,
    layout_id: str,
    gravity: str,
    environment_path: Path,
) -> dict[str, Any]:
    family, env_id = layout_id.rsplit("_", 1)
    return {
        "balanced_goal_id": f"temporal_vqa_{layout_id}_{gravity}",
        "puzzle_key": f"{layout_id}_{gravity}",
        "family": family,
        "env_id": env_id,
        "condition": gravity,
        "signature": "temporal_vqa_no_goal_scoring",
        "environment_json": str(environment_path.resolve()),
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    runner = load_module(args.runner, "temporal_vqa_feedback_runner")
    goal_bank = runner.base.GoalBank(
        Path("/nonexistent-goal-bank-not-used"),
        args.goal_builder,
    )
    cells = unique_cells(args.scene_manifest)
    cells = cells[args.start_index :]
    if args.limit is not None:
        cells = cells[: args.limit]

    placement_by_layout: dict[str, tuple[int, int]] = {}
    cell_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    for relative_index, cell in enumerate(cells):
        absolute_index = args.start_index + relative_index
        layout_id = str(cell["layout_id"])
        gravity = str(cell["gravity_mode"])
        cell_id = str(cell["cell_id"])
        family, env_id = layout_id.rsplit("_", 1)
        environment_path = args.environment_root / family / f"{env_id}.json"
        if not environment_path.exists():
            raise FileNotFoundError(environment_path)
        goal = goal_for_cell(
            layout_id=layout_id,
            gravity=gravity,
            environment_path=environment_path,
        )
        if layout_id not in placement_by_layout:
            placement_by_layout[layout_id] = (
                cached_geometry_point(
                    output_root=args.output,
                    layout_id=layout_id,
                )
                or geometry_only_clear_path_point(
                    goal_bank=goal_bank,
                    representative_goal=goal,
                )
            )
        coords = placement_by_layout[layout_id]
        cell_dir = args.output / "cells" / cell_id
        trace_dir = cell_dir / "truth_trace"
        trace_path = trace_dir / f"trace_{coords[0]}_{coords[1]}.json"
        if not trace_path.exists():
            truth = goal_bank.evaluate_goal_at(goal, coords, trace_dir=trace_dir)
            if not truth.get("valid") or not trace_path.exists():
                raise RuntimeError(f"Diagnostic placement failed for {cell_id}: {coords}")
        frames, render = runner.render_structured_trace_frames(
            goal=goal,
            goal_bank=goal_bank,
            coords=coords,
            trace_path=trace_path,
            out_dir=cell_dir / "frames",
            frame_count=FRAME_COUNT,
        )
        trace = read_json(trace_path)
        paths = sampled_object_paths(trace, render["sample_indices"])
        questions = questions_for_trace(
            cell_id=cell_id,
            cell_index=absolute_index,
            paths=paths,
        )
        question_rows.append({"cell_id": cell_id, "questions": questions})
        payload = {
            "system": system_prompt(),
            "user_text": user_prompt(questions),
            "image_paths": [str(path) for path in frames],
            "image_sequence_labels": True,
            "scene_json": None,
            "response_schema": {
                "type": "object",
                "required_question_ids": [
                    question["question_id"] for question in questions
                ],
                "values": "one option letter",
            },
        }
        payload_path = args.output / "payloads" / f"{cell_id}.json"
        write_json(payload_path, payload)
        cell_rows.append(
            {
                "cell_id": cell_id,
                "layout_id": layout_id,
                "gravity_mode": gravity,
                "diagnostic_placement": list(coords),
                "placement_selection": (
                    "geometry-valid 10-pixel lattice point maximizing the "
                    "smaller initial vertical clearance above/below, then total "
                    "vertical clearance, then center proximity; stable x/y ties; "
                    "same point for both gravities"
                ),
                "selection_uses_model_outputs": False,
                "selection_uses_goal_success": False,
                "environment_path": str(environment_path.resolve()),
                "environment_sha256": sha256_file(environment_path),
                "trace_path": str(trace_path.resolve()),
                "trace_sha256": sha256_file(trace_path),
                "frame_count": len(frames),
                "sample_indices": render["sample_indices"],
                "sample_times_s": render["sample_times_s"],
                "payload_path": str(payload_path.resolve()),
            }
        )
        for model_key in DEFAULT_MODELS:
            calls.append(
                {
                    "call_id": f"{cell_id}:{model_key}:temporal_rollout_images",
                    "layout_id": layout_id,
                    "cell_id": cell_id,
                    "model_key": model_key,
                    "input_condition": "temporal_rollout_images",
                    "benchmark_cell_aliases": [cell_id],
                    "payload_path": str(payload_path.resolve()),
                    "batching": "one call contains six retrospective questions for one 32-frame rollout",
                }
            )

    write_jsonl(args.output / "cell_manifest.jsonl", cell_rows)
    write_jsonl(args.output / "questions.jsonl", question_rows)
    write_jsonl(args.output / "call_manifest.jsonl", calls)
    summary = {
        "schema_version": 1,
        "generated_cell_count": len(cell_rows),
        "expected_full_cell_count": 132,
        "frame_count_per_cell": FRAME_COUNT,
        "questions_per_cell": 6,
        "model_keys": list(DEFAULT_MODELS),
        "call_count": len(calls),
        "selection_uses_model_outputs": False,
        "selection_uses_goal_success": False,
        "retrospective_only": True,
        "coordinate_convention": {
            "canvas": [600, 600],
            "origin": "bottom-left",
            "x_direction": "right",
            "y_direction": "up",
        },
        "no_paid_calls_made": True,
    }
    write_json(args.output / "manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--scene-manifest", type=Path, default=STATIC_VQA_ROOT / "scene_manifest.jsonl")
    parser.add_argument("--runner", type=Path, default=RUNNER_PATH)
    parser.add_argument("--goal-builder", type=Path, default=DEFAULT_GOAL_BUILDER)
    parser.add_argument("--environment-root", type=Path, default=DEFAULT_ENVIRONMENT_ROOT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for field in ("output", "scene_manifest", "runner", "goal_builder", "environment_root"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(generate(parse_args()), indent=2, sort_keys=True))
