#!/usr/bin/env python3
"""Build an answer-balanced temporal VQA battery from existing rollout assets.

The builder is intentionally simulation-free: it reads the exact trace and
32-frame stimulus recorded in the v1 cell manifest, derives four retrospective
facts, balances correct option slots within question family, and emits new
payloads that reference the existing images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_KEYS = ("gpt", "gemini", "qwen")
LETTERS = "ABCDEFGHI"
QUESTION_VERSION = "balanced_v2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sampled_path(
    trace: dict[str, Any],
    object_id: str,
    sample_indices: Sequence[int],
) -> list[tuple[float, float]]:
    samples = (trace.get("pose_samples") or {}).get(object_id) or []
    if not samples:
        raise RuntimeError(f"Trace lacks pose samples for {object_id}")
    return [
        (
            float(samples[min(index, len(samples) - 1)]["x"]),
            float(samples[min(index, len(samples) - 1)]["y"]),
        )
        for index in sample_indices
    ]


def role_id(trace: dict[str, Any], role: str, fallback: str) -> str:
    for object_id, metadata in (trace.get("objects") or {}).items():
        if str((metadata or {}).get("role") or "").lower() == role:
            return str(object_id)
    if fallback in (trace.get("pose_samples") or {}):
        return fallback
    raise RuntimeError(f"Trace lacks object with role={role!r}")


def first_vertical_direction(
    path: Sequence[tuple[float, float]],
    threshold: float = 30.0,
) -> str:
    initial_y = path[0][1]
    for _, y in path[1:]:
        if y - initial_y >= threshold:
            return "upward"
        if y - initial_y <= -threshold:
            return "downward"
    return "no vertical displacement of 30 pixels"


def horizontal_endpoint_change(
    path: Sequence[tuple[float, float]],
    threshold: float = 30.0,
) -> str:
    delta = path[-1][0] - path[0][0]
    if delta > threshold:
        return "more than 30 pixels right"
    if delta < -threshold:
        return "more than 30 pixels left"
    return "within 30 pixels horizontally"


def first_sample_reaching_distance(
    path: Sequence[tuple[float, float]],
    threshold: float = 30.0,
) -> int | None:
    for index, point in enumerate(path[1:], start=1):
        if math.dist(path[0], point) >= threshold:
            return index
    return None


def tool_crossed_before_red(
    tool: Sequence[tuple[float, float]],
    red: Sequence[tuple[float, float]],
) -> str:
    tool_index = first_sample_reaching_distance(tool)
    red_index = first_sample_reaching_distance(red)
    if tool_index is None:
        return "no"
    if red_index is None or tool_index < red_index:
        return "yes"
    return "no"


def red_endpoint_at_least_200(
    red: Sequence[tuple[float, float]],
) -> str:
    return "yes" if math.dist(red[0], red[-1]) >= 200.0 else "no"


def raw_questions(
    cell_id: str,
    tool: Sequence[tuple[float, float]],
    red: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "question_id": f"{cell_id}:tool_initial_direction_v2",
            "category": "initial_motion",
            "question": (
                "In which vertical direction did the orange tool's center "
                "first move at least 30 pixels from its position in FRAME 01?"
            ),
            "option_values": [
                "upward",
                "downward",
                "no vertical displacement of 30 pixels",
            ],
            "answer_value": first_vertical_direction(tool),
        },
        {
            "question_id": f"{cell_id}:red_horizontal_change_v2",
            "category": "endpoint_change",
            "question": (
                "Where was the red ball's center horizontally in FRAME 32 "
                "relative to its center in FRAME 01?"
            ),
            "option_values": [
                "more than 30 pixels right",
                "more than 30 pixels left",
                "within 30 pixels horizontally",
            ],
            "answer_value": horizontal_endpoint_change(red),
        },
        {
            "question_id": f"{cell_id}:tool_before_red_30_v2",
            "category": "temporal_order",
            "question": (
                "Did the orange tool's center first reach a distance of at "
                "least 30 pixels from its FRAME 01 position in a strictly "
                "earlier shown frame than the red ball's center did?"
            ),
            "option_values": ["yes", "no"],
            "answer_value": tool_crossed_before_red(tool, red),
        },
        {
            "question_id": f"{cell_id}:red_endpoint_200_v2",
            "category": "trajectory_magnitude",
            "question": (
                "Was the red ball's center in FRAME 32 at least 200 pixels "
                "away from its center in FRAME 01?"
            ),
            "option_values": ["yes", "no"],
            "answer_value": red_endpoint_at_least_200(red),
        },
    ]


def stable_digest(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def assign_balanced_slots(
    question_rows: list[dict[str, Any]],
    seed: int,
) -> None:
    """Assign nearly equal correct-letter counts per category.

    Cells are first placed in a deterministic pseudorandom order that does not
    use model outputs. Round-robin slots then differ by at most one count.
    """
    by_category: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in question_rows:
        for question in row["questions"]:
            by_category[str(question["category"])].append(
                (str(row["cell_id"]), question)
            )
    for category, items in by_category.items():
        items.sort(
            key=lambda item: stable_digest(
                str(seed), QUESTION_VERSION, category, item[0]
            )
        )
        for ordinal, (cell_id, question) in enumerate(items):
            values = list(question.pop("option_values"))
            answer_value = str(question["answer_value"])
            correct_slot = ordinal % len(values)
            distractors = [value for value in values if value != answer_value]
            distractors.sort(
                key=lambda value: stable_digest(
                    str(seed),
                    QUESTION_VERSION,
                    category,
                    cell_id,
                    value,
                )
            )
            ordered = list(distractors)
            ordered.insert(correct_slot, answer_value)
            question["options"] = {
                LETTERS[index]: value for index, value in enumerate(ordered)
            }
            question["answer"] = LETTERS[correct_slot]


def system_prompt(representation: str = "images") -> str:
    evidence = (
        "the 32 attached full-resolution frames"
        if representation == "images"
        else "the 32 ordered observable JSON states"
    )
    return (
        "You are answering retrospective perception questions about one "
        "observed 2D rollout. Do not predict unshown physics and do not choose "
        f"an action. Use {evidence} as "
        "evidence and return one JSON object mapping every question_id to "
        "exactly one option letter."
    )


def user_prompt(
    questions: Sequence[dict[str, Any]],
    representation: str = "images",
) -> str:
    if representation == "images":
        evidence_line = (
            "The 32 attached images are consecutive sampled observations "
            "labeled FRAME 01 through FRAME 32 in chronological order."
        )
        evidence_noun = "shown frames"
        question_rows = questions
    else:
        evidence_line = (
            "The supplied JSON contains 32 uniformly sampled observable "
            "states in chronological order. Its state_index field is "
            "zero-based: STATE 01 in the questions means state_index 0, and "
            "STATE 32 means state_index 31."
        )
        evidence_noun = "supplied states"
        question_rows = [
            {
                **question,
                "question": str(question["question"])
                .replace("FRAME ", "STATE ")
                .replace("shown frame", "supplied state"),
            }
            for question in questions
        ]
    lines = [
        evidence_line,
        (
            "The scene is 600 by 600 pixels. World coordinate (0,0) is at "
            "bottom-left; x increases right and y increases up."
        ),
        (
            "The orange circle is the placed tool, the red circle is the "
            "target ball, blue filled objects can move, and black objects are "
            "fixed."
        ),
        (
            "Distances in the questions refer to Euclidean center-to-center "
            "pixel distance unless a horizontal or vertical direction is "
            "explicitly named."
        ),
        (
            "This is a retrospective reading test: answer only what happened "
            f"in the {evidence_noun}. No goal, gravity label, or success/failure "
            "label is supplied."
        ),
        "",
    ]
    for question in question_rows:
        lines.append(f"{question['question_id']}: {question['question']}")
        lines.extend(
            f"  {letter}. {value}"
            for letter, value in question["options"].items()
        )
    return "\n".join(lines)


def observable_trace_json(
    trace: dict[str, Any],
    *,
    state_count: int = 32,
) -> dict[str, Any]:
    """Match forked_feedback_runner.serialize_observable_trace schema v1."""
    pose_samples = trace.get("pose_samples") or {}
    objects = trace.get("objects") or {}
    nonempty = {
        str(name): list(samples)
        for name, samples in pose_samples.items()
        if isinstance(samples, list)
        and samples
        and bool((objects.get(name) or {}).get("is_dynamic"))
    }
    if not nonempty:
        raise RuntimeError("Structured trace has no dynamic pose samples")
    sample_total = max(len(samples) for samples in nonempty.values())
    indices = [
        int(round(index * (sample_total - 1) / max(state_count - 1, 1)))
        for index in range(state_count)
    ]
    states = []
    for state_index, trace_index in enumerate(indices):
        state_objects = []
        state_time = 0.0
        for raw_id in sorted(nonempty):
            samples = nonempty[raw_id]
            sample = samples[min(trace_index, len(samples) - 1)]
            state_time = max(
                state_time,
                float(sample.get("time_s") or 0.0),
            )
            state_objects.append(
                {
                    "id": raw_id,
                    "label": str(
                        (objects.get(raw_id) or {}).get("label")
                        or raw_id
                    ),
                    "center": [
                        round(float(sample["x"]), 3),
                        round(float(sample["y"]), 3),
                    ],
                    "orientation_deg": round(
                        math.degrees(
                            float(sample.get("angle_rad") or 0.0)
                        ),
                        3,
                    ),
                }
            )
        states.append(
            {
                "state_index": state_index,
                "time_s": round(state_time, 3),
                "objects": state_objects,
            }
        )
    return {
        "schema_version": 1,
        "sampling": "uniform_over_native_trace",
        "source_pose_sample_count": sample_total,
        "sample_indices": indices,
        "state_count": len(states),
        "states": states,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    cells = read_jsonl(args.source_root / "cell_manifest.jsonl")
    if len(cells) != 132 and not args.allow_partial:
        raise ValueError(f"Expected 132 source cells, found {len(cells)}")

    question_rows: list[dict[str, Any]] = []
    source_payload_by_cell: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        trace = read_json(Path(cell["trace_path"]))
        sample_indices = [int(value) for value in cell["sample_indices"]]
        tool = sampled_path(
            trace,
            role_id(trace, "tool", "PLACED"),
            sample_indices,
        )
        red = sampled_path(
            trace,
            role_id(trace, "target", "Ball"),
            sample_indices,
        )
        question_rows.append(
            {"cell_id": cell_id, "questions": raw_questions(cell_id, tool, red)}
        )
        source_payload_by_cell[cell_id] = read_json(Path(cell["payload_path"]))

    assign_balanced_slots(question_rows, args.seed)
    questions_by_cell = {
        str(row["cell_id"]): row["questions"] for row in question_rows
    }

    cell_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        questions = questions_by_cell[cell_id]
        source_payload = source_payload_by_cell[cell_id]
        if args.representation == "images":
            image_paths = source_payload["image_paths"]
            scene_json = None
            scene_json_label = None
            condition = "temporal_rollout_images"
            suffix = f"temporal_rollout_images_{QUESTION_VERSION}"
        else:
            image_paths = []
            scene_json = observable_trace_json(
                read_json(Path(cell["trace_path"])),
                state_count=32,
            )
            scene_json_label = "VISIBLE ROLLOUT JSON (32 ordered states)"
            condition = "temporal_rollout_json"
            suffix = f"temporal_rollout_json_{QUESTION_VERSION}"
        payload = {
            "system": system_prompt(args.representation),
            "user_text": user_prompt(questions, args.representation),
            "image_paths": image_paths,
            "image_sequence_labels": args.representation == "images",
            "scene_json": scene_json,
            "scene_json_label": scene_json_label,
            "response_schema": {
                "type": "object",
                "required_question_ids": [
                    question["question_id"] for question in questions
                ],
                "values": "one option letter",
            },
        }
        payload_path = args.output_root / "payloads" / f"{cell_id}.json"
        write_json(payload_path, payload)
        cell_row = dict(cell)
        cell_row.update(
            {
                "payload_path": str(payload_path.resolve()),
                "source_payload_path": str(Path(cell["payload_path"])),
                "question_version": QUESTION_VERSION,
                "question_selection_uses_model_outputs": False,
                "question_selection_uses_goal_success": False,
                "representation": args.representation,
                "observable_trace_serializer": (
                    "forked_feedback_runner.serialize_observable_trace_schema_v1"
                    if args.representation == "json"
                    else None
                ),
            }
        )
        cell_rows.append(cell_row)
        for model_key in MODEL_KEYS:
            calls.append(
                {
                    "call_id": (
                        f"{cell_id}:{model_key}:{suffix}"
                    ),
                    "layout_id": cell["layout_id"],
                    "cell_id": cell_id,
                    "model_key": model_key,
                    "input_condition": condition,
                    "benchmark_cell_aliases": [cell_id],
                    "payload_path": str(payload_path.resolve()),
                    "batching": (
                        f"one call contains {len(questions)} retrospective "
                        "questions for one existing 32-frame rollout"
                    ),
                }
            )

    answer_values: dict[str, Counter[str]] = defaultdict(Counter)
    answer_letters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in question_rows:
        for question in row["questions"]:
            category = str(question["category"])
            answer_values[category][str(question["answer_value"])] += 1
            answer_letters[category][str(question["answer"])] += 1

    write_jsonl(args.output_root / "cell_manifest.jsonl", cell_rows)
    write_jsonl(args.output_root / "questions.jsonl", question_rows)
    write_jsonl(args.output_root / "call_manifest.jsonl", calls)
    summary = {
        "schema_version": 2,
        "question_version": QUESTION_VERSION,
        "source_root": str(args.source_root),
        "generated_cell_count": len(cells),
        "expected_full_cell_count": 132,
        "frame_count_per_cell": 32,
        "observable_state_count_per_cell": (
            32 if args.representation == "json" else None
        ),
        "questions_per_cell": {
            "minimum": min(len(row["questions"]) for row in question_rows),
            "maximum": max(len(row["questions"]) for row in question_rows),
            "mean": (
                sum(len(row["questions"]) for row in question_rows)
                / len(question_rows)
            ),
        },
        "model_keys": list(MODEL_KEYS),
        "call_count": len(calls),
        "selection_uses_model_outputs": False,
        "selection_uses_goal_success": False,
        "retrospective_only": True,
        "representation": args.representation,
        "reuses_existing_frame_assets": args.representation == "images",
        "reuses_existing_trace_assets": args.representation == "json",
        "answer_value_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(answer_values.items())
        },
        "correct_letter_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(answer_letters.items())
        },
        "coordinate_convention": {
            "canvas": [600, 600],
            "origin": "bottom-left",
            "x_direction": "right",
            "y_direction": "up",
        },
        "no_paid_calls_made": True,
    }
    write_json(args.output_root / "manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--representation",
        choices=("images", "json"),
        default="images",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
