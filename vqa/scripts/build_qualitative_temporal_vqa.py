#!/usr/bin/env python3
"""Build the qualitative, high-margin temporal-perception VQA battery.

Model-facing questions contain no pixel thresholds. Simulator-derived margins
are used only to include visibly unambiguous items. Selection is frozen under
seed 2026 and never examines model outputs, goal success, or benchmark scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import build_balanced_v2 as base


QUESTION_VERSION = "qualitative_v3"
VISIBLE_DISPLACEMENT_PX = 60.0
STATIONARY_DISPLACEMENT_PX = 15.0
MIN_ONSET_GAP_STATES = 3
ORIGINAL_ASSIGN_BALANCED_SLOTS = base.assign_balanced_slots


def first_visible_motion(
    path: Sequence[tuple[float, float]],
) -> int | None:
    for index, point in enumerate(path[1:], start=1):
        if math.dist(path[0], point) >= VISIBLE_DISPLACEMENT_PX:
            return index
    return None


def maximum_displacement(
    path: Sequence[tuple[float, float]],
) -> float:
    return max(math.dist(path[0], point) for point in path)


def first_visible_vertical_direction(
    path: Sequence[tuple[float, float]],
) -> str | None:
    initial_y = path[0][1]
    for _, y in path[1:]:
        if y - initial_y >= VISIBLE_DISPLACEMENT_PX:
            return "upward"
        if y - initial_y <= -VISIBLE_DISPLACEMENT_PX:
            return "downward"
    return None


def raw_questions(
    cell_id: str,
    tool: Sequence[tuple[float, float]],
    red: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    tool_direction = first_visible_vertical_direction(tool)
    if tool_direction is not None:
        questions.append(
            {
                "question_id": f"{cell_id}:tool_vertical_direction_v3",
                "category": "tool_vertical_direction",
                "question": (
                    "When the orange tool visibly moves away from its starting "
                    "position, does it move upward or downward?"
                ),
                "option_values": ["upward", "downward"],
                "answer_value": tool_direction,
                "hidden_inclusion_rule": (
                    "first sampled vertical displacement reaching 60 px from "
                    "the initial center determines direction"
                ),
            }
        )

    red_delta_x = red[-1][0] - red[0][0]
    if abs(red_delta_x) >= VISIBLE_DISPLACEMENT_PX:
        questions.append(
            {
                "question_id": f"{cell_id}:red_endpoint_horizontal_v3",
                "category": "red_endpoint_horizontal",
                "question": (
                    "Compared with where it started, does the red ball finish "
                    "clearly to the left or clearly to the right?"
                ),
                "option_values": ["left", "right"],
                "answer_value": "right" if red_delta_x > 0 else "left",
                "hidden_inclusion_rule": (
                    "absolute red endpoint horizontal displacement >= 60 px"
                ),
            }
        )

    red_delta_y = red[-1][1] - red[0][1]
    if abs(red_delta_y) >= VISIBLE_DISPLACEMENT_PX:
        questions.append(
            {
                "question_id": f"{cell_id}:red_endpoint_vertical_v3",
                "category": "red_endpoint_vertical",
                "question": (
                    "Compared with where it started, does the red ball finish "
                    "clearly above or clearly below?"
                ),
                "option_values": ["above", "below"],
                "answer_value": "above" if red_delta_y > 0 else "below",
                "hidden_inclusion_rule": (
                    "absolute red endpoint vertical displacement >= 60 px"
                ),
            }
        )

    red_max = maximum_displacement(red)
    if (
        red_max >= VISIBLE_DISPLACEMENT_PX
        or red_max <= STATIONARY_DISPLACEMENT_PX
    ):
        questions.append(
            {
                "question_id": f"{cell_id}:red_visibly_moves_v3",
                "category": "red_visible_motion",
                "question": (
                    "Does the red ball visibly change position during the "
                    "shown rollout?"
                ),
                "option_values": ["yes", "no"],
                "answer_value": (
                    "yes" if red_max >= VISIBLE_DISPLACEMENT_PX else "no"
                ),
                "hidden_inclusion_rule": (
                    "maximum red displacement >= 60 px for yes or <= 15 px "
                    "for no; intermediate cases omitted"
                ),
            }
        )

    tool_onset = first_visible_motion(tool)
    red_onset = first_visible_motion(red)
    if (
        tool_onset is not None
        and red_onset is not None
        and abs(tool_onset - red_onset) >= MIN_ONSET_GAP_STATES
    ):
        questions.append(
            {
                "question_id": f"{cell_id}:visible_motion_order_v3",
                "category": "visible_motion_order",
                "question": (
                    "Which visibly begins moving first in the shown rollout: "
                    "the orange tool or the red ball?"
                ),
                "option_values": ["orange tool", "red ball"],
                "answer_value": (
                    "orange tool" if tool_onset < red_onset else "red ball"
                ),
                "hidden_inclusion_rule": (
                    "both objects reach 60 px displacement and their first "
                    "qualifying sampled states differ by at least 3"
                ),
            }
        )

    if not questions:
        raise RuntimeError(f"No unambiguous qualitative questions for {cell_id}")
    return questions


def stable_digest(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def balance_semantics_then_slots(
    question_rows: list[dict[str, Any]],
    seed: int,
) -> None:
    """Balance answer meanings within family before balancing option letters."""
    by_category: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for row in question_rows:
        for question in row["questions"]:
            by_category[str(question["category"])].append((row, question))

    keep_ids: set[str] = set()
    for category, items in sorted(by_category.items()):
        answer_counts = Counter(
            str(question["answer_value"]) for _, question in items
        )
        # A family with only one possible answer cannot distinguish perception
        # from a constant response and is omitted from the final battery.
        if len(answer_counts) < 2:
            continue
        target_per_answer = min(answer_counts.values())
        by_answer: dict[
            str, list[tuple[dict[str, Any], dict[str, Any]]]
        ] = defaultdict(list)
        for row, question in items:
            by_answer[str(question["answer_value"])].append((row, question))
        for answer_value, answer_items in sorted(by_answer.items()):
            answer_items.sort(
                key=lambda item: stable_digest(
                    str(seed),
                    QUESTION_VERSION,
                    category,
                    answer_value,
                    str(item[0]["cell_id"]),
                )
            )
            keep_ids.update(
                str(question["question_id"])
                for _, question in answer_items[:target_per_answer]
            )

    for row in question_rows:
        row["questions"] = [
            question
            for question in row["questions"]
            if str(question["question_id"]) in keep_ids
        ]
        if not row["questions"]:
            raise RuntimeError(
                f"Semantic balancing removed every question for {row['cell_id']}"
            )
    ORIGINAL_ASSIGN_BALANCED_SLOTS(question_rows, seed)


def system_prompt(representation: str = "images") -> str:
    evidence = (
        "the 32 attached full-resolution frames"
        if representation == "images"
        else "the 32 ordered observable JSON states"
    )
    return (
        "You are answering simple retrospective perception questions about "
        f"one observed 2D rollout. Use {evidence} only. Do not predict unshown "
        "physics, estimate pixel distances, infer a hidden goal, or choose an "
        "action. Return one JSON object mapping every question_id to exactly "
        "one option letter."
    )


def user_prompt(
    questions: Sequence[dict[str, Any]],
    representation: str = "images",
) -> str:
    if representation == "images":
        evidence_line = (
            "The 32 attached images are sampled observations labeled FRAME 01 "
            "through FRAME 32 in chronological order."
        )
        evidence_noun = "shown frames"
        rows = questions
    else:
        evidence_line = (
            "The supplied JSON contains 32 uniformly sampled observable states "
            "in chronological order. STATE 01 means state_index 0 and STATE 32 "
            "means state_index 31. Coordinates use a bottom-left origin: x "
            "increases to the right and y increases upward."
        )
        evidence_noun = "supplied states"
        rows = [
            {
                **question,
                "question": str(question["question"]).replace(
                    "shown rollout", "supplied states"
                ),
            }
            for question in questions
        ]
    lines = [
        evidence_line,
        (
            "The orange circle/tool is the placed tool and the red circle/ball "
            "is the target ball."
        ),
        (
            "This is a qualitative retrospective reading test. Answer only "
            f"what is clearly visible in the {evidence_noun}; no goal, gravity "
            "label, or success/failure label is supplied."
        ),
        "",
    ]
    for question in rows:
        lines.append(f"{question['question_id']}: {question['question']}")
        lines.extend(
            f"  {letter}. {value}"
            for letter, value in question["options"].items()
        )
    return "\n".join(lines)


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
    parser.add_argument("--include-qwen", action="store_true")
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.allow_partial = False
    return args


if __name__ == "__main__":
    args = parse_args()
    base.QUESTION_VERSION = QUESTION_VERSION
    base.MODEL_KEYS = (
        ("gpt", "gemini", "qwen")
        if args.include_qwen
        else ("gpt", "gemini")
    )
    base.raw_questions = raw_questions
    base.system_prompt = system_prompt
    base.user_prompt = user_prompt
    base.assign_balanced_slots = balance_semantics_then_slots
    report = base.build(args)
    report["question_version"] = QUESTION_VERSION
    report["hidden_margin_policy"] = {
        "visible_displacement_px": VISIBLE_DISPLACEMENT_PX,
        "stationary_displacement_px": STATIONARY_DISPLACEMENT_PX,
        "minimum_onset_gap_states": MIN_ONSET_GAP_STATES,
        "thresholds_shown_to_models": False,
        "intermediate_cases_omitted": True,
    }
    report["semantic_answer_balancing"] = {
        "enabled": True,
        "rule": (
            "within each non-invariant family, deterministically retain the "
            "same number of items for each answer value under seed 2026"
        ),
        "invariant_families_omitted": True,
        "selection_uses_model_outputs": False,
    }
    base.write_json(args.output_root / "manifest_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
