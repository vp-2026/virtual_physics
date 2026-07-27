#!/usr/bin/env python3
"""Score the corrected static VQA run against its task-asset question file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--coordinate-ground-truth",
        type=Path,
        required=True,
        help="Verified centers and strict 60-pixel divider eligibility.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth = {
        row["layout_id"]: {
            question["question_id"]: question for question in row["questions"]
        }
        for row in read_jsonl(args.questions)
    }
    coordinate_truth = {
        row["layout_id"]: row["targets"]
        for row in read_jsonl(args.coordinate_ground_truth)
    }
    rows: list[dict[str, Any]] = []
    raw_call_rows = read_jsonl(args.responses)
    by_call_id: dict[str, dict[str, Any]] = {}
    for response in raw_call_rows:
        call_id = response["call_id"]
        incumbent = by_call_id.get(call_id)
        if incumbent is None or (
            bool(response.get("format_valid"))
            and not response.get("error")
            and (
                not bool(incumbent.get("format_valid"))
                or bool(incumbent.get("error"))
            )
        ):
            by_call_id[call_id] = response
    call_rows = list(by_call_id.values())
    for response in call_rows:
        model = response["model_key"]
        condition = response["input_condition"]
        parsed = response.get("parsed_response") or {}
        for question_id, question in truth[response["layout_id"]].items():
            observed = str(parsed.get(question_id, "")).upper()
            primary_scored = True
            exclusion_reason = ""
            if question["category"] == "coarse_location":
                target_id = (
                    "red_ball_center"
                    if question_id.endswith(":red_quadrant")
                    else "green_container_center"
                )
                primary_scored = bool(
                    coordinate_truth[response["layout_id"]][target_id][
                        "combined_halves_scored"
                    ]
                )
                if not primary_scored:
                    exclusion_reason = "within_60px_of_x_or_y_divider"
            rows.append(
                {
                    "call_id": response["call_id"],
                    "layout_id": response["layout_id"],
                    "model_key": model,
                    "input_condition": condition,
                    "question_id": question_id,
                    "category": question["category"],
                    "observed": observed,
                    "expected": question["answer"],
                    "correct": int(observed == question["answer"]),
                    "primary_scored": int(primary_scored),
                    "exclusion_reason": exclusion_reason,
                    "format_valid": int(bool(response.get("format_valid"))),
                    "call_error": response.get("error") or "",
                }
            )

    detail_path = args.output_dir / "question_scores.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        if not row["primary_scored"]:
            continue
        for category in ("__overall__", row["category"]):
            groups[
                (row["model_key"], row["input_condition"], category)
            ].append(row["correct"])
    summary = [
        {
            "model_key": key[0],
            "input_condition": key[1],
            "category": key[2],
            "questions": len(values),
            "correct": sum(values),
            "accuracy": sum(values) / len(values),
        }
        for key, values in sorted(groups.items())
    ]
    summary_path = args.output_dir / "accuracy_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    valid_calls = sum(
        int(bool(row.get("format_valid")) and not row.get("error"))
        for row in call_rows
    )
    report = {
        "calls": len(call_rows),
        "raw_response_rows": len(raw_call_rows),
        "duplicate_retry_rows": len(raw_call_rows) - len(call_rows),
        "valid_calls": valid_calls,
        "format_valid_rate": valid_calls / len(call_rows),
        "questions_scored": len(rows),
        "primary_questions_scored": sum(row["primary_scored"] for row in rows),
        "ambiguous_quadrant_answers_excluded": sum(
            not row["primary_scored"] for row in rows
        ),
        "question_scores": str(detail_path.resolve()),
        "accuracy_summary": str(summary_path.resolve()),
    }
    (args.output_dir / "score_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
