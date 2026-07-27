#!/usr/bin/env python3
"""Score the retrospective 32-frame rollout-perception control."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("--questions", type=Path, default=ROOT / "questions.jsonl")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.responses = args.responses.expanduser().resolve()
    args.questions = args.questions.expanduser().resolve()
    output_dir = (args.output_dir or args.responses.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = {
        row["cell_id"]: {
            question["question_id"]: question for question in row["questions"]
        }
        for row in read_jsonl(args.questions)
    }
    raw_responses = read_jsonl(args.responses)
    response_attempt_counts: dict[str, int] = defaultdict(int)
    response_by_call: dict[str, dict[str, Any]] = {}
    for response in raw_responses:
        call_id = str(response["call_id"])
        response_attempt_counts[call_id] += 1
        valid = not response.get("error") and bool(
            response.get("format_valid")
        )
        prior = response_by_call.get(call_id)
        prior_valid = (
            prior is not None
            and not prior.get("error")
            and bool(prior.get("format_valid"))
        )
        if prior is None or valid or not prior_valid:
            response_by_call[call_id] = response

    rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    for response in (
        response_by_call[call_id] for call_id in sorted(response_by_call)
    ):
        call_id = str(response["call_id"])
        cell_id = str(response.get("cell_id") or call_id.split(":", 1)[0])
        parsed = response.get("parsed_response") or {}
        valid = not response.get("error") and bool(response.get("format_valid"))
        expected_questions = truth.get(cell_id)
        if expected_questions is None:
            raise KeyError(f"No temporal VQA ground truth for {cell_id}")
        correct_count = 0
        for question_id, question in expected_questions.items():
            observed = str(parsed.get(question_id, "")).strip().upper()
            correct = int(valid and observed == question["answer"])
            correct_count += correct
            rows.append(
                {
                    "call_id": call_id,
                    "cell_id": cell_id,
                    "layout_id": response.get("layout_id"),
                    "gravity_mode": cell_id.rsplit("_", 1)[-1],
                    "model_key": response["model_key"],
                    "model_id": response.get("model_id_returned")
                    or response.get("model_id_requested"),
                    "question_id": question_id,
                    "category": question["category"],
                    "observed": observed,
                    "expected": question["answer"],
                    "correct": correct,
                    "format_valid": int(valid),
                    "response_attempt_count": response_attempt_counts[
                        call_id
                    ],
                }
            )
        call_rows.append(
            {
                "call_id": call_id,
                "cell_id": cell_id,
                "layout_id": response.get("layout_id"),
                "gravity_mode": cell_id.rsplit("_", 1)[-1],
                "model_key": response["model_key"],
                "format_valid": int(valid),
                "response_attempt_count": response_attempt_counts[call_id],
                "correct": correct_count,
                "questions": len(expected_questions),
                "accuracy": correct_count / len(expected_questions),
                "error": response.get("error"),
                "provider_cost_usd": float(
                    (response.get("usage") or {}).get("cost") or 0.0
                ),
                "prompt_tokens": int(
                    (response.get("usage") or {}).get("prompt_tokens") or 0
                ),
                "completion_tokens": int(
                    (response.get("usage") or {}).get("completion_tokens")
                    or 0
                ),
            }
        )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for gravity in (row["gravity_mode"], "all"):
            for category in (row["category"], "__overall__"):
                groups[(row["model_key"], gravity, category)].append(row)
    summary = []
    for (model, gravity, category), group in sorted(groups.items()):
        layout_values: dict[str, list[int]] = defaultdict(list)
        for row in group:
            layout_values[str(row["layout_id"])].append(int(row["correct"]))
        layout_means = [
            sum(values) / len(values) for values in layout_values.values()
        ]
        summary.append(
            {
                "model_key": model,
                "gravity_mode": gravity,
                "category": category,
                "questions": len(group),
                "cells": len({row["cell_id"] for row in group}),
                "micro_accuracy": sum(int(row["correct"]) for row in group)
                / len(group),
                "layout_macro_accuracy": sum(layout_means) / len(layout_means),
            }
        )

    write_csv(output_dir / "question_scores.csv", rows)
    write_csv(output_dir / "call_scores.csv", call_rows)
    write_csv(output_dir / "accuracy_summary.csv", summary)
    raw_cost_by_model: dict[str, float] = defaultdict(float)
    raw_calls_by_model: Counter[str] = Counter()
    for response in raw_responses:
        model_key = str(response.get("model_key") or "unknown")
        raw_calls_by_model[model_key] += 1
        raw_cost_by_model[model_key] += float(
            (response.get("usage") or {}).get("cost") or 0.0
        )
    report = {
        "response_file": str(args.responses),
        "question_file": str(args.questions),
        "raw_response_rows": len(raw_responses),
        "calls": len(call_rows),
        "questions": len(rows),
        "format_valid_calls": sum(row["format_valid"] for row in call_rows),
        "raw_provider_calls_by_model": dict(sorted(raw_calls_by_model.items())),
        "raw_provider_cost_usd_by_model": dict(
            sorted(raw_cost_by_model.items())
        ),
        "raw_provider_cost_usd_total": sum(raw_cost_by_model.values()),
        "question_scores": str(output_dir / "question_scores.csv"),
        "call_scores": str(output_dir / "call_scores.csv"),
        "accuracy_summary": str(output_dir / "accuracy_summary.csv"),
    }
    (output_dir / "score_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
