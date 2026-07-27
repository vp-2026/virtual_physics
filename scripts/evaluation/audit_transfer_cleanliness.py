#!/usr/bin/env python3
"""Audit action and transfer response coverage without scoring accuracy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def predictions_from(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    predictions = payload.get("coordinate_predictions")
    if not isinstance(predictions, list):
        parsed = payload.get("parsed")
        predictions = (
            parsed.get("coordinate_predictions")
            if isinstance(parsed, dict)
            else None
        )
    return [item for item in (predictions or []) if isinstance(item, dict)]


def prediction_issues(
    predictions: list[dict[str, Any]],
    expected_ids: set[str],
) -> Counter:
    issues: Counter = Counter()
    ids = [str(item.get("id") or "") for item in predictions]
    issues["missing_ids"] += len(expected_ids - set(ids))
    issues["unexpected_ids"] += len(set(ids) - expected_ids)
    issues["duplicate_ids"] += len(ids) - len(set(ids))
    for item in predictions:
        state = item.get("state")
        point = item.get("point")
        if state == "in_scene":
            valid_point = (
                isinstance(point, list)
                and len(point) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 0 <= float(value) <= 600
                    for value in point
                )
            )
            if not valid_point:
                issues["coordinate_issues"] += 1
        elif state == "exited":
            if point is not None:
                issues["coordinate_issues"] += 1
        else:
            issues["state_issues"] += 1
    return issues


def valid_action_point(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    point = payload.get("point")
    if point is None and isinstance(payload.get("parsed"), dict):
        point = payload["parsed"].get("point")
    return (
        isinstance(point, list)
        and len(point) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= float(value) <= 600
            for value in point
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.expanduser().resolve()

    by_model: dict[str, Counter] = defaultdict(Counter)
    unit_dirs = sorted(path.parent for path in root.glob("*/seed_*/**/unit_summary.json"))
    for unit_dir in unit_dirs:
        model = unit_dir.parents[1].name
        metrics = by_model[model]
        metrics["units"] += 1

        shared_state_path = unit_dir / "shared_attempt_1" / "state.json"
        if not shared_state_path.exists():
            metrics["units_missing_shared_action"] += 1
            continue
        shared = json.loads(shared_state_path.read_text(encoding="utf-8"))
        shared_response = (shared.get("attempt") or {}).get("model_response")
        expected_ids = {
            str(item.get("id") or "")
            for item in predictions_from(shared_response)
        }
        if not expected_ids:
            metrics["units_missing_expected_ids"] += 1

        action_paths = [
            path
            for path in unit_dir.rglob("state.json")
            if "transfer_sidecars" not in path.parts
        ]
        for path in action_paths:
            state = json.loads(path.read_text(encoding="utf-8"))
            response = (state.get("attempt") or {}).get("model_response")
            if not isinstance(response, dict):
                continue
            metrics["action_payloads"] += 1
            if not response.get("schema_valid"):
                metrics["action_schema_invalid"] += 1
            if not valid_action_point(response):
                metrics["action_point_issues"] += 1
            metrics.update(prediction_issues(predictions_from(response), expected_ids))

        for path in (unit_dir / "transfer_sidecars").rglob("result.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if path.parent.name == "goal_transfer":
                metrics["goal_transfer_payloads"] += 1
                response = (data.get("attempt") or {}).get("model_response")
                if not isinstance(response, dict) or not response.get("schema_valid"):
                    metrics["goal_transfer_schema_invalid"] += 1
                if not valid_action_point(response):
                    metrics["goal_transfer_action_point_issues"] += 1
                issues = prediction_issues(
                    predictions_from(response),
                    expected_ids,
                )
                for name, count in issues.items():
                    metrics[f"goal_transfer_{name}"] += count
                if data.get("error"):
                    metrics["goal_transfer_errors"] += 1
            else:
                metrics["heldout_payloads"] += 1
                if not data.get("schema_valid"):
                    metrics["heldout_schema_invalid"] += 1
                validation = data.get("validation") or {}
                for name in ("missing_ids", "unexpected_ids", "duplicate_ids"):
                    value = validation.get(name) or []
                    metrics[f"heldout_{name}"] += len(value)
                issues = prediction_issues(
                    predictions_from(data.get("parsed_payload")),
                    {
                        str(item.get("id") or "")
                        for item in (validation.get("predictions") or [])
                        if isinstance(item, dict)
                    },
                )
                metrics["heldout_coordinate_issues"] += issues["coordinate_issues"]
                metrics["heldout_state_issues"] += issues["state_issues"]

    totals: Counter = Counter()
    for metrics in by_model.values():
        totals.update(metrics)
    print(
        json.dumps(
            {
                "result_root": str(root),
                "models": {
                    model: dict(sorted(metrics.items()))
                    for model, metrics in sorted(by_model.items())
                },
                "totals": dict(sorted(totals.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
