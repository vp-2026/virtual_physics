#!/usr/bin/env python3
"""Extract the 132 canonical source cells for prompt/seed robustness."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build(source: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_goals = payload.get("goals") if isinstance(payload, dict) else None
    if not isinstance(raw_goals, list):
        raise ValueError("Source manifest does not contain a goals list")
    goals = []
    for full_index, raw in enumerate(raw_goals):
        if (
            str(raw.get("source") or "") != "canonical_world_gcond"
            and not bool(raw.get("transfer_source"))
        ):
            continue
        goal = copy.deepcopy(raw)
        goal["full_benchmark_index"] = full_index
        goals.append(goal)
    if len(goals) != 132:
        raise RuntimeError(f"Expected 132 canonical goals, found {len(goals)}")
    gravity = Counter(
        str(
            goal.get("condition")
            or goal.get("gravity")
            or str(goal.get("run_name") or "").rsplit("_", 1)[-1]
        )
        for goal in goals
    )
    if gravity != Counter({"downward": 66, "upward": 66}):
        raise RuntimeError(f"Unexpected canonical gravity counts: {gravity}")
    output_payload = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key != "goals"
    }
    output_payload.update(
        {
            "schema_version": max(
                int(output_payload.get("schema_version") or 1),
                1,
            ),
            "purpose": "canonical_attempt1_prompt_and_seed_robustness",
            "source_manifest": str(source.resolve()),
            "selection_uses_model_outputs": False,
            "goal_count": len(goals),
            "gravity_counts": dict(sorted(gravity.items())),
            "goals": goals,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "goal_count": len(goals),
        "gravity_counts": dict(sorted(gravity.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
