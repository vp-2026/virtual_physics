#!/usr/bin/env python3
"""Build a deterministic 5% same-scene A/B transfer engineering pilot.

The 76 selected goals form 38 unordered pairs.  Both directions are tested:
each goal is a fresh-context baseline for its partner and a source context for
one terminal one-shot transfer.  Selection never reads model outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE = REPO_ROOT / "task_configs" / "paper_goals_1560.json"
OUTPUT = REPO_ROOT / "task_configs" / "transfer_pair_manifest_76_seed2026.json"
SEED = 2026
PAIRS_PER_GRAVITY = 19
# A 1.25 ratio is a tight enough caliper to preserve comparable solution
# density while allowing both gravity conditions to include the Basic family.
MAX_DENSITY_RATIO = 1.25


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def gravity(run_name: str) -> str:
    value = run_name.rsplit("_", 1)[-1]
    if value not in {"downward", "upward"}:
        raise ValueError(f"Cannot infer gravity from {run_name!r}")
    return value


def family(run_name: str) -> str:
    return run_name.split("_", 1)[0]


def quantile_targets(values: list[float], count: int) -> list[float]:
    ordered = sorted(values)
    output = []
    for index in range(count):
        position = (len(ordered) - 1) * ((index + 0.5) / count)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        output.append(
            (1.0 - weight) * ordered[lower] + weight * ordered[upper]
        )
    return output


def candidate_pairs(
    goals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for goal in goals:
        if goal.get("source") == "canonical_world_gcond":
            continue
        if not goal.get("p_success"):
            continue
        by_cell[str(goal["run_name"])].append(goal)

    output: list[dict[str, Any]] = []
    for run_name, rows in sorted(by_cell.items()):
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                if str(left["category_5"]) == str(right["category_5"]):
                    continue
                if str(left["signature"]) == str(right["signature"]):
                    continue
                if str(left.get("equivalence_key")) == str(
                    right.get("equivalence_key")
                ):
                    continue
                left_p = float(left["p_success"])
                right_p = float(right["p_success"])
                ratio = max(left_p, right_p) / min(left_p, right_p)
                if ratio > MAX_DENSITY_RATIO:
                    continue
                categories = tuple(
                    sorted((str(left["category_5"]), str(right["category_5"])))
                )
                output.append(
                    {
                        "run_name": run_name,
                        "gravity": gravity(run_name),
                        "family": family(run_name),
                        "category_pair": categories,
                        "left": left,
                        "right": right,
                        "density_ratio": ratio,
                        "log_density_gap": abs(
                            math.log(left_p) - math.log(right_p)
                        ),
                        "mean_log_density": (
                            math.log(left_p) + math.log(right_p)
                        )
                        / 2.0,
                    }
                )
    return output


def select_for_gravity(
    candidates: list[dict[str, Any]], value: str
) -> list[dict[str, Any]]:
    pool = [
        item for item in candidates if str(item["gravity"]) == value
    ]
    if not pool:
        raise RuntimeError(f"No candidate pairs for {value}")
    targets = quantile_targets(
        [float(item["mean_log_density"]) for item in pool],
        PAIRS_PER_GRAVITY,
    )
    selected: list[dict[str, Any]] = []
    selected_cells: set[str] = set()
    family_counts: Counter[str] = Counter()
    category_pair_counts: Counter[tuple[str, str]] = Counter()
    for target in targets:
        eligible = [
            item
            for item in pool
            if str(item["run_name"]) not in selected_cells
        ]
        if not eligible:
            raise RuntimeError(
                f"Could not select {PAIRS_PER_GRAVITY} unique cells for {value}"
            )
        chosen = min(
            eligible,
            key=lambda item: (
                family_counts[str(item["family"])],
                category_pair_counts[tuple(item["category_pair"])],
                abs(float(item["mean_log_density"]) - target),
                float(item["log_density_gap"]),
                stable_hash(
                    "|".join(
                        (
                            str(item["run_name"]),
                            str(item["left"]["balanced_goal_id"]),
                            str(item["right"]["balanced_goal_id"]),
                        )
                    )
                ),
            ),
        )
        selected.append(chosen)
        selected_cells.add(str(chosen["run_name"]))
        family_counts[str(chosen["family"])] += 1
        category_pair_counts[tuple(chosen["category_pair"])] += 1
    return selected


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    goals = [dict(goal) for goal in source["goals"]]
    all_candidates = candidate_pairs(goals)
    pairs = []
    for value in ("downward", "upward"):
        pairs.extend(select_for_gravity(all_candidates, value))
    pairs.sort(
        key=lambda item: (
            str(item["gravity"]),
            str(item["run_name"]),
            str(item["left"]["balanced_goal_id"]),
            str(item["right"]["balanced_goal_id"]),
        )
    )

    selected_goals: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    for pair_index, item in enumerate(pairs, start=1):
        pair_id = f"pair_{pair_index:02d}"
        left = dict(item["left"])
        right = dict(item["right"])
        for source_goal, target_goal, direction in (
            (left, right, "left_to_right"),
            (right, left, "right_to_left"),
        ):
            source_goal["transfer_pair_id"] = pair_id
            source_goal["transfer_direction"] = direction
            source_goal["transfer_target_goal_id"] = str(
                target_goal["balanced_goal_id"]
            )
            selected_goals.append(source_goal)
        pair_records.append(
            {
                "pair_id": pair_id,
                "run_name": item["run_name"],
                "gravity": item["gravity"],
                "family": item["family"],
                "category_pair": list(item["category_pair"]),
                "goal_ids": [
                    str(left["balanced_goal_id"]),
                    str(right["balanced_goal_id"]),
                ],
                "goal_texts": [
                    str(
                        left.get("goal_text_cleaned")
                        or left.get("goal_text_original")
                    ),
                    str(
                        right.get("goal_text_cleaned")
                        or right.get("goal_text_original")
                    ),
                ],
                "p_success": [
                    float(left["p_success"]),
                    float(right["p_success"]),
                ],
                "density_ratio": float(item["density_ratio"]),
                "log_density_gap": float(item["log_density_gap"]),
            }
        )

    selected_goals.sort(
        key=lambda goal: (
            gravity(str(goal["run_name"])),
            str(goal["run_name"]),
            str(goal["balanced_goal_id"]),
        )
    )
    goal_ids = [str(goal["balanced_goal_id"]) for goal in selected_goals]
    if len(selected_goals) != 76 or len(set(goal_ids)) != 76:
        raise RuntimeError("Expected 76 unique selected goals")
    by_id = {str(goal["balanced_goal_id"]): goal for goal in selected_goals}
    for goal in selected_goals:
        target = by_id[str(goal["transfer_target_goal_id"])]
        if str(target["run_name"]) != str(goal["run_name"]):
            raise RuntimeError("Transfer pair crosses a layout/gravity cell")
        if str(target["transfer_target_goal_id"]) != str(
            goal["balanced_goal_id"]
        ):
            raise RuntimeError("Transfer assignment is not a two-cycle")

    payload = {
        "schema_version": 1,
        "name": "vtools_transfer_engineering_pilot_5pct_seed2026",
        "source_manifest": str(SOURCE),
        "source_manifest_sha256": hashlib.sha256(
            SOURCE.read_bytes()
        ).hexdigest(),
        "selection_seed": SEED,
        "selection_uses_model_outcomes": False,
        "diagnostic_only_until_persistence_resweep": True,
        "goal_count": len(selected_goals),
        "pair_count": len(pair_records),
        "pairing_rules": {
            "same_layout_and_gravity": True,
            "different_goal_category": True,
            "different_signature_and_equivalence_key": True,
            "maximum_solution_density_ratio": MAX_DENSITY_RATIO,
            "both_directions_collected": True,
            "fresh_target_baseline_is_partner_shared_attempt_1": True,
        },
        "summary": {
            "gravity_goal_counts": dict(
                sorted(
                    Counter(
                        gravity(str(goal["run_name"]))
                        for goal in selected_goals
                    ).items()
                )
            ),
            "gravity_pair_counts": dict(
                sorted(Counter(item["gravity"] for item in pair_records).items())
            ),
            "family_pair_counts": dict(
                sorted(Counter(item["family"] for item in pair_records).items())
            ),
            "category_pair_counts": {
                "|".join(key): value
                for key, value in sorted(
                    Counter(
                        tuple(item["category_pair"]) for item in pair_records
                    ).items()
                )
            },
            "unique_layout_gravity_cells": len(
                {str(goal["run_name"]) for goal in selected_goals}
            ),
            "max_density_ratio": max(
                float(item["density_ratio"]) for item in pair_records
            ),
        },
        "transfer_pairs": pair_records,
        "goals": selected_goals,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
