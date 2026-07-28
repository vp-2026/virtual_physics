#!/usr/bin/env python3
"""Build path-clean public task manifests from the frozen run manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def public_goal(
    raw: dict[str, Any], environment_ids: dict[str, str]
) -> dict[str, Any]:
    item = dict(raw)
    item.pop("environment_json_source", None)
    for key in (
        "goal_transfer_eligible",
        "transfer_pair_selection",
        "transfer_source",
        "transfer_target_goal_id",
    ):
        item.pop(key, None)
    environment_id = environment_ids[str(item["run_name"])]
    item["environment_id"] = environment_id
    item["environment_path"] = (
        f"../132_base_environments/cells/{environment_id}/environment.json"
    )
    item["tool_path"] = (
        f"../132_base_environments/cells/{environment_id}/tool.json"
    )
    item["initial_scene_path"] = (
        f"../132_base_environments/cells/{environment_id}/initial_scene.png"
    )
    return item


def public_top_level(
    source: dict[str, Any], source_path: Path
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in source.items()
        if key
        not in {
            "goals",
            "goal_transfer_eligible_count",
            "goal_transfer_ineligible_count",
            "transfer_pairing_rules",
            "transfer_pairs",
            "transfer_source_count",
        }
    }
    for key in ("source_manifest", "source_main_manifest"):
        if key in result:
            result[key] = Path(str(result[key])).name
    result["release_source_filename"] = source_path.name
    result["release_source_sha256"] = sha256(source_path)
    return result


def density_row(goal: dict[str, Any]) -> dict[str, Any]:
    numerator = goal.get("event_count")
    denominator = goal.get("valid_placement_count")
    if numerator is None:
        numerator = goal.get("count")
    if denominator is None:
        denominator = goal.get("valid_placements")
    density = goal.get("p_success")
    density_kind = "paper_saved_solution_density"
    if density is None and goal.get("paper_solution_density_proxy") is not None:
        density = goal["paper_solution_density_proxy"]
        numerator = goal.get("paper_saved_solution_count")
        denominator = 10000
        density_kind = "paper_saved_canonical_density_proxy"
    return {
        "goal_id": goal["balanced_goal_id"],
        "environment_id": goal["environment_id"],
        "run_name": goal["run_name"],
        "gravity": str(goal["run_name"]).rsplit("_", 1)[-1],
        "category": goal["category_5"],
        "internal_subtype": goal["internal_subtype"],
        "solution_density": density,
        "success_count": numerator,
        "valid_placement_count": denominator,
        "density_kind": density_kind,
        "selection_uses_model_outcomes": False,
    }


def build_asset_index(
    environment_root: Path,
    run_asset_index: Path,
) -> dict[str, Any]:
    expected = json.loads(run_asset_index.read_text())["puzzles"]
    puzzles: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(environment_root / "index.jsonl"):
        cell_id = str(row["legacy_cell_id"])
        environment_id = str(row["environment_id"])
        cell_dir = environment_root / "cells" / environment_id
        world = cell_dir / "environment.json"
        screenshot = cell_dir / "initial_scene.png"
        world_hash = sha256(world)
        screenshot_hash = sha256(screenshot)
        if world_hash != expected[cell_id]["world_sha256"]:
            raise ValueError(f"World does not match model-run asset: {cell_id}")
        if screenshot_hash != expected[cell_id]["screenshot_sha256"]:
            raise ValueError(
                f"Screenshot does not match model-run asset: {cell_id}"
            )
        puzzles[cell_id] = {
            "environment_id": environment_id,
            "world_path": (
                f"../132_base_environments/cells/{environment_id}/"
                "environment.json"
            ),
            "screenshot_path": (
                f"../132_base_environments/cells/{environment_id}/"
                "initial_scene.png"
            ),
            "world_sha256": world_hash,
            "screenshot_sha256": screenshot_hash,
            "tool_path": (
                f"../132_base_environments/cells/{environment_id}/tool.json"
            ),
        }
    return {
        "schema_version": 2,
        "provenance": (
            "Path-clean mirror of the frozen asset index used by the model "
            "runs. Hashes are verified against the run index."
        ),
        "puzzle_count": len(puzzles),
        "puzzles": puzzles,
    }


def check_release(repo_root: Path) -> None:
    output = repo_root / "task_configs"
    environment_root = repo_root / "132_base_environments"
    paper = json.loads((output / "paper_goals_1560.json").read_text())
    benchmark = json.loads(
        (output / "benchmark_1692_seed2026.json").read_text()
    )
    assets = json.loads((output / "asset_index_132.json").read_text())
    if len(paper.get("goals") or []) != 1560:
        raise ValueError("paper_goals_1560.json does not contain 1,560 goals")
    goals = benchmark.get("goals") or []
    if len(goals) != 1692:
        raise ValueError(
            "benchmark_1692_seed2026.json does not contain 1,692 goals"
        )
    forbidden = {
        "goal_transfer_eligible_count",
        "goal_transfer_ineligible_count",
        "transfer_pairing_rules",
        "transfer_pairs",
        "transfer_source_count",
    }
    if forbidden.intersection(benchmark):
        raise ValueError("Goal-to-goal transfer metadata must not be released")
    for goal in goals:
        if any(
            key in goal
            for key in (
                "goal_transfer_eligible",
                "transfer_pair_selection",
                "transfer_source",
                "transfer_target_goal_id",
            )
        ):
            raise ValueError("Goal-to-goal transfer fields remain in a goal")
    cutoff = float(benchmark["paper_easy_goal_cutoff"])
    densities = [float(goal["p_success"]) for goal in goals[:1560]]
    if min(densities) <= 0 or max(densities) > cutoff:
        raise ValueError("Noncanonical solution densities violate the policy")
    if benchmark.get("selection_uses_model_outcomes") is not False:
        raise ValueError("Manifest must declare model-independent selection")
    puzzles = assets.get("puzzles") or {}
    if len(puzzles) != 132:
        raise ValueError("Expected 132 asset-index entries")
    for puzzle_key, item in puzzles.items():
        for field, hash_field in (
            ("world_path", "world_sha256"),
            ("screenshot_path", "screenshot_sha256"),
        ):
            path = (output / str(item[field])).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{puzzle_key}: missing {path}")
            if sha256(path) != item[hash_field]:
                raise ValueError(f"{puzzle_key}: {field} hash mismatch")
        tool = (output / str(item["tool_path"])).resolve()
        if not tool.is_file():
            raise FileNotFoundError(f"{puzzle_key}: missing {tool}")
    density_rows = list(
        csv.DictReader(
            (output / "solution_density_summary.csv").open(newline="")
        )
    )
    if len(density_rows) != 1692:
        raise ValueError("Expected 1,692 solution-density rows")
    print(
        "release check passed: 132 cells, 1,560 paper goals, "
        "1,692 expanded goals"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--run-asset-index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.check:
        check_release(repo_root)
        return
    required = {
        "--source-manifest": args.source_manifest,
        "--environment-root": args.environment_root,
        "--run-asset-index": args.run_asset_index,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    assert args.source_manifest is not None
    assert args.environment_root is not None
    assert args.run_asset_index is not None
    assert args.output_dir is not None
    source_path = args.source_manifest.resolve()
    environment_root = args.environment_root.resolve()
    output = args.output_dir.resolve()
    source = json.loads(source_path.read_text())
    environment_ids = {
        str(row["legacy_cell_id"]): str(row["environment_id"])
        for row in read_jsonl(environment_root / "index.jsonl")
    }
    goals = [
        public_goal(goal, environment_ids) for goal in source["goals"]
    ]
    if len(goals) != 1692:
        raise ValueError(f"Expected 1692 goals, found {len(goals)}")

    original = {
        **public_top_level(source, source_path),
        "name": "paper_saved_diverse_goals_1560",
        "goal_count": 1560,
        "goals": goals[:1560],
    }
    augmented = {**public_top_level(source, source_path), "goals": goals}
    write_json(output / "paper_goals_1560.json", original)
    write_json(output / "benchmark_1692_seed2026.json", augmented)
    write_json(
        output / "asset_index_132.json",
        build_asset_index(environment_root, args.run_asset_index.resolve()),
    )

    density_rows = [density_row(goal) for goal in goals]
    with (output / "solution_density_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(density_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(density_rows)

    write_json(
        output / "manifest_summary.json",
        {
            "goal_counts": {
                "paper_saved_noncanonical": 1560,
                "canonical_added": 132,
                "total": 1692,
            },
            "gravity_counts": Counter(
                str(goal["run_name"]).rsplit("_", 1)[-1] for goal in goals
            ),
            "category_counts": Counter(goal["category_5"] for goal in goals),
            "noncanonical_solution_density": {
                "minimum": min(goal["p_success"] for goal in goals[:1560]),
                "maximum": max(goal["p_success"] for goal in goals[:1560]),
                "easy_goal_exclusion_cutoff": source[
                    "paper_easy_goal_cutoff"
                ],
            },
            "selection_uses_model_outcomes": False,
            "selection_seed": source.get("selection_seed"),
            "source_manifest_sha256": sha256(source_path),
        },
    )


if __name__ == "__main__":
    main()
