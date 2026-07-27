#!/usr/bin/env python3
"""Regenerate the static VQA control from screenshots used by the task runner.

The original VQA generator used a separate showcase image directory. This
version takes the source image/world pairs already verified by the coordinate
localization audit so every question is grounded in the exact task asset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_vqa import (
    DEFAULT_MODELS,
    build_user_prompt,
    numeric,
    questions_for_scene,
    read_json,
    role_objects,
    sanitized_scene,
    sha256_file,
    system_prompt,
    visible_objects,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_COORDINATE_ROOT = ROOT / "coordinate_localization_v3_task_assets"
DEFAULT_OUTPUT = ROOT / "static_vqa_v2_task_assets"
CONDITIONS = ("image_only", "json_only")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate(coordinate_root: Path, output: Path) -> dict[str, Any]:
    sources = read_jsonl(coordinate_root / "ground_truth.jsonl")
    if len(sources) != 66 or len({row["layout_id"] for row in sources}) != 66:
        raise ValueError("Expected exactly 66 unique verified task assets")

    question_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []

    for scene_index, source in enumerate(sources):
        layout_id = source["layout_id"]
        image_path = coordinate_root / "images" / f"{layout_id}.png"
        world_path = Path(source["source_world"])
        if not image_path.exists() or not world_path.exists():
            raise FileNotFoundError(f"Missing verified asset for {layout_id}")
        if sha256_file(image_path) != source["source_image_sha256"]:
            raise ValueError(f"Task screenshot hash changed for {layout_id}")
        if sha256_file(world_path) != source["source_world_sha256"]:
            raise ValueError(f"Task world hash changed for {layout_id}")

        world = read_json(world_path)
        objects = visible_objects(world)
        _, green, _ = role_objects(objects)
        scene_json = sanitized_scene(world)
        scene_json_path = output / "scenes" / f"{layout_id}.json"
        write_json(scene_json_path, scene_json)

        questions = questions_for_scene(layout_id, scene_index, world)
        question_rows.append({"layout_id": layout_id, "questions": questions})
        green_movable = numeric(green.get("density")) > 0

        asset_rows.append(
            {
                "layout_id": layout_id,
                "image_path": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "world_path": str(world_path.resolve()),
                "world_sha256": sha256_file(world_path),
                "coordinate_audit_max_center_disagreement_px": max(
                    target["world_pixel_center_disagreement_px"]
                    for target in source["targets"].values()
                ),
            }
        )

        for condition in CONDITIONS:
            representation_note = {
                "image_only": (
                    "The attached clean 600-by-600 image contains the initial scene. "
                    "Image coordinates use (0,0) at bottom-left; x increases to the "
                    "right and y increases upward."
                ),
                "json_only": (
                    "The attached JSON contains the visible initial scene geometry. "
                    "Coordinates use (0,0) at bottom-left; x increases to the right "
                    "and y increases upward."
                ),
            }[condition]
            payload = {
                "system": system_prompt(),
                "user_text": build_user_prompt(
                    questions,
                    representation_note,
                    green_movable,
                ),
                "image_path": str(image_path.resolve())
                if condition == "image_only"
                else None,
                "scene_json": scene_json if condition == "json_only" else None,
                "response_schema": {
                    "type": "object",
                    "required_question_ids": [
                        question["question_id"] for question in questions
                    ],
                    "values": "one option letter",
                },
            }
            payload_path = output / "payloads" / condition / f"{layout_id}.json"
            write_json(payload_path, payload)
            for model_key in DEFAULT_MODELS:
                calls.append(
                    {
                        "call_id": f"task_asset_vqa_v2:{layout_id}:{model_key}:{condition}",
                        "layout_id": layout_id,
                        "benchmark_cell_aliases": [
                            f"{layout_id}_upward",
                            f"{layout_id}_downward",
                        ],
                        "model_key": model_key,
                        "input_condition": condition,
                        "payload_path": str(payload_path.resolve()),
                        "batching": (
                            "one call contains all static questions for one unique layout"
                        ),
                    }
                )

    write_jsonl(output / "assets.jsonl", asset_rows)
    write_jsonl(output / "questions.jsonl", question_rows)
    write_jsonl(output / "call_manifest.jsonl", calls)
    summary = {
        "status": "valid",
        "unique_layouts": len(asset_rows),
        "questions_per_layout": len(question_rows[0]["questions"]),
        "question_count": sum(len(row["questions"]) for row in question_rows),
        "input_conditions": list(CONDITIONS),
        "model_keys": list(DEFAULT_MODELS),
        "call_count": len(calls),
        "asset_source": str(coordinate_root.resolve()),
        "asset_provenance": (
            "Exact initial screenshots and simulation worlds indexed by the task runner; "
            "red/green world centers independently checked against screenshot color masks."
        ),
        "coordinate_convention": {
            "canvas": [600, 600],
            "origin": "bottom-left",
            "x": "increases right",
            "y": "increases upward",
        },
        "coarse_location_scoring": {
            "divider": [300, 300],
            "exclusion_band_px": 60,
            "rule": (
                "A quadrant response is included only when the target center is "
                "strictly more than 60 pixels from both dividers."
            ),
            "primary_coordinate_evidence": (
                "Free-response coordinate_localization_v3_task_assets probe"
            ),
        },
        "no_motion": True,
        "paid_calls_made_by_generator": False,
    }
    write_json(output / "manifest_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coordinate-root",
        type=Path,
        default=DEFAULT_COORDINATE_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(args.coordinate_root.resolve(), args.output.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
