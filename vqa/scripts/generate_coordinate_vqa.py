#!/usr/bin/env python3
"""Generate a free-response coordinate-localization control for 66 VTools layouts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "coordinate_localization_v3_task_assets"
ASSET_INDEX = (
    Path.home()
    / "Library/Caches/vtools_goal_probe_runner/forked_feedback_baseline/asset_index.json"
)
TARGETS = {
    "red_ball_center": "red target ball",
    "green_container_center": "green goal container",
}
MODELS = ("gpt", "gemini", "qwen")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def center_of_geometry(geometry: dict[str, Any]) -> list[float]:
    if "position" in geometry:
        return [float(geometry["position"][0]), float(geometry["position"][1])]
    points = geometry.get("points") or geometry.get("vertices")
    if not points:
        polygons = (
            geometry.get("polygons")
            or geometry.get("polylist")
            or geometry.get("polys")
            or []
        )
        points = [point for polygon in polygons for point in polygon]
    if not points:
        raise ValueError(f"Cannot find a center for geometry: {geometry}")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2]


def color_of(obj: dict[str, Any]) -> str:
    return str(obj.get("color") or obj.get("innerColor") or "black").lower()


def image_color_center(path: Path, color: str) -> list[float]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    selected: list[tuple[int, int]] = []
    for image_y in range(image.height):
        for image_x in range(image.width):
            red, green, blue = image.getpixel((image_x, image_y))
            if color == "red":
                matches = red > green + 30 and red > blue + 30 and red > 90
            elif color == "green":
                matches = green > red + 20 and green > blue + 20 and green > 90
            else:
                raise ValueError(color)
            if matches:
                selected.append((image_x, image_y))
    if not selected:
        raise RuntimeError(f"No {color} pixels found in {path}")
    xs = [point[0] for point in selected]
    ys = [point[1] for point in selected]
    image_center_x = (min(xs) + max(xs)) / 2
    image_center_y = (min(ys) + max(ys)) / 2
    return [image_center_x, (image.height - 1) - image_center_y]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    scene_paths = sorted((ROOT / "scenes").glob("*.json"))
    if len(scene_paths) != 66:
        raise RuntimeError(f"Expected 66 scenes, found {len(scene_paths)}")
    truth_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    assets = read_json(ASSET_INDEX)["puzzles"]
    for scene_path in scene_paths:
        layout_id = scene_path.stem
        candidate_keys = (
            f"{layout_id}_downward",
            f"{layout_id}_upward",
        )
        asset_key = next((key for key in candidate_keys if key in assets), None)
        if asset_key is None:
            raise RuntimeError(f"Missing task asset for {layout_id}")
        asset = assets[asset_key]
        source_image = Path(asset["screenshot_path"])
        world = read_json(Path(asset["world_path"]))["world"]
        visible = {
            object_id: obj
            for object_id, obj in world["objects"].items()
            if not str(object_id).startswith("_")
        }
        red_objects = [
            obj for obj in visible.values() if color_of(obj) == "red"
        ]
        green_objects = [
            obj for obj in visible.values() if color_of(obj) == "green"
        ]
        if len(red_objects) != 1 or len(green_objects) != 1:
            raise RuntimeError(
                f"{layout_id}: expected one red and one green object, found "
                f"{len(red_objects)} and {len(green_objects)}"
            )
        image_path = OUTPUT_ROOT / "images" / f"{layout_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_path)
        objects = {
            "red target ball": red_objects[0],
            "green goal container": green_objects[0],
        }
        target_truth: dict[str, Any] = {}
        for target_id, label in TARGETS.items():
            center = center_of_geometry(objects[label])
            pixel_center = image_color_center(
                image_path,
                "red" if target_id == "red_ball_center" else "green",
            )
            center_disagreement = (
                (center[0] - pixel_center[0]) ** 2
                + (center[1] - pixel_center[1]) ** 2
            ) ** 0.5
            if center_disagreement > 3:
                raise RuntimeError(
                    f"{layout_id} {target_id}: task screenshot center {pixel_center} "
                    f"does not match world center {center}; disagreement={center_disagreement:.3f}"
                )
            target_truth[target_id] = {
                "label": label,
                "center": center,
                "pixel_mask_center": pixel_center,
                "world_pixel_center_disagreement_px": center_disagreement,
                "distance_from_x_divider": abs(center[0] - 300),
                "distance_from_y_divider": abs(center[1] - 300),
                "x_half_scored": abs(center[0] - 300) > 60,
                "y_half_scored": abs(center[1] - 300) > 60,
                "combined_halves_scored": (
                    abs(center[0] - 300) > 60 and abs(center[1] - 300) > 60
                ),
            }
        truth_rows.append(
            {
                "layout_id": layout_id,
                "canvas": {
                    "width": 600,
                    "height": 600,
                    "coordinate_origin": "bottom-left",
                    "x_direction": "right",
                    "y_direction": "up",
                },
                "targets": target_truth,
                "source_world": str(Path(asset["world_path"]).resolve()),
                "source_image": str(source_image.resolve()),
                "source_world_sha256": asset["world_sha256"],
                "source_image_sha256": asset["screenshot_sha256"],
            }
        )
        payload = {
            "system": (
                "You are localizing objects in one INITIAL STATIC 2D scene. "
                "No action has occurred and no physics prediction is requested. "
                "Return only the requested JSON coordinate object."
            ),
            "user_text": """The attached image is exactly 600 pixels wide and 600 pixels tall.

Coordinate convention:
- The bottom-left corner is (0,0).
- x increases from left to right.
- y increases from bottom to top.
- Valid coordinates range from 0 through 599.

Estimate the center-point coordinates of:
1. the red target ball
2. the green goal container

For the green open container, use the center of its overall axis-aligned bounding box, including the empty interior. Reasonable visual estimates are expected; pixel-perfect answers are not required.

Return exactly:
{"red_ball_center":[x,y],"green_container_center":[x,y]}""",
            "image_path": str(image_path.resolve()),
            "scene_json": None,
            "response_schema": {
                "type": "coordinate_localization",
                "required_targets": list(TARGETS),
            },
        }
        payload_path = OUTPUT_ROOT / "payloads" / f"{layout_id}.json"
        write_json(payload_path, payload)
        for model in MODELS:
            call_rows.append(
                {
                    "call_id": f"{layout_id}:{model}:image_coordinates_v2",
                    "layout_id": layout_id,
                    "model_key": model,
                    "input_condition": "image_coordinates",
                    "benchmark_cell_aliases": [
                        f"{layout_id}_upward",
                        f"{layout_id}_downward",
                    ],
                    "payload_path": str(payload_path.resolve()),
                    "batching": "one image call returns both target center estimates",
                }
            )
    write_jsonl(OUTPUT_ROOT / "ground_truth.jsonl", truth_rows)
    write_jsonl(OUTPUT_ROOT / "call_manifest.jsonl", call_rows)
    write_json(
        OUTPUT_ROOT / "manifest_summary.json",
        {
            "layouts": len(truth_rows),
            "models": list(MODELS),
            "paid_calls": len(call_rows),
            "targets_per_call": list(TARGETS),
            "coordinate_tolerance_px": [60, 100, 150],
            "half_location_exclusion_band_px": 60,
            "half_location_rule": "strictly more than 60 px from the relevant divider",
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "layouts": len(truth_rows),
                "calls": len(call_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
