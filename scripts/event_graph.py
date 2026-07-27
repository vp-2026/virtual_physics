#!/usr/bin/env python3
"""Build a small deterministic event graph from a saved rollout JSON.

This is a compact, dependency-light version of the backend event graph used in
the experiments. It converts simulator trajectories into object timelines and
then emits movement, final-state, and contact events with ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def object_center(spec: dict) -> Optional[List[float]]:
    if spec.get("type") == "Ball" and "position" in spec:
        return [float(spec["position"][0]), float(spec["position"][1])]
    points = spec.get("vertices") or spec.get("points")
    if isinstance(points, list) and points:
        return [
            sum(float(point[0]) for point in points) / len(points),
            sum(float(point[1]) for point in points) / len(points),
        ]
    return None


def label_object(name: str, spec: dict) -> str:
    color = str(spec.get("color") or spec.get("innerColor") or "").lower()
    typ = str(spec.get("type") or "object").lower()
    if color == "red" and typ == "ball":
        return "red ball"
    if color == "blue":
        return f"blue {typ}"
    if str(spec.get("innerColor") or "").lower() == "green":
        return "green container"
    if name.startswith("_BottomWall"):
        return "floor"
    if name.startswith("_TopWall"):
        return "ceiling"
    if name.startswith("_LeftWall"):
        return "left wall"
    if name.startswith("_RightWall"):
        return "right wall"
    return name


def radius_or_bbox(meta: dict, point: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if point is None:
        return None
    spec = meta["spec"]
    cx, cy = float(point[0]), float(point[1])
    if spec.get("type") == "Ball" or "radius" in spec:
        radius = float(spec.get("radius", meta.get("radius", 0.0)) or 0.0)
        return cx - radius, cy - radius, cx + radius, cy + radius
    points = spec.get("vertices") or spec.get("points") or []
    center = meta.get("start") or [0.0, 0.0]
    if not points:
        return None
    dx = cx - float(center[0])
    dy = cy - float(center[1])
    xs = [float(p[0]) + dx for p in points]
    ys = [float(p[1]) + dy for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def boxes_touch(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], margin: float = 4.0) -> bool:
    return not (
        a[2] < b[0] - margin
        or b[2] < a[0] - margin
        or a[3] < b[1] - margin
        or b[3] < a[1] - margin
    )


def timeline_point(meta: dict, index: int) -> Optional[List[float]]:
    timeline = meta.get("timeline") or []
    if timeline:
        return timeline[min(index, len(timeline) - 1)]
    return meta.get("end") or meta.get("start")


def build_timelines(world: dict, rollout: dict) -> Dict[str, dict]:
    placement = (rollout.get("placements") or [{}])[0]
    trajectory = placement.get("trajectory_data") or {}
    landing = placement.get("landing_positions") or {}
    coords = placement.get("placement_coords") or [0, 0]
    out: Dict[str, dict] = {}

    for name, spec in (world.get("world", {}).get("objects") or {}).items():
        start = object_center(spec)
        traj = trajectory.get(name) or []
        end = landing.get(name) if isinstance(landing.get(name), list) else start
        timeline = [[float(p[0]), float(p[1])] for p in traj if isinstance(p, list) and len(p) == 2]
        if start and not timeline:
            timeline = [[float(start[0]), float(start[1])]]
        if end and timeline and timeline[-1] != end:
            timeline.append([float(end[0]), float(end[1])])
        out[name] = {
            "name": name,
            "label": label_object(name, spec),
            "spec": spec,
            "start": start,
            "end": [float(end[0]), float(end[1])] if isinstance(end, list) and len(end) == 2 else start,
            "timeline": timeline,
        }

    tool_traj = trajectory.get("PLACED") or trajectory.get("placed_tool") or []
    tool_end = landing.get("PLACED") or landing.get("placed_tool") or coords
    out["placed_tool"] = {
        "name": "placed_tool",
        "label": "big orange ball",
        "spec": {"type": "Ball", "radius": 36, "color": "orange"},
        "start": [float(coords[0]), float(coords[1])],
        "end": [float(tool_end[0]), float(tool_end[1])],
        "timeline": [[float(p[0]), float(p[1])] for p in tool_traj if isinstance(p, list) and len(p) == 2],
    }
    return out


def movement_events(meta: dict, threshold: float = 50.0) -> List[dict]:
    start, end = meta.get("start"), meta.get("end")
    if not start or not end:
        return []
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    events = []
    for axis, delta in (("right", dx), ("left", -dx), ("up", dy), ("down", -dy)):
        if delta >= threshold:
            events.append(
                {
                    "type": "final_movement",
                    "object": meta["label"],
                    "axis": axis,
                    "threshold_px": threshold,
                    "delta_px": round(delta, 3),
                    "answer": "yes",
                }
            )
    return events


def contact_events(a: dict, b: dict) -> List[dict]:
    total = max(len(a.get("timeline") or []), len(b.get("timeline") or []), 1)
    frames = []
    for index in range(total):
        abox = radius_or_bbox(a, timeline_point(a, index))
        bbox = radius_or_bbox(b, timeline_point(b, index))
        if abox and bbox and boxes_touch(abox, bbox):
            frames.append(index)
    final_touch = False
    abox = radius_or_bbox(a, a.get("end"))
    bbox = radius_or_bbox(b, b.get("end"))
    if abox and bbox:
        final_touch = boxes_touch(abox, bbox)
    events = []
    if final_touch:
        events.append({"type": "final_contact", "object": a["label"], "reference": b["label"], "answer": "yes"})
    if len(frames) >= 30:
        events.append(
            {
                "type": "sustained_contact",
                "object": a["label"],
                "reference": b["label"],
                "min_seconds": 0.5,
                "contact_frames": len(frames),
                "answer": "yes",
            }
        )
    return events


def build_event_graph(world: dict, rollout: dict) -> dict:
    timelines = build_timelines(world, rollout)
    movable = [meta for meta in timelines.values() if meta.get("timeline")]
    events = []
    for meta in movable:
        events.extend(movement_events(meta))
    for i, a in enumerate(movable):
        for b in movable[i + 1 :]:
            events.extend(contact_events(a, b))
    return {
        "objects": [
            {
                "name": meta["name"],
                "label": meta["label"],
                "start": meta.get("start"),
                "end": meta.get("end"),
                "trajectory_frames": len(meta.get("timeline") or []),
            }
            for meta in timelines.values()
        ],
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    world = json.loads(Path(args.world).read_text())
    rollout = json.loads(Path(args.rollout).read_text())
    event_graph = build_event_graph(world, rollout)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(event_graph, indent=2))


if __name__ == "__main__":
    main()
