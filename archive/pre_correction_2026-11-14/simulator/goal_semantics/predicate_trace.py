#!/usr/bin/env python3
"""Structured rollout traces and causal summary helpers.

The simulator's native trace is intentionally lightweight: sampled poses,
rotations, collision intervals, and goal intervals. This module converts that
into a stable JSON schema and computes unit-testable latency/depth metrics.
The causal-depth routines are conservative approximations over observed
temporal contacts; they are designed for later statistical analyses rather
than formal counterfactual proof of necessity.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from predicates import PREDICATE_REGISTRY, PredicateSpec, get_predicate_spec


TOOL_ID = "PLACED"
DEFAULT_LINEAR_VELOCITY_EPS = 2.0
DEFAULT_ANGULAR_VELOCITY_EPS = 0.05
DEFAULT_SETTLE_WINDOW_S = 0.5
DEFAULT_MOVE_ONSET_PX = 5.0


@dataclass(frozen=True)
class TraceThresholds:
    linear_velocity_epsilon: float = DEFAULT_LINEAR_VELOCITY_EPS
    angular_velocity_epsilon: float = DEFAULT_ANGULAR_VELOCITY_EPS
    settle_window_s: float = DEFAULT_SETTLE_WINDOW_S
    move_onset_px: float = DEFAULT_MOVE_ONSET_PX
    goal_hold_s: float = 2.0


@dataclass(frozen=True)
class TraceObjectMeta:
    id: str
    role: str
    label: str
    is_dynamic: bool = True


def _pose(path: dict[str, list[list[float]]], rot: dict[str, list[float]], obj_id: str, idx: int) -> dict[str, float] | None:
    pts = path.get(obj_id)
    if not pts:
        return None
    idx = max(0, min(idx, len(pts) - 1))
    angle = (rot.get(obj_id) or [0.0])[idx if idx < len(rot.get(obj_id, [])) else -1]
    return {"x": float(pts[idx][0]), "y": float(pts[idx][1]), "angle_rad": float(angle)}


def _displacement(path: dict[str, list[list[float]]], obj_id: str) -> float | None:
    pts = path.get(obj_id)
    if not pts:
        return None
    p0 = np.array(pts[0], dtype=float)
    p1 = np.array(pts[-1], dtype=float)
    return float(np.linalg.norm(p1 - p0))


def _velocity_samples(path: dict[str, list[list[float]]], obj_id: str, step_s: float) -> list[dict[str, float]]:
    pts = path.get(obj_id)
    if not pts or step_s <= 0:
        return []
    out: list[dict[str, float]] = []
    prev = np.array(pts[0], dtype=float)
    out.append({"time_s": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0})
    for idx, point in enumerate(pts[1:], start=1):
        cur = np.array(point, dtype=float)
        delta = (cur - prev) / step_s
        vx = float(delta[0])
        vy = float(delta[1])
        out.append(
            {
                "time_s": round(idx * step_s, 6),
                "vx": vx,
                "vy": vy,
                "speed": float(math.hypot(vx, vy)),
            }
        )
        prev = cur
    return out


def _angular_velocity_samples(rot: dict[str, list[float]], obj_id: str, step_s: float) -> list[dict[str, float]]:
    vals = rot.get(obj_id) or []
    if not vals or step_s <= 0:
        return []
    out: list[dict[str, float]] = []
    out.append({"time_s": 0.0, "omega_rad_s": 0.0})
    prev = float(vals[0])
    for idx, angle in enumerate(vals[1:], start=1):
        cur = float(angle)
        out.append(
            {
                "time_s": round(idx * step_s, 6),
                "omega_rad_s": float((cur - prev) / step_s),
            }
        )
        prev = cur
    return out


def first_move_time_from_path(
    path: dict[str, list[list[float]]],
    obj_id: str,
    step_s: float,
    move_onset_px: float = DEFAULT_MOVE_ONSET_PX,
) -> float | None:
    pts = path.get(obj_id)
    if not pts:
        return None
    p0 = np.array(pts[0], dtype=float)
    for idx, point in enumerate(pts):
        if float(np.linalg.norm(np.array(point, dtype=float) - p0)) >= move_onset_px:
            return round(idx * step_s, 6)
    return None


def first_rotation_time_from_rot(
    rot: dict[str, list[float]],
    obj_id: str,
    step_s: float,
    min_rotation_rad: float = math.radians(5.0),
) -> float | None:
    vals = rot.get(obj_id) or []
    if not vals:
        return None
    start = float(vals[0])
    for idx, angle in enumerate(vals):
        if abs(float(angle) - start) >= min_rotation_rad:
            return round(idx * step_s, 6)
    return None


def compute_settle_time(trace: dict[str, Any]) -> float:
    thresholds = trace.get("metadata", {}).get("thresholds", {})
    linear_eps = float(thresholds.get("linear_velocity_epsilon", DEFAULT_LINEAR_VELOCITY_EPS))
    angular_eps = float(thresholds.get("angular_velocity_epsilon", DEFAULT_ANGULAR_VELOCITY_EPS))
    settle_window = float(thresholds.get("settle_window_s", DEFAULT_SETTLE_WINDOW_S))
    step = float(trace.get("metadata", {}).get("step_s", 0.05))
    states = trace.get("_state_samples", {})
    path = states.get("path", {})
    rot = states.get("rot", {})
    if not path:
        return float(trace.get("settle_time_s") or trace.get("duration_s") or 0.0)
    n = max(len(v) for v in path.values())
    needed = max(1, int(math.ceil(settle_window / step)))
    still_count = 0
    for idx in range(1, n):
        moving = False
        for obj_id, pts in path.items():
            if idx >= len(pts):
                continue
            dx = float(pts[idx][0] - pts[idx - 1][0])
            dy = float(pts[idx][1] - pts[idx - 1][1])
            linear_v = math.hypot(dx, dy) / step
            rots = rot.get(obj_id) or []
            angular_v = abs(float(rots[idx] - rots[idx - 1])) / step if idx < len(rots) else 0.0
            if linear_v > linear_eps or angular_v > angular_eps:
                moving = True
                break
        still_count = 0 if moving else still_count + 1
        if still_count >= needed:
            return round((idx - needed + 1) * step, 6)
    return float(trace.get("duration_s") or (n - 1) * step)


def _roles(trace: dict[str, Any], role: str | None) -> set[str]:
    if role is None:
        return set()
    if role == "tool":
        return {"tool"} if "tool" in trace.get("objects", {}) else {TOOL_ID}
    out: set[str] = set()
    for key, obj in trace.get("objects", {}).items():
        if obj.get("role") == role or obj.get("label") == role:
            out.add(key)
        if role == "blue" and str(obj.get("role", "")).startswith("blue"):
            out.add(key)
        if role == "blue_rectangle" and "blue rectangle" in str(obj.get("label", "")):
            out.add(key)
        if role == "blue_square" and "blue square" in str(obj.get("label", "")):
            out.add(key)
        if role == "movable" and obj.get("is_dynamic"):
            out.add(key)
    return out


def _collisions_until(trace: dict[str, Any], terminal_time: float | None) -> list[dict[str, Any]]:
    contacts = trace.get("contact_pairs", [])
    if terminal_time is None:
        return list(contacts)
    return [c for c in contacts if float(c.get("time_s", 0.0)) <= terminal_time + 1e-9]


def _pair_matches(contact: dict[str, Any], a_set: set[str], b_set: set[str]) -> bool:
    a = str(contact.get("a"))
    b = str(contact.get("b"))
    return (a in a_set and b in b_set) or (a in b_set and b in a_set)


def _first_collision_time(trace: dict[str, Any], a_set: set[str], b_set: set[str]) -> float | None:
    for contact in sorted(trace.get("contact_pairs", []), key=lambda c: float(c.get("time_s", 0.0))):
        if _pair_matches(contact, a_set, b_set):
            return float(contact.get("time_s", 0.0))
    return None


def _direct_tool_contact(trace: dict[str, Any], objects: set[str], before_time: float | None = None) -> bool:
    tool_ids = _roles(trace, "tool")
    return any(_pair_matches(c, tool_ids, objects) for c in _collisions_until(trace, before_time))


def _terminal_objects_and_time(
    trace: dict[str, Any],
    spec: PredicateSpec,
    metadata: dict[str, Any] | None = None,
) -> tuple[set[str], float | None]:
    metadata = metadata or {}
    if spec.event_type == "collision":
        a_set = _roles(trace, spec.primary_role)
        b_set = _roles(trace, spec.secondary_role)
        return a_set | b_set, _first_collision_time(trace, a_set, b_set)
    terminal = _roles(trace, metadata.get("terminal_role") or spec.terminal_role)
    if metadata.get("object"):
        terminal = {str(metadata["object"])}
    if spec.event_type == "goal_entry":
        terminal = _roles(trace, "goal") or terminal
        hold_s = float(metadata.get("goal_hold_s", trace.get("metadata", {}).get("thresholds", {}).get("goal_hold_s", 2.0)))
        for interval in trace.get("goal_intervals", []):
            enter = float(interval.get("enter_time_s", 0.0))
            exit_val = interval.get("exit_time_s")
            exit_time = float(exit_val) if exit_val is not None else float(trace.get("duration_s", trace.get("settle_time_s", enter)))
            if exit_time - enter >= hold_s:
                return terminal, enter
        return terminal, None
    if spec.event_type == "move_onset":
        times = [
            obj.get("first_move_time_s")
            for key, obj in trace.get("objects", {}).items()
            if key in terminal and obj.get("first_move_time_s") is not None
        ]
        return terminal, min((float(t) for t in times), default=None)
    if spec.event_type == "settled":
        return terminal, float(trace.get("settle_time_s")) if trace.get("settle_time_s") is not None else None
    return terminal, None


def compute_event_latency(
    trace: dict[str, Any],
    predicate_name: str,
    predicate_metadata: dict[str, Any] | None = None,
) -> float | None:
    spec = get_predicate_spec(predicate_name)
    if spec is None:
        return None
    predicate_metadata = predicate_metadata or {}
    if spec.requires_no_tool_baseline and not predicate_metadata.get("baseline_available", False):
        return None
    terminal_objects, event_time = _terminal_objects_and_time(trace, spec, predicate_metadata)
    if event_time is None:
        return None
    if spec.direct_tool_contact_invalidates and _direct_tool_contact(trace, terminal_objects, event_time):
        return None
    return round(event_time - float(trace.get("tool_release_time_s", 0.0)), 6)


def _raw_chain(trace: dict[str, Any], terminal_objects: set[str], terminal_time: float | None) -> list[dict[str, Any]] | None:
    tool_ids = _roles(trace, "tool")
    if not tool_ids:
        return [] if not trace.get("has_tool") else None
    reached = set(tool_ids)
    chain: list[dict[str, Any]] = []
    for contact in sorted(_collisions_until(trace, terminal_time), key=lambda c: float(c.get("time_s", 0.0))):
        a = str(contact.get("a"))
        b = str(contact.get("b"))
        if a in reached or b in reached:
            chain.append(contact)
            reached.add(a)
            reached.add(b)
            if reached & terminal_objects:
                return chain
    return None


def compute_raw_causal_depth(
    trace: dict[str, Any],
    predicate_name: str,
    predicate_metadata: dict[str, Any] | None = None,
) -> int | None:
    spec = get_predicate_spec(predicate_name)
    if spec is None:
        return None
    predicate_metadata = predicate_metadata or {}
    if spec.requires_no_tool_baseline and not predicate_metadata.get("baseline_available", False):
        return None
    terminal_objects, terminal_time = _terminal_objects_and_time(trace, spec, predicate_metadata)
    if terminal_time is None and spec.event_type != "settled":
        return None
    if spec.direct_tool_contact_invalidates and _direct_tool_contact(trace, terminal_objects, terminal_time):
        return None
    chain = _raw_chain(trace, terminal_objects, terminal_time)
    return len(chain) if chain is not None else None


def compute_unique_causal_depth(
    trace: dict[str, Any],
    predicate_name: str,
    predicate_metadata: dict[str, Any] | None = None,
) -> int | None:
    spec = get_predicate_spec(predicate_name)
    if spec is None:
        return None
    predicate_metadata = predicate_metadata or {}
    if spec.requires_no_tool_baseline and not predicate_metadata.get("baseline_available", False):
        return None
    terminal_objects, terminal_time = _terminal_objects_and_time(trace, spec, predicate_metadata)
    if terminal_time is None and spec.event_type != "settled":
        return None
    chain = _raw_chain(trace, terminal_objects, terminal_time)
    if chain is None:
        return None
    return len({tuple(sorted((str(c.get("a")), str(c.get("b"))))) for c in chain})


def compute_minimal_causal_depth(
    trace: dict[str, Any],
    predicate_name: str,
    predicate_metadata: dict[str, Any] | None = None,
) -> int | None:
    spec = get_predicate_spec(predicate_name)
    if spec is None:
        return None
    predicate_metadata = predicate_metadata or {}
    if spec.requires_no_tool_baseline and not predicate_metadata.get("baseline_available", False):
        return None
    terminal_objects, terminal_time = _terminal_objects_and_time(trace, spec, predicate_metadata)
    if terminal_time is None and spec.event_type != "settled":
        return None
    if spec.direct_tool_contact_invalidates and _direct_tool_contact(trace, terminal_objects, terminal_time):
        return None
    tool_ids = _roles(trace, "tool")
    if not tool_ids:
        return 0 if not trace.get("has_tool") else None
    release_time = float(trace.get("tool_release_time_s", 0.0))
    # A direct tool contact should count as one interaction, not zero.
    for contact in _collisions_until(trace, terminal_time):
        t = float(contact.get("time_s", 0.0))
        if t < release_time:
            continue
        a = str(contact.get("a"))
        b = str(contact.get("b"))
        if ((a in tool_ids and b in terminal_objects) or (b in tool_ids and a in terminal_objects)):
            return 1
    contacts = sorted(_collisions_until(trace, terminal_time), key=lambda c: float(c.get("time_s", 0.0)))
    queue = deque((tool, 0, release_time) for tool in tool_ids)
    best: dict[str, int] = {tool: 0 for tool in tool_ids}
    while queue:
        obj, depth, earliest = queue.popleft()
        if obj in terminal_objects:
            return depth
        for contact in contacts:
            t = float(contact.get("time_s", 0.0))
            if t < earliest:
                continue
            a = str(contact.get("a"))
            b = str(contact.get("b"))
            nxt = b if a == obj else a if b == obj else None
            if nxt is None:
                continue
            if best.get(nxt, 10**9) <= depth + 1:
                continue
            best[nxt] = depth + 1
            queue.append((nxt, depth + 1, t))
    return None


def build_structured_trace(
    *,
    puzzle_id: str,
    rollout_id: str,
    has_tool: bool,
    simple_trace: dict[str, Any],
    objects: Iterable[TraceObjectMeta],
    step_s: float,
    thresholds: TraceThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or TraceThresholds()
    path = simple_trace.get("path", {})
    rot = simple_trace.get("rot", {})
    object_map: dict[str, Any] = {}
    pose_samples: dict[str, list[dict[str, float]]] = {}
    kinematics: dict[str, dict[str, Any]] = {}
    for meta in objects:
        first_move = first_move_time_from_path(path, meta.id, step_s, thresholds.move_onset_px)
        first_rotation = first_rotation_time_from_rot(rot, meta.id, step_s)
        vel_samples = _velocity_samples(path, meta.id, step_s)
        omega_samples = _angular_velocity_samples(rot, meta.id, step_s)
        max_speed = max((sample["speed"] for sample in vel_samples), default=0.0)
        max_omega = max((abs(sample["omega_rad_s"]) for sample in omega_samples), default=0.0)
        object_map[meta.id] = {
            **asdict(meta),
            "initial_pose": _pose(path, rot, meta.id, 0),
            "final_pose": _pose(path, rot, meta.id, -1),
            "first_move_time_s": first_move,
            "first_rotation_time_s": first_rotation,
            "total_displacement_px": _displacement(path, meta.id),
            "max_speed_px_s": float(max_speed),
            "max_angular_speed_rad_s": float(max_omega),
        }
        pts = path.get(meta.id) or []
        rots = rot.get(meta.id) or []
        pose_samples[meta.id] = [
            {
                "time_s": round(idx * step_s, 6),
                "x": float(point[0]),
                "y": float(point[1]),
                "angle_rad": float(rots[idx] if idx < len(rots) else rots[-1] if rots else 0.0),
            }
            for idx, point in enumerate(pts)
        ]
        kinematics[meta.id] = {
            "velocity_samples": vel_samples,
            "angular_velocity_samples": omega_samples,
        }
    collisions = sorted(simple_trace.get("collisions", []), key=lambda ev: float(ev[2] or 0.0))
    contact_pairs = [
        {"time_s": float(ev[2] or 0.0), "a": str(ev[0]), "b": str(ev[1])}
        for ev in collisions
    ]
    goal_intervals = [
        {"object": "red_target", "enter_time_s": float(start), "exit_time_s": float(end) if end is not None else None}
        for start, end in simple_trace.get("in_goal_intervals", [])
    ]
    events: list[dict[str, Any]] = [{"time_s": 0.0, "type": "placement", "actor": TOOL_ID if has_tool else None, "details": {}}]
    events.extend(
        {"time_s": c["time_s"], "type": "contact", "objects": [c["a"], c["b"]], "details": {}}
        for c in contact_pairs
    )
    for obj_id, obj in object_map.items():
        if obj.get("first_move_time_s") is not None:
            events.append({"time_s": obj["first_move_time_s"], "type": "move_onset", "object": obj_id, "details": {}})
        if obj.get("first_rotation_time_s") is not None:
            events.append({"time_s": obj["first_rotation_time_s"], "type": "rotation_start", "object": obj_id, "details": {}})
    for interval in goal_intervals:
        events.append({"time_s": interval["enter_time_s"], "type": "goal_entry", "object": interval["object"], "details": {}})
        if interval["exit_time_s"] is not None:
            events.append({"time_s": interval["exit_time_s"], "type": "goal_exit", "object": interval["object"], "details": {}})
        hold_s = float(thresholds.goal_hold_s)
        exit_time = float(interval["exit_time_s"]) if interval["exit_time_s"] is not None else float(simple_trace.get("duration", 0.0))
        if exit_time - float(interval["enter_time_s"]) >= hold_s:
            events.append({"time_s": round(float(interval["enter_time_s"]) + hold_s, 6), "type": "goal_hold_success", "object": interval["object"], "details": {}})
    trace = {
        "puzzle_id": puzzle_id,
        "rollout_id": rollout_id,
        "has_tool": bool(has_tool),
        "tool_release_time_s": 0.0,
        "settle_time_s": None,
        "settled_within_horizon": None,
        "duration_s": float(simple_trace.get("duration", 0.0)),
        "metadata": {"step_s": step_s, "thresholds": asdict(thresholds)},
        "events": [],
        "objects": object_map,
        "contact_pairs": contact_pairs,
        "goal_intervals": goal_intervals,
        "final_contacts": simple_trace.get("final_contacts", {}),
        "pose_samples": pose_samples,
        "kinematics": kinematics,
        "_state_samples": {"path": path, "rot": rot},
    }
    settle_time = compute_settle_time(trace)
    trace["settle_time_s"] = settle_time
    trace["settled_within_horizon"] = bool(
        trace.get("duration_s") is not None and float(settle_time) + 1e-9 < float(trace.get("duration_s"))
    )
    events.append({"time_s": settle_time, "type": "settled", "details": {}})
    trace["events"] = sorted(events, key=lambda e: (float(e.get("time_s", 0.0)), str(e.get("type", ""))))
    trace["predicate_metrics"] = {
        pred: {
            "event_latency_s": compute_event_latency(trace, pred, {}),
            "raw_causal_depth": compute_raw_causal_depth(trace, pred, {}),
            "unique_causal_depth": compute_unique_causal_depth(trace, pred, {}),
            "minimal_causal_depth": compute_minimal_causal_depth(trace, pred, {}),
        }
        for pred in PREDICATE_REGISTRY.keys()
    }
    return trace


def write_trace(trace: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in trace.items() if not k.startswith("_")}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def summarize_trace(trace: dict[str, Any], predicates: Iterable[str] = PREDICATE_REGISTRY.keys()) -> dict[str, Any]:
    row: dict[str, Any] = {
        "puzzle_id": trace.get("puzzle_id"),
        "rollout_id": trace.get("rollout_id"),
        "has_tool": trace.get("has_tool"),
        "success": any(i.get("exit_time_s") is None or float(i.get("exit_time_s")) > float(i.get("enter_time_s", 0.0)) for i in trace.get("goal_intervals", [])),
        "settle_time_s": trace.get("settle_time_s"),
        "settled_within_horizon": trace.get("settled_within_horizon"),
    }
    for pred in predicates:
        row[f"event_latency_{pred}"] = compute_event_latency(trace, pred, {})
        row[f"raw_depth_{pred}"] = compute_raw_causal_depth(trace, pred, {})
        row[f"unique_depth_{pred}"] = compute_unique_causal_depth(trace, pred, {})
        row[f"minimal_depth_{pred}"] = compute_minimal_causal_depth(trace, pred, {})
    row["raw_depth_target_in_goal"] = row.get("raw_depth_TARGET_IN_GOAL_AFTER_SETTLE")
    row["unique_depth_target_in_goal"] = row.get("unique_depth_TARGET_IN_GOAL_AFTER_SETTLE")
    row["minimal_depth_target_in_goal"] = row.get("minimal_depth_TARGET_IN_GOAL_AFTER_SETTLE")
    return row


def write_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
