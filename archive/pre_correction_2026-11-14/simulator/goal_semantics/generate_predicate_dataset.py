#!/usr/bin/env python3
"""Generate yes/no predicate labels for the newer one-ball virtual-tool worlds.

Each candidate placement is simulated once per noisy rollout and all predicates
are evaluated from that same trajectory/collision trace. A no-tool baseline is
also simulated for counterfactual and indirect-transfer predicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SIM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SIM_DIR.parents[1]
RUNTIME_ROOT = SIM_DIR.parent / "runtime"
SCRIPT_ROOT = RUNTIME_ROOT
ENV_SET_ROOT = PROJECT_ROOT / "132_base_environments" / "cells"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import pymunk  # noqa: E402
import pygame as pg  # noqa: E402
from make_trial_onetool_3 import _target_fully_inside_goal, pick_container_or_none, point_in_poly  # noqa: E402
from pyGameWorld.helpers import filterCollisionEvents, word2Color  # noqa: E402
from pyGameWorld.noisyWorld import noisifyWorld  # noqa: E402
from pyGameWorld.viewer import drawWorld  # noqa: E402
from pyGameWorld.world import loadFromDict  # noqa: E402
from run_rollout import simulate_rollout  # noqa: E402
from predicate_trace import (  # noqa: E402
    TraceObjectMeta,
    TraceThresholds,
    build_structured_trace,
    summarize_trace,
    write_summary_csv,
    write_trace,
)

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None

WALL_NAMES = {"_LeftWall", "_BottomWall", "_RightWall", "_TopWall"}
TOOL_NAME = "PLACED"


@dataclass
class Thresholds:
    yes_px: float = 30.0
    no_px: float = 5.0
    contact_min_s: float = 0.10
    goal_hold_s: float = 2.0
    first_move_px: float = 5.0
    rotation_rad: float = math.radians(15.0)
    before_event_min_gap_s: float = 0.35
    goal_touch_min_fraction: float = 0.20
    initial_contact_min_gap_px: float = 10.0
    event_time_std_max_s: float = 0.20
    summary_contact_min_gap_px: float = 2.0
    visible_contact_gap_px: float = 0.1
    settle_linear_velocity_eps: float = 1.0
    settle_angular_velocity_eps: float = 0.1
    settle_window_s: float = 0.5
    temporal_question_min_gap_s: float = 0.5


@dataclass
class ObjInfo:
    name: str
    role: str
    label: str
    density: float
    color: str
    typ: str
    is_movable: bool


def regular_ngon(radius: float, n: int = 32) -> List[List[List[float]]]:
    return [[
        [float(radius * math.cos(2.0 * math.pi * (1.0 - i / n))),
         float(radius * math.sin(2.0 * math.pi * (1.0 - i / n)))]
        for i in range(n)
    ]]


def load_world_file(path: Path) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"{path} is empty")
            data = json.loads(text)
            break
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise RuntimeError(f"Failed to load world file {path}: {exc}") from exc
            time.sleep(0.5 * (attempt + 1))
    else:
        assert last_error is not None
        raise RuntimeError(f"Failed to load world file {path}: {last_error}") from last_error
    if "world" not in data:
        raise ValueError(f"{path} does not contain a top-level world object")
    return data


def discover_worlds(root: Path, include_sets: Sequence[str]) -> List[Path]:
    include = {s for s in include_sets if s}
    paths: List[Path] = []
    for p in sorted(root.glob("*/*.json")):
        if p.parent.name == "Warmup":
            continue
        if include and p.parent.name not in include:
            continue
        paths.append(p)
    return paths


def filter_worlds(paths: List[Path], include_ids: Sequence[str]) -> List[Path]:
    ids = {str(v) for v in include_ids if str(v)}
    if not ids:
        return paths
    return [p for p in paths if p.stem in ids]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _slug(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def object_center(spec: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for key in ("position", "pos", "center"):
        val = spec.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return float(val[0]), float(val[1])
    pts: List[Tuple[float, float]] = []
    for key in ("points", "vertices"):
        for p in spec.get(key, []) or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
    for poly in spec.get("polys", []) or []:
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def shape_label(name: str, spec: Dict[str, Any]) -> str:
    typ = str(spec.get("type", "")).lower()
    low = name.lower()
    if typ == "container":
        return "blue U-shape container"
    if typ == "ball" or "radius" in spec:
        return "blue ball"
    pts = spec.get("vertices") or spec.get("points")
    if spec.get("polys"):
        pts = [pt for poly in spec.get("polys", []) for pt in poly]
    if isinstance(pts, list) and len(pts) >= 3:
        xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [float(p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        if xs and ys:
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect >= 3.0:
                return "blue plank"
            if abs(w - h) <= 0.18 * max(w, h):
                return "blue square"
            if len(pts) == 4:
                if aspect >= 2.0:
                    return "blue rectangle"
                return "blue trapezoid"
            if aspect >= 2.0:
                return "blue rectangle"
    if "platform" in low or "plank" in low:
        return "blue plank"
    if "trap" in low or "trape" in low:
        return "blue trapezoid"
    if "square" in low:
        return "blue square"
    if "rect" in low:
        return "blue rectangle"
    return "blue object"


def black_shape_label(name: str, spec: Dict[str, Any]) -> str:
    typ = str(spec.get("type", "")).lower()
    low = name.lower()
    if typ == "ball" or "radius" in spec:
        return "black ball"
    pts = spec.get("vertices") or spec.get("points")
    if spec.get("polys"):
        pts = [pt for poly in spec.get("polys", []) for pt in poly]
    if isinstance(pts, list) and len(pts) >= 3:
        xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [float(p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        if xs and ys:
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            aspect = max(w, h) / max(1.0, min(w, h))
            if len(pts) == 3:
                return "black triangle"
            if aspect >= 3.0:
                return "black rectangle"
            if abs(w - h) <= 0.18 * max(w, h):
                return "black square"
            if len(pts) == 4:
                if aspect >= 1.4:
                    return "black rectangle"
                return "black trapezoid"
            if aspect >= 1.4:
                return "black rectangle"
    if "trap" in low or "trape" in low:
        return "black trapezoid"
    if "tri" in low:
        return "black triangle"
    if "pillar" in low or "post" in low or "peg" in low:
        return "black pillar"
    if "rect" in low:
        return "black rectangle"
    return "black object"


def describe_objects(
    world_dict: Dict[str, Any],
) -> Tuple[ObjInfo, Optional[ObjInfo], List[ObjInfo], List[ObjInfo], List[ObjInfo]]:
    world = world_dict["world"]
    objects = world.get("objects", {}) or {}
    defaults = world.get("defaults", {}) or {}
    default_density = _float(defaults.get("density", 1.0), 1.0)
    gcond = world.get("gcond", {}) or {}
    target_name = str(gcond.get("obj") or "Ball")
    goal_name = str(gcond.get("goal") or "Goal")

    infos: List[ObjInfo] = []
    used_labels: Dict[str, int] = {}
    for name, spec in objects.items():
        color = str(spec.get("color") or spec.get("innerColor") or "").lower()
        density = _float(spec.get("density", default_density), default_density)
        typ = str(spec.get("type", ""))
        role = "other"
        label = name
        if name == target_name or color == "red":
            role = "target"
            label = "red target"
        elif name == goal_name or (typ == "Container" and str(spec.get("innerColor", "")).lower() == "green"):
            role = "goal"
            label = "green goal container"
        elif color == "blue":
            role = "blue"
            label = shape_label(name, spec)
        if role == "blue":
            used_labels[label] = used_labels.get(label, 0) + 1
            if used_labels[label] > 1:
                label = f"{label} {used_labels[label]}"
        infos.append(ObjInfo(name, role, label, density, color, typ, density > 0))

    target = next((i for i in infos if i.role == "target"), None)
    if target is None:
        raise ValueError("Could not identify the red target object")
    goal = next((i for i in infos if i.name == goal_name), None) or next((i for i in infos if i.role == "goal"), None)
    blues = [i for i in infos if i.role == "blue" and i.is_movable]
    movable = [i for i in infos if i.is_movable and i.name not in WALL_NAMES]
    static_black = [
        i
        for i in infos
        if i.name not in WALL_NAMES and not i.is_movable and i.color == "black" and i.role not in {"target", "goal", "blue"}
    ]
    return target, goal, blues, movable, static_black


def place_ball_tool(world_obj: Any, pos_xy: Tuple[float, float], radius: float, gravity_mode: str, color: str = "orange") -> bool:
    polys = regular_ngon(radius, 32)
    for poly in polys:
        if world_obj.checkCollision(pos_xy, poly):
            return False
    translated = [[(vx + pos_xy[0], vy + pos_xy[1]) for vx, vy in poly] for poly in polys]
    world_obj.addPlacedCompound(TOOL_NAME, translated, word2Color(color), density=1.0, friction=0.5, elasticity=0.5)
    if gravity_mode == "upward":
        placed = world_obj.objects.get(TOOL_NAME)
        gx, gy = world_obj._cpSpace.gravity
        gmag = max(abs(float(gx)), abs(float(gy)))

        def _velocity_func(cp_body, gravity, damping, dt):
            pymunk.Body.update_velocity(cp_body, (0.0, gmag), damping, dt)

        placed._cpBody.velocity_func = _velocity_func
    return True


def add_noise(world_obj: Any, rng: np.random.Generator, noise_scale: float) -> Any:
    if noise_scale <= 0:
        return world_obj.copy()
    np_state = np.random.get_state()
    py_state = random.getstate()
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)
    random.seed(seed)
    try:
        return noisifyWorld(
            world_obj,
            # Keep the semantic rollout's initial scene geometry identical to the
            # frontend scene. We only perturb dynamics, not initial object layout.
            noise_position_static=0.0,
            noise_position_moving=0.0,
            noise_collision_direction=1.0 * noise_scale,
            noise_collision_elasticity=1.0 * noise_scale,
            noise_gravity=0.5 * noise_scale,
            noise_object_friction=0.5 * noise_scale,
            noise_object_density=0.5 * noise_scale,
            noise_object_elasticity=0.5 * noise_scale,
        )
    except Exception:
        w = world_obj.copy()
        try:
            gx, gy = w._cpSpace.gravity
            scale = max(0.0, 1.0 + float(rng.normal(0.0, 0.5 * noise_scale)))
            w._cpSpace.gravity = (float(gx) * scale, float(gy) * scale)
        except Exception:
            pass
        for name, obj in list(getattr(w, "objects", {}).items()):
            if name in WALL_NAMES or name == TOOL_NAME:
                continue
            try:
                if obj.isStatic():
                    continue
                dx = float(rng.normal(0.0, 25.0 * noise_scale))
                dy = float(rng.normal(0.0, 25.0 * noise_scale))
                obj.setPos((float(obj.position[0]) + dx, float(obj.position[1]) + dy))
            except Exception:
                pass
        return w
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def _pos(obj: Any) -> List[float]:
    p = obj.position
    return [float(p[0]), float(p[1])]


DISK_SAMPLE_OFFSETS: List[Tuple[float, float]] = [
    (0.0, 0.0),
    *[
        (r * math.cos(theta), r * math.sin(theta))
        for r in (0.35, 0.6, 0.82, 0.95)
        for theta in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    ],
]


def objects_in_contact(world_obj: Any, a_name: str, b_name: str) -> bool:
    a = world_obj.objects.get(a_name)
    b = world_obj.objects.get(b_name)
    if a is None or b is None:
        return False
    try:
        return bool(a.checkContact(b))
    except Exception:
        return False


def any_pair_in_contact(world_obj: Any, a_name: str, candidates: Iterable[str]) -> bool:
    return any(objects_in_contact(world_obj, a_name, other) for other in candidates)


def any_initial_contact_for_any(world_obj: Any, names: Iterable[str], candidate_names: Iterable[str]) -> bool:
    candidate_set = set(candidate_names)
    for name in names:
        for other in candidate_set:
            if other == name:
                continue
            if objects_in_contact(world_obj, name, other):
                return True
    return False


def object_bbox(world_obj: Any, world_dict: Dict[str, Any], name: str) -> Optional[Tuple[float, float, float, float]]:
    obj = world_obj.objects.get(name)
    if obj is None:
        return None
    pts: List[Tuple[float, float]] = []
    try:
        if hasattr(obj, "radius"):
            pos = obj.getPos()
            if hasattr(pos, "x") and hasattr(pos, "y"):
                cx, cy = float(pos.x), float(pos.y)
            elif hasattr(pos, "tolist"):
                cx, cy = [float(value) for value in pos.tolist()[:2]]
            else:
                cx, cy = float(pos[0]), float(pos[1])
            r = float(obj.radius)
            return (cx - r, cy - r, cx + r, cy + r)
    except Exception:
        pass
    try:
        if hasattr(obj, "getVertices"):
            verts = obj.getVertices()
            pts = [
                (float(v.x), float(v.y)) if hasattr(v, "x") else (float(v[0]), float(v[1]))
                for v in verts
            ]
    except Exception:
        pts = []
    if not pts:
        spec = (world_dict.get("world", {}).get("objects", {}) or {}).get(name, {}) or {}
        pts = []
        for key in ("points", "vertices"):
            for p in spec.get(key, []) or []:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
        for poly in spec.get("polys", []) or []:
            for p in poly:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
    if not pts:
        return None
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def object_boundary_points(world_obj: Any, world_dict: Dict[str, Any], name: str, circle_samples: int = 32) -> List[Tuple[float, float]]:
    obj = world_obj.objects.get(name)
    if obj is None:
        return []
    try:
        if hasattr(obj, "radius"):
            pos = obj.getPos()
            if hasattr(pos, "x") and hasattr(pos, "y"):
                cx, cy = float(pos.x), float(pos.y)
            elif hasattr(pos, "tolist"):
                cx, cy = [float(value) for value in pos.tolist()[:2]]
            else:
                cx, cy = float(pos[0]), float(pos[1])
            r = float(obj.radius)
            return [
                (cx + r * math.cos(theta), cy + r * math.sin(theta))
                for theta in np.linspace(0.0, 2.0 * math.pi, circle_samples, endpoint=False)
            ]
    except Exception:
        pass
    try:
        if hasattr(obj, "getVertices"):
            verts = obj.getVertices()
            return [
                (float(v.x), float(v.y)) if hasattr(v, "x") else (float(v[0]), float(v[1]))
                for v in verts
            ]
    except Exception:
        pass
    spec = (world_dict.get("world", {}).get("objects", {}) or {}).get(name, {}) or {}
    out: List[Tuple[float, float]] = []
    for key in ("points", "vertices"):
        for p in spec.get(key, []) or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
    for poly in spec.get("polys", []) or []:
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
    return out


def min_object_gap_px(world_obj: Any, world_dict: Dict[str, Any], a_name: str, b_name: str) -> Optional[float]:
    if objects_in_contact(world_obj, a_name, b_name):
        return 0.0
    a = world_obj.objects.get(a_name)
    b = world_obj.objects.get(b_name)
    if a is None or b is None:
        return None
    pts_a = object_boundary_points(world_obj, world_dict, a_name)
    pts_b = object_boundary_points(world_obj, world_dict, b_name)
    best = float("inf")
    for px, py in pts_a:
        try:
            best = min(best, max(float(b.distanceFromPoint([px, py])), 0.0))
        except Exception:
            pass
    for px, py in pts_b:
        try:
            best = min(best, max(float(a.distanceFromPoint([px, py])), 0.0))
        except Exception:
            pass
    return best if math.isfinite(best) else None


def vertical_gap_if_aligned_px(world_obj: Any, world_dict: Dict[str, Any], a_name: str, b_name: str) -> Optional[float]:
    bbox_a = object_bbox(world_obj, world_dict, a_name)
    bbox_b = object_bbox(world_obj, world_dict, b_name)
    if bbox_a is None or bbox_b is None:
        return None
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    horizontal_overlap = min(ax1, bx1) - max(ax0, bx0)
    if horizontal_overlap <= 0:
        return None
    if ay1 < by0:
        return by0 - ay1
    if by1 < ay0:
        return ay0 - by1
    return 0.0


def pair_too_close_for_contact(
    world_obj: Any,
    world_dict: Dict[str, Any],
    a_name: str,
    b_name: str,
    thr: Thresholds,
    *,
    require_vertical_for_tool: bool = False,
) -> bool:
    gap = min_object_gap_px(world_obj, world_dict, a_name, b_name)
    if gap is not None and gap < thr.initial_contact_min_gap_px:
        return True
    if require_vertical_for_tool:
        vgap = vertical_gap_if_aligned_px(world_obj, world_dict, a_name, b_name)
        if vgap is not None and vgap < thr.initial_contact_min_gap_px:
            return True
    return False


def any_pair_too_close_for_contact(
    world_obj: Any,
    world_dict: Dict[str, Any],
    a_name: str,
    candidates: Iterable[str],
    thr: Thresholds,
    *,
    require_vertical_for_tool: bool = False,
) -> bool:
    return any(
        pair_too_close_for_contact(
            world_obj,
            world_dict,
            a_name,
            other,
            thr,
            require_vertical_for_tool=require_vertical_for_tool,
        )
        for other in candidates
    )


def final_contact_names(world_obj: Any, names: Iterable[str]) -> Dict[str, List[str]]:
    tracked = [name for name in names if name in world_obj.objects]
    out: Dict[str, List[str]] = {}
    for name in tracked:
        others: List[str] = []
        for other in tracked:
            if other == name:
                continue
            if objects_in_contact(world_obj, name, other):
                others.append(other)
        out[name] = sorted(others)
    return out


def initial_near_contact_pairs(
    world_obj: Any,
    world_dict: Dict[str, Any],
    thr: Thresholds,
    *,
    include_walls: bool = False,
) -> set[Tuple[str, str]]:
    names = [
        name for name in world_obj.objects.keys()
        if include_walls or name not in WALL_NAMES
    ]
    out: set[Tuple[str, str]] = set()
    for idx, a_name in enumerate(names):
        for b_name in names[idx + 1:]:
            if objects_in_contact(world_obj, a_name, b_name):
                out.add(tuple(sorted((a_name, b_name))))
                continue
            gap = min_object_gap_px(world_obj, world_dict, a_name, b_name)
            if gap is not None and gap < thr.summary_contact_min_gap_px:
                out.add(tuple(sorted((a_name, b_name))))
    return out


def object_goal_touch_fraction(world_obj: Any, world_dict: Dict[str, Any], object_name: str, goal_name: Optional[str]) -> float:
    if not goal_name or object_name not in world_obj.objects or goal_name not in world_obj.objects:
        return 0.0
    target = world_obj.objects[object_name]
    goal = world_obj.objects[goal_name]
    shape_def = (world_dict.get("world", {}).get("objects", {}) or {}).get(object_name, {}) or {}
    pos = target.getPos()
    if hasattr(pos, "x") and hasattr(pos, "y"):
        cx, cy = float(pos.x), float(pos.y)
    elif hasattr(pos, "tolist"):
        cx, cy = [float(value) for value in pos.tolist()[:2]]
    else:
        cx, cy = float(pos[0]), float(pos[1])

    if hasattr(target, "radius") or ("radius" in shape_def):
        radius = float(getattr(target, "radius", shape_def.get("radius", 0.0) or 0.0))
        if radius <= 0:
            return 0.0
        def point_inside_goal(px: float, py: float) -> bool:
            try:
                if hasattr(goal, "pointIn"):
                    return bool(goal.pointIn([px, py]))
            except Exception:
                pass
            try:
                goal_vertices = goal.getVertices()
            except Exception:
                goal_vertices = None
            return bool(goal_vertices) and point_in_poly(px, py, goal_vertices)

        inside = 0
        total = 0
        for ox, oy in DISK_SAMPLE_OFFSETS:
            px = cx + radius * ox
            py = cy + radius * oy
            total += 1
            if point_inside_goal(px, py):
                inside += 1
        return float(inside) / float(total or 1)

    poly_world = None
    try:
        poly_world = [
            (float(v.x), float(v.y)) if hasattr(v, "x") else (float(v[0]), float(v[1]))
            for v in target.getVertices()
        ]
    except Exception:
        poly_world = None
    if not poly_world:
        return 0.0
    def poly_point_inside_goal(px: float, py: float) -> bool:
        try:
            if hasattr(goal, "pointIn"):
                return bool(goal.pointIn([px, py]))
        except Exception:
            pass
        return point_in_poly(px, py, goal.getVertices())

    inside_points = list(poly_world)
    try:
        cx_poly = sum(px for px, _ in poly_world) / float(len(poly_world))
        cy_poly = sum(py for _, py in poly_world) / float(len(poly_world))
        inside_points.append((cx_poly, cy_poly))
    except Exception:
        pass
    inside = sum(1 for px, py in inside_points if poly_point_inside_goal(px, py))
    return float(inside) / float(len(inside_points) or 1)


def target_goal_touch_fraction(world_obj: Any, world_dict: Dict[str, Any], target_name: str, goal_name: Optional[str]) -> float:
    return object_goal_touch_fraction(world_obj, world_dict, target_name, goal_name)


def transformed_polygon_at_final_time(
    trace: Dict[str, Any],
    world_dict: Dict[str, Any],
    object_name: str,
    points: List[List[float]],
) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    if not pts:
        return []
    final_path = (trace.get("path", {}) or {}).get(object_name) or []
    final_rot = (trace.get("rot", {}) or {}).get(object_name) or []
    if not final_path:
        return pts
    final_pos = final_path[-1]
    angle = float(final_rot[-1]) if final_rot else 0.0
    spec = ((world_dict.get("world", {}) or {}).get("objects", {}) or {}).get(object_name, {}) or {}
    center = object_center(spec) or (
        float(np.mean([p[0] for p in pts])),
        float(np.mean([p[1] for p in pts])),
    )
    cx, cy = center
    tx, ty = float(final_pos[0]), float(final_pos[1])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    out: List[Tuple[float, float]] = []
    for px, py in pts:
        lx = px - cx
        ly = py - cy
        wx = tx + lx * cos_a - ly * sin_a
        wy = ty + lx * sin_a + ly * cos_a
        out.append((wx, wy))
    return out


def final_goal_interior_polygon(trace: Dict[str, Any], world_dict: Dict[str, Any], goal_name: Optional[str]) -> List[Tuple[float, float]]:
    if not goal_name:
        return []
    goal_spec = ((world_dict.get("world", {}) or {}).get("objects", {}) or {}).get(goal_name, {}) or {}
    points = goal_spec.get("points") or goal_spec.get("inner_vertices") or []
    if not isinstance(points, list) or len(points) < 3:
        return []
    goal_poly = transformed_polygon_at_final_time(trace, world_dict, goal_name, points)
    if str(goal_spec.get("type", "")) == "Container":
        width = float(goal_spec.get("width", 0.0) or 0.0)
        if width > 0 and len(goal_poly) >= 4:
            xs = [p[0] for p in goal_poly]
            ys = [p[1] for p in goal_poly]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            inset = width
            if max_x - min_x > 2 * inset and max_y - min_y > inset:
                # Treat VTool containers as open-top U-shapes whose green interior is the
                # axis-aligned region inside the side walls and above the bottom wall.
                return [
                    (min_x + inset, max_y),
                    (min_x + inset, min_y + inset),
                    (max_x - inset, min_y + inset),
                    (max_x - inset, max_y),
                ]
    return goal_poly


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> bool:
    eps = 1e-6
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(a1: Tuple[float, float], a2: Tuple[float, float], b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)
    eps = 1e-6
    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps):
        return True
    if abs(o1) <= eps and _on_segment(a1, a2, b1):
        return True
    if abs(o2) <= eps and _on_segment(a1, a2, b2):
        return True
    if abs(o3) <= eps and _on_segment(b1, b2, a1):
        return True
    if abs(o4) <= eps and _on_segment(b1, b2, a2):
        return True
    return False


def polygons_overlap(poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]]) -> bool:
    if not poly_a or not poly_b:
        return False
    if any(point_in_poly(x, y, poly_b) for x, y in poly_a):
        return True
    if any(point_in_poly(x, y, poly_a) for x, y in poly_b):
        return True
    edges_a = list(zip(poly_a, poly_a[1:] + poly_a[:1]))
    edges_b = list(zip(poly_b, poly_b[1:] + poly_b[:1]))
    for a1, a2 in edges_a:
        for b1, b2 in edges_b:
            if segments_intersect(a1, a2, b1, b2):
                return True
    return False


def final_object_touches_goal_interior(trace: Dict[str, Any], world_dict: Dict[str, Any], object_name: str, goal_name: Optional[str]) -> bool:
    goal_poly = final_goal_interior_polygon(trace, world_dict, goal_name)
    if not goal_poly:
        return False
    spec = ((world_dict.get("world", {}) or {}).get("objects", {}) or {}).get(object_name, {}) or {}
    path_pts = (trace.get("path", {}) or {}).get(object_name) or []
    if not path_pts:
        return False
    cx, cy = float(path_pts[-1][0]), float(path_pts[-1][1])
    if "radius" in spec:
        radius = float(spec.get("radius", 0.0) or 0.0)
        dense_offsets = [(0.0, 0.0)] + [
            (r * math.cos(theta), r * math.sin(theta))
            for r in (0.2, 0.4, 0.6, 0.8, 0.95)
            for theta in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        ]
        sample_points = [(cx + radius * ox, cy + radius * oy) for ox, oy in dense_offsets]
        return any(point_in_poly(px, py, goal_poly) for px, py in sample_points)
    if any(key in spec for key in ("vertices", "points")):
        points = spec.get("vertices") or spec.get("points") or []
        final_poly = transformed_polygon_at_final_time(trace, world_dict, object_name, points)
        if not final_poly:
            return point_in_poly(cx, cy, goal_poly)
        # For polygonal objects, count as inside when any of the object's vertices
        # are in the green interior, or when the polygon overlaps the interior.
        if any(point_in_poly(px, py, goal_poly) for px, py in final_poly):
            return True
        if polygons_overlap(final_poly, goal_poly):
            return True
        center_x = float(np.mean([p[0] for p in final_poly]))
        center_y = float(np.mean([p[1] for p in final_poly]))
        return point_in_poly(center_x, center_y, goal_poly)
    return point_in_poly(cx, cy, goal_poly)


def run_trace(
    world_obj: Any,
    *,
    world_dict: Dict[str, Any],
    maxtime: float,
    step: float,
    collision_slop: float,
    target_name: str,
    goal_name: Optional[str],
    thr: Optional[Thresholds] = None,
    video_path: Optional[Path] = None,
    video_fps: Optional[float] = None,
) -> Dict[str, Any]:
    thr = thr or Thresholds()
    t = 0.0
    path: Dict[str, List[List[float]]] = {}
    rot: Dict[str, List[float]] = {}
    track = [name for name, obj in world_obj.objects.items() if not obj.isStatic()]
    for name in track:
        path[name] = [_pos(world_obj.objects[name])]
        rot[name] = [float(world_obj.objects[name].rotation)]

    in_goal_since: Optional[float] = None
    in_goal_intervals: List[Tuple[float, float]] = []
    goal_touch_intervals: List[Tuple[float, float]] = []
    object_in_goal_since: Dict[str, Optional[float]] = {name: None for name in track}
    object_in_goal_intervals: Dict[str, List[Tuple[float, float]]] = {name: [] for name in track}
    object_goal_touch_since: Dict[str, Optional[float]] = {name: None for name in track}
    object_goal_touch_intervals: Dict[str, List[Tuple[float, float]]] = {name: [] for name in track}
    object_goal_interior_since: Dict[str, Optional[float]] = {name: None for name in track}
    object_goal_interior_intervals: Dict[str, List[Tuple[float, float]]] = {name: [] for name in track}
    max_goal_touch_fraction = 0.0
    current_start: Optional[float] = None
    current_touch_start: Optional[float] = None
    ended = False
    initial_near_pairs = initial_near_contact_pairs(world_obj, world_dict, thr)
    initial_near_pairs_with_walls = initial_near_contact_pairs(world_obj, world_dict, thr, include_walls=True)
    visible_contact_times: Dict[Tuple[str, str], float] = {}
    visible_contact_pairs = [
        tuple(sorted((a_name, b_name)))
        for idx, a_name in enumerate(world_obj.objects.keys())
        for b_name in list(world_obj.objects.keys())[idx + 1:]
        if not ({a_name, b_name} <= WALL_NAMES)
    ]
    _goal_name, goal_obj = pick_container_or_none(world_dict.get("objects", {}) or {}, preferred_name=goal_name)
    goal_pts_world = goal_obj.get("points") if goal_obj else None
    still_time = 0.0
    writer = None
    screen = None
    if video_path is not None and imageio is not None:
        pg.init()
        dims = world_dict.get("dims", [600, 600])
        screen = pg.display.set_mode((int(dims[0]), int(dims[1])), flags=pg.HIDDEN)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(video_path),
            fps=float(video_fps or (1.0 / step)),
            format="FFMPEG",
            codec="libx264",
            quality=8,
            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    while t < maxtime:
        if writer is not None and screen is not None:
            frame_surf = drawWorld(world_obj)
            screen.blit(frame_surf, (0, 0))
            pg.display.flip()
            arr = pg.surfarray.array3d(screen).swapaxes(0, 1)
            writer.append_data(arr)
        world_obj.step(step)
        t += step
        for name in track:
            if name in world_obj.objects:
                path[name].append(_pos(world_obj.objects[name]))
                rot[name].append(float(world_obj.objects[name].rotation))
        for a_name, b_name in visible_contact_pairs:
            pair = tuple(sorted((a_name, b_name)))
            if pair in visible_contact_times:
                continue
            if pair in initial_near_pairs_with_walls:
                continue
            gap = min_object_gap_px(world_obj, world_dict, a_name, b_name)
            if gap is not None and gap <= thr.visible_contact_gap_px:
                visible_contact_times[pair] = t
        inside = bool(_target_fully_inside_goal(world_obj, {"world": world_dict}, target_name, goal_name, goal_pts_world))
        goal_touch_fraction = object_goal_touch_fraction(world_obj, {"world": world_dict}, target_name, goal_name)
        max_goal_touch_fraction = max(max_goal_touch_fraction, goal_touch_fraction)
        if inside and current_start is None:
            current_start = t
        if not inside and current_start is not None:
            in_goal_intervals.append((current_start, t))
            current_start = None
        if goal_touch_fraction >= 0.20 and current_touch_start is None:
            current_touch_start = t
        if goal_touch_fraction < 0.20 and current_touch_start is not None:
            goal_touch_intervals.append((current_touch_start, t))
            current_touch_start = None
        if inside:
            if in_goal_since is None:
                in_goal_since = t
            ended = True
        else:
            in_goal_since = None
        for object_name in track:
            object_inside = bool(_target_fully_inside_goal(world_obj, {"world": world_dict}, object_name, goal_name, goal_pts_world))
            object_goal_fraction = object_goal_touch_fraction(world_obj, {"world": world_dict}, object_name, goal_name)
            if object_inside and object_in_goal_since[object_name] is None:
                object_in_goal_since[object_name] = t
            if not object_inside and object_in_goal_since[object_name] is not None:
                object_in_goal_intervals[object_name].append((object_in_goal_since[object_name], t))
                object_in_goal_since[object_name] = None
            if object_goal_fraction > 1e-6 and object_goal_interior_since[object_name] is None:
                object_goal_interior_since[object_name] = t
            if object_goal_fraction <= 1e-6 and object_goal_interior_since[object_name] is not None:
                object_goal_interior_intervals[object_name].append((object_goal_interior_since[object_name], t))
                object_goal_interior_since[object_name] = None
            if object_goal_fraction >= 0.20 and object_goal_touch_since[object_name] is None:
                object_goal_touch_since[object_name] = t
            if object_goal_fraction < 0.20 and object_goal_touch_since[object_name] is not None:
                object_goal_touch_intervals[object_name].append((object_goal_touch_since[object_name], t))
                object_goal_touch_since[object_name] = None
        all_still_this_frame = True
        for obj in world_obj.getDynamicObjects():
            if obj._cpBody.is_sleeping:
                continue
            lin_sq = obj._cpBody.velocity.length ** 2
            ang_v = abs(obj._cpBody.angular_velocity)
            if lin_sq > (thr.settle_linear_velocity_eps ** 2) or ang_v > thr.settle_angular_velocity_eps:
                all_still_this_frame = False
                break
        if all_still_this_frame:
            still_time += step
        else:
            still_time = 0.0
        waiting_on_goal_duration = (
            (not ended)
            and current_touch_start is not None
            and (thr.goal_hold_s > 0.0)
        )
        if still_time >= thr.settle_window_s and not waiting_on_goal_duration:
            break
    if current_start is not None:
        in_goal_intervals.append((current_start, t))
    if current_touch_start is not None:
        goal_touch_intervals.append((current_touch_start, t))
    for object_name in track:
        if object_in_goal_since[object_name] is not None:
            object_in_goal_intervals[object_name].append((object_in_goal_since[object_name], t))
        if object_goal_interior_since[object_name] is not None:
            object_goal_interior_intervals[object_name].append((object_goal_interior_since[object_name], t))
        if object_goal_touch_since[object_name] is not None:
            object_goal_touch_intervals[object_name].append((object_goal_touch_since[object_name], t))
    if writer is not None:
        writer.close()
    if screen is not None:
        pg.quit()
    return {
        "path": path,
        "rot": rot,
        "collisions": filterCollisionEvents(world_obj.collisionEvents, collision_slop),
        "ended": ended,
        "duration": t,
        "in_goal_intervals": in_goal_intervals,
        "goal_touch_intervals": goal_touch_intervals,
        "object_in_goal_intervals": {name: [(float(start), float(end)) for start, end in spans] for name, spans in object_in_goal_intervals.items()},
        "object_goal_interior_intervals": {name: [(float(start), float(end)) for start, end in spans] for name, spans in object_goal_interior_intervals.items()},
        "object_goal_touch_intervals": {name: [(float(start), float(end)) for start, end in spans] for name, spans in object_goal_touch_intervals.items()},
        "max_goal_touch_fraction": max_goal_touch_fraction,
        "final_contacts": final_contact_names(world_obj, world_obj.objects.keys()),
        "initial_near_contact_pairs": [list(pair) for pair in sorted(initial_near_pairs)],
        "initial_near_contact_pairs_with_walls": [list(pair) for pair in sorted(initial_near_pairs_with_walls)],
        "visible_contact_times": [{"a": a, "b": b, "t": tm} for (a, b), tm in sorted(visible_contact_times.items(), key=lambda item: item[1])],
    }


def event_duration(ev: Sequence[Any], sim_duration: float) -> float:
    start = ev[2]
    end = ev[3] if len(ev) > 3 else None
    if start is None:
        return 0.0
    return float((sim_duration if end is None else end) - start)


def contact(collisions: List[Any], a: str, b: str, thr: Thresholds, sim_duration: float) -> bool:
    for ev in collisions:
        o1, o2 = str(ev[0]), str(ev[1])
        if {o1, o2} == {a, b} and event_duration(ev, sim_duration) >= thr.contact_min_s:
            return True
    return False


def first_contact_time(collisions: List[Any], a: str, candidates: Iterable[str], thr: Thresholds, sim_duration: float) -> Optional[float]:
    cands = set(candidates)
    best: Optional[float] = None
    for ev in collisions:
        o1, o2 = str(ev[0]), str(ev[1])
        if event_duration(ev, sim_duration) < thr.contact_min_s:
            continue
        if (o1 == a and o2 in cands) or (o2 == a and o1 in cands):
            t = float(ev[2] or 0.0)
            best = t if best is None else min(best, t)
    return best


def first_contact_event(
    collisions: List[Any],
    focal_names: Iterable[str],
    candidate_names: Iterable[str],
    thr: Thresholds,
    sim_duration: float,
) -> Optional[Tuple[float, str]]:
    focal = set(focal_names)
    candidates = set(candidate_names)
    best: Optional[Tuple[float, str]] = None
    for ev in collisions:
        if event_duration(ev, sim_duration) < thr.contact_min_s:
            continue
        o1, o2 = str(ev[0]), str(ev[1])
        other = None
        if o1 in focal and o2 in candidates:
            other = o2
        elif o2 in focal and o1 in candidates:
            other = o1
        if other is None:
            continue
        t = float(ev[2] or 0.0)
        if best is None or t < best[0]:
            best = (t, other)
    return best


def ordered_before_label(time_a: Optional[float], time_b: Optional[float], thr: Thresholds) -> Optional[bool]:
    if time_a is None or time_b is None:
        return None
    if abs(time_a - time_b) < thr.before_event_min_gap_s:
        return None
    return time_a < time_b


def first_contact_category_label(
    collisions: List[Any],
    focal_names: Iterable[str],
    positive_names: Iterable[str],
    thr: Thresholds,
    sim_duration: float,
) -> Optional[bool]:
    focal = set(focal_names)
    positives = set(positive_names)
    earliest: Optional[Tuple[float, str]] = None
    ambiguous = False
    for ev in collisions:
        if event_duration(ev, sim_duration) < thr.contact_min_s:
            continue
        o1, o2 = str(ev[0]), str(ev[1])
        other = None
        if o1 in focal:
            other = o2
        elif o2 in focal:
            other = o1
        if other is None:
            continue
        t = float(ev[2] or 0.0)
        if earliest is None:
            earliest = (t, other)
            continue
        if t < earliest[0] - 1e-9:
            earliest = (t, other)
            ambiguous = False
            continue
        if abs(t - earliest[0]) < thr.before_event_min_gap_s and other != earliest[1]:
            ambiguous = True
    if earliest is None:
        return False
    if ambiguous:
        return None
    return earliest[1] in positives


def entered_goal_label(trace: Dict[str, Any], thr: Thresholds) -> Optional[bool]:
    max_fraction = float(trace.get("max_goal_touch_fraction", 0.0) or 0.0)
    if max_fraction >= thr.goal_touch_min_fraction:
        return True
    if max_fraction > 0.0:
        return None
    return False


def exited_goal_after_entry_label(trace: Dict[str, Any], thr: Thresholds) -> Optional[bool]:
    entry = entered_goal_label(trace, thr)
    if entry is not True:
        return None
    duration = float(trace.get("duration", 0.0) or 0.0)
    for start, end in trace.get("goal_touch_intervals", []) or []:
        if float(end) < duration - 1e-9:
            return True
    return False


def displacement(path: Dict[str, List[List[float]]], name: str) -> Optional[Tuple[float, float, float]]:
    pts = path.get(name)
    if not pts:
        return None
    dx = float(pts[-1][0] - pts[0][0])
    dy = float(pts[-1][1] - pts[0][1])
    return dx, dy, float(math.hypot(dx, dy))


def final_pos(path: Dict[str, List[List[float]]], name: str) -> Optional[np.ndarray]:
    pts = path.get(name)
    if not pts:
        return None
    return np.array(pts[-1], dtype=float)


def direction_label(delta: float, direction: str, thr: Thresholds) -> Optional[bool]:
    if direction == "right":
        if delta >= thr.yes_px:
            return True
        if delta < thr.no_px:
            return False
        return None
    if delta <= -thr.yes_px:
        return True
    if delta > -thr.no_px:
        return False
    return None


def moves_label(dist: Optional[float], thr: Thresholds) -> Optional[bool]:
    if dist is None:
        return False
    if dist >= thr.yes_px:
        return True
    if dist < thr.no_px:
        return False
    return None


def first_move_time(path: Dict[str, List[List[float]]], name: str, step: float, first_move_px: float) -> Optional[float]:
    pts = path.get(name)
    if not pts:
        return None
    p0 = np.array(pts[0], dtype=float)
    for i, p in enumerate(pts):
        if float(np.linalg.norm(np.array(p, dtype=float) - p0)) >= first_move_px:
            return i * step
    return None


def object_motion_radius(world_dict: Dict[str, Any], name: str) -> float:
    spec = (world_dict.get("world", {}).get("objects", {}) or {}).get(name, {}) or {}
    if "radius" in spec:
        return float(spec.get("radius") or 0.0)
    center = object_center(spec)
    if center is None:
        return 0.0
    cx, cy = center
    pts: List[Tuple[float, float]] = []
    for key in ("points", "vertices"):
        for p in spec.get(key, []) or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
    for poly in spec.get("polys", []) or []:
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
    if not pts:
        return 0.0
    return max(math.hypot(px - cx, py - cy) for px, py in pts)


def first_visible_motion_time(
    world_dict: Dict[str, Any],
    path: Dict[str, List[List[float]]],
    rot: Dict[str, List[float]],
    name: str,
    step: float,
    first_move_px: float,
) -> Optional[float]:
    """First time any visible part of the object has moved about first_move_px.

    Center displacement misses early rotation of planks/rectangles. For temporal
    order predicates, count approximate edge motion from rotation too.
    """
    pts = path.get(name)
    if not pts:
        return None
    p0 = np.array(pts[0], dtype=float)
    r0 = float((rot.get(name) or [0.0])[0])
    radius = object_motion_radius(world_dict, name)
    rots = rot.get(name) or []
    for i, p in enumerate(pts):
        center_disp = float(np.linalg.norm(np.array(p, dtype=float) - p0))
        angle_disp = 0.0
        if i < len(rots) and radius > 0:
            angle_disp = abs(float(rots[i]) - r0) * radius
        if max(center_disp, angle_disp) >= first_move_px:
            return i * step
    return None


def distance_to_goal(world_dict: Dict[str, Any], obj_name: str, goal: ObjInfo, path: Dict[str, List[List[float]]]) -> Optional[Tuple[float, float]]:
    objects = world_dict["world"].get("objects", {}) or {}
    goal_center = object_center(objects.get(goal.name, {}))
    pts = path.get(obj_name)
    if goal_center is None or not pts:
        return None
    g = np.array(goal_center, dtype=float)
    d0 = float(np.linalg.norm(np.array(pts[0], dtype=float) - g))
    d1 = float(np.linalg.norm(np.array(pts[-1], dtype=float) - g))
    return d0, d1


def held_in_goal(trace: Dict[str, Any], thr: Thresholds) -> bool:
    return any(float(end - start) >= thr.goal_hold_s for start, end in trace["in_goal_intervals"])


def rotation_label(rot: Dict[str, List[float]], name: str, clockwise: bool, thr: Thresholds) -> Optional[bool]:
    vals = rot.get(name)
    if not vals:
        return False
    da = float(vals[-1] - vals[0])
    # Pymunk positive is counterclockwise.
    if clockwise:
        if da <= -thr.rotation_rad:
            return True
        if da > -math.radians(3):
            return False
    else:
        if da >= thr.rotation_rad:
            return True
        if da < math.radians(3):
            return False
    return None


def robust_label(values: List[Optional[bool]], required_votes: int) -> Tuple[Optional[bool], float, int, int, int]:
    yes = sum(v is True for v in values)
    no = sum(v is False for v in values)
    gray = sum(v is None for v in values)
    denom = max(1, yes + no + gray)
    if yes >= required_votes:
        return True, yes / denom, yes, no, gray
    if no >= required_votes:
        return False, yes / denom, yes, no, gray
    return None, yes / denom, yes, no, gray


def q(text: str) -> str:
    return text


def black_phrase_for_scene(env_set: str, env_id: str | int) -> str:
    env = str(env_set)
    eid = str(env_id)
    if env == "BackUp":
        return "the black trapezoid"
    if env == "Balance":
        return "the black pillar"
    if env == "BalanceUnder":
        return "a black support object: the black trapezoid or black rectangle connected to the green goal container, or the short black pillar"
    if env == "Basic":
        return "the black trapezoid or black rectangle"
    if env == "Falling":
        return "the black rectangle or black trapezoid"
    if env == "FallingAlt":
        return "a black support object: the black rectangle or black trapezoid, or one of the short black pegs"
    if env == "Prevention":
        return "one of the black obstacles: the left black trapezoid, middle black triangle, or right black rectangle"
    if env == "Remove":
        if eid in {"9", "10"}:
            return "one of the black obstacles: the black triangle, the big black rectangle attached to the left, or the small black rectangle in the middle/right"
        if eid in {"3", "4", "8"}:
            return "one of the black obstacles: the black triangle or the big black rectangle attached to the left"
        return "the black triangle"
    return "a static black object"


def blue_phrase(blues: List[ObjInfo], *, article: str = "definite") -> str:
    if len(blues) == 1:
        label = blues[0].label
        return f"the {label}" if article == "definite" else label
    return "any movable blue object"


def blue_subject_phrase(blues: List[ObjInfo]) -> str:
    if len(blues) == 1:
        return f"the {blues[0].label}"
    return "any movable blue object"


def add_pred(store: Dict[str, List[Optional[bool]]], questions: Dict[str, str], key: str, value: Optional[bool], question: str) -> None:
    store.setdefault(key, []).append(value)
    questions.setdefault(key, question)


def evaluate_rollout(
    *,
    trace: Dict[str, Any],
    baseline: Dict[str, Any],
    world_dict: Dict[str, Any],
    target: ObjInfo,
    goal: Optional[ObjInfo],
    blues: List[ObjInfo],
    movable: List[ObjInfo],
    static_black: List[ObjInfo],
    env_set: str,
    env_id: str,
    step: float,
    thr: Thresholds,
    values: Dict[str, List[Optional[bool]]],
    questions: Dict[str, str],
    placement_flags: Dict[str, bool],
) -> None:
    path = trace["path"]
    base_path = baseline["path"]
    collisions = trace["collisions"]
    sim_duration = trace["duration"]
    container_movable = bool(goal and goal.is_movable)
    static_black_names = [obj.name for obj in static_black]
    blue_obj_phrase = blue_phrase(blues)
    blue_subject = blue_subject_phrase(blues)
    black_obj_phrase = black_phrase_for_scene(env_set, env_id)

    if placement_flags.get("CONTACT_TOOL_TARGET", True):
        add_pred(values, questions, "CONTACT_TOOL_TARGET", contact(collisions, TOOL_NAME, target.name, thr, sim_duration),
                 q("Will the big orange ball (dropped tool) physically collide with or touch the red ball at any point during the rollout? Count only actual contact, not passing nearby."))
    if blues:
        if placement_flags.get("CONTACT_TOOL_ANY_BLUE_MOVABLE", True):
            add_pred(values, questions, "CONTACT_TOOL_ANY_BLUE_MOVABLE", any(contact(collisions, TOOL_NAME, b.name, thr, sim_duration) for b in blues),
                     q(f"Will the big orange ball (dropped tool) physically collide with or touch {blue_obj_phrase} at any point during the rollout? Count only actual contact, not passing nearby."))
    if goal:
        if placement_flags.get("CONTACT_TOOL_GOAL_CONTAINER", True):
            add_pred(values, questions, "CONTACT_TOOL_GOAL_CONTAINER", contact(collisions, TOOL_NAME, goal.name, thr, sim_duration),
                     q("Will the big orange ball (dropped tool) physically collide with or touch the green goal container at any point during the rollout? Count only actual contact, not passing nearby. The big orange ball (dropped tool) does not need to be inside the green goal container; touching an outside edge or wall counts."))
        if placement_flags.get("CONTACT_TARGET_GOAL_CONTAINER", True):
            add_pred(values, questions, "CONTACT_TARGET_GOAL_CONTAINER", contact(collisions, target.name, goal.name, thr, sim_duration),
                     q("Will the red ball physically collide with or touch the green goal container at any point during the rollout? Count only actual contact, not passing nearby. The red ball does not need to be inside the green goal container; touching an outside edge or wall counts."))
    if blues:
        if placement_flags.get("CONTACT_TARGET_ANY_BLUE", True):
            add_pred(values, questions, "CONTACT_TARGET_ANY_BLUE", any(contact(collisions, target.name, b.name, thr, sim_duration) for b in blues),
                     q(f"Will the red ball physically collide with or touch {blue_obj_phrase} at any point during the rollout? Count only actual contact, not passing nearby."))
    if goal and blues:
        if placement_flags.get("CONTACT_BLUE_GOAL_CONTAINER", True):
            add_pred(values, questions, "CONTACT_BLUE_GOAL_CONTAINER", any(contact(collisions, b.name, goal.name, thr, sim_duration) for b in blues),
                     q(f"Will {blue_subject} physically collide with or touch the green goal container at any point during the rollout? Count only actual contact, not passing nearby. The blue object does not need to be inside the green goal container; touching an outside edge or wall counts."))

    if placement_flags.get("FIRST_CONTACT_OF_TARGET_IS_BLUE", True):
        add_pred(values, questions, "FIRST_CONTACT_OF_TARGET_IS_BLUE",
                 first_contact_category_label(collisions, [target.name], [b.name for b in blues], thr, sim_duration),
                 q(f"Will the first object that the red ball physically collides with or touches during the rollout be {blue_obj_phrase}? Count only actual contact, not passing nearby."))
    if placement_flags.get("FIRST_CONTACT_OF_TARGET_IS_BLACK", True):
        add_pred(values, questions, "FIRST_CONTACT_OF_TARGET_IS_BLACK",
                 first_contact_category_label(collisions, [target.name], static_black_names, thr, sim_duration),
                 q(f"Will the first object that the red ball physically collides with or touches during the rollout be {black_obj_phrase}? Walls do not count as static black objects. Count only actual contact, not passing nearby."))
    if placement_flags.get("FIRST_CONTACT_OF_TOOL_IS_BLUE", True):
        add_pred(values, questions, "FIRST_CONTACT_OF_TOOL_IS_BLUE",
                 first_contact_category_label(collisions, [TOOL_NAME], [b.name for b in blues], thr, sim_duration),
                 q(f"Will the first object that the big orange ball (dropped tool) physically collides with or touches during the rollout be {blue_obj_phrase}? Count only actual contact, not passing nearby."))
    if placement_flags.get("FIRST_CONTACT_OF_TOOL_IS_BLACK", True):
        add_pred(values, questions, "FIRST_CONTACT_OF_TOOL_IS_BLACK",
                 first_contact_category_label(collisions, [TOOL_NAME], static_black_names, thr, sim_duration),
                 q(f"Will the first object that the big orange ball (dropped tool) physically collides with or touches during the rollout be {black_obj_phrase}? Walls do not count as static black objects. Count only actual contact, not passing nearby."))
    if placement_flags.get("FIRST_CONTACT_OF_BLUE_IS_ANOTHER_BLUE", True):
        add_pred(values, questions, "FIRST_CONTACT_OF_BLUE_IS_ANOTHER_BLUE",
                 first_contact_category_label(collisions, [b.name for b in blues], [b.name for b in blues], thr, sim_duration),
                 q(f"Will the first object that {blue_subject} physically collides with or touches during the rollout be another movable blue object? Count only actual contact, not passing nearby."))
    if placement_flags.get("FIRST_CONTACT_OF_BLUE_IS_BLACK", True):
        add_pred(values, questions, "FIRST_CONTACT_OF_BLUE_IS_BLACK",
                 first_contact_category_label(collisions, [b.name for b in blues], static_black_names, thr, sim_duration),
                 q(f"Will the first object that {blue_subject} physically collides with or touches during the rollout be {black_obj_phrase}? Walls do not count as static black objects. Count only actual contact, not passing nearby."))

    if blues and placement_flags.get("CONTACT_TOOL_BLUE_BEFORE_TOOL_TARGET", True):
        add_pred(values, questions, "CONTACT_TOOL_BLUE_BEFORE_TOOL_TARGET",
                 ordered_before_label(
                     first_contact_time(collisions, TOOL_NAME, [b.name for b in blues], thr, sim_duration),
                     first_contact_time(collisions, TOOL_NAME, [target.name], thr, sim_duration),
                     thr,
                 ),
                 q(f"Will the big orange ball (dropped tool) physically collide with or touch {blue_obj_phrase} before it collides with or touches the red ball at any point during the rollout? Both contact events must happen clearly and be visibly separated in time. Count only actual contact, not passing nearby."))
    if placement_flags.get("CONTACT_TOOL_BLACK_BEFORE_TOOL_TARGET", True):
        add_pred(values, questions, "CONTACT_TOOL_BLACK_BEFORE_TOOL_TARGET",
                 ordered_before_label(
                     first_contact_time(collisions, TOOL_NAME, static_black_names, thr, sim_duration),
                     first_contact_time(collisions, TOOL_NAME, [target.name], thr, sim_duration),
                     thr,
                 ),
                 q(f"Will the big orange ball (dropped tool) physically collide with or touch {black_obj_phrase} before it collides with or touches the red ball at any point during the rollout? Walls do not count as static black objects. Both contact events must happen clearly and be visibly separated in time. Count only actual contact, not passing nearby."))
    if blues and placement_flags.get("CONTACT_TOOL_BLUE_BEFORE_TOOL_BLACK", True):
        add_pred(values, questions, "CONTACT_TOOL_BLUE_BEFORE_TOOL_BLACK",
                 ordered_before_label(
                     first_contact_time(collisions, TOOL_NAME, [b.name for b in blues], thr, sim_duration),
                     first_contact_time(collisions, TOOL_NAME, static_black_names, thr, sim_duration),
                     thr,
                 ),
                 q(f"Will the big orange ball (dropped tool) physically collide with or touch {blue_obj_phrase} before it collides with or touches {black_obj_phrase} at any point during the rollout? Walls do not count as static black objects. Both contact events must happen clearly and be visibly separated in time. Count only actual contact, not passing nearby."))
    if blues and placement_flags.get("CONTACT_TARGET_BLUE_BEFORE_TOOL_TARGET", True):
        add_pred(values, questions, "CONTACT_TARGET_BLUE_BEFORE_TOOL_TARGET",
                 ordered_before_label(
                     first_contact_time(collisions, target.name, [b.name for b in blues], thr, sim_duration),
                     first_contact_time(collisions, TOOL_NAME, [target.name], thr, sim_duration),
                     thr,
                 ),
                 q(f"Will the red ball physically collide with or touch {blue_obj_phrase} before the big orange ball (dropped tool) collides with or touches the red ball at any point during the rollout? Both contact events must happen clearly and be visibly separated in time. Count only actual contact, not passing nearby."))
    if placement_flags.get("CONTACT_TARGET_BLACK_BEFORE_TOOL_TARGET", True):
        add_pred(values, questions, "CONTACT_TARGET_BLACK_BEFORE_TOOL_TARGET",
                 ordered_before_label(
                     first_contact_time(collisions, target.name, static_black_names, thr, sim_duration),
                     first_contact_time(collisions, TOOL_NAME, [target.name], thr, sim_duration),
                     thr,
                 ),
                 q(f"Will the red ball physically collide with or touch {black_obj_phrase} before the big orange ball (dropped tool) collides with or touches the red ball at any point during the rollout? Walls do not count as static black objects. Both contact events must happen clearly and be visibly separated in time. Count only actual contact, not passing nearby."))

    d_target = displacement(path, target.name)
    if d_target:
        dx, _dy, dist = d_target
        add_pred(values, questions, "TARGET_MOVES_LEFT", direction_label(dx, "left", thr),
                 q("After the scene settles, will the red ball end up more than 30 pixels to the left of where it started? Use yes for more than 30 pixels of leftward displacement and no for less than 5 pixels."))
        add_pred(values, questions, "TARGET_MOVES_RIGHT", direction_label(dx, "right", thr),
                 q("After the scene settles, will the red ball end up more than 30 pixels to the right of where it started? Use yes for more than 30 pixels of rightward displacement and no for less than 5 pixels."))
    for b in blues:
        dd = displacement(path, b.name)
        if dd:
            dx, _dy, dist = dd
            slug = _slug(b.label)
            add_pred(values, questions, f"{slug}_MOVES_LEFT", direction_label(dx, "left", thr),
                     q(f"After the scene settles, will the {b.label} end up more than 30 pixels to the left of where it started? Use yes for more than 30 pixels of leftward displacement and no for less than 5 pixels."))
            add_pred(values, questions, f"{slug}_MOVES_RIGHT", direction_label(dx, "right", thr),
                     q(f"After the scene settles, will the {b.label} end up more than 30 pixels to the right of where it started? Use yes for more than 30 pixels of rightward displacement and no for less than 5 pixels."))
            if env_set.lower().startswith(("falling", "balance", "prevention")) and any(token in b.label for token in ("rectangle", "plank")):
                add_pred(values, questions, f"{slug}_ROTATES_CLOCKWISE", rotation_label(trace["rot"], b.name, True, thr),
                         q(f"After the scene settles, will the {b.label} have rotated clockwise by a clearly visible amount, at least 15 degrees?"))
                add_pred(values, questions, f"{slug}_ROTATES_COUNTERCLOCKWISE", rotation_label(trace["rot"], b.name, False, thr),
                         q(f"After the scene settles, will the {b.label} have rotated counterclockwise by a clearly visible amount, at least 15 degrees?"))
    if goal and container_movable:
        dg = displacement(path, goal.name)
        if dg:
            dx, _dy, _dist = dg
            add_pred(values, questions, "GOAL_CONTAINER_MOVES_LEFT", direction_label(dx, "left", thr),
                     q("The green goal container is movable. After the scene settles, will it end up more than 30 pixels to the left of where it started? Use yes for more than 30 pixels of leftward displacement and no for less than 5 pixels."))
            add_pred(values, questions, "GOAL_CONTAINER_MOVES_RIGHT", direction_label(dx, "right", thr),
                     q("The green goal container is movable. After the scene settles, will it end up more than 30 pixels to the right of where it started? Use yes for more than 30 pixels of rightward displacement and no for less than 5 pixels."))

    all_move_vals = [moves_label((displacement(path, m.name) or (0, 0, 0))[2], thr) for m in movable]
    if not all_move_vals:
        all_move: Optional[bool] = False
    elif any(v is False for v in all_move_vals):
        all_move = False
    elif all(v is True for v in all_move_vals):
        all_move = True
    else:
        all_move = None
    add_pred(values, questions, "ALL_MOVABLE_OBJECTS_MOVE", all_move,
             q("Will every movable object in the scene move more than 30 pixels from its starting position? Include the red ball, any movable blue objects that are present, and the green goal container only if it is movable."))

    if goal:
        d_goal = distance_to_goal(world_dict, target.name, goal, path)
        if d_goal:
            d0, d1 = d_goal
            closer = d0 - d1
            farther = d1 - d0
            add_pred(values, questions, "TARGET_MOVES_CLOSER_TO_GOAL", True if closer >= thr.yes_px else (False if closer < thr.no_px else None),
                     q("After the scene settles, will the red ball be more than 30 pixels closer to the green goal container than it was at the start? Use yes for a decrease in distance greater than 30 pixels and no for a decrease less than 5 pixels."))
            add_pred(values, questions, "TARGET_MOVES_FARTHER_FROM_GOAL", True if farther >= thr.yes_px else (False if farther < thr.no_px else None),
                     q("After the scene settles, will the red ball be more than 30 pixels farther from the green goal container than it was at the start? Use yes for an increase in distance greater than 30 pixels and no for an increase less than 5 pixels."))
    add_pred(values, questions, "TARGET_IN_GOAL_AFTER_SETTLE", held_in_goal(trace, thr),
             q("After the scene settles, will the red ball remain fully in the green goal container for at least 2 continuous seconds?"))
    add_pred(values, questions, "TARGET_ENTERS_GOAL_AT_ANY_POINT", entered_goal_label(trace, thr),
             q("Will the red ball enter the green goal container at any point during the rollout? Count entry only when at least 20% of the red ball overlaps the green interior of the container. Touching only the outside wall, edge, or less than 20% of the ball does not count."))
    exit_label = exited_goal_after_entry_label(trace, thr)
    if exit_label is not None:
        add_pred(values, questions, "TARGET_EXITS_GOAL_AFTER_ENTRY", exit_label,
                 q("After the red ball has entered the green goal container by at least the 20% interior-overlap threshold, will it later exit the green interior? Touching the wall alone does not count as being in the goal."))

    target_moves = moves_label((d_target or (0, 0, 0))[2], thr)
    base_target_moves = moves_label((displacement(base_path, target.name) or (0, 0, 0))[2], thr)
    add_pred(values, questions, "INDIRECT_TRANSFER_TO_TARGET",
             (target_moves is True and base_target_moves is False and not contact(collisions, TOOL_NAME, target.name, thr, sim_duration)),
             q("Will the red ball move more than 30 pixels without being directly touched by the big orange ball (dropped tool)? This counts only if the big orange ball (dropped tool) first affects another object, which then causes the red ball to move; in the no-tool rollout, the red ball would not move."))
    any_blue_indirect = False
    for b in blues:
        bm = moves_label((displacement(path, b.name) or (0, 0, 0))[2], thr)
        bb = moves_label((displacement(base_path, b.name) or (0, 0, 0))[2], thr)
        if bm is True and bb is False and not contact(collisions, TOOL_NAME, b.name, thr, sim_duration):
            any_blue_indirect = True
    if blues:
        add_pred(values, questions, "INDIRECT_TRANSFER_TO_BLUE",
                 any_blue_indirect,
                 q("Will any movable blue object move more than 30 pixels without being directly touched by the big orange ball (dropped tool)? This counts only if the big orange ball (dropped tool) first affects another object, which then causes that blue object to move; in the no-tool rollout, that blue object would not move."))
    if goal and container_movable:
        gm = moves_label((displacement(path, goal.name) or (0, 0, 0))[2], thr)
        gb = moves_label((displacement(base_path, goal.name) or (0, 0, 0))[2], thr)
        add_pred(values, questions, "INDIRECT_TRANSFER_TO_CONTAINER",
                 (gm is True and gb is False and not contact(collisions, TOOL_NAME, goal.name, thr, sim_duration)),
                 q("The green goal container is movable. Will it move more than 30 pixels without being directly touched by the big orange ball (dropped tool)? This counts only if the big orange ball (dropped tool) first affects another object, which then causes the container to move; in the no-tool rollout, the container would not move."))

    for obj in [target, *blues, *([goal] if goal and container_movable else [])]:
        fp = final_pos(path, obj.name)
        bp = final_pos(base_path, obj.name)
        if fp is None or bp is None:
            continue
        changed = float(np.linalg.norm(fp - bp))
        pred = f"COUNTERFACTUAL_{_slug(obj.label)}_DIFFERENT_FINAL_POSITION"
        add_pred(values, questions, pred, True if changed >= thr.yes_px else (False if changed < thr.no_px else None),
                 q(f"Relative to a no-tool rollout of the same environment, will the {obj.label.replace('red target', 'red ball')} finish more than 30 pixels away from its no-tool final position? Use yes for a difference greater than 30 pixels and no for a difference less than 5 pixels."))


def sample_xy(rng: np.random.Generator, dims: Sequence[float], margin: float, anchors: List[Tuple[float, float]]) -> Tuple[float, float]:
    w, h = float(dims[0]), float(dims[1])
    if anchors and rng.random() < 0.70:
        ax, ay = anchors[int(rng.integers(0, len(anchors)))]
        x = ax + float(rng.normal(0.0, 85.0))
        y = ay + float(rng.normal(0.0, 85.0))
    else:
        x = float(rng.uniform(margin, w - margin))
        y = float(rng.uniform(margin, h - margin))
    return float(np.clip(x, margin, w - margin)), float(np.clip(y, margin, h - margin))


def load_success_heatmap_points(root: Optional[Path], env_set: str, env_id: str) -> List[Tuple[float, float]]:
    if root is None:
        return []
    csv_path = root / env_set / f"env_{env_id}" / "heatmap_samples.csv"
    if not csv_path.exists():
        return []
    points: List[Tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("valid_drop", "1")).strip() not in {"1", "true", "True"}:
                continue
            if str(row.get("success", "0")).strip() not in {"1", "true", "True"}:
                continue
            try:
                points.append((float(row["x"]), float(row["y"])))
            except Exception:
                continue
    return points


def sample_xy_with_heatmap(
    rng: np.random.Generator,
    dims: Sequence[float],
    margin: float,
    anchors: List[Tuple[float, float]],
    heatmap_points: List[Tuple[float, float]],
) -> Tuple[float, float]:
    w, h = float(dims[0]), float(dims[1])
    if heatmap_points and rng.random() < 0.60:
        hx, hy = heatmap_points[int(rng.integers(0, len(heatmap_points)))]
        x = hx + float(rng.normal(0.0, 18.0))
        y = hy + float(rng.normal(0.0, 18.0))
        return float(np.clip(x, margin, w - margin)), float(np.clip(y, margin, h - margin))
    return sample_xy(rng, dims, margin, anchors)


def far_enough_from_kept(x: float, y: float, kept_points: Sequence[Tuple[float, float]], min_distance: float) -> bool:
    if min_distance <= 0:
        return True
    return all(math.hypot(float(x) - float(px), float(y) - float(py)) >= min_distance for px, py in kept_points)


def make_record_id(env_set: str, env_id: str, gravity: str, x: float, y: float) -> str:
    h = hashlib.sha1(f"{env_set}|{env_id}|{gravity}|{x:.3f}|{y:.3f}".encode()).hexdigest()[:10]
    return f"{env_set}_{env_id}_{gravity}_{h}"


def make_slot_record_id(env_set: str, env_id: str, gravity: str, slot_index: int) -> str:
    return f"{env_set}_{env_id}_{gravity}_slot_{slot_index:04d}"


def load_resume_state(out_path: Path) -> Tuple[set[str], Dict[Tuple[str, str, str], int]]:
    """Return completed record ids and next slot index per env/condition."""
    completed: set[str] = set()
    next_slot: Dict[Tuple[str, str, str], int] = {}
    if not out_path.exists():
        return completed, next_slot
    with out_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] ignoring malformed JSONL row {line_no} in {out_path}")
                continue
            record_id = str(rec.get("record_id") or "")
            env_set = str(rec.get("env_set") or "")
            env_id = str(rec.get("env_id") or "")
            gravity = str(rec.get("gravity_mode") or "")
            slot_index = rec.get("placement_slot")
            if record_id:
                completed.add(record_id)
            if env_set and env_id and gravity:
                key = (env_set, env_id, gravity)
                try:
                    idx = int(slot_index)
                except Exception:
                    idx = next_slot.get(key, -1)
                next_slot[key] = max(next_slot.get(key, 0), idx + 1)
    return completed, next_slot


def render_dotted_screenshot(
    *,
    source_image: Path,
    world_dict: Optional[Dict[str, Any]] = None,
    out_path: Path,
    x: float,
    y: float,
    radius: float,
) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = None
    if world_dict is not None:
        try:
            pg.init()
            world_obj = loadFromDict(world_dict["world"] if "world" in world_dict else world_dict)
            surf = drawWorld(world_obj)
            arr = pg.surfarray.array3d(surf).swapaxes(0, 1)
            im = Image.fromarray(arr.astype("uint8"), mode="RGB").convert("RGBA")
            pg.quit()
        except Exception:
            im = None
    if im is None:
        if not source_image.exists():
            return None
        im = Image.open(source_image).convert("RGBA")
    draw = ImageDraw.Draw(im)
    cx = float(x)
    cy = float(im.height) - float(y)
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    for start in range(0, 360, 14):
        draw.arc(bbox, start=start, end=start + 7, fill=(127, 127, 127, 255), width=4)
    im.save(out_path)
    return str(out_path)


def render_representative_video(
    *,
    env_set: str,
    env_id: str,
    gravity: str,
    x: float,
    y: float,
    tool_color: str,
    out_dir: Path,
    basename: str,
    max_seconds: float,
    fps: float,
) -> Optional[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "worldPath": f"simulation/runtime/environment_sets/{env_set}/{env_id}.json",
        "selectedTool": "obj1",
        "dropCoords": [int(round(x)), int(round(y))],
        "gravityMode": gravity,
        "toolColor": tool_color,
        "videoOutputDir": str(out_dir),
        "videoBasename": basename,
        "maxSimSeconds": max_seconds,
        "recordWindowSeconds": max_seconds,
        "goalDurationSeconds": 2.0,
        "fps": fps,
    }
    try:
        result = simulate_rollout(payload)
    except Exception:
        return None
    return result.get("rolloutVideoPath")


def trace_object_metas(target: ObjInfo, goal: Optional[ObjInfo], blues: List[ObjInfo], movable: List[ObjInfo], has_tool: bool) -> List[TraceObjectMeta]:
    metas: List[TraceObjectMeta] = []
    if has_tool:
        metas.append(TraceObjectMeta(TOOL_NAME, "tool", "big orange ball (dropped tool)", True))
    known = {TOOL_NAME}
    for obj in [target, *blues, *([goal] if goal else [])]:
        if obj and obj.name not in known:
            role = "target" if obj.role == "target" else "goal" if obj.role == "goal" else obj.label.replace(" ", "_")
            metas.append(TraceObjectMeta(obj.name, role, obj.label.replace("red target", "red ball"), obj.is_movable))
            known.add(obj.name)
    for obj in movable:
        if obj.name not in known:
            metas.append(TraceObjectMeta(obj.name, obj.role, obj.label, obj.is_movable))
            known.add(obj.name)
    return metas


def rounded_time(value: float) -> float:
    return round(float(value), 2)


def final_contacts_for(trace: Dict[str, Any], obj_name: str) -> List[str]:
    return list((trace.get("final_contacts", {}) or {}).get(obj_name, []) or [])


def rotation_start_time(rot: Dict[str, List[float]], name: str, step: float, thr: Thresholds) -> Optional[float]:
    vals = rot.get(name)
    if not vals:
        return None
    start = float(vals[0])
    trigger = math.radians(5.0)
    for idx, angle in enumerate(vals):
        if abs(float(angle) - start) >= trigger:
            return idx * step
    return None


def goal_hold_success_time(trace: Dict[str, Any], thr: Thresholds) -> Optional[float]:
    for start, end in trace.get("in_goal_intervals", []) or []:
        if float(end) - float(start) >= thr.goal_hold_s:
            return float(start) + float(thr.goal_hold_s)
    return None


def object_kinematics_summary(trace: Dict[str, Any], name: str, step: float) -> Dict[str, Any]:
    pts = trace.get("path", {}).get(name) or []
    rots = trace.get("rot", {}).get(name) or []
    max_speed = 0.0
    max_ang_speed = 0.0
    for idx in range(1, len(pts)):
        dx = float(pts[idx][0] - pts[idx - 1][0]) / step
        dy = float(pts[idx][1] - pts[idx - 1][1]) / step
        max_speed = max(max_speed, float(math.hypot(dx, dy)))
    for idx in range(1, len(rots)):
        max_ang_speed = max(max_ang_speed, abs(float(rots[idx] - rots[idx - 1])) / step)
    start_pose = pts[0] if pts else [None, None]
    final_pose = pts[-1] if pts else [None, None]
    dx = float(final_pose[0] - start_pose[0]) if pts else 0.0
    dy = float(final_pose[1] - start_pose[1]) if pts else 0.0
    final_angle = rots[-1] if rots else 0.0
    start_angle = rots[0] if rots else 0.0
    angle_delta = float(final_angle - start_angle)
    max_positive_angle_delta = 0.0
    max_negative_angle_delta = 0.0
    for angle in rots:
        delta = float(angle) - float(start_angle)
        max_positive_angle_delta = max(max_positive_angle_delta, delta)
        max_negative_angle_delta = min(max_negative_angle_delta, delta)
    if angle_delta <= -math.radians(15.0):
        rotation_direction = "clockwise"
    elif angle_delta >= math.radians(15.0):
        rotation_direction = "counterclockwise"
    else:
        rotation_direction = "none"
    max_angular_displacement_rad = max(abs(max_positive_angle_delta), abs(max_negative_angle_delta))
    if abs(max_negative_angle_delta) > abs(max_positive_angle_delta) and abs(max_negative_angle_delta) >= math.radians(15.0):
        max_rotation_direction = "clockwise"
    elif abs(max_positive_angle_delta) >= math.radians(15.0):
        max_rotation_direction = "counterclockwise"
    else:
        max_rotation_direction = "none"
    return {
        "initial_position_xy": [float(start_pose[0]), float(start_pose[1])] if pts else None,
        "final_position_xy": [float(final_pose[0]), float(final_pose[1])] if pts else None,
        "displacement_xy": [dx, dy],
        "displacement_px": float(math.hypot(dx, dy)),
        "final_angle_rad": float(final_angle),
        "final_angle_deg": float(math.degrees(final_angle)),
        "rotation_delta_rad": angle_delta,
        "rotation_delta_deg": float(math.degrees(angle_delta)),
        "rotation_direction": rotation_direction,
        "max_angular_displacement_rad": float(max_angular_displacement_rad),
        "max_angular_displacement_deg": float(math.degrees(max_angular_displacement_rad)),
        "max_rotation_direction": max_rotation_direction,
        "max_speed_px_s": float(max_speed),
        "max_angular_speed_rad_s": float(max_ang_speed),
        "max_angular_speed_deg_s": float(math.degrees(max_ang_speed)),
    }


def dominant_displacement_direction(dx: float, dy: float, min_dist_px: float) -> Optional[str]:
    if math.hypot(dx, dy) < min_dist_px:
        return None
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "up" if dy > 0 else "down"


def stage_label(t: float, duration: float) -> str:
    if duration <= 0:
        return "early"
    frac = max(0.0, min(1.0, float(t) / float(duration)))
    if frac < (1.0 / 3.0):
        return "early"
    if frac < (2.0 / 3.0):
        return "middle"
    return "late"


def overlaps_goal_interior_at_time(
    trace: Dict[str, Any],
    object_name: str,
    t: float,
    *,
    tol: float = 0.06,
) -> bool:
    intervals = ((trace.get("object_goal_interior_intervals") or {}).get(object_name) or [])
    tt = float(t)
    for start, end in intervals:
        if float(start) - tol <= tt <= float(end) + tol:
            return True
    return False


def pascal_name(label: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", label)
    return "".join(word[:1].upper() + word[1:] for word in words) or "Object"


def static_black_object_names(
    world_dict: Dict[str, Any],
    static_black: List[ObjInfo],
    env_set: str,
    env_id: str,
) -> Dict[str, str]:
    objects = (world_dict.get("world", {}).get("objects", {}) or {})
    def poly_kind(spec: Dict[str, Any]) -> str:
        pts = spec.get("vertices") or spec.get("points") or []
        if len(pts) == 3:
            return "triangle"
        if len(pts) >= 4:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w >= h * 1.4:
                return "rectangle"
        return "polygon"
    if env_set == "Balance":
        return {obj.name: "BlackPillar" for obj in static_black}
    if env_set == "BackUp":
        labels: Dict[str, str] = {}
        for obj in static_black:
            spec = objects.get(obj.name, {}) or {}
            label = black_shape_label(obj.name, spec)
            labels[obj.name] = pascal_name(label)
        return labels
    if env_set == "Prevention":
        ordered = sorted(
            static_black,
            key=lambda obj: (object_center(objects.get(obj.name, {}) or {}) or (0.0, 0.0))[0],
        )
        labels = ["LeftBlackTrapezoid", "MiddleBlackTriangle", "RightBlackRectangle"]
        return {obj.name: labels[min(idx, len(labels) - 1)] for idx, obj in enumerate(ordered)}
    if env_set == "Remove":
        eid = str(env_id)
        if eid in {"1", "2", "4", "5", "6", "8", "9", "10"}:
            ordered = sorted(
                static_black,
                key=lambda obj: (object_center(objects.get(obj.name, {}) or {}) or (0.0, 0.0))[0],
            )
            labels: Dict[str, str] = {}
            if ordered:
                labels[ordered[0].name] = "BlackTrapezoid"
            if len(ordered) >= 2:
                labels[ordered[-1].name] = "BigBlackRectangleLeft" if eid in {"4", "8", "9", "10"} else pascal_name(black_shape_label(ordered[-1].name, objects.get(ordered[-1].name, {}) or {}))
            if len(ordered) >= 3:
                middle = [obj for obj in ordered if obj.name not in labels]
                for obj in middle:
                    labels[obj.name] = "SmallBlackRectangleMiddleRight"
            for obj in ordered:
                labels.setdefault(obj.name, "BlackTrapezoid")
            return labels
        if eid == "3":
            ordered = sorted(
                static_black,
                key=lambda obj: (object_center(objects.get(obj.name, {}) or {}) or (0.0, 0.0))[0],
            )
            labels: Dict[str, str] = {}
            if ordered:
                labels[ordered[0].name] = "BlackRectangle"
            if len(ordered) >= 2:
                labels[ordered[-1].name] = "BigBlackRectangleLeft"
            for obj in ordered:
                labels.setdefault(obj.name, "BlackRectangle")
            return labels
        if eid == "7":
            return {obj.name: pascal_name(black_shape_label(obj.name, objects.get(obj.name, {}) or {})) for obj in static_black}
        centers = {
            obj.name: (object_center(objects.get(obj.name, {}) or {}) or (0.0, 0.0))[0]
            for obj in static_black
        }
        ordered = sorted(static_black, key=lambda obj: centers[obj.name])
        labels: Dict[str, str] = {}
        for idx, obj in enumerate(ordered):
            spec = objects.get(obj.name, {}) or {}
            kind = poly_kind(spec)
            if idx == 0 and kind == "rectangle":
                labels[obj.name] = "BigBlackRectangleLeft"
            elif idx == len(ordered) - 1 and len(ordered) >= 3 and kind == "rectangle":
                labels[obj.name] = "SmallBlackRectangleMiddleRight"
            elif kind == "triangle":
                labels[obj.name] = "BlackTriangle"
            elif kind == "rectangle":
                labels[obj.name] = "BlackRectangle"
            else:
                labels[obj.name] = "BlackObstacle"
        return labels
    return {obj.name: pascal_name(black_phrase_for_scene(env_set, env_id).replace("the ", "").replace("a ", "")) for obj in static_black}


def summary_name_map(
    *,
    world_dict: Dict[str, Any],
    target: ObjInfo,
    goal: Optional[ObjInfo],
    blues: List[ObjInfo],
    static_black: List[ObjInfo],
    env_set: str,
    env_id: str,
    has_tool: bool,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {
        "_BottomWall": "Floor",
        "BottomWall": "Floor",
        "_TopWall": "TopWall",
        "TopWall": "TopWall",
        "_LeftWall": "LeftWall",
        "LeftWall": "LeftWall",
        "_RightWall": "RightWall",
        "RightWall": "RightWall",
        TOOL_NAME: "PlacedTool" if has_tool else TOOL_NAME,
        target.name: "RedTarget",
    }
    if goal is not None:
        mapping[goal.name] = "GoalContainer"
    for obj in blues:
        if env_set == "Remove" and str(env_id) == "7":
            mapping[obj.name] = "BlueTrapezoid"
        else:
            mapping[obj.name] = pascal_name(obj.label)
    mapping.update(static_black_object_names(world_dict, static_black, env_set, env_id))
    return mapping


def rename_summary_event(event: Dict[str, Any], name_map: Dict[str, str]) -> Dict[str, Any]:
    out = dict(event)
    if "actor" in out:
        out["actor"] = name_map.get(str(out["actor"]), str(out["actor"]))
    if "obj" in out:
        out["obj"] = name_map.get(str(out["obj"]), str(out["obj"]))
    if "goal" in out:
        out["goal"] = name_map.get(str(out["goal"]), str(out["goal"]))
    if "a" in out:
        out["a"] = name_map.get(str(out["a"]), str(out["a"]))
    if "b" in out:
        out["b"] = name_map.get(str(out["b"]), str(out["b"]))
    return out


def object_final_state_flags(
    *,
    trace: Dict[str, Any],
    world_dict: Dict[str, Any],
    name: str,
    goal_name: Optional[str],
    target_name: str,
    blue_names: List[str],
    black_names: List[str],
    name_map: Dict[str, str],
    is_target: bool,
    thr: Thresholds,
) -> Dict[str, bool]:
    contacts = set(final_contacts_for(trace, name))
    floor_names = {"_BottomWall", "BottomWall"}
    ceiling_names = {"_TopWall", "TopWall"}
    in_goal = final_object_touches_goal_interior(trace, world_dict, name, goal_name)
    out = {
        "in_goal": in_goal,
        "on_floor": any(other in floor_names for other in contacts),
        "on_ceiling": any(other in ceiling_names for other in contacts),
        "touching_goal": bool(goal_name and goal_name in contacts and not in_goal),
        "touching_goal_exterior": bool(goal_name and goal_name in contacts and not in_goal),
        "touching_floor": any(other in floor_names for other in contacts),
        "touching_ceiling": any(other in ceiling_names for other in contacts),
        "touching_target": target_name in contacts and name != target_name,
        "touching_placed_tool": TOOL_NAME in contacts and name != TOOL_NAME,
    }
    blue_touch = any(other in blue_names and other != name for other in contacts)
    black_touch = any(other in black_names for other in contacts)
    if len(blue_names) == 1:
        blue_label = name_map.get(blue_names[0], blue_names[0]).lower()
        out[f"touching_{re.sub(r'[^a-z0-9]+', '_', blue_label).strip('_')}"] = blue_touch
    else:
        out["touching_any_blue_object"] = blue_touch
    if len(black_names) == 1:
        black_label = name_map.get(black_names[0], black_names[0]).lower()
        out[f"touching_{re.sub(r'[^a-z0-9]+', '_', black_label).strip('_')}"] = black_touch
    else:
        out["touching_any_black_object"] = black_touch
    return out


def summarize_rollout_for_record(
    *,
    trace: Dict[str, Any],
    world_dict: Dict[str, Any],
    placement_xy: Tuple[float, float],
    target: ObjInfo,
    goal: Optional[ObjInfo],
    blues: List[ObjInfo],
    static_black: List[ObjInfo],
    env_set: str,
    env_id: str,
    thr: Thresholds,
    step: float,
    has_tool: bool = True,
) -> Dict[str, Any]:
    clean_initial_near_pairs_raw: set[Tuple[str, str]] = set()
    try:
        clean_base_world = loadFromDict(world_dict["world"])
        clean_initial_near_pairs_raw |= initial_near_contact_pairs(clean_base_world, world_dict, thr)
        if has_tool:
            clean_tool_world = loadFromDict(world_dict["world"])
            if place_ball_tool(clean_tool_world, placement_xy, 36.0, "downward", "orange"):
                clean_initial_near_pairs_raw |= initial_near_contact_pairs(clean_tool_world, world_dict, thr)
    except Exception:
        pass
    name_map = summary_name_map(
        world_dict=world_dict,
        target=target,
        goal=goal,
        blues=blues,
        static_black=static_black,
        env_set=env_set,
        env_id=env_id,
        has_tool=has_tool,
    )
    initial_near_pairs = {
        tuple(sorted((name_map.get(a, a), name_map.get(b, b))))
        for a, b in (tuple(pair) for pair in trace.get("initial_near_contact_pairs", []) or [])
    }
    initial_near_pairs_with_walls = {
        tuple(sorted((name_map.get(a, a), name_map.get(b, b))))
        for a, b in (tuple(pair) for pair in trace.get("initial_near_contact_pairs_with_walls", []) or [])
    }
    initial_near_pairs |= {
        tuple(sorted((name_map.get(a, a), name_map.get(b, b))))
        for a, b in clean_initial_near_pairs_raw
    }
    initial_near_pairs_with_walls |= initial_near_pairs
    object_kinematics = {
        name_map.get(target.name, target.name): object_kinematics_summary(trace, target.name, step),
    }
    if has_tool and TOOL_NAME in trace.get("path", {}):
        object_kinematics[name_map.get(TOOL_NAME, TOOL_NAME)] = object_kinematics_summary(trace, TOOL_NAME, step)
    if goal is not None and goal.is_movable:
        object_kinematics[name_map.get(goal.name, goal.name)] = object_kinematics_summary(trace, goal.name, step)
    for obj in [*blues, *static_black]:
        if obj.name in trace.get("path", {}):
            object_kinematics[name_map.get(obj.name, obj.name)] = object_kinematics_summary(trace, obj.name, step)

    events: List[Dict[str, Any]] = []
    if has_tool:
        events.append({"t": 0.0, "type": "placement", "actor": name_map.get(TOOL_NAME, TOOL_NAME), "stage": "early"})
    for item in sorted(trace.get("visible_contact_times", []) or [], key=lambda item: float(item.get("t", 0.0))):
        raw_a = str(item["a"])
        raw_b = str(item["b"])
        if goal is not None and goal.name in {raw_a, raw_b}:
            other_raw = raw_b if raw_a == goal.name else raw_a
            if overlaps_goal_interior_at_time(trace, other_raw, float(item.get("t", 0.0))):
                continue
        a, b = name_map.get(raw_a, raw_a), name_map.get(raw_b, raw_b)
        pair = tuple(sorted((a, b)))
        if pair in initial_near_pairs_with_walls:
            continue
        t_contact = rounded_time(item["t"])
        events.append({"t": t_contact, "type": "contact", "a": a, "b": b, "stage": stage_label(t_contact, trace.get("duration", 0.0) or 0.0)})
    motion_event_names: List[str] = [target.name, *[obj.name for obj in blues]]
    if has_tool and TOOL_NAME in trace.get("path", {}):
        motion_event_names.append(TOOL_NAME)
    if goal is not None and goal.is_movable and goal.name in trace.get("path", {}):
        motion_event_names.append(goal.name)
    for obj_name in motion_event_names:
        summary_obj_name = name_map.get(obj_name, obj_name)
        kin = object_kinematics.get(summary_obj_name, {})
        start_t = first_visible_motion_time(world_dict, trace.get("path", {}), trace.get("rot", {}), obj_name, step, thr.first_move_px)
        if start_t is not None:
            event_t = rounded_time(start_t)
            disp_xy = kin.get("displacement_xy") or [0.0, 0.0]
            move_direction = dominant_displacement_direction(float(disp_xy[0]), float(disp_xy[1]), thr.yes_px)
            events.append({
                "t": event_t,
                "type": "move_start",
                "obj": summary_obj_name,
                "movement_direction": move_direction,
                "max_displacement_px": round(float(kin.get("displacement_px", 0.0) or 0.0), 3),
                "stage": stage_label(event_t, trace.get("duration", 0.0) or 0.0),
            })
        rot_t = rotation_start_time(trace.get("rot", {}), obj_name, step, thr)
        if rot_t is not None:
            event_t = rounded_time(rot_t)
            events.append({
                "t": event_t,
                "type": "rotation_start",
                "obj": summary_obj_name,
                "rotation_direction": kin.get("max_rotation_direction"),
                "max_angular_displacement_deg": round(float(kin.get("max_angular_displacement_deg", 0.0) or 0.0), 3),
                "stage": stage_label(event_t, trace.get("duration", 0.0) or 0.0),
            })
    for start, _end in trace.get("goal_touch_intervals", []) or []:
        if goal is not None:
            event_t = rounded_time(start)
            events.append({
                "t": event_t,
                "type": "enter_goal",
                "obj": name_map.get(target.name, target.name),
                "goal": name_map.get(goal.name, goal.name),
                "stage": stage_label(event_t, trace.get("duration", 0.0) or 0.0),
            })
        break
    hold_t = goal_hold_success_time(trace, thr)
    if hold_t is not None and goal is not None:
        event_t = rounded_time(hold_t)
        events.append({
            "t": event_t,
            "type": "goal_hold_success",
            "obj": name_map.get(target.name, target.name),
            "goal": name_map.get(goal.name, goal.name),
            "stage": stage_label(event_t, trace.get("duration", 0.0) or 0.0),
        })
    settle_t = float(trace.get("duration", 0.0) or 0.0)
    events.append({"t": rounded_time(settle_t), "type": "settled", "stage": "late"})
    events = sorted(events, key=lambda item: (float(item.get("t", 0.0)), str(item.get("type", ""))))

    blue_names_raw = [obj.name for obj in blues]
    black_names_raw = [obj.name for obj in static_black]
    target_in_goal_final = object_final_state_flags(
        trace=trace,
        world_dict=world_dict,
        name=target.name,
        goal_name=(goal.name if goal else None),
        target_name=target.name,
        blue_names=blue_names_raw,
        black_names=black_names_raw,
        name_map=name_map,
        is_target=True,
        thr=thr,
    )["in_goal"]
    final_state = {
        "target_in_goal": target_in_goal_final,
        "target_entered_goal": entered_goal_label(trace, thr) is True,
        "target_on_floor": any(name in {"_BottomWall", "bottom_wall"} for name in final_contacts_for(trace, target.name)),
        "target_on_blue": any(b.name in final_contacts_for(trace, target.name) for b in blues),
        "target_touching_goal": bool(goal) and goal.name in final_contacts_for(trace, target.name) and not target_in_goal_final,
        "target_touching_goal_exterior": bool(goal) and goal.name in final_contacts_for(trace, target.name) and not target_in_goal_final,
        "target_touching_black": any(s.name in final_contacts_for(trace, target.name) for s in static_black),
    }
    final_state_by_object: Dict[str, Dict[str, bool]] = {
        name_map.get(target.name, target.name): object_final_state_flags(
            trace=trace,
            world_dict=world_dict,
            name=target.name,
            goal_name=(goal.name if goal else None),
            target_name=target.name,
            blue_names=blue_names_raw,
            black_names=black_names_raw,
            name_map=name_map,
            is_target=True,
            thr=thr,
        )
    }
    if has_tool and TOOL_NAME in trace.get("path", {}):
        final_state_by_object[name_map.get(TOOL_NAME, TOOL_NAME)] = object_final_state_flags(
            trace=trace,
            world_dict=world_dict,
            name=TOOL_NAME,
            goal_name=(goal.name if goal else None),
            target_name=target.name,
            blue_names=blue_names_raw,
            black_names=black_names_raw,
            name_map=name_map,
            is_target=False,
            thr=thr,
        )
    for obj in blues:
        final_state_by_object[name_map.get(obj.name, obj.name)] = object_final_state_flags(
            trace=trace,
            world_dict=world_dict,
            name=obj.name,
            goal_name=(goal.name if goal else None),
            target_name=target.name,
            blue_names=blue_names_raw,
            black_names=black_names_raw,
            name_map=name_map,
            is_target=False,
            thr=thr,
        )
    return {
        "placement": [int(round(float(placement_xy[0]))), int(round(float(placement_xy[1])))],
        "objects": {
            "tool": name_map.get(TOOL_NAME, TOOL_NAME) if has_tool else None,
            "target": name_map.get(target.name, target.name),
            "blue_objects": [name_map.get(obj.name, obj.name) for obj in blues],
            "goal": name_map.get(goal.name, goal.name) if goal else None,
            "black_static_objects": [name_map.get(obj.name, obj.name) for obj in static_black],
        },
        "events": events,
        "final_state": final_state,
        "final_state_by_object": final_state_by_object,
        "final_contacts": {
            "tool": [name_map.get(name, name) for name in final_contacts_for(trace, TOOL_NAME)],
            "target": [name_map.get(name, name) for name in final_contacts_for(trace, target.name)],
            "goal": [name_map.get(name, name) for name in final_contacts_for(trace, goal.name)] if goal else [],
        },
        "object_kinematics": object_kinematics,
    }


def event_alignment_key(event: Dict[str, Any]) -> str:
    etype = str(event.get("type", ""))
    if etype == "contact":
        a = str(event.get("a", ""))
        b = str(event.get("b", ""))
        return f"contact:{'|'.join(sorted((a, b)))}"
    if etype in {"move_start", "rotation_start"}:
        return f"{etype}:{event.get('obj')}"
    if etype in {"enter_goal", "goal_hold_success"}:
        return f"{etype}:{event.get('obj')}->{event.get('goal')}"
    if etype == "placement":
        return "placement"
    if etype == "settled":
        return "settled"
    return f"{etype}:{json.dumps(event, sort_keys=True)}"


def human_object_name(name: str) -> str:
    raw = str(name)
    specials = {
        "PlacedTool": "the big orange ball",
        "RedTarget": "the red target ball",
        "GoalContainer": "the green goal container",
        "Floor": "the floor",
        "TopWall": "the ceiling",
        "LeftWall": "the left wall",
        "RightWall": "the right wall",
        "BlackSupportObjectBlackTrapezoidOrBlackRectangleConnectedToGreenGoalContainerOrShortBlackPillar": "at least one of the black support objects (the black trapezoid or black rectangle connected to the green goal container, or the short black pillar)",
    }
    if raw in specials:
        return specials[raw]
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", raw).replace("_", " ").strip().lower()
    return f"the {words}"


def prettify_flag_object_name(token: str) -> str:
    text = str(token).replace("_", " ").strip().lower()
    replacements = [
        ("blacksupportobjectblacktrapezoidorblackrectangleconnectedtogreengoalcontainerorshortblackpillar", "at least one of the black support objects (the black trapezoid or black rectangle connected to the green goal container, or the short black pillar)"),
        ("blackpillar", "black pillar"),
        ("bigblackrectangleleft", "big black rectangle attached to the left"),
        ("smallblackrectanglemiddleright", "small black rectangle in the middle/right"),
        ("blackrectangle", "black rectangle"),
        ("blacktriangle", "black triangle"),
        ("blackobstacle", "black obstacle"),
        ("blueplank", "blue plank"),
        ("bluesquare", "blue square"),
        ("bluerectangle", "blue rectangle"),
        ("bluetrapezoid", "blue trapezoid"),
        ("goalcontainer", "green goal container"),
        ("redtarget", "red target ball"),
        ("placedtool", "big orange ball"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def object_priority(name: str) -> int:
    raw = str(name)
    if raw == "RedTarget":
        return 10
    if raw == "PlacedTool":
        return 9
    if raw.startswith("Blue"):
        return 8
    if raw == "GoalContainer":
        return 6
    if "Black" in raw:
        return 5
    if raw in {"Floor", "TopWall", "LeftWall", "RightWall"}:
        return 0
    return 3


def align_rollout_events(
    rollout_summaries: List[Dict[str, Any]],
    *,
    thr: Thresholds,
) -> Dict[str, Any]:
    total = len(rollout_summaries)
    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, summary in enumerate(rollout_summaries):
        for event in summary.get("events", []) or []:
            key = event_alignment_key(event)
            bucket = grouped.setdefault(
                key,
                {
                    "event_key": key,
                    "event_type": event.get("type"),
                    "prototype": event,
                    "times": [],
                    "rollout_indices": [],
                },
            )
            bucket["times"].append(float(event.get("t", 0.0) or 0.0))
            bucket["rollout_indices"].append(idx)
    aligned_events: List[Dict[str, Any]] = []
    stable_events: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        times = [float(t) for t in bucket["times"]]
        count = len(times)
        probability = float(count) / float(max(1, total))
        time_mean = float(np.mean(times)) if times else None
        time_std = float(np.std(times)) if len(times) >= 2 else 0.0
        stable = (
            count == total
            and time_mean is not None
            and time_std <= thr.event_time_std_max_s
        )
        item = {
            "event_key": bucket["event_key"],
            "event_type": bucket["event_type"],
            "prototype": bucket["prototype"],
            "count": count,
            "probability": probability,
            "time_mean_s": round(time_mean, 4) if time_mean is not None else None,
            "time_std_s": round(time_std, 4) if time_std is not None else None,
            "rollout_indices": bucket["rollout_indices"],
            "stable": stable,
        }
        aligned_events.append(item)
        if stable:
            stable_events.append(item)
    aligned_events.sort(key=lambda item: (-item["count"], item["time_mean_s"] if item["time_mean_s"] is not None else 1e9, item["event_key"]))
    stable_events.sort(key=lambda item: (item["time_mean_s"] if item["time_mean_s"] is not None else 1e9, item["event_key"]))
    return {
        "num_rollouts": total,
        "stability_rule": {
            "min_frequency": f"{total}/{total}",
            "max_time_std_s": thr.event_time_std_max_s,
            "semantics": "Only keep rollout events that appear in every noisy rollout and have temporally consistent timings.",
        },
        "aligned_events": aligned_events,
        "stable_events": stable_events,
    }


def yes_no_answer(values: List[Any]) -> Optional[bool]:
    if not values:
        return None
    if all(v is True for v in values):
        return True
    if all(v is False for v in values):
        return False
    return None


def suppress_floor_questions(env_set: Optional[str], env_id: Optional[str]) -> bool:
    env_set = str(env_set or "")
    env_id = str(env_id or "")
    if env_set in {"Prevention", "BalanceUnder"}:
        return True
    return (env_set, env_id) in {
        ("BackUp", "3"),
        ("Falling", "7"),
        ("Falling", "8"),
        ("Falling", "9"),
    }


def contact_question_text(a: str, b: str, stage: Optional[str], env_set: Optional[str] = None, env_id: Optional[str] = None) -> Optional[str]:
    del stage
    if object_priority(a) >= object_priority(b):
        subject, other = a, b
    else:
        subject, other = b, a
    if "Floor" in {subject, other} and suppress_floor_questions(env_set, env_id):
        return None
    if other == "GoalContainer":
        if env_set in {"Remove", "BalanceUnder", "Prevention"}:
            return None
        return f"Does {human_object_name(subject)} physically collide with or touch the outside wall of the green goal container at any point?"
    if subject == "GoalContainer":
        if env_set in {"Remove", "BalanceUnder", "Prevention"}:
            return None
        return f"Does {human_object_name(other)} physically collide with or touch the outside wall of the green goal container at any point?"
    return f"Does {human_object_name(subject)} physically collide with or touch {human_object_name(other)} at any point?"


def move_direction_question_text(obj: str, direction: str, stage: Optional[str]) -> str:
    del stage
    return f"Does {human_object_name(obj)} move {direction} in a clearly visible way?"


def rotation_question_text(obj: str, direction: str, stage: Optional[str]) -> str:
    del stage
    return f"Does {human_object_name(obj)} rotate {direction} by at least 15 degrees?"


def event_phrase(event: Dict[str, Any]) -> Optional[str]:
    etype = str(event.get("type", ""))
    if etype == "contact":
        return f"{event.get('a')} touching {event.get('b')}"
    if etype == "move_start":
        direction = event.get("movement_direction")
        if direction:
            return f"{event.get('obj')} starting to move {direction}"
        return f"{event.get('obj')} starting to move"
    if etype == "rotation_start":
        direction = event.get("rotation_direction")
        if direction and direction != "none":
            return f"{event.get('obj')} starting to rotate {direction}"
        return f"{event.get('obj')} starting to rotate"
    if etype == "enter_goal":
        return f"{event.get('obj')} entering {event.get('goal')}"
    if etype == "goal_hold_success":
        return f"{event.get('obj')} successfully staying in {event.get('goal')}"
    return None


def event_objects(event: Dict[str, Any]) -> List[str]:
    etype = str(event.get("type", ""))
    if etype == "contact":
        return [str(event.get("a", "")), str(event.get("b", ""))]
    if etype in {"move_start", "rotation_start"}:
        return [str(event.get("obj", ""))]
    if etype in {"enter_goal", "goal_hold_success"}:
        return [str(event.get("obj", "")), str(event.get("goal", ""))]
    if etype == "placement":
        return [str(event.get("actor", ""))]
    return []


def action_phrase_for_temporal(event: Dict[str, Any], common_obj: str, env_set: Optional[str] = None, env_id: Optional[str] = None) -> Optional[str]:
    etype = str(event.get("type", ""))
    if etype == "move_start" and str(event.get("obj")) == common_obj:
        direction = event.get("movement_direction")
        return f"move {direction}" if direction else "start moving"
    if etype == "rotation_start" and str(event.get("obj")) == common_obj:
        direction = event.get("rotation_direction")
        if direction and direction != "none":
            return f"start rotating {direction}"
        return "start rotating"
    if etype == "contact" and common_obj in {str(event.get("a")), str(event.get("b"))}:
        other = str(event.get("b")) if str(event.get("a")) == common_obj else str(event.get("a"))
        if other == "Floor" and suppress_floor_questions(env_set, env_id):
            return None
        if other == "GoalContainer":
            if env_set in {"Remove", "BalanceUnder", "Prevention"}:
                return None
            return "touch the outside wall of the green goal container"
        return f"touch {human_object_name(other)}"
    if etype == "enter_goal" and str(event.get("obj")) == common_obj:
        return f"enter {human_object_name(str(event.get('goal')))}"
    if etype == "goal_hold_success" and str(event.get("obj")) == common_obj:
        return f"stay inside {human_object_name(str(event.get('goal')))} successfully"
    return None


def conjugate_action_phrase(action: str) -> str:
    text = str(action).strip()
    if text.startswith("touch "):
        return "touches " + text[len("touch "):]
    if text.startswith("move "):
        return "moves " + text[len("move "):]
    if text.startswith("enter "):
        return "enters " + text[len("enter "):]
    if text.startswith("stay "):
        return "stays " + text[len("stay "):]
    if text.startswith("start rotating "):
        return "starts rotating " + text[len("start rotating "):]
    if text == "start rotating":
        return "starts rotating"
    if text == "start moving":
        return "starts moving"
    return text


def temporal_question_from_events(first_event: Dict[str, Any], second_event: Dict[str, Any], env_set: Optional[str] = None, env_id: Optional[str] = None) -> Optional[str]:
    banned = {"Floor", "TopWall", "LeftWall", "RightWall"}
    if any(obj in banned for obj in event_objects(first_event) + event_objects(second_event)):
        if any(obj == "Floor" for obj in event_objects(first_event) + event_objects(second_event)) and not suppress_floor_questions(env_set, env_id):
            pass
        else:
            return None
    # "move ... before touch ..." is usually degenerate and not a useful
    # temporal reasoning question for these gravity-driven scenes.
    if str(first_event.get("type")) == "move_start" and str(second_event.get("type")) == "contact":
        return None
    if str(first_event.get("type")) == "contact" and str(second_event.get("type")) == "move_start":
        return None
    first_objs = set(event_objects(first_event))
    second_objs = set(event_objects(second_event))
    shared = [
        obj
        for obj in sorted(first_objs & second_objs, key=lambda name: -object_priority(name))
        if obj not in {"GoalContainer", "Floor", "TopWall", "LeftWall", "RightWall"}
    ]
    if not shared:
        return None
    common_obj = shared[0]
    if "Black" in str(common_obj):
        return None
    first_action = action_phrase_for_temporal(first_event, common_obj, env_set, env_id)
    second_action = action_phrase_for_temporal(second_event, common_obj, env_set, env_id)
    if not first_action or not second_action:
        return None
    return f"Does {human_object_name(common_obj)} {first_action} before it {conjugate_action_phrase(second_action)}?"


def bbox_from_spec(spec: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    pts = spec.get("vertices") or spec.get("points")
    if spec.get("polys"):
        pts = [pt for poly in spec.get("polys", []) for pt in poly]
    if not isinstance(pts, list) or len(pts) < 2:
        center = object_center(spec)
        radius = float(spec.get("radius", 0.0) or 0.0)
        if center and radius > 0:
            cx, cy = center
            return (cx - radius, cy - radius, cx + radius, cy + radius)
        return None
    xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [float(p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def final_state_question_text(obj: str, flag: str, env_set: Optional[str] = None, env_id: Optional[str] = None) -> Optional[str]:
    if flag == "in_goal":
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} inside the green goal container? Partly inside counts."
    if flag == "on_floor":
        if suppress_floor_questions(env_set, env_id):
            return None
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} on the floor?"
    if flag == "on_ceiling":
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} on the ceiling?"
    if flag == "touching_goal":
        if env_set in {"Remove", "BalanceUnder", "Prevention"}:
            return None
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching the outside wall of the green goal container?"
    if flag == "touching_goal_exterior":
        if env_set in {"Remove", "BalanceUnder", "Prevention"}:
            return None
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching the outside wall of the green goal container?"
    if flag == "touching_floor":
        if suppress_floor_questions(env_set, env_id):
            return None
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching the floor?"
    if flag == "touching_ceiling":
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching the ceiling?"
    if flag == "touching_target":
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching the red target ball?"
    if flag == "touching_placed_tool":
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching the big orange ball?"
    if flag.startswith("touching_"):
        other = prettify_flag_object_name(flag[len("touching_"):])
        if other in {"red target ball", "big orange ball", "blue plank", "blue square", "blue rectangle", "blue trapezoid", "blue object", "green goal container"}:
            if human_object_name(obj) == f"the {other}":
                return None
        if other.startswith("any "):
            return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching {other}?"
        return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} touching the {other}?"
    return None


def settled_displacement_question_text(obj: str, direction: str, pixels: int) -> str:
    return f"After the scene has settled and all motion has stopped, is {human_object_name(obj)} about {pixels} pixels {direction} from its starting position?"


def candidate_contact_pairs(
    *,
    target: ObjInfo,
    goal: Optional[ObjInfo],
    blues: List[ObjInfo],
    static_black: List[ObjInfo],
    has_tool: bool,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if has_tool:
        pairs.append((TOOL_NAME, target.name))
        if goal is not None:
            pairs.append((TOOL_NAME, goal.name))
        for b in blues:
            pairs.append((TOOL_NAME, b.name))
        for blk in static_black:
            pairs.append((TOOL_NAME, blk.name))
    if goal is not None:
        pairs.append((target.name, goal.name))
    for b in blues:
        pairs.append((target.name, b.name))
        if goal is not None:
            pairs.append((b.name, goal.name))
        for blk in static_black:
            pairs.append((b.name, blk.name))
    for blk in static_black:
        pairs.append((target.name, blk.name))
    for i, b1 in enumerate(blues):
        for b2 in blues[i + 1:]:
            pairs.append((b1.name, b2.name))
    seen: set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for a, b in pairs:
        pair = tuple(sorted((a, b)))
        if pair not in seen and a != b:
            seen.add(pair)
            out.append((a, b))
    return out


def rounded_displacement_target(values: List[float], base: int = 10) -> Optional[int]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    if np.max(arr) - np.min(arr) > 25.0:
        return None
    mean_val = float(np.mean(arr))
    rounded = int(base * round(mean_val / float(base)))
    return rounded if rounded >= base else None


def add_final_displacement_questions(
    *,
    questions: List[Dict[str, Any]],
    seen_questions: set[Tuple[str, str, Optional[bool]]],
    rollout_summaries: List[Dict[str, Any]],
) -> None:
    def add_question(category: str, question: Optional[str], answer: Optional[bool], evidence: Dict[str, Any]) -> None:
        if answer is None or not question:
            return
        key = (category, question, answer)
        if key in seen_questions:
            return
        seen_questions.add(key)
        questions.append({
            "category": category,
            "question": question,
            "answer": bool(answer),
            "evidence": evidence,
        })

    first_summary = rollout_summaries[0]
    for obj_name in (first_summary.get("object_kinematics") or {}).keys():
        dxs: List[float] = []
        dys: List[float] = []
        for summary in rollout_summaries:
            kin = ((summary.get("object_kinematics") or {}).get(obj_name) or {})
            disp_xy = kin.get("displacement_xy") or [0.0, 0.0]
            dxs.append(float(disp_xy[0] or 0.0))
            dys.append(float(disp_xy[1] or 0.0))
        mean_dx = float(np.mean(dxs))
        mean_dy = float(np.mean(dys))
        if max(abs(mean_dx), abs(mean_dy)) < 30.0:
            continue
        if abs(mean_dx) >= abs(mean_dy):
            direction = "right" if mean_dx > 0 else "left"
            rounded = rounded_displacement_target([abs(v) for v in dxs])
            opposite = "left" if direction == "right" else "right"
            axis_vals = dxs
        else:
            direction = "up" if mean_dy > 0 else "down"
            rounded = rounded_displacement_target([abs(v) for v in dys])
            opposite = "down" if direction == "up" else "up"
            axis_vals = dys
        if rounded is None:
            continue
        add_question(
            "final_state",
            settled_displacement_question_text(obj_name, direction, rounded),
            True,
            {"object": obj_name, "axis_values": [round(float(v), 3) for v in axis_vals], "rounded_pixels": rounded, "direction": direction},
        )
        add_question(
            "final_state",
            settled_displacement_question_text(obj_name, opposite, rounded),
            False,
            {"object": obj_name, "rounded_pixels": rounded, "direction": direction, "negated_with": "opposite_direction"},
        )


def choose_question_subset(
    all_questions: List[Dict[str, Any]],
    *,
    max_questions: int = 4,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    movable_priority = ["RedTarget", "BluePlank", "BlueRectangle", "BlueSquare", "BlueTrapezoid", "GoalContainer", "PlacedTool"]

    def canonical_slot(q: Dict[str, Any]) -> str:
        category = str(q.get("category", ""))
        evidence = q.get("evidence") or {}
        stage = str(evidence.get("stage", ""))
        if category == "temporal_order":
            return "temporal_order"
        if category == "final_state":
            return "final_state"
        if category.startswith("mechanism_"):
            if stage == "late":
                return "late"
            if stage == "middle":
                return "middle"
            return "early"
        return "early"

    def mentioned_objects(q: Dict[str, Any]) -> set[str]:
        text = str(q.get("question", ""))
        names = set()
        for canonical in (*movable_priority, "GoalContainer"):
            human = human_object_name(canonical)
            if human in text:
                names.add(canonical)
        evidence = q.get("evidence") or {}
        for key in ("object",):
            val = evidence.get(key)
            if isinstance(val, str):
                names.add(val)
        for key in ("event_key", "first_event_key", "second_event_key", "paired_event_key"):
            val = str(evidence.get(key, ""))
            for canonical in (*movable_priority, "GoalContainer"):
                if canonical in val:
                    names.add(canonical)
        return names

    def temporal_signature(q: Dict[str, Any]) -> Optional[frozenset[str]]:
        if str(q.get("category", "")) != "temporal_order":
            return None
        evidence = q.get("evidence") or {}
        a = str(evidence.get("first_event_key") or "")
        b = str(evidence.get("second_event_key") or "")
        if not a or not b:
            return None
        return frozenset({a, b})

    def is_murky_contact(q: Dict[str, Any]) -> bool:
        category = str(q.get("category", ""))
        if category != "mechanism_contact":
            return False
        text = str(q.get("question", "")).lower()
        if "outside wall of the green goal container" in text and any(token in text for token in ("blue plank", "blue rectangle", "blue square", "blue trapezoid")):
            return True
        return False

    def preference_score(q: Dict[str, Any]) -> float:
        question = str(q.get("question", "")).lower()
        category = str(q.get("category", ""))
        evidence = q.get("evidence") or {}
        score = 0.0
        if "redtarget" in question:
            score += 4.0
        if "goalcontainer" in question or "green goal container" in question:
            score += 3.0
        if "blue" in question:
            score += 2.5
        if "placedtool" in question or "placed tool" in question:
            score += 2.0
        if "black" in question:
            score += 1.5
        if any(token in question for token in ("floor", "leftwall", "rightwall", "topwall", "ceiling")):
            score -= 3.0
        if category == "final_state":
            flag = str(evidence.get("flag", ""))
            if flag in {"in_goal", "touching_goal"}:
                score += 4.0
            if flag in {"on_floor", "touching_floor", "on_ceiling", "touching_ceiling"}:
                score -= 1.5
            if "rounded_pixels" in evidence:
                score += 2.5
            score -= 0.75
        if category == "temporal_order":
            gap = float(evidence.get("time_gap_s", 0.0) or 0.0)
            score += min(gap, 3.0)
        if category.startswith("mechanism_"):
            stage = str(evidence.get("stage", ""))
            if stage == "middle":
                score += 1.0
            if stage == "early":
                score += 0.5
        if category == "mechanism_contact" and any(token in question for token in ("floor", "wall", "ceiling")):
            score -= 2.5
        if category == "mechanism_rotation":
            score += 3.5
        if category == "mechanism_motion":
            score += 4.0
            if "big orange ball" in question:
                score += 1.0
            if "green goal container" in question:
                score += 1.5
        if is_murky_contact(q):
            score -= 8.0
        return score

    def movable_mentions(q: Dict[str, Any]) -> List[str]:
        return [obj for obj in movable_priority if obj in mentioned_objects(q)]

    def mechanism_time(q: Dict[str, Any]) -> float:
        evidence = q.get("evidence") or {}
        t = evidence.get("t_mean_s")
        try:
            if t is not None:
                return float(t)
        except Exception:
            pass
        return 1e9

    def category_rank(cat: str) -> int:
        if cat == "mechanism_rotation":
            return 0
        if cat == "mechanism_motion":
            return 1
        if cat == "mechanism_contact":
            return 2
        if cat == "temporal_order":
            return 3
        return 4

    def slot_penalty(q: Dict[str, Any], slot_name: str, already_selected: List[Dict[str, Any]]) -> float:
        penalty = 0.0
        q_objects = set(movable_mentions(q))
        selected_objects = set().union(*(movable_mentions(item) for item in already_selected)) if already_selected else set()
        if slot_name in {"early", "middle"}:
            if str(q.get("category", "")).startswith("mechanism_"):
                penalty -= 2.0
            else:
                penalty += 10.0
        if slot_name == "middle":
            t = mechanism_time(q)
            if t < 0.75:
                penalty += 4.0
            elif t < 1.5:
                penalty += 1.0
            if q_objects and q_objects.issubset(selected_objects):
                penalty += 2.5
        if slot_name == "final_state":
            if str(q.get("category", "")) != "final_state":
                penalty += 8.0
        if slot_name == "temporal_order":
            if str(q.get("category", "")) != "temporal_order":
                penalty += 8.0
        return penalty

    def candidate_sort_key(q: Dict[str, Any], slot_name: str, already_selected: List[Dict[str, Any]]) -> Tuple[float, float, int]:
        score = preference_score(q) - slot_penalty(q, slot_name, already_selected)
        return (
            score,
            -mechanism_time(q) if slot_name == "middle" else 0.0,
            -category_rank(str(q.get("category", ""))),
        )

    def by_cat(cat: str) -> List[Dict[str, Any]]:
        return [q for q in all_questions if str(q.get("category")) == cat]

    mechanism_pool = [
        q for q in all_questions
        if str(q.get("category", "")).startswith("mechanism_")
    ]
    early_pool = [
        q for q in mechanism_pool
        if str((q.get("evidence") or {}).get("stage")) == "early"
    ]
    middle_pool = [
        q for q in mechanism_pool
        if str((q.get("evidence") or {}).get("stage")) == "middle"
    ]
    if not early_pool:
        early_pool = sorted(mechanism_pool, key=mechanism_time)[: max(1, min(6, len(mechanism_pool)))]
    if not middle_pool:
        middle_pool = [q for q in mechanism_pool if q not in early_pool]
    if not middle_pool:
        middle_pool = mechanism_pool
    temporal_pool = by_cat("temporal_order")
    final_pool = by_cat("final_state")

    def pick(
        pool: List[Dict[str, Any]],
        used_questions: set[str],
        already_selected: List[Dict[str, Any]],
        slot_name: str,
        target_answer: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        used_temporal = {temporal_signature(item) for item in already_selected if temporal_signature(item) is not None}
        candidates = [q for q in pool if q.get("question") not in used_questions and not is_murky_contact(q)]
        candidates = [q for q in candidates if temporal_signature(q) is None or temporal_signature(q) not in used_temporal]
        if target_answer is not None:
            preferred = [q for q in candidates if bool(q.get("answer")) is target_answer]
            if preferred:
                candidates = preferred
        if not candidates:
            return None
        # Uniformly sample within the valid slot/category pool so we cover a
        # broader question space across attempts and puzzles.
        return rng.choice(candidates)

    selected: List[Dict[str, Any]] = []
    used_questions: set[str] = set()
    answer_targets = [True, False, True, False]
    slots: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("early", early_pool),
        ("middle", middle_pool),
        ("temporal_order", temporal_pool),
        ("final_state", final_pool),
    ]
    for idx, (slot_name, pool) in enumerate(slots):
        item = pick(pool, used_questions, selected, slot_name, answer_targets[idx])
        if item is None and slot_name == "temporal_order":
            fallback_pool = middle_pool + [q for q in mechanism_pool if canonical_slot(q) == "late"] + final_pool
            item = pick(fallback_pool, used_questions, selected, slot_name, answer_targets[idx])
        if item is None:
            continue
        item = dict(item)
        item["selected_slot"] = canonical_slot(item)
        selected.append(item)
        used_questions.add(str(item.get("question")))
        if len(selected) >= max_questions:
            break

    covered = set().union(*(mentioned_objects(q) for q in selected)) if selected else set()
    red_candidates = [q for q in all_questions if "RedTarget" in mentioned_objects(q) and q.get("question") not in used_questions and not is_murky_contact(q)]
    if "RedTarget" not in covered and red_candidates:
        replacement_index = None
        replacement_score = None
        for idx, q in enumerate(selected):
            score = preference_score(q)
            if replacement_index is None or score < replacement_score:
                replacement_index = idx
                replacement_score = score
        if replacement_index is not None:
            best_red = rng.choice(red_candidates)
            best_red = dict(best_red)
            best_red["selected_slot"] = canonical_slot(best_red)
            selected[replacement_index] = best_red
            used_questions.add(str(best_red.get("question")))

    covered = set().union(*(mentioned_objects(q) for q in selected)) if selected else set()
    movable_covered = [obj for obj in movable_priority if obj in covered]
    if len(movable_covered) < 2:
        coverage_candidates = [
            q for q in all_questions
            if q.get("question") not in {item.get("question") for item in selected}
            and not is_murky_contact(q)
            and any(obj in mentioned_objects(q) for obj in movable_priority)
        ]
        if coverage_candidates and selected:
            best_cov = rng.choice(coverage_candidates)
            replace_idx = min(range(len(selected)), key=lambda idx: preference_score(selected[idx]))
            best_cov = dict(best_cov)
            best_cov["selected_slot"] = canonical_slot(best_cov)
            selected[replace_idx] = best_cov
            used_questions.add(str(best_cov.get("question")))

    final_count = sum(1 for q in selected if str(q.get("category", "")) == "final_state")
    if final_count > 1:
        replacement_pool = [
            q for q in mechanism_pool + temporal_pool
            if q.get("question") not in {item.get("question") for item in selected}
            and not is_murky_contact(q)
        ]
        for idx, item in enumerate(list(selected)):
            if final_count <= 1:
                break
            if str(item.get("category", "")) != "final_state":
                continue
            replacement_candidates = [
                q for q in replacement_pool
                if not set(movable_mentions(q)).issubset(set().union(*(movable_mentions(s) for j, s in enumerate(selected) if j != idx)))
                or str(q.get("category", "")) == "temporal_order"
            ] or replacement_pool
            if not replacement_candidates:
                break
            replacement = rng.choice(replacement_candidates)
            replacement = dict(replacement)
            replacement["selected_slot"] = canonical_slot(replacement)
            selected[idx] = replacement
            replacement_pool = [q for q in replacement_pool if q.get("question") != replacement.get("question")]
            final_count -= 1

    if len(selected) >= max_questions:
        return selected[:max_questions]

    leftovers = sorted(
        [q for q in all_questions if q.get("question") not in used_questions and not is_murky_contact(q)],
        key=preference_score,
        reverse=True,
    )
    for q in leftovers:
        item = dict(q)
        item["selected_slot"] = canonical_slot(item)
        selected.append(item)
        used_questions.add(str(item.get("question")))
        if len(selected) >= max_questions:
            break
    return selected


def generate_event_graph_questions(
    *,
    rollout_summaries: List[Dict[str, Any]],
    aligned_event_graph: Dict[str, Any],
    world_dict: Dict[str, Any],
    target: ObjInfo,
    goal: Optional[ObjInfo],
    blues: List[ObjInfo],
    static_black: List[ObjInfo],
    env_set: str,
    env_id: str,
    clean_base_world: Any,
    clean_tool_world: Any,
    thr: Thresholds,
    has_tool: bool,
) -> List[Dict[str, Any]]:
    if not rollout_summaries:
        return []
    first_summary = rollout_summaries[0]
    name_map = summary_name_map(
        world_dict=world_dict,
        target=target,
        goal=goal,
        blues=blues,
        static_black=static_black,
        env_set=env_set,
        env_id=env_id,
        has_tool=has_tool,
    )
    inv_name_map = {v: k for k, v in name_map.items()}
    raw_objects = (world_dict.get("world", {}).get("objects", {}) or {})

    def is_ball_summary_name(summary_name: str) -> bool:
        raw_name = inv_name_map.get(summary_name, summary_name)
        spec = raw_objects.get(raw_name, {}) or {}
        return str(spec.get("type", "")).lower() == "ball" or "radius" in spec

    stable_events = aligned_event_graph.get("stable_events", []) or []
    aligned_events = aligned_event_graph.get("aligned_events", []) or []
    aligned_by_key = {str(item.get("event_key")): item for item in aligned_events}
    questions: List[Dict[str, Any]] = []
    seen_questions: set[Tuple[str, str, Optional[bool]]] = set()

    def add_question(category: str, question: Optional[str], answer: Optional[bool], evidence: Dict[str, Any]) -> None:
        if answer is None or not question:
            return
        key = (category, question, answer)
        if key in seen_questions:
            return
        seen_questions.add(key)
        questions.append({
            "category": category,
            "question": question,
            "answer": bool(answer),
            "evidence": evidence,
        })

    stable_contact_keys = {
        str(item.get("event_key"))
        for item in stable_events
        if str(item.get("event_type")) == "contact"
    }
    for item in stable_events:
        prototype = dict(item.get("prototype") or {})
        etype = str(item.get("event_type"))
        stage = prototype.get("stage")
        if etype == "contact":
            add_question(
                "mechanism_contact",
                contact_question_text(str(prototype.get("a")), str(prototype.get("b")), stage, env_set, env_id),
                True,
                {"event_key": item.get("event_key"), "t_mean_s": item.get("time_mean_s"), "stage": stage},
            )
        elif etype == "move_start":
            direction = prototype.get("movement_direction")
            max_disp = float(prototype.get("max_displacement_px", 0.0) or 0.0)
            if direction and max_disp >= thr.yes_px:
                add_question(
                    "mechanism_motion",
                    move_direction_question_text(str(prototype.get("obj")), str(direction), stage),
                    True,
                    {"event_key": item.get("event_key"), "t_mean_s": item.get("time_mean_s"), "stage": stage, "max_displacement_px": max_disp},
                )
                opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}.get(str(direction))
                if opposite:
                    add_question(
                        "mechanism_motion",
                        move_direction_question_text(str(prototype.get("obj")), opposite, stage),
                        False,
                        {"paired_event_key": item.get("event_key"), "t_mean_s": item.get("time_mean_s"), "stage": stage, "max_displacement_px": max_disp},
                    )
        elif etype == "rotation_start":
            if is_ball_summary_name(str(prototype.get("obj"))):
                continue
            direction = prototype.get("rotation_direction")
            max_angle = float(prototype.get("max_angular_displacement_deg", 0.0) or 0.0)
            if direction in {"clockwise", "counterclockwise"} and max_angle >= math.degrees(thr.rotation_rad):
                add_question(
                    "mechanism_rotation",
                    rotation_question_text(str(prototype.get("obj")), str(direction), stage),
                    True,
                    {"event_key": item.get("event_key"), "t_mean_s": item.get("time_mean_s"), "stage": stage, "max_angular_displacement_deg": max_angle},
                )
                opposite = "counterclockwise" if direction == "clockwise" else "clockwise"
                add_question(
                    "mechanism_rotation",
                    rotation_question_text(str(prototype.get("obj")), opposite, stage),
                        False,
                        {"paired_event_key": item.get("event_key"), "t_mean_s": item.get("time_mean_s"), "stage": stage, "max_angular_displacement_deg": max_angle},
                    )
        elif etype == "enter_goal":
            add_question(
                "mechanism_contact",
                f"Does {human_object_name(str(prototype.get('obj')))} touch or enter the inside of the green goal container at any point?",
                True,
                {"event_key": item.get("event_key"), "t_mean_s": item.get("time_mean_s"), "stage": stage},
            )

    # Also support explicit "no" rotation questions when an object never rotates
    # by a clearly visible amount in any rollout.
    first_object_kinematics = first_summary.get("object_kinematics") or {}
    for obj_name in first_object_kinematics.keys():
        if is_ball_summary_name(str(obj_name)):
            continue
        max_angles: List[float] = []
        max_dirs: List[str] = []
        for summary in rollout_summaries:
            kin = ((summary.get("object_kinematics") or {}).get(obj_name) or {})
            max_angles.append(float(kin.get("max_angular_displacement_deg", 0.0) or 0.0))
            max_dirs.append(str(kin.get("max_rotation_direction", "none") or "none"))
        if not max_angles:
            continue
        if all(angle < math.degrees(thr.rotation_rad) for angle in max_angles):
            for direction in ("clockwise", "counterclockwise"):
                add_question(
                    "mechanism_rotation",
                    rotation_question_text(str(obj_name), direction, None),
                    False,
                    {
                        "object": obj_name,
                        "t_mean_s": None,
                        "stage": None,
                        "max_angular_displacement_deg": round(float(max(max_angles) if max_angles else 0.0), 3),
                        "max_rotation_direction": "none",
                        "negated_with": "no_visible_rotation",
                    },
                )

    for raw_a, raw_b in candidate_contact_pairs(target=target, goal=goal, blues=blues, static_black=static_black, has_tool=has_tool):
        a = name_map.get(raw_a, raw_a)
        b = name_map.get(raw_b, raw_b)
        pair_key = f"contact:{'|'.join(sorted((a, b)))}"
        if pair_key in stable_contact_keys:
            continue
        near = (
            pair_too_close_for_contact(clean_tool_world if TOOL_NAME in {raw_a, raw_b} else clean_base_world,
                                       world_dict,
                                       raw_a,
                                       raw_b,
                                       thr,
                                       require_vertical_for_tool=(TOOL_NAME in {raw_a, raw_b}))
            or objects_in_contact(clean_tool_world if TOOL_NAME in {raw_a, raw_b} else clean_base_world, raw_a, raw_b)
        )
        if near:
            continue
        aligned_item = aligned_by_key.get(pair_key)
        if aligned_item is None or int(aligned_item.get("count", 0) or 0) == 0:
            add_question(
                "mechanism_contact",
                contact_question_text(a, b, None, env_set, env_id),
                False,
                {"event_key": pair_key, "t_mean_s": None, "stage": None},
            )

    stable_sorted = sorted(
        stable_events,
        key=lambda item: float(item.get("time_mean_s") if item.get("time_mean_s") is not None else 1e9),
    )

    world_objects = raw_objects

    def is_gravity_degenerate_temporal_pair(first_proto: Dict[str, Any], second_proto: Dict[str, Any]) -> bool:
        pairs = [(first_proto, second_proto), (second_proto, first_proto)]
        for move_evt, contact_evt in pairs:
            if str(move_evt.get("type")) != "move_start" or str(contact_evt.get("type")) != "contact":
                continue
            common_name = str(move_evt.get("obj") or "")
            if common_name not in {str(contact_evt.get("a") or ""), str(contact_evt.get("b") or "")}:
                continue
            if str(move_evt.get("movement_direction") or "") != "down":
                continue
            raw_common = inv_name_map.get(common_name, common_name)
            other_name = str(contact_evt.get("b") if str(contact_evt.get("a")) == common_name else contact_evt.get("a"))
            raw_other = inv_name_map.get(other_name, other_name)
            common_spec = world_objects.get(raw_common, {}) or {}
            other_spec = world_objects.get(raw_other, {}) or {}
            if _float(other_spec.get("density", 1.0), 1.0) > 0.0:
                continue
            if str(other_spec.get("color", "")).lower() != "black":
                continue
            common_box = bbox_from_spec(common_spec)
            other_box = bbox_from_spec(other_spec)
            if common_box is None or other_box is None:
                continue
            cminx, cminy, cmaxx, cmaxy = common_box
            ominx, ominy, omaxx, omaxy = other_box
            x_overlap = max(0.0, min(cmaxx, omaxx) - max(cminx, ominx))
            common_width = max(1.0, cmaxx - cminx)
            if x_overlap < 0.2 * common_width:
                continue
            if cminy >= omaxy:
                return True
        return False

    def temporal_event_is_visibly_strong(proto: Dict[str, Any]) -> bool:
        etype = str(proto.get("type") or "")
        if etype == "move_start":
            max_disp = float(proto.get("max_displacement_px", 0.0) or 0.0)
            return max_disp >= 40.0
        if etype == "rotation_start":
            max_angle = float(proto.get("max_angular_displacement_deg", 0.0) or 0.0)
            return max_angle >= 15.0
        return True

    temporal_seen_pairs: set[frozenset[str]] = set()
    for i, first in enumerate(stable_sorted):
        for second in stable_sorted[i + 1:]:
            first_proto = dict(first.get("prototype") or {})
            second_proto = dict(second.get("prototype") or {})
            if not temporal_event_is_visibly_strong(first_proto):
                continue
            if not temporal_event_is_visibly_strong(second_proto):
                continue
            if is_gravity_degenerate_temporal_pair(first_proto, second_proto):
                continue
            question_text = temporal_question_from_events(first_proto, second_proto, env_set, env_id)
            if not question_text:
                continue
            ta = float(first.get("time_mean_s") or 0.0)
            tb = float(second.get("time_mean_s") or 0.0)
            if tb - ta < thr.temporal_question_min_gap_s:
                continue
            pair_sig = frozenset({str(first.get("event_key")), str(second.get("event_key"))})
            if pair_sig in temporal_seen_pairs:
                continue
            add_question(
                "temporal_order",
                question_text,
                True,
                {
                    "first_event_key": first.get("event_key"),
                    "second_event_key": second.get("event_key"),
                    "first_time_s": round(ta, 3),
                    "second_time_s": round(tb, 3),
                    "time_gap_s": round(tb - ta, 3),
                },
            )
            temporal_seen_pairs.add(pair_sig)

    object_names = list((first_summary.get("final_state_by_object") or {}).keys())
    for obj_name in object_names:
        flags_union = set()
        for summary in rollout_summaries:
            flags_union.update((summary.get("final_state_by_object") or {}).get(obj_name, {}).keys())
        for flag in sorted(flags_union):
            vals = [
                ((summary.get("final_state_by_object") or {}).get(obj_name, {}) or {}).get(flag)
                for summary in rollout_summaries
            ]
            answer = yes_no_answer(vals)
            question = final_state_question_text(obj_name, flag, env_set, env_id)
            if question is None:
                continue
            add_question(
                "final_state",
                question,
                answer,
                {"object": obj_name, "flag": flag},
            )

    add_final_displacement_questions(
        questions=questions,
        seen_questions=seen_questions,
        rollout_summaries=rollout_summaries,
    )

    return questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment-root", type=Path, default=ENV_SET_ROOT)
    ap.add_argument("--world-json", type=Path, default=None, help="Optional exact world JSON file to use instead of discovering by environment root.")
    ap.add_argument("--include-env-set", action="append", default=[])
    ap.add_argument("--include-env-id", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--placements-per-env-condition", type=int, default=80)
    ap.add_argument("--candidate-multiplier", type=int, default=8)
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gravity-mode", choices=["downward", "upward", "both"], default="both")
    ap.add_argument("--maxtime", type=float, default=12.0)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--collision-slop", type=float, default=0.05)
    ap.add_argument("--tool-radius", type=float, default=36.0)
    ap.add_argument("--tool-color", type=str, default="orange")
    ap.add_argument("--fixed-placement", type=str, default="", help="Optional fixed x,y placement for smoke tests.")
    ap.add_argument("--heatmap-assets-root-downward", type=Path, default=None)
    ap.add_argument("--heatmap-assets-root-upward", type=Path, default=None)
    ap.add_argument("--media-dir", type=Path, default=None, help="If set, write a dotted screenshot and one representative rollout video per kept placement.")
    ap.add_argument("--media-max-seconds", type=float, default=12.0)
    ap.add_argument("--media-fps", type=float, default=60.0)
    ap.add_argument("--trace-dir", type=Path, default=None, help="If set, write one structured JSON trace per noisy rollout and no-tool baseline.")
    ap.add_argument("--trace-summary-csv", type=Path, default=None, help="If set with --trace-dir, write a summary CSV for traces generated in this run.")
    ap.add_argument("--margin", type=float, default=40.0)
    ap.add_argument(
        "--placement-min-distance",
        type=float,
        default=0.0,
        help="Reject new placements closer than this many pixels to an already-kept placement in the same env/gravity.",
    )
    ap.add_argument("--noise-scale", type=float, default=0.2)
    ap.add_argument("--visible-contact-gap-px", type=float, default=None, help="Override the summary-side visible near-contact threshold in pixels.")
    ap.add_argument("--allow-ambiguous", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Append to --out and continue from completed placement slots.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.world_json is not None:
        worlds = [args.world_json.resolve()]
    else:
        worlds = filter_worlds(discover_worlds(args.environment_root, args.include_env_set), args.include_env_id)
    gravities = ["downward", "upward"] if args.gravity_mode == "both" else [args.gravity_mode]
    thr = Thresholds()
    if args.visible_contact_gap_px is not None:
        thr.visible_contact_gap_px = float(args.visible_contact_gap_px)

    total = 0
    trace_summary_rows: List[Dict[str, Any]] = []
    completed_ids: set[str] = set()
    resume_next_slot: Dict[Tuple[str, str, str], int] = {}
    if args.resume:
        completed_ids, resume_next_slot = load_resume_state(args.out)
        print(f"[resume] loaded {len(completed_ids)} completed records from {args.out}")

    out_mode = "a" if args.resume else "w"
    with args.out.open(out_mode, encoding="utf-8") as f:
        for world_path in worlds:
            data = load_world_file(world_path)
            world_dict = data["world"]
            target, goal, blues, movable, static_black = describe_objects(data)
            dims = world_dict["dims"]
            anchors = [p for p in [object_center(world_dict["objects"].get(target.name, {}))]
                       + [object_center(world_dict["objects"].get(b.name, {})) for b in blues]
                       + ([object_center(world_dict["objects"].get(goal.name, {}))] if goal else [])
                       if p is not None]
            base_clean = loadFromDict(world_dict)

            for gravity in gravities:
                env_set = world_path.parent.name
                env_id = world_path.stem
                heatmap_root = args.heatmap_assets_root_upward if gravity == "upward" else args.heatmap_assets_root_downward
                heatmap_points = load_success_heatmap_points(heatmap_root, env_set, env_id)
                start_slot = resume_next_slot.get((env_set, env_id, gravity), 0) if args.resume else 0
                kept = start_slot
                kept_points: List[Tuple[float, float]] = []
                if args.resume and args.placement_min_distance > 0 and args.out.exists():
                    with args.out.open("r", encoding="utf-8") as rf:
                        for line in rf:
                            try:
                                prev = json.loads(line)
                            except Exception:
                                continue
                            if (
                                str(prev.get("env_set")) == env_set
                                and str(prev.get("env_id")) == env_id
                                and str(prev.get("gravity_mode")) == gravity
                            ):
                                xy = prev.get("position_xy") or []
                                if len(xy) >= 2:
                                    kept_points.append((float(xy[0]), float(xy[1])))
                attempts = 0
                max_attempts = max(args.placements_per_env_condition * args.candidate_multiplier, args.placements_per_env_condition)
                while kept < args.placements_per_env_condition and attempts < max_attempts:
                    attempts += 1
                    if args.fixed_placement:
                        raw_xy = [float(v.strip()) for v in args.fixed_placement.split(",")]
                        if len(raw_xy) != 2:
                            raise ValueError("--fixed-placement must be formatted as x,y")
                        x, y = raw_xy
                    else:
                        x, y = sample_xy_with_heatmap(rng, dims, args.margin, anchors, heatmap_points)
                    if not far_enough_from_kept(x, y, kept_points, args.placement_min_distance):
                        continue
                    if not place_ball_tool(base_clean.copy(), (x, y), args.tool_radius, gravity, args.tool_color):
                        continue

                    values: Dict[str, List[Optional[bool]]] = {}
                    questions: Dict[str, str] = {}
                    valid = 0
                    placement_trace_paths: List[str] = []
                    baseline_trace_paths: List[str] = []
                    rollout_summaries: List[Dict[str, Any]] = []
                    baseline_rollout_summaries: List[Dict[str, Any]] = []
                    candidate_record_id = make_slot_record_id(env_set, env_id, gravity, kept)
                    media_root = (args.media_dir / env_set / env_id / gravity / candidate_record_id) if args.media_dir is not None else None
                    rollout_video_path = str(media_root / "representative_noisy_rollout.mp4") if media_root is not None else None
                    clean_base_world = base_clean.copy()
                    clean_tool_world = base_clean.copy()
                    placed_clean_tool = place_ball_tool(clean_tool_world, (x, y), args.tool_radius, gravity, args.tool_color)
                    base_object_names = list(clean_base_world.objects.keys())
                    tool_object_names = list(clean_tool_world.objects.keys())
                    placement_flags = {
                        "CONTACT_TOOL_TARGET": placed_clean_tool
                        and not objects_in_contact(clean_tool_world, TOOL_NAME, target.name)
                        and not pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, target.name, thr, require_vertical_for_tool=True),
                        "CONTACT_TOOL_ANY_BLUE_MOVABLE": placed_clean_tool
                        and not any_pair_in_contact(clean_tool_world, TOOL_NAME, [b.name for b in blues])
                        and not any_pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, [b.name for b in blues], thr, require_vertical_for_tool=True),
                        "CONTACT_TOOL_GOAL_CONTAINER": bool(goal)
                        and placed_clean_tool
                        and not objects_in_contact(clean_tool_world, TOOL_NAME, goal.name)
                        and not pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, goal.name, thr, require_vertical_for_tool=True),
                        "CONTACT_TARGET_GOAL_CONTAINER": bool(goal)
                        and not objects_in_contact(clean_base_world, target.name, goal.name)
                        and not pair_too_close_for_contact(clean_base_world, data, target.name, goal.name, thr),
                        "CONTACT_TARGET_ANY_BLUE": not any_pair_in_contact(clean_base_world, target.name, [b.name for b in blues])
                        and not any_pair_too_close_for_contact(clean_base_world, data, target.name, [b.name for b in blues], thr),
                        "CONTACT_BLUE_GOAL_CONTAINER": bool(goal) and not any(
                            objects_in_contact(clean_base_world, b.name, goal.name) for b in blues
                        ) and not any(
                            pair_too_close_for_contact(clean_base_world, data, b.name, goal.name, thr) for b in blues
                        ),
                        "FIRST_CONTACT_OF_TARGET_IS_BLUE": not any_pair_in_contact(
                            clean_base_world,
                            target.name,
                            [name for name in base_object_names if name != target.name],
                        ) and not any_pair_too_close_for_contact(clean_base_world, data, target.name, [b.name for b in blues], thr),
                        "FIRST_CONTACT_OF_TARGET_IS_BLACK": not any_pair_in_contact(
                            clean_base_world,
                            target.name,
                            [name for name in base_object_names if name != target.name],
                        ) and not any_pair_too_close_for_contact(clean_base_world, data, target.name, [s.name for s in static_black], thr),
                        "FIRST_CONTACT_OF_TOOL_IS_BLUE": placed_clean_tool and not any_pair_in_contact(
                            clean_tool_world,
                            TOOL_NAME,
                            [name for name in tool_object_names if name != TOOL_NAME],
                        ) and not any_pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, [b.name for b in blues], thr, require_vertical_for_tool=True),
                        "FIRST_CONTACT_OF_TOOL_IS_BLACK": placed_clean_tool and not any_pair_in_contact(
                            clean_tool_world,
                            TOOL_NAME,
                            [name for name in tool_object_names if name != TOOL_NAME],
                        ) and not any_pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, [s.name for s in static_black], thr, require_vertical_for_tool=True),
                        "FIRST_CONTACT_OF_BLUE_IS_ANOTHER_BLUE": not any_initial_contact_for_any(
                            clean_base_world,
                            [b.name for b in blues],
                            base_object_names,
                        ) and not any(
                            any_pair_too_close_for_contact(clean_base_world, data, b.name, [o.name for o in blues if o.name != b.name], thr)
                            for b in blues
                        ),
                        "FIRST_CONTACT_OF_BLUE_IS_BLACK": not any_initial_contact_for_any(
                            clean_base_world,
                            [b.name for b in blues],
                            base_object_names,
                        ) and not any(
                            any_pair_too_close_for_contact(clean_base_world, data, b.name, [s.name for s in static_black], thr)
                            for b in blues
                        ),
                        "CONTACT_TOOL_BLUE_BEFORE_TOOL_TARGET": placed_clean_tool
                        and not any_pair_in_contact(clean_tool_world, TOOL_NAME, [b.name for b in blues])
                        and not objects_in_contact(clean_tool_world, TOOL_NAME, target.name)
                        and not any_pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, [b.name for b in blues], thr, require_vertical_for_tool=True)
                        and not pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, target.name, thr, require_vertical_for_tool=True),
                        "CONTACT_TOOL_BLACK_BEFORE_TOOL_TARGET": placed_clean_tool
                        and not any_pair_in_contact(clean_tool_world, TOOL_NAME, [s.name for s in static_black])
                        and not objects_in_contact(clean_tool_world, TOOL_NAME, target.name)
                        and not any_pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, [s.name for s in static_black], thr, require_vertical_for_tool=True)
                        and not pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, target.name, thr, require_vertical_for_tool=True),
                        "CONTACT_TOOL_BLUE_BEFORE_TOOL_BLACK": placed_clean_tool
                        and not any_pair_in_contact(clean_tool_world, TOOL_NAME, [b.name for b in blues])
                        and not any_pair_in_contact(clean_tool_world, TOOL_NAME, [s.name for s in static_black])
                        and not any_pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, [b.name for b in blues], thr, require_vertical_for_tool=True)
                        and not any_pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, [s.name for s in static_black], thr, require_vertical_for_tool=True),
                        "CONTACT_TARGET_BLUE_BEFORE_TOOL_TARGET": not any_pair_in_contact(clean_base_world, target.name, [b.name for b in blues])
                        and placed_clean_tool
                        and not objects_in_contact(clean_tool_world, TOOL_NAME, target.name)
                        and not any_pair_too_close_for_contact(clean_base_world, data, target.name, [b.name for b in blues], thr)
                        and not pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, target.name, thr, require_vertical_for_tool=True),
                        "CONTACT_TARGET_BLACK_BEFORE_TOOL_TARGET": not any_pair_in_contact(clean_base_world, target.name, [s.name for s in static_black])
                        and placed_clean_tool
                        and not objects_in_contact(clean_tool_world, TOOL_NAME, target.name)
                        and not any_pair_too_close_for_contact(clean_base_world, data, target.name, [s.name for s in static_black], thr)
                        and not pair_too_close_for_contact(clean_tool_world, data, TOOL_NAME, target.name, thr, require_vertical_for_tool=True),
                    }
                    for rollout_index in range(args.K):
                        base_world = add_noise(base_clean, rng, args.noise_scale)
                        baseline_trace = run_trace(
                            base_world.copy(),
                            world_dict=world_dict,
                            maxtime=args.maxtime,
                            step=args.step,
                            collision_slop=args.collision_slop,
                            target_name=target.name,
                            goal_name=(goal.name if goal else None),
                            thr=thr,
                        )
                        if args.trace_dir is not None:
                            baseline_rollout_id = f"{make_slot_record_id(env_set, env_id, gravity, kept)}_baseline_{rollout_index:02d}"
                            structured_baseline = build_structured_trace(
                                puzzle_id=f"{env_set}/{env_id}",
                                rollout_id=baseline_rollout_id,
                                has_tool=False,
                                simple_trace=baseline_trace,
                                objects=trace_object_metas(target, goal, blues, movable, has_tool=False),
                                step_s=args.step,
                                thresholds=TraceThresholds(goal_hold_s=thr.goal_hold_s),
                            )
                            trace_path = args.trace_dir / env_set / env_id / gravity / f"{baseline_rollout_id}.json"
                            write_trace(structured_baseline, trace_path)
                            baseline_trace_paths.append(str(trace_path))
                            if args.trace_summary_csv is not None:
                                trace_summary_rows.append(summarize_trace(structured_baseline))
                        baseline_rollout_summaries.append(
                            summarize_rollout_for_record(
                                trace=baseline_trace,
                                world_dict=data,
                                placement_xy=(x, y),
                                target=target,
                                goal=goal,
                                blues=blues,
                                static_black=static_black,
                                env_set=env_set,
                                env_id=env_id,
                                thr=thr,
                                step=args.step,
                                has_tool=False,
                            )
                        )
                        tool_world = base_world.copy()
                        if not place_ball_tool(tool_world, (x, y), args.tool_radius, gravity, args.tool_color):
                            continue
                        trace = run_trace(
                            tool_world,
                            world_dict=world_dict,
                            maxtime=args.maxtime,
                            step=args.step,
                            collision_slop=args.collision_slop,
                            target_name=target.name,
                            goal_name=(goal.name if goal else None),
                            thr=thr,
                            video_path=(Path(rollout_video_path) if rollout_video_path is not None and rollout_index == 0 else None),
                            video_fps=args.media_fps,
                        )
                        if args.trace_dir is not None:
                            tool_rollout_id = f"{make_slot_record_id(env_set, env_id, gravity, kept)}_tool_{rollout_index:02d}"
                            structured_tool = build_structured_trace(
                                puzzle_id=f"{env_set}/{env_id}",
                                rollout_id=tool_rollout_id,
                                has_tool=True,
                                simple_trace=trace,
                                objects=trace_object_metas(target, goal, blues, movable, has_tool=True),
                                step_s=args.step,
                                thresholds=TraceThresholds(goal_hold_s=thr.goal_hold_s),
                            )
                            trace_path = args.trace_dir / env_set / env_id / gravity / f"{tool_rollout_id}.json"
                            write_trace(structured_tool, trace_path)
                            placement_trace_paths.append(str(trace_path))
                            if args.trace_summary_csv is not None:
                                trace_summary_rows.append(summarize_trace(structured_tool))
                        rollout_summaries.append(
                            summarize_rollout_for_record(
                                trace=trace,
                                world_dict=data,
                                placement_xy=(x, y),
                                target=target,
                                goal=goal,
                                blues=blues,
                                static_black=static_black,
                                env_set=env_set,
                                env_id=env_id,
                                thr=thr,
                                step=args.step,
                                has_tool=True,
                            )
                        )
                        valid += 1
                        evaluate_rollout(
                            trace=trace,
                            baseline=baseline_trace,
                            world_dict=data,
                            target=target,
                            goal=goal,
                            blues=blues,
                            movable=movable,
                            static_black=static_black,
                            env_set=world_path.parent.name,
                            env_id=world_path.stem,
                            step=args.step,
                            thr=thr,
                            values=values,
                            questions=questions,
                            placement_flags=placement_flags,
                        )
                    if valid < args.K:
                        continue

                    labels: Dict[str, bool] = {}
                    p_true: Dict[str, float] = {}
                    vote_counts: Dict[str, Dict[str, int]] = {}
                    ambiguous = False
                    for key, vals in sorted(values.items()):
                        lab, p, yes, no, gray = robust_label(vals, args.K)
                        vote_counts[key] = {"yes": yes, "no": no, "gray": gray}
                        if lab is None:
                            ambiguous = True
                            if not args.allow_ambiguous:
                                break
                            continue
                        labels[key] = bool(lab)
                        p_true[key] = float(p)
                    if ambiguous and not args.allow_ambiguous:
                        continue

                    slot_index = kept
                    record_id = make_slot_record_id(env_set, env_id, gravity, slot_index)
                    if record_id in completed_ids:
                        kept += 1
                        continue
                    env_image_path = PROJECT_ROOT / "public" / "stimuli" / env_set / env_id / "environment.png"
                    dotted_screenshot_path = None
                    if args.media_dir is not None:
                        dotted_screenshot_path = render_dotted_screenshot(
                            source_image=env_image_path,
                            world_dict=data,
                            out_path=media_root / "dotted_first_frame.png",
                            x=x,
                            y=y,
                            radius=args.tool_radius,
                        )

                    aligned_event_graph = align_rollout_events(rollout_summaries, thr=thr)
                    event_graph_questions = generate_event_graph_questions(
                        rollout_summaries=rollout_summaries,
                        aligned_event_graph=aligned_event_graph,
                        world_dict=data,
                        target=target,
                        goal=goal,
                        blues=blues,
                        static_black=static_black,
                        env_set=env_set,
                        env_id=env_id,
                        clean_base_world=clean_base_world,
                        clean_tool_world=clean_tool_world,
                        thr=thr,
                        has_tool=True,
                    )
                    selected_event_graph_questions = choose_question_subset(
                        event_graph_questions,
                        max_questions=4,
                        seed=args.seed + slot_index,
                    )

                    rec = {
                        "record_id": record_id,
                        "placement_slot": int(slot_index),
                        "env_set": env_set,
                        "env_id": env_id,
                        "trial_path": str(world_path),
                        "image_path": dotted_screenshot_path or str(Path("public/stimuli") / env_set / env_id / "environment.png"),
                        "environment_image_path": str(env_image_path),
                        "dotted_screenshot_path": dotted_screenshot_path,
                        "rollout_video_path": rollout_video_path,
                        "video_path": rollout_video_path,
                        "tool": f"{args.tool_color}_ball",
                        "tool_radius": float(args.tool_radius),
                        "gravity_mode": gravity,
                        "position_xy": [float(x), float(y)],
                        "target_name": target.name,
                        "goal_name": goal.name if goal else None,
                        "goal_container_density": goal.density if goal else None,
                        "blue_objects": [{"name": b.name, "label": b.label} for b in blues],
                        "movable_objects": [{"name": m.name, "label": m.label, "role": m.role} for m in movable],
                        "labels": labels,
                        "p_true": p_true,
                        "vote_counts": vote_counts,
                        "predicate_questions": questions,
                        "event_graph_questions": event_graph_questions,
                        "selected_event_graph_questions": selected_event_graph_questions,
                        "trace_paths": placement_trace_paths,
                        "baseline_trace_paths": baseline_trace_paths,
                        "representative_rollout_summary": rollout_summaries[0] if rollout_summaries else None,
                        "all_rollout_summaries": rollout_summaries,
                        "baseline_rollout_summaries": baseline_rollout_summaries,
                        "aligned_event_graph": aligned_event_graph,
                        "black_static_objects": [{"name": s.name, "label": s.label} for s in static_black],
                        "config": {
                            "K": args.K,
                            "yes_rule": f"yes iff all {args.K} of {args.K} noisy rollouts are yes",
                            "no_rule": f"no iff all {args.K} of {args.K} noisy rollouts are no",
                            "movement_yes_px": thr.yes_px,
                            "movement_no_px": thr.no_px,
                            "contact_min_s": thr.contact_min_s,
                            "goal_hold_s": thr.goal_hold_s,
                            "before_event_min_gap_s": thr.before_event_min_gap_s,
                            "goal_touch_min_fraction": thr.goal_touch_min_fraction,
                            "initial_contact_min_gap_px": thr.initial_contact_min_gap_px,
                            "visible_contact_gap_px": thr.visible_contact_gap_px,
                            "event_time_std_max_s": thr.event_time_std_max_s,
                            "temporal_question_min_gap_s": thr.temporal_question_min_gap_s,
                            "selected_event_graph_question_count": 4,
                            "upward_tool_note": "In upward condition the big orange ball (dropped tool) accelerates upward with equal and opposite magnitude to normal downward gravity.",
                        },
                    }
                    if labels.get("TARGET_ENTERS_GOAL_AT_ANY_POINT") is not True:
                        rec["labels"].pop("TARGET_EXITS_GOAL_AFTER_ENTRY", None)
                        rec["p_true"].pop("TARGET_EXITS_GOAL_AFTER_ENTRY", None)
                        rec["vote_counts"].pop("TARGET_EXITS_GOAL_AFTER_ENTRY", None)
                        rec["predicate_questions"].pop("TARGET_EXITS_GOAL_AFTER_ENTRY", None)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    total += 1
                    kept += 1
                    kept_points.append((float(x), float(y)))
                if kept < args.placements_per_env_condition:
                    print(f"[warn] {env_set}/{world_path.name} {gravity}: kept {kept}/{args.placements_per_env_condition}")
                else:
                    print(f"[ok] {env_set}/{world_path.name} {gravity}: kept {kept}/{args.placements_per_env_condition}")
    print(f"[ok] wrote {total} new placement records to {args.out}")
    if args.trace_summary_csv is not None and trace_summary_rows:
        write_summary_csv(trace_summary_rows, args.trace_summary_csv)
        print(f"[ok] wrote {len(trace_summary_rows)} trace summary rows to {args.trace_summary_csv}")


if __name__ == "__main__":
    main()
