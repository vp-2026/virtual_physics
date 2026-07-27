#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", os.environ.get("SDL_VIDEODRIVER", "dummy"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = (
    os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    if os.path.basename(SCRIPT_DIR) == "gemini_scripts.py"
    else SCRIPT_DIR
)
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

try:
    import imageio  # type: ignore
except ModuleNotFoundError:
    imageio = None
import pygame as pg
import pygame.gfxdraw
import pymunk

from pyGameWorld.world import loadFromDict
from pyGameWorld.helpers import word2Color
from pyGameWorld.viewer import drawWorld
from three_ball_config import (
    ALLOWED_TRIALS,
    VALID_TOOL_COLORS,
    VALID_TOOL_NAMES,
    build_three_ball_tools,
    normalize_trial_name,
)


SIM_WIDTH = 600
SIM_HEIGHT = 600
PANEL_WIDTH = 300
TRIAL_SEARCH_DIRS = (
    os.path.join(ENV_DIR, "Trials", "Pilot"),
    os.path.join(ENV_DIR, "Trials", "New"),
)
OUT_JSON_DIR = os.path.join(ENV_DIR, "participant_onetool")
OUT_VID_DIR = os.path.join(ENV_DIR, "interaction_video_onetool")
DEFAULT_MAX_ATTEMPTS = 6
MAX_RECORDED_SECONDS = 10.0
MAX_SIM_SECONDS = 20.0
NO_TOOL_LABEL = "don't use tool"


def regular_ngon(radius: float, n: int = 32) -> List[List[List[float]]]:
    verts = []
    for i in range(n):
        theta = 2.0 * math.pi * (1.0 - (i / n))
        verts.append([float(radius * math.cos(theta)), float(radius * math.sin(theta))])
    return [verts]


def _signed_area(poly: List[List[float]]) -> float:
    area = 0.0
    count = len(poly)
    for index in range(count):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % count]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _assert_valid(polys: List[List[List[float]]], name: str) -> None:
    for poly in polys:
        area = _signed_area(poly)
        if not math.isfinite(area) or abs(area) < 1e-6:
            raise ValueError(f"{name} polygon has near-zero/invalid area ({area}).")


def pick_three_colors() -> Tuple[str, str, str]:
    colors = random.sample(list(VALID_TOOL_COLORS), 3)
    return colors[0], colors[1], colors[2]


def build_tools(color_obj1: str, color_obj2: str, color_obj3: str) -> Dict[str, dict]:
    return build_three_ball_tools(color_obj1, color_obj2, color_obj3)


def _placement_radius_from_polys(polys: List[List[List[float]]]) -> float:
    best = 0.0
    for poly in polys:
        for vx, vy in poly:
            best = max(best, math.hypot(float(vx), float(vy)))
    return best if best > 0 else 36.0


def build_tools_from_world(world_dict: dict, color_map: Dict[str, str]) -> Optional[Dict[str, dict]]:
    raw_tools = world_dict.get("tools")
    if not isinstance(raw_tools, dict):
        return None
    tools_dict: Dict[str, dict] = {}
    for tool_name in VALID_TOOL_NAMES:
        polys = raw_tools.get(tool_name)
        if not polys:
            continue
        tool_polys: List[List[List[float]]] = []
        for poly in polys:
            cast_poly = [[float(x), float(y)] for x, y in poly]
            _assert_valid([cast_poly], tool_name)
            tool_polys.append(cast_poly)
        tools_dict[tool_name] = {
            "polys": tool_polys,
            "density": 1.0,
            "friction": 0.5,
            "elasticity": 0.5,
            "color": color_map.get(tool_name, "pink"),
            "placement_radius": _placement_radius_from_polys(tool_polys),
            "kind": "from_world",
        }
    return tools_dict or None


def preferred_three_ball_order() -> List[str]:
    non_obj1 = ["obj2", "obj3"]
    random.shuffle(non_obj1)
    if random.random() < 0.5:
        return [non_obj1[0], "obj1", non_obj1[1]]
    return [non_obj1[0], non_obj1[1], "obj1"]


def resolve_trial_name(trial_name: str) -> Tuple[str, str]:
    raw_name = trial_name.strip()
    candidate_names = []

    try:
        normalized = normalize_trial_name(raw_name)
        candidate_names.append(normalized)
    except ValueError:
        pass

    if raw_name not in candidate_names:
        candidate_names.append(raw_name)

    matches = []
    seen = set()
    for candidate in candidate_names:
        for trial_dir in TRIAL_SEARCH_DIRS:
            trial_path = os.path.join(trial_dir, f"{candidate}.json")
            if os.path.exists(trial_path) and trial_path not in seen:
                matches.append((candidate, trial_path))
                seen.add(trial_path)

    if not matches:
        searched = ", ".join(
            os.path.join(trial_dir, f"{candidate}.json")
            for candidate in candidate_names
            for trial_dir in TRIAL_SEARCH_DIRS
        )
        raise FileNotFoundError(
            f"Could not find trial '{trial_name}'. Checked: {searched}"
        )

    if len(matches) > 1:
        locations = ", ".join(path for _, path in matches)
        raise ValueError(
            f"Trial name '{trial_name}' is ambiguous across folders: {locations}"
        )

    return matches[0]


def point_in_poly(x: float, y: float, poly: List[List[float]]) -> bool:
    inside = False
    count = len(poly)
    for index in range(count):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-12)
            if x_int > x:
                inside = not inside
    return inside


def dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy)


def min_dist_to_polygon_edges(px: float, py: float, poly: List[List[float]]) -> float:
    best = float("inf")
    for index in range(len(poly)):
        ax, ay = poly[index]
        bx, by = poly[(index + 1) % len(poly)]
        best = min(best, dist_point_to_segment(px, py, ax, ay, bx, by))
    return best


def circle_fully_in_polygon(cx: float, cy: float, radius: float, poly: List[List[float]], eps: float = 1e-6) -> bool:
    if not point_in_poly(cx, cy, poly):
        return False
    return min_dist_to_polygon_edges(cx, cy, poly) + eps >= radius


def _safe_pos_list(obj_instance) -> Optional[List[float]]:
    try:
        if hasattr(obj_instance, "isStatic") and obj_instance.isStatic():
            return None
        pos = obj_instance.getPos()
        if hasattr(pos, "x") and hasattr(pos, "y"):
            return [float(pos.x), float(pos.y)]
        if hasattr(pos, "tolist"):
            xy = pos.tolist()
            return [float(xy[0]), float(xy[1])]
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return [float(pos[0]), float(pos[1])]
    except Exception:
        return None
    return None


def _snapshot_positions_dynamic(pyworld) -> Dict[str, List[float]]:
    out = {}
    for name, obj in pyworld.objects.items():
        pos_list = _safe_pos_list(obj)
        if pos_list is not None:
            out[name] = pos_list
    return out


def _snapshot_positions_movable(pyworld, movable_names: set) -> Dict[str, List[float]]:
    out = {}
    for name, obj in pyworld.objects.items():
        if name not in movable_names:
            continue
        pos_list = _safe_pos_list(obj)
        if pos_list is not None:
            out[name] = pos_list
    return out


def _color_payload(color_value):
    if color_value is None:
        return None
    if isinstance(color_value, str):
        color_value = word2Color(color_value)
    if isinstance(color_value, (list, tuple)):
        values = [int(round(float(component))) for component in list(color_value)]
        if len(values) == 3:
            values.append(255)
        return values[:4]
    return None


def _point_payload(point) -> List[float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return [float(point.x), float(point.y)]
    if hasattr(point, "tolist"):
        values = point.tolist()
        return [float(values[0]), float(values[1])]
    return [float(point[0]), float(point[1])]


def _poly_payload(poly) -> List[List[float]]:
    return [_point_payload(vertex) for vertex in poly]


def _snapshot_drawables(pyworld) -> List[dict]:
    drawables: List[dict] = []

    def append_drawable(name: str, obj) -> None:
        base = {
            "name": name,
            "kind": str(getattr(obj, "type", "")).lower(),
            "color": _color_payload(getattr(obj, "color", None)),
            "is_static": bool(obj.isStatic()) if hasattr(obj, "isStatic") else True,
        }

        if getattr(obj, "type", None) == "Ball":
            base["position"] = _point_payload(obj.getPos())
            base["radius"] = float(getattr(obj, "radius", obj.getRadius()))
        elif getattr(obj, "type", None) in ("Poly", "Goal", "Blocker"):
            base["vertices"] = _poly_payload(obj.getVertices())
        elif getattr(obj, "type", None) == "Container":
            base["polys"] = [_poly_payload(poly) for poly in obj.getPolys()]
            base["inner_vertices"] = _poly_payload(obj.getVertices())
            base["inner_color"] = _color_payload(getattr(obj, "inner_color", None))
            base["outer_color"] = _color_payload(getattr(obj, "outer_color", getattr(obj, "color", None)))
        elif getattr(obj, "type", None) == "Compound":
            base["polys"] = [_poly_payload(poly) for poly in obj.getPolys()]
        elif getattr(obj, "type", None) == "Segment":
            p1, p2 = obj.getPoints()
            base["points"] = [_point_payload(p1), _point_payload(p2)]
            base["radius"] = float(getattr(obj, "r", 0.0))
        else:
            return

        drawables.append(base)

    for name, obj in pyworld.objects.items():
        append_drawable(name, obj)
    for name, obj in pyworld.blockers.items():
        append_drawable(name, obj)

    drawables.sort(key=lambda item: (0 if item.get("is_static") else 1, item.get("name", "")))
    return drawables


def build_movable_name_set_from_env(world_dict: dict) -> set:
    movable = set()
    objs = world_dict.get("world", {}).get("objects", {})
    for name, spec in objs.items():
        try:
            if float(spec.get("density", 0)) > 0:
                movable.add(name)
        except Exception:
            pass
    movable.add("PLACED")
    return movable


def pick_container_or_none(objs_dict: dict, preferred_name: Optional[str] = None) -> Tuple[Optional[str], Optional[dict]]:
    if preferred_name and preferred_name in objs_dict and objs_dict[preferred_name].get("type") == "Container":
        return preferred_name, objs_dict[preferred_name]
    for name, spec in objs_dict.items():
        if spec.get("type") == "Container" and str(spec.get("innerColor", "")).lower() == "green":
            return name, spec
    for name, spec in objs_dict.items():
        if spec.get("type") == "Container":
            return name, spec
    return None, None


def current_goal_polygon(pyworld, goal_name: Optional[str], fallback_points):
    """Return live goal polygon vertices if available, else fallback points."""
    if goal_name and goal_name in pyworld.objects:
        goal_obj = pyworld.objects.get(goal_name)
        if goal_obj is not None and hasattr(goal_obj, "getVertices"):
            try:
                poly = [
                    (float(v.x), float(v.y)) if hasattr(v, "x") else (float(v[0]), float(v[1]))
                    for v in goal_obj.getVertices()
                ]
                if len(poly) >= 3:
                    return poly
            except Exception:
                pass
    return fallback_points


def _goal_distance_diagnostics(pyworld, world_dict: dict, target_obj_name: str, goal_name: Optional[str]) -> dict:
    """
    Diagnostics for "how far is the target from satisfying the goal".

    Notes:
    - `pyworld.distanceToGoal(point)` clamps any "inside" distance to 0.
    - `goal_signed_distance` is the raw signed distance (negative means inside) from the
      target center to the goal's sensor shape via `distanceFromPoint`.
    - For circular targets, `goal_clearance` is -signed_distance when inside else 0, and
      `goal_margin` is clearance - radius (positive means the entire circle fits).
    """
    out = {
        # Back-compat: `goal_distance` is the clamped distance used by PGWorld.distanceToGoal
        # (0 whenever the point is inside).
        "goal_distance": None,
        # `goal_signed_distance` is the raw signed distance from the target center to the goal
        # sensor shape (negative means inside).
        "goal_signed_distance": None,
        # Convenience: unclamped outside-only distance (same as max(signed, 0)).
        "goal_outside_distance": None,
        "goal_clearance": None,
        "goal_margin": None,
        # For circular targets: distance from the *ball boundary* to goal interior (0 if touching/inside).
        "goal_ball_outside_distance": None,
        # If the goal is a Container with wall thickness, approximate distance to the "green free space"
        # by insetting the sensor polygon inward by `goal_wall_radius` (width/2). This avoids treating
        # the walls themselves as valid target area.
        "goal_wall_radius": None,
        "goal_green_signed_distance": None,
        "goal_green_outside_distance": None,
        "goal_green_ball_outside_distance": None,
    }

    try:
        if not getattr(pyworld, "goalCond", None):
            gcond = world_dict.get("world", {}).get("gcond", {}) or {}
            if gcond.get("type") == "SpecificInGoal":
                pyworld.attachSpecificInGoal(goal_name or gcond.get("goal", "Goal"), target_obj_name, 0.0)
    except Exception:
        pass

    if target_obj_name not in getattr(pyworld, "objects", {}):
        return out

    target = pyworld.objects[target_obj_name]
    pos = target.getPos()
    if hasattr(pos, "x") and hasattr(pos, "y"):
        point = [float(pos.x), float(pos.y)]
    elif hasattr(pos, "tolist"):
        values = pos.tolist()
        point = [float(values[0]), float(values[1])]
    else:
        point = [float(pos[0]), float(pos[1])]

    try:
        out["goal_distance"] = float(pyworld.distanceToGoal(point))
    except Exception:
        out["goal_distance"] = None

    signed = None
    goal_wall_radius = None
    try:
        gcond = getattr(pyworld, "goalCond", None)
        if gcond is not None and getattr(gcond, "goal", None):
            goal_obj = pyworld.getObject(gcond.goal)
            signed = float(goal_obj.distanceFromPoint(point))
            if getattr(goal_obj, "type", None) == "Container" and hasattr(goal_obj, "r"):
                goal_wall_radius = float(getattr(goal_obj, "r"))
    except Exception:
        signed = None
    out["goal_signed_distance"] = signed
    if signed is not None:
        out["goal_outside_distance"] = float(max(signed, 0.0))
    if goal_wall_radius is not None:
        out["goal_wall_radius"] = goal_wall_radius
        green_signed = float(signed + goal_wall_radius) if signed is not None else None
        out["goal_green_signed_distance"] = green_signed
        if green_signed is not None:
            out["goal_green_outside_distance"] = float(max(green_signed, 0.0))

    shape_def = world_dict.get("world", {}).get("objects", {}).get(target_obj_name, {}) or {}
    if signed is not None and (hasattr(target, "radius") or ("radius" in shape_def)):
        radius = float(getattr(target, "radius", shape_def.get("radius", 0.0) or 0.0))
        clearance = (-signed) if signed < 0 else 0.0
        out["goal_clearance"] = float(clearance)
        out["goal_margin"] = float(clearance - radius)
        out["goal_ball_outside_distance"] = float(max(signed - radius, 0.0))
        if out.get("goal_green_signed_distance") is not None:
            out["goal_green_ball_outside_distance"] = float(max(out["goal_green_signed_distance"] - radius, 0.0))

    return out


def _target_fully_inside_goal(pyworld, world_dict: dict, target_obj_name: str, goal_name: Optional[str], goal_pts_world):
    if goal_pts_world is None or target_obj_name not in pyworld.objects:
        return False

    target = pyworld.objects[target_obj_name]
    shape_def = world_dict["world"]["objects"].get(target_obj_name, {})
    live_goal_pts_world = current_goal_polygon(pyworld, goal_name, goal_pts_world)
    pos = target.getPos()
    if hasattr(pos, "x") and hasattr(pos, "y"):
        cx, cy = float(pos.x), float(pos.y)
    elif hasattr(pos, "tolist"):
        cx, cy = [float(value) for value in pos.tolist()[:2]]
    else:
        cx, cy = float(pos[0]), float(pos[1])

    if hasattr(target, "radius") or ("radius" in shape_def):
        radius = float(getattr(target, "radius", shape_def.get("radius", 0.0)))
        return circle_fully_in_polygon(cx, cy, radius, live_goal_pts_world)

    poly_world = None
    if hasattr(target, "getVertices"):
        try:
            poly_world = [
                (float(v.x), float(v.y)) if hasattr(v, "x") else (float(v[0]), float(v[1]))
                for v in target.getVertices()
            ]
        except Exception:
            poly_world = None
    if poly_world is None and "vertices" in shape_def:
        poly_world = [(float(x), float(y)) for x, y in shape_def["vertices"]]
    return poly_world is not None and all(point_in_poly(x, y, live_goal_pts_world) for x, y in poly_world)


def _apply_hidden_tool_behavior(pyworld, tool_name: str, tool_def: dict) -> None:
    placed_obj = pyworld.objects.get("PLACED")
    if placed_obj is None or placed_obj.isStatic():
        return
    if tool_def.get("inverse_gravity", False):
        try:
            gx, gy = pyworld._cpSpace.gravity
            gmag = max(abs(float(gx)), abs(float(gy)))

            def _custom_velocity_func(cp_body, gravity, damping, dt):
                pymunk.Body.update_velocity(cp_body, (0.0, gmag), damping, dt)

            placed_obj._cpBody.velocity_func = _custom_velocity_func
        except Exception:
            pass
    if "body_damping" in tool_def:
        placed_obj._cpBody.damping = tool_def["body_damping"]
    if all(k in tool_def for k in ("launch_speed_x", "launch_speed_y", "launch_spin")):
        placed_obj.setVel((
            random.uniform(-tool_def.get("launch_speed_x", 0.0), tool_def.get("launch_speed_x", 0.0)),
            random.uniform(-tool_def.get("launch_speed_y", 0.0), tool_def.get("launch_speed_y", 0.0)),
        ))
        placed_obj._cpBody.angular_velocity = random.uniform(
            -tool_def.get("launch_spin", 0.0), tool_def.get("launch_spin", 0.0)
        )


def _translate_tool_polys(tool_def: dict, position: Tuple[float, float]) -> List[List[Tuple[float, float]]]:
    px, py = float(position[0]), float(position[1])
    translated = []
    for poly in tool_def["polys"]:
        translated.append([(float(vx) + px, float(vy) + py) for vx, vy in poly])
    return translated


def _tool_fits_within_world(world_dims: Tuple[float, float], translated_polys: List[List[Tuple[float, float]]], eps: float = 1e-6) -> bool:
    width, height = float(world_dims[0]), float(world_dims[1])
    for poly in translated_polys:
        for vx, vy in poly:
            if vx < -eps or vy < -eps or vx > width + eps or vy > height + eps:
                return False
    return True


def _placement_collision_reason(pyworld, tool_def: dict, position: Tuple[float, float]) -> Optional[str]:
    translated_polys = _translate_tool_polys(tool_def, position)
    if not _tool_fits_within_world(pyworld.dims, translated_polys):
        return "That spot is too close to the edge."
    for poly in tool_def["polys"]:
        if pyworld.checkCollision(position, poly):
            return "That spot is blocked by another object."
    return None


def _placement_collides(pyworld, tool_def: dict, position: Tuple[float, float]) -> bool:
    return _placement_collision_reason(pyworld, tool_def, position) is not None


def load_world(trial_name: str) -> dict:
    normalized, trial_path = resolve_trial_name(trial_name)
    with open(trial_path, "r") as handle:
        world_dict = json.load(handle)
    world_dict["_source_name"] = normalized
    return world_dict


def load_world_from_path(trial_path: str) -> dict:
    path = os.path.abspath(trial_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trial JSON not found: {path}")
    with open(path, "r") as handle:
        world_dict = json.load(handle)
    world_dict["_source_name"] = os.path.splitext(os.path.basename(path))[0]
    return world_dict


def run_headless_episode(
    world_dict: dict,
    *,
    tool_name: Optional[str],
    tools_dict: Dict[str, dict],
    drop_xy: Optional[Tuple[int, int]],
    no_tool: bool = False,
    fps: float = 60.0,
    stillness_duration: float = 0.5,
    linear_threshold: float = 1.0,
    angular_threshold: float = 0.1,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    video_basename: Optional[str] = None,
    accepted_callback=None,
    frame_callback=None,
) -> Tuple[dict, Optional[str]]:
    sim_width, sim_height = world_dict["world"]["dims"]
    pg.init()
    screen = pg.display.set_mode((sim_width, sim_height), flags=pg.HIDDEN)
    clock = pg.time.Clock()

    world = deepcopy(world_dict)
    pyworld = loadFromDict(world["world"])
    sx = sy = None
    tool_def = None
    if not no_tool:
        if tool_name is None or drop_xy is None:
            raise ValueError("tool_name and drop_xy are required when no_tool is False")
        tool_def = tools_dict[tool_name]
        _assert_valid(tool_def["polys"], tool_name)
        sx, sy = int(drop_xy[0]), int(drop_xy[1])
        placement_collision_reason = _placement_collision_reason(pyworld, tool_def, (sx, sy))
        if placement_collision_reason:
            payload = {
                "placements": [{
                    "placement_coords": [sx, sy],
                    "landing_positions": {},
                    "trajectory_data": {},
                    "trial_completed": False,
                    "selected_tool": tool_name,
                    "selected_color": tool_def["color"],
                    "success": False,
                    "obstruction_detected": True,
                    "error": placement_collision_reason,
                    "phase": "goal",
                }],
            }
            pg.quit()
            return payload, None

    gcond = world["world"].get("gcond") or {}
    target_obj_name = gcond.get("obj", "Ball")
    goal_name = gcond.get("goal", "Goal")
    _, goal_obj = pick_container_or_none(world["world"]["objects"], preferred_name=goal_name)
    goal_pts_world = goal_obj["points"] if goal_obj else None

    if not no_tool:
        translated_polys = _translate_tool_polys(tool_def, (sx, sy))

        pyworld.addPlacedCompound(
            "PLACED",
            translated_polys,
            word2Color(tool_def["color"]),
            density=tool_def.get("density", 1.0),
            friction=tool_def.get("friction", 0.5),
            elasticity=tool_def.get("elasticity", 0.5),
        )
        _apply_hidden_tool_behavior(pyworld, tool_name, tool_def)

    if accepted_callback is not None:
        accepted_callback({
            "placement_coords": [sx, sy] if (sx is not None and sy is not None) else None,
            "selected_tool": tool_name if not no_tool else NO_TOOL_LABEL,
            "selected_color": (tool_def["color"] if tool_def is not None else None),
            "no_tool_rollout": bool(no_tool),
            "phase": "goal",
        })

    if frame_callback is not None:
        frame_callback({
            "elapsed": 0.0,
            "drawables": _snapshot_drawables(pyworld),
        })

    saved_video_path = None
    writer = None
    if record_video:
        if imageio is None:
            raise ModuleNotFoundError(
                "imageio is required for record_video=True. Install dependencies from requirements.txt."
            )
        os.makedirs(video_dir, exist_ok=True)
        base = video_basename or (world_dict.get("_source_name") or "episode")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_video_path = os.path.join(video_dir, f"{base}_{ts}.mp4")
        try:
            writer = imageio.get_writer(
                saved_video_path,
                fps=fps,
                format="FFMPEG",
                codec="libx264",
                quality=8,
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        except Exception:
            writer = imageio.get_writer(
                saved_video_path,
                fps=fps,
                format="FFMPEG",
                codec="mpeg4",
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )

    movable_set = build_movable_name_set_from_env(world_dict)
    last_pos = {}
    moved_flag = {}
    trajectory = {}
    rotation_trajectory = {}
    success_seen = False
    time_to_success = None
    goal_duration_seconds = float(gcond.get("duration", 0.0) or 0.0)
    target_in_goal_since = None
    elapsed = 0.0
    last_movement_time = time.time()
    all_still = False
    move_eps_sq = 0.25 * 0.25

    running = True
    while running:
        pyworld.step(1.0 / fps)

        frame_surf = drawWorld(pyworld)
        screen.blit(frame_surf, (0, 0))
        pg.display.flip()

        if writer is not None and elapsed < MAX_RECORDED_SECONDS:
            arr = pg.surfarray.array3d(screen).swapaxes(0, 1)
            writer.append_data(arr)

        if frame_callback is not None:
            frame_callback({
                "elapsed": elapsed,
                "drawables": _snapshot_drawables(pyworld),
            })

        frame_pos = _snapshot_positions_dynamic(pyworld)
        for name, pos_list in frame_pos.items():
            if elapsed < MAX_RECORDED_SECONDS:
                trajectory.setdefault(name, []).append(pos_list)
            prev = last_pos.get(name)
            if prev is None:
                last_pos[name] = pos_list
            else:
                dx = pos_list[0] - prev[0]
                dy = pos_list[1] - prev[1]
                if dx * dx + dy * dy > move_eps_sq:
                    moved_flag[name] = True
                last_pos[name] = pos_list
        for obj in pyworld.getDynamicObjects():
            name = getattr(obj, "name", None)
            if not name:
                continue
            if elapsed < MAX_RECORDED_SECONDS:
                rotation_trajectory.setdefault(name, []).append(float(obj.rotation))

        target_inside_goal = _target_fully_inside_goal(
            pyworld, world_dict, target_obj_name, goal_name, goal_pts_world
        )
        if target_inside_goal:
            if target_in_goal_since is None:
                target_in_goal_since = elapsed
            if (not success_seen) and (elapsed - target_in_goal_since >= goal_duration_seconds):
                success_seen = True
                time_to_success = elapsed
        else:
            target_in_goal_since = None

        if success_seen:
            running = False

        dynamic_objects = pyworld.getDynamicObjects()
        all_still_this_frame = True
        for obj in dynamic_objects:
            if obj._cpBody.is_sleeping:
                continue
            lin_sq = obj._cpBody.velocity.length ** 2
            ang_v = abs(obj._cpBody.angular_velocity)
            if lin_sq > (linear_threshold ** 2) or ang_v > angular_threshold:
                all_still_this_frame = False
                last_movement_time = time.time()
                break

        if all_still_this_frame and not all_still:
            all_still = True
            last_movement_time = time.time()
        elif not all_still_this_frame:
            all_still = False

        waiting_on_goal_duration = (
            (not success_seen) and
            (target_in_goal_since is not None) and
            (goal_duration_seconds > 0)
        )

        if all_still and (time.time() - last_movement_time >= stillness_duration) and (not waiting_on_goal_duration):
            running = False

        elapsed += 1.0 / fps
        if elapsed >= MAX_SIM_SECONDS:
            running = False
        # Only pace to real-time when recording video; in streaming-only mode the
        # client uses the elapsed timestamps to schedule playback at the correct speed.
        if writer is not None:
            clock.tick(fps)

    landing_positions = _snapshot_positions_movable(pyworld, movable_set)
    movers_only = {name: traj for name, traj in trajectory.items() if moved_flag.get(name, False)}
    goal_diagnostics = _goal_distance_diagnostics(pyworld, world_dict, target_obj_name, goal_name)

    payload = {
        "placements": [{
            "placement_coords": [sx, sy] if (sx is not None and sy is not None) else None,
            "landing_positions": landing_positions,
            "trajectory_data": movers_only,
            "rotation_data": rotation_trajectory,
            "trajectory_window_seconds": MAX_RECORDED_SECONDS,
            "trial_completed": True,
            "selected_tool": tool_name if not no_tool else NO_TOOL_LABEL,
            "selected_color": (tool_def["color"] if tool_def is not None else None),
            "success": success_seen,
            "time_to_success": time_to_success if success_seen else None,
            **goal_diagnostics,
            "phase": "goal",
            "no_tool_rollout": bool(no_tool),
        }],
    }

    if writer is not None:
        writer.close()
    pg.quit()
    return payload, saved_video_path


def _sanitize_attempt_payload(payload: dict) -> dict:
    copy_payload = json.loads(json.dumps(payload))
    placement = copy_payload["placements"][0]
    placement["trajectory_frame_count"] = len(placement.get("trajectory_data", {}).get("PLACED", []))
    return copy_payload


def _ensure_drive_parent(service, folder_name: str, drive_root_id: str) -> str:
    query = (
        f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
        f"and '{drive_root_id}' in parents and trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [drive_root_id],
    }
    created = service.files().create(body=metadata, fields="id").execute()
    return created["id"]


def upload_file_to_google_drive(local_path: str, prolific_id: str, drive_root_id: str, service_account_json: str) -> None:
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive upload requires google-api-python-client and google-auth. "
            "Install them before using --drive-root-id/--service-account-json."
        ) from exc

    scopes = ["https://www.googleapis.com/auth/drive.file"]
    credentials = service_account.Credentials.from_service_account_file(service_account_json, scopes=scopes)
    service = build("drive", "v3", credentials=credentials)
    folder_id = _ensure_drive_parent(service, prolific_id, drive_root_id)
    media = MediaFileUpload(local_path, mimetype="application/json", resumable=False)
    metadata = {"name": os.path.basename(local_path), "parents": [folder_id]}
    service.files().create(body=metadata, media_body=media, fields="id").execute()


def _tool_buttons(display_order: List[str], start_y: int) -> List[dict]:
    buttons = []
    for index, tool_name in enumerate(display_order):
        buttons.append(
            {
                "name": tool_name,
                "rect": pg.Rect(SIM_WIDTH + 20, start_y + index * 112, PANEL_WIDTH - 40, 82),
                "center": (SIM_WIDTH + (PANEL_WIDTH // 2), start_y + index * 112 + 41),
            }
        )
    return buttons


def _wrap_text(text: str, font, max_width: int) -> List[pg.Surface]:
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(font.render(current, True, (20, 20, 20)))
            current = word
    lines.append(font.render(current, True, (20, 20, 20)))
    return lines


def run_human_goal_session(
    trial_name: str,
    tools_dict: Dict[str, dict],
    display_order: List[str],
    prolific_id: str,
    world_template: Optional[dict] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    drive_root_id: Optional[str] = None,
    service_account_json: Optional[str] = None,
) -> str:
    os.environ.pop("SDL_VIDEODRIVER", None)
    if world_template is None:
        world_template = load_world(trial_name)
    gcond = world_template["world"].get("gcond") or {}
    target_obj_name = gcond.get("obj", "Ball")
    goal_name = gcond.get("goal", "Goal")
    try:
        goal_duration_seconds = float(gcond.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        goal_duration_seconds = 0.0
    _, goal_obj = pick_container_or_none(world_template["world"]["objects"], preferred_name=goal_name)
    goal_pts_world = goal_obj["points"] if goal_obj else None

    pg.init()
    try:
        screen = pg.display.set_mode((SIM_WIDTH + PANEL_WIDTH, SIM_HEIGHT), vsync=1)
    except TypeError:
        screen = pg.display.set_mode((SIM_WIDTH + PANEL_WIDTH, SIM_HEIGHT))
    pg.display.set_caption("Drop and Discover - Three Ball Goal Task")
    font = pg.font.Font(None, 24)
    small_font = pg.font.Font(None, 18)
    clock = pg.time.Clock()

    goal_text = world_template.get("sucText", "Your goal is to get a red thing into the green area.")
    goal_surfaces = _wrap_text(goal_text, small_font, PANEL_WIDTH - 56)
    goal_line_height = 18
    goal_card_height = max(92, 24 + len(goal_surfaces) * goal_line_height + 16)
    goal_card_rect = pg.Rect(SIM_WIDTH + 20, 24, PANEL_WIDTH - 40, goal_card_height)
    tools_header_y = goal_card_rect.bottom + 14
    buttons = _tool_buttons(display_order, tools_header_y + 28)
    reset_button_rect = pg.Rect(SIM_WIDTH + 20, SIM_HEIGHT - 70, PANEL_WIDTH - 40, 46)
    selected_tool = display_order[0]
    attempts = []
    current_world = ToolPicker(deepcopy(world_template))
    movable_set = build_movable_name_set_from_env(world_template)
    placement_active = False
    placed_tool_name = None
    attempt_start = None
    placement_coords = None
    success_seen = False
    target_in_goal_since = None
    last_pos = {}
    moved_flag = {}
    trajectory = {}
    last_movement_time = time.time()
    all_still = False
    move_eps_sq = 0.25 * 0.25
    success_flash_until = None
    running = True

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if reset_button_rect.collidepoint(mx, my):
                    current_world = ToolPicker(deepcopy(world_template))
                    placement_active = False
                    placed_tool_name = None
                    attempt_start = None
                    placement_coords = None
                    success_seen = False
                    target_in_goal_since = None
                    last_pos = {}
                    moved_flag = {}
                    trajectory = {}
                    last_movement_time = time.time()
                    all_still = False
                    continue

                if placement_active:
                    continue
                for button in buttons:
                    if button["rect"].collidepoint(mx, my):
                        selected_tool = button["name"]
                        break
                else:
                    if 0 <= mx < SIM_WIDTH and 0 <= my < SIM_HEIGHT and len(attempts) < max_attempts:
                        tool_def = tools_dict[selected_tool]
                        if _placement_collides(current_world._pyworld, tool_def, (mx, my)):
                            continue
                        translated_polys = _translate_tool_polys(tool_def, (mx, my))
                        current_world._pyworld.addPlacedCompound(
                            "PLACED",
                            translated_polys,
                            word2Color(tool_def["color"]),
                            density=tool_def.get("density", 1.0),
                            friction=tool_def.get("friction", 0.5),
                            elasticity=tool_def.get("elasticity", 0.5),
                        )
                        _apply_hidden_tool_behavior(current_world, selected_tool, tool_def)
                        placement_active = True
                        placed_tool_name = selected_tool
                        attempt_start = time.time()
                        placement_coords = [mx, my]
                        success_seen = False
                        target_in_goal_since = None
                        last_pos = {}
                        moved_flag = {}
                        trajectory = {}
                        last_movement_time = time.time()
                        all_still = False

        if placement_active:
            current_world._pyworld.step(1.0 / 60.0)
            frame_pos = _snapshot_positions_dynamic(current_world._pyworld)
            for name, pos_list in frame_pos.items():
                trajectory.setdefault(name, []).append(pos_list)
                prev = last_pos.get(name)
                if prev is None:
                    last_pos[name] = pos_list
                else:
                    dx = pos_list[0] - prev[0]
                    dy = pos_list[1] - prev[1]
                    if dx * dx + dy * dy > move_eps_sq:
                        moved_flag[name] = True
                        last_pos[name] = pos_list

            target_inside_goal = _target_fully_inside_goal(
                current_world._pyworld, world_template, target_obj_name, goal_name, goal_pts_world
            )
            if target_inside_goal:
                if target_in_goal_since is None:
                    target_in_goal_since = time.time()
                if (not success_seen) and (time.time() - target_in_goal_since >= goal_duration_seconds):
                    success_seen = True
                    success_flash_until = time.time() + 2.0
            else:
                target_in_goal_since = None

            dynamic_objects = current_world._pyworld.getDynamicObjects()
            all_still_this_frame = True
            for obj in dynamic_objects:
                if obj._cpBody.is_sleeping:
                    continue
                lin_sq = obj._cpBody.velocity.length ** 2
                ang_v = abs(obj._cpBody.angular_velocity)
                if lin_sq > 1.0 or ang_v > 0.1:
                    all_still_this_frame = False
                    last_movement_time = time.time()
                    break

            if all_still_this_frame and not all_still:
                all_still = True
                last_movement_time = time.time()
            elif not all_still_this_frame:
                all_still = False

            if all_still and (time.time() - last_movement_time >= 0.5):
                landing_positions = _snapshot_positions_movable(current_world._pyworld, movable_set)
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "selected_tool": placed_tool_name,
                        "selected_color": tools_dict[placed_tool_name]["color"],
                        "placement_coords": placement_coords,
                        "landing_positions": landing_positions,
                        "trajectory_data": {
                            name: traj for name, traj in trajectory.items() if moved_flag.get(name, False)
                        },
                        "success": success_seen,
                        "time_elapsed": (time.time() - attempt_start) if attempt_start else None,
                    }
                )
                if success_seen or len(attempts) >= max_attempts:
                    running = False
                else:
                    current_world = ToolPicker(deepcopy(world_template))
                    placement_active = False
                    placed_tool_name = None
                    attempt_start = None
                    placement_coords = None
                    target_in_goal_since = None

        frame = drawWorld(current_world._pyworld)
        screen.fill((242, 242, 242))
        screen.blit(frame, (0, 0))
        pg.draw.rect(screen, (232, 232, 232), (SIM_WIDTH, 0, PANEL_WIDTH, SIM_HEIGHT))
        pg.draw.line(screen, (70, 70, 70), (SIM_WIDTH, 0), (SIM_WIDTH, SIM_HEIGHT), width=2)

        title = font.render("Goal", True, (20, 20, 20))
        screen.blit(title, (SIM_WIDTH + 20, 4))
        pg.draw.rect(screen, (246, 246, 246), goal_card_rect, border_radius=16)
        pg.draw.rect(screen, (45, 45, 45), goal_card_rect, width=3, border_radius=16)
        y_offset = goal_card_rect.top + 12
        for surface in goal_surfaces:
            screen.blit(surface, (goal_card_rect.left + 16, y_offset))
            y_offset += goal_line_height
        tools_header = font.render("Tools", True, (20, 20, 20))
        screen.blit(tools_header, (SIM_WIDTH + 20, tools_header_y))
        attempts_y = buttons[-1]["rect"].bottom + 18
        budget = font.render(f"Attempts left: {max_attempts - len(attempts)}", True, (20, 20, 20))
        screen.blit(budget, (SIM_WIDTH + 20, attempts_y))
        pid_text = small_font.render(f"Prolific ID: {prolific_id}", True, (50, 50, 50))
        screen.blit(pid_text, (SIM_WIDTH + 20, attempts_y + 28))
        hint_lines = [
            "Pick a ball, then click",
            "inside the white area",
            "to drop it.",
            "Reset clears the current try.",
        ]
        for index, line in enumerate(hint_lines):
            hint = small_font.render(line, True, (60, 60, 60))
            screen.blit(hint, (SIM_WIDTH + 20, attempts_y + 56 + index * 20))

        for button in buttons:
            tool_def = tools_dict[button["name"]]
            color_rgb = word2Color(tool_def["color"])
            border = (30, 30, 30) if button["name"] == selected_tool else (150, 150, 150)
            fill = (246, 246, 246) if button["name"] == selected_tool else (241, 241, 241)
            pg.draw.rect(screen, fill, button["rect"], border_radius=18)
            pg.draw.rect(screen, border, button["rect"], width=3 if button["name"] == selected_tool else 2, border_radius=18)
            shadow_center = (button["center"][0] + 3, button["center"][1] + 3)
            pg.gfxdraw.filled_circle(screen, shadow_center[0], shadow_center[1], 28, (185, 185, 185))
            pg.gfxdraw.aacircle(screen, shadow_center[0], shadow_center[1], 28, (185, 185, 185))
            pg.gfxdraw.filled_circle(screen, button["center"][0], button["center"][1], 26, color_rgb)
            pg.gfxdraw.aacircle(screen, button["center"][0], button["center"][1], 26, border)

        pg.draw.rect(screen, (180, 50, 50), reset_button_rect, border_radius=10)
        reset_label = small_font.render("Reset", True, (255, 255, 255))
        reset_rect = reset_label.get_rect(center=reset_button_rect.center)
        screen.blit(reset_label, reset_rect)

        if success_flash_until and time.time() < success_flash_until:
            success_surface = font.render("SUCCESS", True, (0, 160, 0))
            success_rect = success_surface.get_rect(center=(SIM_WIDTH // 2, 24))
            screen.blit(success_surface, success_rect)

        pg.display.flip()
        clock.tick(60)

    pg.quit()

    resolved_trial_name, _ = resolve_trial_name(trial_name)
    os.makedirs(OUT_JSON_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_JSON_DIR, f"{prolific_id}_{resolved_trial_name}_{timestamp}.json")
    payload = {
        "prolific_id": prolific_id,
        "trial_name": resolved_trial_name,
        "attempt_budget": max_attempts,
        "tool_manifest": {
            tool_name: {"color": tools_dict[tool_name]["color"], "display_index": display_order.index(tool_name)}
            for tool_name in VALID_TOOL_NAMES
        },
        "attempts": [_sanitize_attempt_payload({"placements": [attempt]})["placements"][0] for attempt in attempts],
        "success": any(attempt["success"] for attempt in attempts),
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    if drive_root_id and service_account_json:
        upload_file_to_google_drive(out_path, prolific_id, drive_root_id, service_account_json)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-name", default="Basic_EY", help="Pilot goal environment name.")
    parser.add_argument("--trial-path", default=None, help="Direct path to a trial JSON file.")
    parser.add_argument("--human", action="store_true", help="Run the interactive participant task.")
    parser.add_argument("--drop", type=str, help="Placement coords x,y for headless mode.")
    parser.add_argument("--tool", choices=VALID_TOOL_NAMES, help="Tool name for headless mode.")
    parser.add_argument("--no-tool", action="store_true", help="Run a no-tool rollout in headless mode.")
    parser.add_argument("--color-obj1", choices=VALID_TOOL_COLORS)
    parser.add_argument("--color-obj2", choices=VALID_TOOL_COLORS)
    parser.add_argument("--color-obj3", choices=VALID_TOOL_COLORS)
    parser.add_argument("--video", action="store_true", help="Record an MP4 in headless mode.")
    parser.add_argument("--video-name", default=None)
    parser.add_argument(
        "--respect-world-tools",
        action="store_true",
        help="Use tool polygons from the trial JSON tools section instead of default ball tools.",
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--prolific-id", default=None)
    parser.add_argument("--drive-root-id", default=None)
    parser.add_argument("--service-account-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trial_path:
        trial_name = os.path.splitext(os.path.basename(args.trial_path))[0]
    else:
        trial_name, _ = resolve_trial_name(args.trial_name)

    # Default to interactive only when no explicit headless instruction was provided.
    if (not args.human) and (not args.no_tool) and args.tool is None and args.drop is None:
        args.human = True

    if args.color_obj1 and args.color_obj2 and args.color_obj3:
        if len({args.color_obj1, args.color_obj2, args.color_obj3}) != 3:
            raise SystemExit("Headless colors must be three distinct values.")
        colors = (args.color_obj1, args.color_obj2, args.color_obj3)
    else:
        colors = pick_three_colors()

    world_dict = load_world_from_path(args.trial_path) if args.trial_path else load_world(trial_name)
    color_map = {"obj1": colors[0], "obj2": colors[1], "obj3": colors[2]}
    if args.respect_world_tools:
        tools_dict = build_tools_from_world(world_dict, color_map) or build_tools(*colors)
    else:
        tools_dict = build_tools(*colors)
    display_order = preferred_three_ball_order()

    if args.human:
        prolific_id = args.prolific_id or input("Enter Prolific ID: ").strip()
        if not prolific_id:
            raise SystemExit("A Prolific ID is required for the participant task.")
        out_path = run_human_goal_session(
            trial_name=trial_name,
            tools_dict=tools_dict,
            display_order=display_order,
            prolific_id=prolific_id,
            world_template=world_dict,
            max_attempts=args.max_attempts,
            drive_root_id=args.drive_root_id,
            service_account_json=args.service_account_json,
        )
        print(f"[OK] Session JSON -> {out_path}")
        return

    if args.no_tool:
        drop_xy = None
    else:
        if not args.tool or not args.drop:
            raise SystemExit("Headless mode requires --tool and --drop, or use --no-tool, or use --human.")
        try:
            x_str, y_str = args.drop.split(",")
            drop_xy = (int(x_str.strip()), int(y_str.strip()))
        except Exception as exc:
            raise SystemExit("Bad --drop format. Use: --drop 250,180") from exc

    payload, saved_video_path = run_headless_episode(
        world_dict,
        tool_name=args.tool,
        tools_dict=tools_dict,
        drop_xy=drop_xy,
        no_tool=args.no_tool,
        record_video=args.video,
        video_dir=OUT_VID_DIR,
        video_basename=args.video_name,
    )

    os.makedirs(OUT_JSON_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tool_name = args.tool if args.tool else "no_tool"
    out_json = os.path.join(OUT_JSON_DIR, f"{trial_name}_{out_tool_name}_{timestamp}.json")
    payload["tool_manifest"] = {
        tool_name: {"color": tools_dict[tool_name]["color"], "display_index": display_order.index(tool_name)}
        for tool_name in VALID_TOOL_NAMES
    }
    with open(out_json, "w") as handle:
        json.dump(payload, handle, indent=2)

    obstruction = payload["placements"][0].get("obstruction_detected", False)
    if obstruction:
        print(f"[ERROR] Obstruction detected for {args.tool} at {drop_xy}")
        print(f"[ERROR] {payload['placements'][0].get('error', 'Unknown error')}")
    print(f"[OK] JSON -> {out_json}")
    if saved_video_path:
        print(f"[OK] Video -> {saved_video_path}")


if __name__ == "__main__":
    main()
