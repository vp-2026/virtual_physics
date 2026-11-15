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
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_DIR = (
    REPO_ROOT
    if os.path.basename(SCRIPT_DIR) == "gemini_scripts.py"
    else SCRIPT_DIR
)
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

import imageio
import pygame as pg
import pygame.gfxdraw
import pymunk

from pyGameWorld import ToolPicker
from pyGameWorld.helpers import word2Color
from pyGameWorld.viewer import drawWorld
from tool_config import (
    ALLOWED_TRIALS,
    VALID_TOOL_COLORS,
    VALID_TOOL_NAMES,
    build_default_tools,
    normalize_trial_name,
)


SIM_WIDTH = 600
SIM_HEIGHT = 600
PANEL_WIDTH = 300
TRIAL_SEARCH_DIRS = (
    os.path.join(ENV_DIR, "Trials", "Pilot"),
    os.path.join(ENV_DIR, "Trials", "New"),
)
OUT_JSON_DIR = os.path.join(REPO_ROOT, "outputs", "rollouts")
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
    return build_default_tools(color_obj1, color_obj2, color_obj3)


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


def _safe_rotation(obj_instance) -> Optional[float]:
    try:
        if hasattr(obj_instance, "isStatic") and obj_instance.isStatic():
            return None
        if hasattr(obj_instance, "getRot"):
            return float(obj_instance.getRot())
        if hasattr(obj_instance, "rotation"):
            return float(obj_instance.rotation)
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


def _snapshot_rotations_dynamic(pyworld) -> Dict[str, float]:
    out = {}
    for name, obj in pyworld.objects.items():
        rot = _safe_rotation(obj)
        if rot is not None:
            out[name] = rot
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


def _snapshot_rotations_movable(pyworld, movable_names: set) -> Dict[str, float]:
    out = {}
    for name, obj in pyworld.objects.items():
        if name not in movable_names:
            continue
        rot = _safe_rotation(obj)
        if rot is not None:
            out[name] = rot
    return out


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


def _goal_currently_satisfied(pyworld, world_dict: dict, target_obj_name: str, goal_name: Optional[str], goal_pts_world):
    """Return True only while the target is fully inside the goal right now."""
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


def _goal_satisfied(pyworld, world_dict: dict, target_obj_name: str, goal_name: Optional[str], goal_pts_world):
    """Prefer the engine win condition, with geometric fallback."""
    try:
        if pyworld.checkEnd():
            return True
    except Exception:
        pass
    return _goal_currently_satisfied(pyworld, world_dict, target_obj_name, goal_name, goal_pts_world)


def _set_body_gravity_vector(body, gravity_mode_name: str) -> None:
    def _custom_velocity_func(cp_body, gravity, damping, dt):
        try:
            gx, gy = gravity
        except Exception:
            gx, gy = gravity.x, gravity.y

        gmag = max(abs(float(gx)), abs(float(gy)))
        if gravity_mode_name == "upsidedown":
            custom_gravity = (0.0, gmag)
        else:
            custom_gravity = (float(gx), float(gy))

        pymunk.Body.update_velocity(cp_body, custom_gravity, damping, dt)

    body.velocity_func = _custom_velocity_func


def _apply_hidden_tool_behavior(tp, tool_name: str, tool_def: dict) -> None:
    placed_obj = tp._pyworld.objects.get("PLACED")
    if placed_obj is None or placed_obj.isStatic():
        return
    if tool_def.get("inverse_gravity", False):
        _set_body_gravity_vector(placed_obj._cpBody, "upsidedown")
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
    frame_dir: Optional[str] = None,
    frame_count: int = 32,
) -> dict:
    sim_width, sim_height = world_dict["world"]["dims"]
    pg.init()
    screen = pg.display.set_mode((sim_width, sim_height), flags=pg.HIDDEN)
    clock = pg.time.Clock()

    world = deepcopy(world_dict)
    tp = ToolPicker(world)
    sx = sy = None
    tool_def = None
    if not no_tool:
        if tool_name is None or drop_xy is None:
            raise ValueError("tool_name and drop_xy are required when no_tool is False")
        tool_def = tools_dict[tool_name]
        _assert_valid(tool_def["polys"], tool_name)
        sx, sy = int(drop_xy[0]), int(drop_xy[1])
        if tp._pyworld.checkCircleCollision((sx, sy), tool_def.get("placement_radius", 36)):
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
                    "error": "Drop coordinates overlap with existing object",
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
        translated_polys = []
        for poly in tool_def["polys"]:
            translated_polys.append([(vx + sx, vy + sy) for vx, vy in poly])

        tp._pyworld.addPlacedCompound(
            "PLACED",
            translated_polys,
            word2Color(tool_def["color"]),
            density=tool_def.get("density", 1.0),
            friction=tool_def.get("friction", 0.5),
            elasticity=tool_def.get("elasticity", 0.5),
        )
        _apply_hidden_tool_behavior(tp, tool_name, tool_def)

    sampled_frames = []
    frame_index = 0
    frame_interval = max(1, int(round((MAX_RECORDED_SECONDS * fps) / max(1, int(frame_count)))))

    movable_set = build_movable_name_set_from_env(world_dict)
    last_pos = {}
    moved_flag = {}
    trajectory = {}
    rotations = {}
    success_seen = False
    time_to_success = None
    required_goal_duration = float((world_dict.get("world", {}).get("gcond") or {}).get("duration", 0.0) or 0.0)
    in_goal_for = 0.0
    elapsed = 0.0
    last_movement_time = time.time()
    all_still = False
    move_eps_sq = 0.25 * 0.25

    running = True
    while running:
        tp._pyworld.step(1.0 / fps)

        frame_surf = drawWorld(tp._pyworld)
        screen.blit(frame_surf, (0, 0))
        pg.display.flip()

        if frame_dir is not None and elapsed < MAX_RECORDED_SECONDS and len(sampled_frames) < int(frame_count):
            if frame_index % frame_interval == 0:
                sampled_frames.append(pg.surfarray.array3d(screen).swapaxes(0, 1).copy())
        frame_index += 1

        frame_pos = _snapshot_positions_dynamic(tp._pyworld)
        frame_rot = _snapshot_rotations_dynamic(tp._pyworld)
        for name, pos_list in frame_pos.items():
            if elapsed < MAX_RECORDED_SECONDS:
                trajectory.setdefault(name, []).append(pos_list)
                if name in frame_rot:
                    rotations.setdefault(name, []).append(frame_rot[name])
            prev = last_pos.get(name)
            if prev is None:
                last_pos[name] = pos_list
            else:
                dx = pos_list[0] - prev[0]
                dy = pos_list[1] - prev[1]
                if dx * dx + dy * dy > move_eps_sq:
                    moved_flag[name] = True
                last_pos[name] = pos_list

        if not success_seen:
            if _goal_currently_satisfied(tp._pyworld, world_dict, target_obj_name, goal_name, goal_pts_world):
                in_goal_for += 1.0 / fps
                if in_goal_for >= required_goal_duration:
                    success_seen = True
                    time_to_success = elapsed
            else:
                in_goal_for = 0.0

        dynamic_objects = tp._pyworld.getDynamicObjects()
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

        pending_goal_streak = (not success_seen) and (in_goal_for > 0.0)
        if pending_goal_streak and all_still:
            last_movement_time = time.time()
        if all_still and (time.time() - last_movement_time >= stillness_duration):
            running = False

        elapsed += 1.0 / fps
        if elapsed >= MAX_SIM_SECONDS:
            running = False
        clock.tick(fps)

    landing_positions = _snapshot_positions_movable(tp._pyworld, movable_set)
    landing_rotations = _snapshot_rotations_movable(tp._pyworld, movable_set)
    movers_only = {name: traj for name, traj in trajectory.items() if moved_flag.get(name, False)}
    rotations_only = {
        name: rotations.get(name, [])
        for name in movers_only
        if rotations.get(name)
    }

    payload = {
        "placements": [{
            "placement_coords": [sx, sy] if (sx is not None and sy is not None) else None,
            "landing_positions": landing_positions,
            "landing_rotations": landing_rotations,
            "trajectory_data": movers_only,
            "rotation_data": rotations_only,
            "trajectory_window_seconds": MAX_RECORDED_SECONDS,
            "frame_source": "live_physics_step_drawWorld",
            "frame_sampling_fps": fps,
            "sampled_frame_count": len(sampled_frames),
            "trial_completed": True,
            "selected_tool": tool_name if not no_tool else NO_TOOL_LABEL,
            "selected_color": (tool_def["color"] if tool_def is not None else None),
            "success": success_seen,
            "time_to_success": time_to_success if success_seen else None,
            "phase": "goal",
            "no_tool_rollout": bool(no_tool),
        }],
    }

    if frame_dir is not None:
        os.makedirs(frame_dir, exist_ok=True)
        for index, frame in enumerate(sampled_frames, start=1):
            imageio.imwrite(os.path.join(frame_dir, f"frame_{index:02d}.png"), frame)
    pg.quit()
    return payload


def _sanitize_attempt_payload(payload: dict) -> dict:
    copy_payload = json.loads(json.dumps(payload))
    placement = copy_payload["placements"][0]
    placement["trajectory_frame_count"] = len(placement.get("trajectory_data", {}).get("PLACED", []))
    return copy_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-name", default="Basic", help="Goal environment name.")
    parser.add_argument("--trial-path", default=None, help="Direct path to a trial JSON file.")
    parser.add_argument("--drop", type=str, help="Placement coords x,y for headless mode.")
    parser.add_argument("--tool", choices=VALID_TOOL_NAMES, help="Tool name for headless mode.")
    parser.add_argument("--no-tool", action="store_true", help="Run a no-tool rollout in headless mode.")
    parser.add_argument("--color-obj1", choices=VALID_TOOL_COLORS)
    parser.add_argument("--color-obj2", choices=VALID_TOOL_COLORS)
    parser.add_argument("--color-obj3", choices=VALID_TOOL_COLORS)
    parser.add_argument("--frames-dir", default=None, help="Optional directory for 32 sampled rollout frames.")
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument(
        "--respect-world-tools",
        action="store_true",
        help="Use tool polygons from the trial JSON tools section instead of default ball tools.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trial_path:
        trial_name = os.path.splitext(os.path.basename(args.trial_path))[0]
    else:
        trial_name, _ = resolve_trial_name(args.trial_name)

    if args.color_obj1 and args.color_obj2 and args.color_obj3:
        if len({args.color_obj1, args.color_obj2, args.color_obj3}) != 3:
            raise SystemExit("Headless colors must be three distinct values.")
        colors = (args.color_obj1, args.color_obj2, args.color_obj3)
    else:
        colors = pick_three_colors()

    if args.no_tool:
        drop_xy = None
    else:
        if not args.tool or not args.drop:
            raise SystemExit("Headless mode requires --tool and --drop, or use --no-tool.")
        try:
            x_str, y_str = args.drop.split(",")
            drop_xy = (int(x_str.strip()), int(y_str.strip()))
        except Exception as exc:
            raise SystemExit("Bad --drop format. Use: --drop 250,180") from exc

    world_dict = load_world_from_path(args.trial_path) if args.trial_path else load_world(trial_name)
    color_map = {"obj1": colors[0], "obj2": colors[1], "obj3": colors[2]}
    if args.respect_world_tools:
        tools_dict = build_tools_from_world(world_dict, color_map) or build_tools(*colors)
    else:
        tools_dict = build_tools(*colors)
    display_order = preferred_three_ball_order()
    payload = run_headless_episode(
        world_dict,
        tool_name=args.tool,
        tools_dict=tools_dict,
        drop_xy=drop_xy,
        no_tool=args.no_tool,
        frame_dir=args.frames_dir,
        frame_count=args.frame_count,
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
    if args.frames_dir:
        print(f"[OK] Frames -> {args.frames_dir}")


if __name__ == "__main__":
    main()
