#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
VTOOLS_ROOT = REPO_ROOT / "132_base_environments"
PREDICATE_DATASET_PATH = HERE / "generate_predicate_dataset.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "goal_bank_sweeps"

TOOL_RADIUS_PX = 36.0
TRACE_STEP_S = 0.02
TRACE_MAXTIME_S = 30.0
TRACE_COLLISION_SLOP = 0.05
INITIAL_CONTACT_RELEVANCE_GAP_PX = 15.0
FINAL_VISIBLE_TOUCH_GAP_PX = 2.0
FINAL_GOAL_EXTERIOR_TOUCH_GAP_PX = 3.0
FINAL_TRANSIENT_SEPARATION_GAP_PX = 10.0
CONTAINER_EVENT_MIN_DURATION_S = 0.5
EVALUATION_GOAL_HOLD_S = 2.0
JOINT_MOVEMENT_MIN_DURATION_S = 0.5
FINAL_BOUNDARY_TOUCH_EPS_PX = 2.0
FINAL_ORIENTATION_AXIS_EPS_DEG = 5.0
FINAL_ORIENTATION_SLANTED_MIN_DEG = 15.0
FINAL_UPSIDE_DOWN_EPS_DEG = 10.0
RELATIVE_POSITION_EDGE_GAP_PX = 5.0
RELATIVE_POSITION_ALIGN_OVERLAP_EPS_PX = 2.0
CENTER_RELATIVE_POSITION_THRESHOLD_PX = 50.0
GLIDE_SIDE_MARGIN_PX = 5.0
GLIDE_VERTICAL_CLEARANCE_PX = 10.0
GLIDE_MIN_OVERLAP_SAMPLES = 2
ROTATION_FIRST_DIRECTION_EPS_DEG = 5.0
GOAL_INTERIOR_VISIBLE_SAMPLE_FRAC = 0.25

WALL_LABELS = {
    "_LeftWall": "left wall",
    "_RightWall": "right wall",
    "_BottomWall": "floor",
    "_TopWall": "ceiling",
}

WALL_KEYS = set(WALL_LABELS)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_predicate_module():
    predicate_dir = str(PREDICATE_DATASET_PATH.parent)
    if predicate_dir not in sys.path:
        sys.path.insert(0, predicate_dir)
    return _load_module(PREDICATE_DATASET_PATH, "goal_bank_predicates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep tool placements and build a per-puzzle goal bank from rollout event graphs.")
    parser.add_argument("--environment-json", default=None, help="Absolute path to a specific environment JSON.")
    parser.add_argument("--environment-set", default=None, help="Subfolder in VTools/environment_sets.")
    parser.add_argument("--env-id", default=None, help="Environment id within an environment set (e.g. 1).")
    parser.add_argument("--all-envs", action="store_true", help="When used with --environment-set, process all JSONs in that folder.")
    parser.add_argument("--condition", choices=["upward", "downward", "both"], default="downward")
    parser.add_argument("--grid-step", type=int, default=10, help="Spacing between candidate placement centers in pixels.")
    parser.add_argument("--movement-threshold-px", type=float, default=50.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--contact-min-duration-s", type=float, default=0.5, help="Minimum duration for contact events to count in aggregation.")
    parser.add_argument(
        "--terminal-persistence-s",
        type=float,
        default=0.0,
        help=(
            "Require final-state predicates to hold at every native simulator "
            "tick in the final N seconds; event predicates are unchanged."
        ),
    )
    parser.add_argument(
        "--persistence-signature-manifest",
        default=None,
        help=(
            "Optional benchmark manifest whose per-run final-state signatures "
            "are the only signatures persistence-scored during a resweep."
        ),
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-valid-placements", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--include-tool-events", action="store_true", help="Include tool-centered predicates in aggregate counts.")
    parser.add_argument("--save-structured-traces", action="store_true", help="Write one structured trace JSON per valid placement.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_words(name: str) -> str:
    name = re.sub(r"^_+", "", str(name))
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Za-z])([0-9]+)$", r"\1 \2", name)
    name = name.replace("_", " ")
    return re.sub(r"\s+", " ", name).strip().lower()


def object_points(spec: dict) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for key in ("vertices", "points"):
        values = spec.get(key) or []
        if isinstance(values, list):
            for item in values:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pts.append((float(item[0]), float(item[1])))
    polys = spec.get("polys") or []
    if isinstance(polys, list):
        for poly in polys:
            if isinstance(poly, list):
                for item in poly:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        pts.append((float(item[0]), float(item[1])))
    return pts


def object_center_from_spec(spec: dict) -> Optional[Tuple[float, float]]:
    for key in ("position", "pos", "center"):
        value = spec.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    pts = object_points(spec)
    if not pts:
        return None
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]
    return (sum(xs) / float(len(xs)), sum(ys) / float(len(ys)))


def _vec(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _vec_len(v: Tuple[float, float]) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _parallel(v1: Tuple[float, float], v2: Tuple[float, float], tol: float = 0.08) -> bool:
    n1 = _vec_len(v1)
    n2 = _vec_len(v2)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return False
    cross = abs(float(v1[0]) * float(v2[1]) - float(v1[1]) * float(v2[0]))
    return (cross / (n1 * n2)) <= tol


def classify_shape_name(spec: dict) -> Optional[str]:
    typ = str(spec.get("type", "")).lower()
    if typ == "container":
        return "container"
    if "radius" in spec or typ == "ball":
        return "ball"
    pts = object_points(spec)
    if len(pts) == 3:
        return "triangle"
    if len(pts) == 4:
        e01 = _vec(pts[0], pts[1])
        e12 = _vec(pts[1], pts[2])
        e23 = _vec(pts[2], pts[3])
        e30 = _vec(pts[3], pts[0])
        opp1 = _parallel(e01, e23)
        opp2 = _parallel(e12, e30)
        xs = [float(x) for x, _ in pts]
        ys = [float(y) for _, y in pts]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        shorter = min(width, height)
        longer = max(width, height)
        ratio = (longer / shorter) if shorter > 1e-6 else float("inf")
        if opp1 ^ opp2:
            return "trapezoid"
        if opp1 and opp2:
            if ratio >= 3.0:
                return "plank"
            if ratio <= 1.2:
                return "square"
            return "rectangle"
    return None


def black_object_label(raw_name: str, spec: dict) -> Optional[str]:
    raw_lower = str(raw_name).lower()
    if "support" in raw_lower:
        return "black support"
    if "slope" in raw_lower:
        return "black slope on the left"
    if "side1" in raw_lower:
        return "black left barrier"
    if "side2" in raw_lower:
        return "black right barrier"
    if str(spec.get("type", "")).lower() == "compound":
        return "black trapezoid platform"

    pts = object_points(spec)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return None
    min_x = min(xs)
    min_y = min(ys)
    ratio = width / height
    is_axis_aligned_rect = (
        str(spec.get("type", "")).lower() == "poly"
        and len({round(x, 3) for x in xs}) == 2
        and len({round(y, 3) for y in ys}) == 2
    )
    if is_axis_aligned_rect and width >= 200 and height >= 100 and (min_x <= 2 or min_y <= 2):
        return "black rectangular platform"
    if ratio >= 4.0:
        return "black rectangular platform"
    if ratio <= 0.3:
        return "black pillar"
    return "black obstacle"


def resolve_environment_paths(args: argparse.Namespace) -> List[Path]:
    if args.environment_json:
        return [Path(args.environment_json).expanduser().resolve()]

    if not args.environment_set:
        raise ValueError("Provide either --environment-json or --environment-set.")

    env_dir = (VTOOLS_ROOT / "environment_sets" / args.environment_set).resolve()
    if args.all_envs:
        paths = sorted(env_dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem))
        if not paths:
            raise FileNotFoundError(f"No JSON environments found in {env_dir}")
        return paths

    if not args.env_id:
        raise ValueError("Provide --env-id or use --all-envs.")
    env_path = env_dir / f"{args.env_id}.json"
    if not env_path.exists():
        raise FileNotFoundError(env_path)
    return [env_path.resolve()]


def humanize_name(raw_name: str, world_objects: dict, label_map: Dict[str, str]) -> str:
    if raw_name in label_map:
        return label_map[raw_name]
    wall_label = wall_label_for_raw_name(raw_name, world_objects)
    if wall_label is not None:
        return wall_label
    if raw_name in {"Goal", "GoalContainer"}:
        return "green container"

    obj = world_objects.get(raw_name) or {}
    base = split_words(raw_name)
    color = str(obj.get("color") or obj.get("innerColor") or "").strip().lower()
    shape_name = classify_shape_name(obj)

    if color:
        if color == "green" and "container" in str(obj.get("type", "")).lower():
            return "green container"
        if color == "black":
            special = black_object_label(raw_name, obj)
            if special:
                return special
        if shape_name and (
            raw_name.lower().startswith("object")
            or "rectangle" in base
            or "object" in base
        ):
            return f"{color} {shape_name}".strip()
        if color != "black" and color not in base:
            return f"{color} {base}".strip()
    return base


def canonical_contact_label(raw_name: str, world_objects: dict, label_map: Dict[str, str]) -> str:
    wall_label = wall_label_for_raw_name(raw_name, world_objects)
    if wall_label is not None:
        return wall_label
    if raw_name in {"Goal", "GoalContainer"}:
        return "green container"
    return humanize_name(raw_name, world_objects, label_map)


def goal_exterior_allowed(world_data: dict) -> bool:
    env_set = str(world_data.get("_env_set") or "")
    env_id = str(world_data.get("_env_id") or "")
    if env_set in {"Prevention", "FallingAlt", "BalanceUnder"}:
        return False
    if env_set == "BackUp" and env_id == "3":
        return False
    if env_set == "Falling" and env_id in {"1", "7", "8", "9"}:
        return False
    return True


def glide_allowed(world_data: dict) -> bool:
    env_set = str(world_data.get("_env_set") or "")
    if env_set == "Prevention":
        return False
    return True


def wall_label_for_raw_name(raw_name: str, world_objects: dict) -> Optional[str]:
    if raw_name in {"_LeftWall", "_RightWall", "_BottomWall", "_TopWall"}:
        return WALL_LABELS[raw_name]
    spec = world_objects.get(raw_name) or {}
    raw_lower = str(raw_name).lower()
    pts = object_points(spec)
    xs = [float(x) for x, _ in pts] if pts else []
    ys = [float(y) for _, y in pts] if pts else []
    min_x = min(xs) if xs else None
    max_x = max(xs) if xs else None
    min_y = min(ys) if ys else None
    max_y = max(ys) if ys else None
    if raw_lower == "rightwall":
        if max_x is not None and max_x >= 598.0:
            return "right wall"
        return "black rectangle on the right"
    if raw_lower == "leftwall":
        if min_x is not None and min_x <= 2.0:
            return "left wall"
        return "black rectangle on the left"
    if raw_lower == "topwall":
        return "black upper triangular barrier"
    if raw_lower == "bottomwall":
        if min_y is not None and min_y <= 2.0:
            return "floor"
        return "black lower barrier"
    if raw_name in {"LeftWall", "RightWall", "BottomWall", "TopWall"}:
        return humanize_name(raw_name, world_objects, {})
    return None


def choose_contact_subject(
    raw_a: str,
    raw_b: str,
    *,
    label_a: str,
    label_b: str,
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    include_tool_events: bool,
) -> Tuple[str, str, str, str]:
    def score(raw_name: str) -> Tuple[int, int, int, str]:
        role = role_map.get(raw_name, split_words(raw_name))
        is_dynamic = int(dynamic_map.get(raw_name, False))
        is_tool = int(role == "tool")
        is_goal = int(role == "goal")
        return (
            0 if is_dynamic and not is_tool and not is_goal else 1,
            0 if is_dynamic and not is_goal else 1,
            0 if role != "wall" else 1,
            role,
        )

    first, second = (raw_a, label_a, raw_b, label_b), (raw_b, label_b, raw_a, label_a)
    if score(raw_b) < score(raw_a):
        first, second = second, first
    return first


def contact_display(subject_label: str, other_label: str, lasting_to_final: bool) -> str:
    if other_label == "green container":
        tail = "and is still touching the outside of the green container at the final frame" if lasting_to_final else "touches the outside of the green container transiently"
        return f"{subject_label} {tail}"
    if lasting_to_final:
        return f"{subject_label} contacts {other_label} and is still touching it at the final frame"
    return f"{subject_label} contacts {other_label} transiently"


def build_label_maps(pred: Any, world_data: dict) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, bool]]:
    target, goal, blues, movable, static_black = pred.describe_objects(world_data)
    label_map: Dict[str, str] = {}
    role_map: Dict[str, str] = {}
    dynamic_map: Dict[str, bool] = {}
    env_set = str(world_data.get("_env_set") or "")

    for obj in [target, goal, *blues, *movable, *static_black]:
        if obj is None:
            continue
        raw_name = str(obj.name)
        label = getattr(obj, "label", None) or split_words(raw_name)
        role = getattr(obj, "role", None) or split_words(raw_name)
        if str(role) == "goal":
            label = "green container"
        if str(role) == "target":
            label = "red ball"
        if env_set == "Falling" and str(role) == "blue":
            label = "blue U-shaped cradle"
        label_map[raw_name] = str(label).lower()
        role_map[raw_name] = str(role)
        dynamic_map[raw_name] = raw_name not in WALL_KEYS and raw_name not in [item.name for item in static_black]

    for wall_name, wall_label in WALL_LABELS.items():
        label_map.setdefault(wall_name, wall_label)
        role_map.setdefault(wall_name, "wall")
        dynamic_map.setdefault(wall_name, False)

    world_objects = (world_data.get("world") or {}).get("objects") or {}
    for raw_name in world_objects:
        label_map.setdefault(raw_name, humanize_name(raw_name, world_objects, label_map))
        role_map.setdefault(raw_name, split_words(raw_name))
        dynamic_map.setdefault(raw_name, raw_name not in WALL_KEYS and str(world_objects[raw_name].get("color", "")).lower() != "black")
    for raw_name, spec in world_objects.items():
        if str(spec.get("color", "")).lower() == "black" or str(spec.get("type", "")).lower() == "container":
            label_map[raw_name] = humanize_name(raw_name, world_objects, {})
            continue
        shape_name = classify_shape_name(spec)
        if not shape_name:
            continue
        current_label = str(label_map.get(raw_name, ""))
        raw_lower = raw_name.lower()
        if raw_lower.startswith("object") or "rectangle" in current_label or "object" in current_label:
            color = str(spec.get("color") or "").strip().lower()
            if color:
                label_map[raw_name] = f"{color} {shape_name}".strip()

    if env_set == "Falling":
        for raw_name, role in role_map.items():
            if role == "blue":
                label_map[raw_name] = "blue U-shaped cradle"

    label_map["PLACED"] = "big orange ball (dropped tool)"
    role_map["PLACED"] = "tool"
    dynamic_map["PLACED"] = True
    return label_map, role_map, dynamic_map


def initial_relevant_contact_pairs(
    pred: Any,
    *,
    world_obj: Any,
    world_data: dict,
    min_gap_px: float,
    include_walls: bool = True,
) -> set[Tuple[str, str]]:
    names = [
        str(name)
        for name in world_obj.objects.keys()
        if include_walls or str(name) not in WALL_KEYS
    ]
    out: set[Tuple[str, str]] = set()
    for idx, a_name in enumerate(names):
        for b_name in names[idx + 1:]:
            gap = None
            try:
                gap = pred.min_object_gap_px(world_obj, world_data, a_name, b_name)
            except Exception:
                gap = None
            if gap is not None and float(gap) <= float(min_gap_px):
                out.add(tuple(sorted((a_name, b_name))))
    return out


def generate_grid(dims: Sequence[int], step: int, margin: float) -> List[Tuple[int, int]]:
    width, height = int(dims[0]), int(dims[1])
    low = int(math.ceil(margin))
    high_x = int(math.floor(width - margin))
    high_y = int(math.floor(height - margin))
    xs = list(range(low, high_x + 1, int(step)))
    ys = list(range(low, high_y + 1, int(step)))
    return [(x, y) for y in ys for x in xs]


def interval_ends_at_final(end_s: float, duration_s: float, tol: float = 0.05) -> bool:
    return abs(float(end_s) - float(duration_s)) <= tol


def is_dynamic_candidate(raw_name: str, role_map: Dict[str, str], dynamic_map: Dict[str, bool], include_tool_events: bool) -> bool:
    if raw_name in WALL_KEYS:
        return False
    if not dynamic_map.get(raw_name, False):
        return False
    if not include_tool_events and role_map.get(raw_name) == "tool":
        return False
    if role_map.get(raw_name) == "goal":
        return False
    return True


def build_structured_trace(
    pred: Any,
    *,
    world_data: dict,
    target: Any,
    goal: Any,
    blues: Sequence[Any],
    movable: Sequence[Any],
    trace: dict,
    rollout_id: str,
) -> dict:
    return pred.build_structured_trace(
        puzzle_id=str(world_data.get("_source_name") or "runtime_world"),
        rollout_id=rollout_id,
        has_tool=True,
        simple_trace=trace,
        objects=pred.trace_object_metas(target, goal, blues, movable, has_tool=True),
        step_s=TRACE_STEP_S,
        thresholds=pred.TraceThresholds(goal_hold_s=EVALUATION_GOAL_HOLD_S),
    )


def extract_movement_events(
    *,
    raw_name: str,
    label: str,
    role: str,
    path_samples: Sequence[Sequence[float]],
    movement_threshold_px: float,
) -> Tuple[List[dict], dict]:
    if not path_samples:
        return [], {
            "initial_position_xy": [None, None],
            "final_position_xy": [None, None],
            "displacement_xy": [None, None],
            "max_abs_displacement_xy": [None, None],
        }

    start_x = float(path_samples[0][0])
    start_y = float(path_samples[0][1])
    max_dx = 0.0
    min_dx = 0.0
    max_dy = 0.0
    min_dy = 0.0
    first_hits = {"right": None, "left": None, "up": None, "down": None}

    for idx, sample in enumerate(path_samples):
        dx = float(sample[0]) - start_x
        dy = float(sample[1]) - start_y
        max_dx = max(max_dx, dx)
        min_dx = min(min_dx, dx)
        max_dy = max(max_dy, dy)
        min_dy = min(min_dy, dy)
        if first_hits["right"] is None and dx >= movement_threshold_px:
            first_hits["right"] = idx * TRACE_STEP_S
        if first_hits["left"] is None and dx <= -movement_threshold_px:
            first_hits["left"] = idx * TRACE_STEP_S
        if first_hits["up"] is None and dy >= movement_threshold_px:
            first_hits["up"] = idx * TRACE_STEP_S
        if first_hits["down"] is None and dy <= -movement_threshold_px:
            first_hits["down"] = idx * TRACE_STEP_S

    events = []
    if first_hits["right"] is not None:
        events.append(
            {
                "category": "movement",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "axis": "x",
                "direction": "right",
                "threshold_px": movement_threshold_px,
                "time_s": round(float(first_hits["right"]), 3),
                "display": f"{label}'s center point moves right by at least {int(movement_threshold_px)} px at any point",
            }
        )
    if first_hits["left"] is not None:
        events.append(
            {
                "category": "movement",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "axis": "x",
                "direction": "left",
                "threshold_px": movement_threshold_px,
                "time_s": round(float(first_hits["left"]), 3),
                "display": f"{label}'s center point moves left by at least {int(movement_threshold_px)} px at any point",
            }
        )
    if first_hits["up"] is not None:
        events.append(
            {
                "category": "movement",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "axis": "y",
                "direction": "up",
                "threshold_px": movement_threshold_px,
                "time_s": round(float(first_hits["up"]), 3),
                "display": f"{label}'s center point moves up by at least {int(movement_threshold_px)} px at any point",
            }
        )
    if first_hits["down"] is not None:
        events.append(
            {
                "category": "movement",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "axis": "y",
                "direction": "down",
                "threshold_px": movement_threshold_px,
                "time_s": round(float(first_hits["down"]), 3),
                "display": f"{label}'s center point moves down by at least {int(movement_threshold_px)} px at any point",
            }
        )

    final_x = float(path_samples[-1][0])
    final_y = float(path_samples[-1][1])
    kinematics = {
        "initial_position_xy": [start_x, start_y],
        "final_position_xy": [final_x, final_y],
        "displacement_xy": [final_x - start_x, final_y - start_y],
        "max_abs_displacement_xy": [max(abs(max_dx), abs(min_dx)), max(abs(max_dy), abs(min_dy))],
    }
    return events, kinematics


def movement_threshold_intervals(
    *,
    path_samples: Sequence[Sequence[float]],
    movement_threshold_px: float,
    direction: str,
) -> List[List[float]]:
    if not path_samples:
        return []
    start_x = float(path_samples[0][0])
    start_y = float(path_samples[0][1])
    intervals: List[List[float]] = []
    active_start: Optional[float] = None
    for idx, sample in enumerate(path_samples):
        dx = float(sample[0]) - start_x
        dy = float(sample[1]) - start_y
        is_on = False
        if direction == "right":
            is_on = dx >= movement_threshold_px
        elif direction == "left":
            is_on = dx <= -movement_threshold_px
        elif direction == "up":
            is_on = dy >= movement_threshold_px
        elif direction == "down":
            is_on = dy <= -movement_threshold_px
        t = idx * TRACE_STEP_S
        if is_on and active_start is None:
            active_start = t
        elif (not is_on) and active_start is not None:
            intervals.append([float(active_start), float(t)])
            active_start = None
    if active_start is not None:
        intervals.append([float(active_start), float((len(path_samples) - 1) * TRACE_STEP_S)])
    return merge_intervals(intervals)


def intersect_intervals(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    i = 0
    j = 0
    aa = merge_intervals(a)
    bb = merge_intervals(b)
    while i < len(aa) and j < len(bb):
        start_s = max(float(aa[i][0]), float(bb[j][0]))
        end_s = min(float(aa[i][1]), float(bb[j][1]))
        if end_s > start_s:
            out.append([start_s, end_s])
        if float(aa[i][1]) < float(bb[j][1]):
            i += 1
        else:
            j += 1
    return merge_intervals(out)


def extract_joint_movement_events(
    *,
    path_map: Dict[str, Sequence[Sequence[float]]],
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    movement_threshold_px: float,
    include_tool_events: bool,
) -> List[dict]:
    candidate_names = []
    for raw_name, samples in path_map.items():
        if not samples or not dynamic_map.get(raw_name, False):
            continue
        if (not include_tool_events) and role_map.get(raw_name) == "tool":
            continue
        candidate_names.append(str(raw_name))
    candidate_names = sorted(set(candidate_names))

    intervals_by_obj_dir: Dict[Tuple[str, str], List[List[float]]] = {}
    for raw_name in candidate_names:
        samples = path_map.get(raw_name) or []
        for direction in ("right", "left", "up", "down"):
            intervals_by_obj_dir[(raw_name, direction)] = movement_threshold_intervals(
                path_samples=samples,
                movement_threshold_px=movement_threshold_px,
                direction=direction,
            )

    events: List[dict] = []
    for idx_a in range(len(candidate_names)):
        for idx_b in range(idx_a + 1, len(candidate_names)):
            raw_a = candidate_names[idx_a]
            raw_b = candidate_names[idx_b]
            label_a = label_map.get(raw_a, split_words(raw_a))
            label_b = label_map.get(raw_b, split_words(raw_b))
            object_a, object_b = sorted([label_a, label_b])
            for direction in ("right", "left", "up", "down"):
                overlap = intersect_intervals(
                    intervals_by_obj_dir.get((raw_a, direction), []),
                    intervals_by_obj_dir.get((raw_b, direction), []),
                )
                if not overlap:
                    continue
                total_duration = sum(max(0.0, float(end_s) - float(start_s)) for start_s, end_s in overlap)
                if total_duration < JOINT_MOVEMENT_MIN_DURATION_S:
                    continue
                axis = "x" if direction in {"right", "left"} else "y"
                events.append(
                    {
                        "category": "joint_movement",
                        "object": object_a,
                        "other_object": object_b,
                        "object_ids": [raw_a, raw_b],
                        "roles": [role_map.get(raw_a, split_words(raw_a)), role_map.get(raw_b, split_words(raw_b))],
                        "axis": axis,
                        "direction": direction,
                        "threshold_px": movement_threshold_px,
                        "time_s": round(float(overlap[0][0]), 3),
                        "end_s": round(float(overlap[-1][1]), 3),
                        "duration_s": round(float(total_duration), 3),
                        "display": f"{object_a} and {object_b}'s center points move {direction} by at least {int(movement_threshold_px)} px together for at least {JOINT_MOVEMENT_MIN_DURATION_S:.1f} s",
                    }
                )
    return events


def object_bbox_at_sample(
    *,
    world_data: dict,
    trace: dict,
    object_name: str,
    sample_idx: int,
) -> Optional[Tuple[float, float, float, float]]:
    world_objects = (world_data.get("world") or {}).get("objects") or {}
    spec = world_objects.get(object_name) or {}
    path_map = trace.get("path", {}) or {}
    rot_map = trace.get("rot", {}) or {}

    if "radius" in spec:
        path_samples = path_map.get(object_name) or []
        center = object_center_from_spec(spec)
        if path_samples:
            clamped_idx = min(max(int(sample_idx), 0), len(path_samples) - 1)
            center = (float(path_samples[clamped_idx][0]), float(path_samples[clamped_idx][1]))
        if center is None:
            return None
        radius = float(spec.get("radius", 0.0) or 0.0)
        if radius <= 0:
            return None
        cx, cy = center
        return (cx - radius, cy - radius, cx + radius, cy + radius)

    pts = object_points(spec)
    if not pts:
        return None
    base_center = object_center_from_spec(spec)
    if base_center is None:
        return None
    path_samples = path_map.get(object_name) or []
    rot_samples = rot_map.get(object_name) or []
    if path_samples:
        clamped_idx = min(max(int(sample_idx), 0), len(path_samples) - 1)
        tx = float(path_samples[clamped_idx][0])
        ty = float(path_samples[clamped_idx][1])
    else:
        tx, ty = base_center
    angle = 0.0
    if rot_samples:
        clamped_idx = min(max(int(sample_idx), 0), len(rot_samples) - 1)
        angle = float(rot_samples[clamped_idx])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = base_center
    transformed: List[Tuple[float, float]] = []
    for px, py in pts:
        lx = float(px) - cx
        ly = float(py) - cy
        wx = tx + lx * cos_a - ly * sin_a
        wy = ty + lx * sin_a + ly * cos_a
        transformed.append((wx, wy))
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    return (min(xs), min(ys), max(xs), max(ys))


def _pair_has_any_collision(
    collisions: Sequence[Sequence[Any]],
    a_name: str,
    b_name: str,
) -> bool:
    target = tuple(sorted((str(a_name), str(b_name))))
    for collision in collisions:
        pair = tuple(sorted((str(collision[0]), str(collision[1]))))
        if pair == target:
            return True
    return False


def _glide_event_for_pair(
    *,
    subject_name: str,
    subject_label: str,
    subject_role: str,
    other_name: str,
    other_label: str,
    trace: dict,
    world_data: dict,
    movement_threshold_px: float,
    direction: str,
) -> Optional[dict]:
    path_map = trace.get("path", {}) or {}
    subject_path = path_map.get(subject_name) or []
    if len(subject_path) < 2:
        return None

    sign = 1.0 if direction == "right" else -1.0
    active = False
    entered_overlap = False
    overlap_samples = 0
    start_x = None
    start_time_s = None
    first_clear_cross_time_s = None

    for idx, sample in enumerate(subject_path):
        bbox_other = object_bbox_at_sample(
            world_data=world_data,
            trace=trace,
            object_name=other_name,
            sample_idx=idx,
        )
        if bbox_other is None:
            active = False
            entered_overlap = False
            overlap_samples = 0
            start_x = None
            start_time_s = None
            continue
        ox0, oy0, ox1, oy1 = bbox_other
        sx = float(sample[0])
        sy = float(sample[1])
        left_bound = ox0 - GLIDE_SIDE_MARGIN_PX
        right_bound = ox1 + GLIDE_SIDE_MARGIN_PX
        is_on_start_side = sx <= left_bound if sign > 0 else sx >= right_bound
        is_past_far_side = sx >= right_bound if sign > 0 else sx <= left_bound
        overlaps_x = left_bound <= sx <= right_bound
        above_other = sy >= oy1 + GLIDE_VERTICAL_CLEARANCE_PX

        if not active:
            if is_on_start_side:
                active = True
                entered_overlap = False
                overlap_samples = 0
                start_x = sx
                start_time_s = idx * TRACE_STEP_S
            continue

        if overlaps_x:
            entered_overlap = True
            if not above_other:
                active = False
                entered_overlap = False
                overlap_samples = 0
                start_x = None
                start_time_s = None
                first_clear_cross_time_s = None
                continue
            overlap_samples += 1

        if entered_overlap and is_past_far_side:
            if first_clear_cross_time_s is None:
                first_clear_cross_time_s = idx * TRACE_STEP_S
            horizontal_delta = sign * (sx - float(start_x if start_x is not None else sx))
            if overlap_samples >= GLIDE_MIN_OVERLAP_SAMPLES and horizontal_delta >= movement_threshold_px:
                return {
                    "category": "pass_over",
                    "object": subject_label,
                    "object_id": subject_name,
                    "role": subject_role,
                    "other_object": other_label,
                    "direction": direction,
                    "threshold_px": movement_threshold_px,
                    "time_s": round(float(first_clear_cross_time_s), 3),
                    "display": f"{subject_label}'s center point passes over {other_label} to the {direction} while staying above it",
                }

        if entered_overlap and not above_other:
            active = False
            entered_overlap = False
            overlap_samples = 0
            start_x = None
            start_time_s = None
            first_clear_cross_time_s = None

    return None


def extract_glide_events(
    *,
    trace: dict,
    world_data: dict,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    movement_threshold_px: float,
    initial_contact_pairs: set[Tuple[str, str]],
    include_tool_events: bool,
) -> List[dict]:
    if not glide_allowed(world_data):
        return []
    collisions = trace.get("collisions", []) or []
    path_map = trace.get("path", {}) or {}
    world_objects = (world_data.get("world") or {}).get("objects") or {}
    static_names = [name for name in world_objects.keys() if name not in path_map]
    candidate_others = sorted(set(list(path_map.keys()) + static_names))
    events: List[dict] = []

    for subject_name, subject_path in path_map.items():
        if len(subject_path) < 2:
            continue
        subject_role = role_map.get(subject_name, split_words(subject_name))
        if not is_dynamic_candidate(subject_name, role_map, dynamic_map, include_tool_events):
            continue
        for other_name in candidate_others:
            if other_name == subject_name or other_name in WALL_KEYS:
                continue
            other_role = role_map.get(other_name, split_words(other_name))
            if other_role == "wall":
                continue
            pair_key = tuple(sorted((str(subject_name), str(other_name))))
            if pair_key in initial_contact_pairs:
                continue
            if _pair_has_any_collision(collisions, subject_name, other_name):
                continue
            other_label = canonical_contact_label(other_name, world_objects, label_map)
            if other_label == "floor" or other_label == "ceiling":
                continue
            subject_label = label_map.get(subject_name, split_words(subject_name))
            for direction in ("right", "left"):
                event = _glide_event_for_pair(
                    subject_name=subject_name,
                    subject_label=subject_label,
                    subject_role=subject_role,
                    other_name=other_name,
                    other_label=other_label,
                    trace=trace,
                    world_data=world_data,
                    movement_threshold_px=movement_threshold_px,
                    direction=direction,
                )
                if event is not None:
                    events.append(event)
    deduped: List[dict] = []
    seen = set()
    for event in events:
        sig = (event["object"], event["other_object"], event["direction"], int(event["threshold_px"]))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(event)
    return deduped


def extract_rotation_events(
    *,
    raw_name: str,
    label: str,
    role: str,
    spec: dict,
    rotation_samples: Sequence[float],
    rotation_threshold_deg: float,
) -> Tuple[List[dict], dict]:
    if "radius" in spec or not rotation_samples:
        return [], {
            "initial_angle_rad": None,
            "final_angle_rad": None,
            "rotation_delta_deg": None,
            "max_abs_rotation_deg": None,
        }

    start = float(rotation_samples[0])
    threshold_rad = math.radians(rotation_threshold_deg)
    first_direction_eps_rad = math.radians(ROTATION_FIRST_DIRECTION_EPS_DEG)
    first_direction: Optional[str] = None
    first_direction_time_s: Optional[float] = None
    max_ccw = 0.0
    max_cw = 0.0

    for idx, value in enumerate(rotation_samples):
        delta = float(value) - start
        max_ccw = max(max_ccw, delta)
        max_cw = min(max_cw, delta)
        if first_direction is None:
            if delta >= first_direction_eps_rad:
                first_direction = "counterclockwise"
                first_direction_time_s = idx * TRACE_STEP_S
            elif delta <= -first_direction_eps_rad:
                first_direction = "clockwise"
                first_direction_time_s = idx * TRACE_STEP_S

    events = []
    if first_direction == "counterclockwise" and max_ccw >= threshold_rad:
        events.append(
            {
                "category": "rotation",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "direction": "counterclockwise",
                "threshold_deg": rotation_threshold_deg,
                "time_s": round(float(first_direction_time_s or 0.0), 3),
                "display": f"{label} rotates counterclockwise by at least {int(rotation_threshold_deg)} degrees at any point",
            }
        )
    elif first_direction == "clockwise" and abs(max_cw) >= threshold_rad:
        events.append(
            {
                "category": "rotation",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "direction": "clockwise",
                "threshold_deg": rotation_threshold_deg,
                "time_s": round(float(first_direction_time_s or 0.0), 3),
                "display": f"{label} rotates clockwise by at least {int(rotation_threshold_deg)} degrees at any point",
            }
        )

    final_delta = float(rotation_samples[-1]) - start
    kinematics = {
        "initial_angle_rad": start,
        "final_angle_rad": float(rotation_samples[-1]),
        "rotation_delta_deg": math.degrees(final_delta),
        "max_abs_rotation_deg": max(math.degrees(max_ccw), abs(math.degrees(max_cw))),
    }
    return events, kinematics


def extract_contact_events(
    *,
    pred: Any,
    collisions: Sequence[Sequence[Any]],
    duration_s: float,
    world_data: dict,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    final_contacts: Dict[str, Sequence[str]],
    object_in_goal_intervals: Dict[str, Sequence[Sequence[float]]],
    goal_name: Optional[str],
    initial_contact_pairs: set[Tuple[str, str]],
    final_world_obj: Any,
    world_objects: dict,
    contact_min_duration_s: float,
    include_tool_events: bool,
) -> Tuple[List[dict], List[dict]]:
    final_contact_label_sets: Dict[str, set[str]] = {
        raw_name: {
            canonical_contact_label(other_name, world_objects, label_map)
            for other_name in set(other_names or [])
        }
        for raw_name, other_names in final_contacts.items()
    }
    grouped: Dict[Tuple[str, str], dict] = {}
    ended_contact_events: List[dict] = []
    for collision in collisions:
        raw_a = str(collision[0])
        raw_b = str(collision[1])
        if not (dynamic_map.get(raw_a, False) or dynamic_map.get(raw_b, False)):
            continue
        if (not include_tool_events) and ("tool" in {role_map.get(raw_a), role_map.get(raw_b)}):
            continue
        start_s = float(collision[2] or 0.0)
        end_s = float(collision[3]) if collision[3] is not None else float(duration_s)
        interval_duration = max(0.0, end_s - start_s)
        if interval_duration < float(contact_min_duration_s):
            continue
        pair_raw_key = tuple(sorted((raw_a, raw_b)))
        label_a = canonical_contact_label(raw_a, world_objects, label_map)
        label_b = canonical_contact_label(raw_b, world_objects, label_map)
        if goal_name and goal_name in {raw_a, raw_b}:
            non_goal_name = raw_b if raw_a == goal_name else raw_a
            if object_in_goal_intervals.get(non_goal_name):
                continue
            if not goal_exterior_allowed(world_data):
                continue
        key = tuple(sorted([label_a, label_b]))
        lasting_to_final = label_b in final_contact_label_sets.get(raw_a, set()) or label_a in final_contact_label_sets.get(raw_b, set())
        final_gap = final_gap_px(pred, world_obj=final_world_obj, world_data=world_data, a_name=raw_a, b_name=raw_b)
        visibly_separated_at_final = final_gap is not None and final_gap >= FINAL_TRANSIENT_SEPARATION_GAP_PX
        subject_raw, subject_label, other_raw, other_label = choose_contact_subject(
            raw_a,
            raw_b,
            label_a=label_a,
            label_b=label_b,
            role_map=role_map,
            dynamic_map=dynamic_map,
            include_tool_events=include_tool_events,
        )
        if pair_raw_key in initial_contact_pairs:
            if visibly_separated_at_final:
                ended_contact_events.append(
                    {
                        "category": "contact_change",
                        "subtype": "no_longer_touching",
                        "object": subject_label,
                        "object_id": subject_raw,
                        "role": role_map.get(subject_raw, split_words(subject_raw)),
                        "other_object": other_label,
                        "time_s": None,
                        "display": f"{subject_label} no longer touches {other_label} at the final frame",
                    }
                )
            continue
        if lasting_to_final or not visibly_separated_at_final:
            continue
        bucket = grouped.setdefault(
            key,
            {
                "category": "contact",
                "objects": sorted([label_a, label_b]),
                "object_ids": sorted([subject_raw, other_raw]),
                "roles": sorted([role_map.get(raw_a, split_words(raw_a)), role_map.get(raw_b, split_words(raw_b))]),
                "subject": subject_label,
                "other_object": other_label,
                "time_s": start_s,
                "duration_s": 0.0,
                "lasting_to_final": False,
            },
        )
        bucket["time_s"] = min(float(bucket["time_s"]), start_s)
        bucket["duration_s"] += interval_duration
        bucket["lasting_to_final"] = bool(bucket["lasting_to_final"] or lasting_to_final)

    events: List[dict] = []
    for bucket in grouped.values():
        display = contact_display(bucket["subject"], bucket["other_object"], bool(bucket["lasting_to_final"]))
        events.append(
            {
                **bucket,
                "time_s": round(float(bucket["time_s"]), 3),
                "duration_s": round(float(bucket["duration_s"]), 3),
                "display": display,
            }
        )
    deduped_ended: List[dict] = []
    seen_changes = set()
    for event in ended_contact_events:
        sig = (event["object"], event["other_object"], event["subtype"])
        if sig in seen_changes:
            continue
        seen_changes.add(sig)
        deduped_ended.append(event)
    return events, deduped_ended


def extract_container_events(
    *,
    object_in_goal_intervals: Dict[str, Sequence[Sequence[float]]],
    duration_s: float,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    final_partial_in_goal_object_ids: set[str],
    include_tool_events: bool,
) -> List[dict]:
    events: List[dict] = []
    for raw_name, intervals in object_in_goal_intervals.items():
        if role_map.get(raw_name) == "goal":
            continue
        if (not include_tool_events) and role_map.get(raw_name) == "tool":
            continue
        label = label_map.get(raw_name, split_words(raw_name))
        if not intervals:
            continue
        if raw_name in final_partial_in_goal_object_ids:
            continue
        starts = [float(start_s) for start_s, _ in intervals]
        ends = [float(end_s) for _, end_s in intervals]
        total_duration = sum(max(0.0, float(end_s) - float(start_s)) for start_s, end_s in intervals)
        if total_duration < CONTAINER_EVENT_MIN_DURATION_S:
            continue
        lasting_to_final = any(interval_ends_at_final(float(end_s), duration_s) for _, end_s in intervals)
        if lasting_to_final:
            continue
        events.append(
            {
                "category": "container",
                "subtype": "enter_container",
                "object": label,
                "object_id": raw_name,
                "role": role_map.get(raw_name, split_words(raw_name)),
                "time_s": round(min(starts), 3),
                "end_s": round(max(ends), 3),
                "duration_s": round(total_duration, 3),
                "lasting_to_final": False,
                "display": f"{label} enters the green container transiently",
            }
        )
    return events


def merge_intervals(intervals: Sequence[Sequence[float]], *, eps: float = 1e-6) -> List[List[float]]:
    cleaned: List[Tuple[float, float]] = []
    for item in intervals:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start_s = float(item[0])
        end_s = float(item[1])
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        cleaned.append((start_s, end_s))
    if not cleaned:
        return []
    cleaned.sort()
    merged: List[List[float]] = [[cleaned[0][0], cleaned[0][1]]]
    for start_s, end_s in cleaned[1:]:
        last = merged[-1]
        if start_s <= last[1] + eps:
            last[1] = max(last[1], end_s)
        else:
            merged.append([start_s, end_s])
    return merged


def geometry_object_in_goal_intervals(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    role_map: Dict[str, str],
    include_tool_events: bool,
) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    path_map = trace.get("path", {}) or {}
    duration_s = float(trace.get("duration") or 0.0)
    for raw_name, samples in path_map.items():
        if role_map.get(raw_name) == "goal":
            continue
        if (not include_tool_events) and role_map.get(raw_name) == "tool":
            continue
        intervals: List[List[float]] = []
        active_start: Optional[float] = None
        for idx in range(len(samples)):
            t = min(idx * TRACE_STEP_S, duration_s)
            inside = object_visible_inside_goal_at_time(
                pred,
                trace=trace,
                world_data=world_data,
                object_name=raw_name,
                t=t,
            )
            if inside and active_start is None:
                active_start = t
            elif (not inside) and active_start is not None:
                intervals.append([float(active_start), float(t)])
                active_start = None
        if active_start is not None:
            intervals.append([float(active_start), float(duration_s)])
        if intervals:
            out[str(raw_name)] = merge_intervals(intervals)
    return out


def effective_object_in_goal_intervals(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    role_map: Dict[str, str],
    include_tool_events: bool,
) -> Dict[str, List[List[float]]]:
    existing = {
        str(raw_name): merge_intervals(intervals or [])
        for raw_name, intervals in (trace.get("object_in_goal_intervals", {}) or {}).items()
    }
    geom = geometry_object_in_goal_intervals(
        pred,
        trace=trace,
        world_data=world_data,
        role_map=role_map,
        include_tool_events=include_tool_events,
    )
    merged: Dict[str, List[List[float]]] = {}
    for raw_name in sorted(set(existing.keys()) | set(geom.keys())):
        merged[raw_name] = merge_intervals((existing.get(raw_name) or []) + (geom.get(raw_name) or []))
    return merged


def final_object_polygon(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
) -> List[Tuple[float, float]]:
    spec = ((world_data.get("world") or {}).get("objects") or {}).get(object_name, {}) or {}
    path_pts = (trace.get("path", {}) or {}).get(object_name) or []
    if not path_pts:
        return []
    cx, cy = float(path_pts[-1][0]), float(path_pts[-1][1])
    if "radius" in spec:
        radius = float(spec.get("radius", 0.0) or 0.0)
        if radius <= 0:
            return []
        return [
            (
                cx + radius * math.cos(theta),
                cy + radius * math.sin(theta),
            )
            for theta in [2.0 * math.pi * idx / 48.0 for idx in range(48)]
        ]
    points = spec.get("vertices") or spec.get("points") or []
    if not points:
        return []
    return pred.transformed_polygon_at_final_time(trace, world_data, object_name, points)


def final_object_bbox(
    pred: Any,
    *,
    world_obj: Any = None,
    trace: dict,
    world_data: dict,
    object_name: str,
) -> Optional[Tuple[float, float, float, float]]:
    if world_obj is not None:
        try:
            bbox = pred.object_bbox(world_obj, world_data.get("world") or {}, object_name)
            if bbox is not None:
                return tuple(float(v) for v in bbox)
        except Exception:
            pass
    poly = final_object_polygon(pred, trace=trace, world_data=world_data, object_name=object_name)
    if poly:
        xs = [float(x) for x, _ in poly]
        ys = [float(y) for _, y in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    path_pts = (trace.get("path", {}) or {}).get(object_name) or []
    if not path_pts:
        return None
    spec = ((world_data.get("world") or {}).get("objects") or {}).get(object_name, {}) or {}
    if "radius" in spec:
        cx, cy = float(path_pts[-1][0]), float(path_pts[-1][1])
        radius = float(spec.get("radius", 0.0) or 0.0)
        return (cx - radius, cy - radius, cx + radius, cy + radius)
    return None


def object_polygon_at_sample_index(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
    sample_idx: int,
) -> Optional[List[Tuple[float, float]]]:
    spec = ((world_data.get("world") or {}).get("objects") or {}).get(object_name, {}) or {}
    points = spec.get("vertices") or spec.get("points") or []
    if not isinstance(points, list) or not points:
        return None
    pts = [(float(x), float(y)) for x, y in points]
    path_pts = (trace.get("path", {}) or {}).get(object_name) or []
    if not path_pts:
        return None
    idx = max(0, min(int(sample_idx), len(path_pts) - 1))
    rot_vals = (trace.get("rot", {}) or {}).get(object_name) or []
    angle = float(rot_vals[idx]) if idx < len(rot_vals) else (float(rot_vals[-1]) if rot_vals else 0.0)
    pos = path_pts[idx]
    center = pred.object_center(spec) if hasattr(pred, "object_center") else None
    if center is None:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        center = (sum(xs) / float(len(xs)), sum(ys) / float(len(ys)))
    cx, cy = float(center[0]), float(center[1])
    tx, ty = float(pos[0]), float(pos[1])
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


def object_visible_inside_goal_at_time(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
    t: float,
) -> bool:
    goal_name = None
    world_objects = (world_data.get("world") or {}).get("objects") or {}
    for raw_name, spec in world_objects.items():
        if str((spec or {}).get("type", "")) == "Container":
            goal_name = str(raw_name)
            break
    if not goal_name:
        return False
    goal_poly = pred.final_goal_interior_polygon(trace, world_data, goal_name)
    if not goal_poly:
        return False
    sample_idx = int(round(float(t) / TRACE_STEP_S))
    spec = world_objects.get(object_name, {}) or {}
    path_pts = (trace.get("path", {}) or {}).get(object_name) or []
    if not path_pts:
        return False
    idx = max(0, min(sample_idx, len(path_pts) - 1))
    cx, cy = float(path_pts[idx][0]), float(path_pts[idx][1])

    if "radius" in spec:
        radius = float(spec.get("radius", 0.0) or 0.0)
        sample_points = [(cx, cy)] + [
            (cx + radius * r * math.cos(theta), cy + radius * r * math.sin(theta))
            for r in (0.35, 0.6, 0.85)
            for theta in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0, math.pi, 5.0 * math.pi / 4.0, 3.0 * math.pi / 2.0, 7.0 * math.pi / 4.0)
        ]
    else:
        poly = object_polygon_at_sample_index(pred, trace=trace, world_data=world_data, object_name=object_name, sample_idx=idx)
        if not poly:
            return False
        center_x = sum(px for px, _ in poly) / float(len(poly))
        center_y = sum(py for _, py in poly) / float(len(poly))
        sample_points = list(poly)
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            sample_points.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        sample_points.append((center_x, center_y))

    inside_count = sum(1 for px, py in sample_points if bool(pred.point_in_poly(float(px), float(py), goal_poly)))
    if inside_count <= 0:
        return False
    if bool(pred.point_in_poly(cx, cy, goal_poly)):
        return True
    return (inside_count / float(len(sample_points))) >= GOAL_INTERIOR_VISIBLE_SAMPLE_FRAC


def final_gap_px(
    pred: Any,
    *,
    world_obj: Any,
    world_data: dict,
    a_name: str,
    b_name: str,
) -> Optional[float]:
    try:
        gap = pred.min_object_gap_px(world_obj, world_data, a_name, b_name)
        if gap is not None:
            return float(gap)
    except Exception:
        pass
    return None


def final_visibly_touching_pair(
    pred: Any,
    *,
    world_obj: Any,
    world_data: dict,
    trace: dict,
    a_name: str,
    b_name: str,
    gap_threshold_px: float = FINAL_VISIBLE_TOUCH_GAP_PX,
) -> bool:
    final_contacts = trace.get("final_contacts", {}) or {}
    if b_name in set(final_contacts.get(a_name, [])) or a_name in set(final_contacts.get(b_name, [])):
        return True
    gap = final_gap_px(pred, world_obj=world_obj, world_data=world_data, a_name=a_name, b_name=b_name)
    return gap is not None and gap <= float(gap_threshold_px)


def normalize_angle_deg(angle_deg: float) -> float:
    value = float(angle_deg) % 360.0
    if value < 0:
        value += 360.0
    return value


def normalize_half_turn_deg(angle_deg: float) -> float:
    value = normalize_angle_deg(angle_deg) % 180.0
    return value


def orientation_eligible(spec: dict) -> bool:
    if "radius" in spec:
        return False
    if str(spec.get("type", "")).lower() == "container":
        return True
    pts = object_points(spec)
    if len(pts) < 2:
        return False
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    shorter = min(width, height)
    longer = max(width, height)
    if shorter <= 0:
        return False
    return (longer / shorter) >= 1.5


def upside_down_eligible(spec: dict) -> bool:
    if str(spec.get("type", "")).lower() == "container":
        return True
    return classify_shape_name(spec) == "trapezoid"


def local_major_axis_angle_deg(spec: dict) -> Optional[float]:
    pts = object_points(spec)
    if len(pts) < 2:
        return None
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    shorter = min(width, height)
    longer = max(width, height)
    if shorter <= 0 or (longer / shorter) < 1.5:
        return None
    return 0.0 if width >= height else 90.0


def final_orientation_labels(
    spec: dict,
    initial_angle_rad: Optional[float],
    final_angle_rad: Optional[float],
) -> Dict[str, bool]:
    if final_angle_rad is None:
        return {}
    base_axis_deg = local_major_axis_angle_deg(spec)
    if base_axis_deg is None:
        return {}
    absolute_axis_deg = base_axis_deg + math.degrees(float(final_angle_rad))
    half_turn = normalize_half_turn_deg(absolute_axis_deg)
    dist_horizontal = min(abs(half_turn), abs(180.0 - half_turn))
    dist_vertical = abs(half_turn - 90.0)
    upside_down = False
    shape_name = classify_shape_name(spec)
    if initial_angle_rad is not None and upside_down_eligible(spec):
        delta_deg = math.degrees(float(final_angle_rad) - float(initial_angle_rad))
        upside_down = abs(normalize_angle_deg(delta_deg) - 180.0) <= FINAL_UPSIDE_DOWN_EPS_DEG
    horizontal = dist_horizontal <= FINAL_ORIENTATION_AXIS_EPS_DEG
    if shape_name == "trapezoid":
        horizontal = False
    return {
        "horizontal": horizontal,
        "vertical": dist_vertical <= FINAL_ORIENTATION_AXIS_EPS_DEG,
        "slanted": dist_horizontal >= FINAL_ORIENTATION_SLANTED_MIN_DEG and dist_vertical >= FINAL_ORIENTATION_SLANTED_MIN_DEG,
        "upside_down": upside_down,
        "absolute_axis_deg": absolute_axis_deg,
    }


def bbox_center(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float]]:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def axis_overlap_amount(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def final_relative_position_predicates(
    *,
    pred: Any,
    world_obj: Any,
    trace: dict,
    raw_name: str,
    label: str,
    role: str,
    bbox: Optional[Tuple[float, float, float, float]],
    world_data: dict,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    include_tool_events: bool,
) -> List[dict]:
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    predicates: List[dict] = []
    world_objects = (world_data.get("world") or {}).get("objects") or {}
    subject_inside_goal_names: set[str] = set()
    for maybe_goal_name, maybe_goal_spec in world_objects.items():
        if str((maybe_goal_spec or {}).get("type", "")) != "Container":
            continue
        interior_poly = pred.final_goal_interior_polygon(trace, world_data, str(maybe_goal_name))
        if interior_poly and final_object_overlaps_polygon(
            pred,
            trace=trace,
            world_data=world_data,
            object_name=raw_name,
            polygon=interior_poly,
        ):
            subject_inside_goal_names.add(str(maybe_goal_name))

    for other_name in sorted(world_objects.keys()):
        other_name = str(other_name)
        if other_name == raw_name or other_name in WALL_KEYS:
            continue
        other_role = role_map.get(other_name, split_words(other_name))
        if other_role == "wall":
            continue
        if (not include_tool_events) and other_role == "tool":
            continue
        if other_name in subject_inside_goal_names:
            continue
        other_bbox = final_object_bbox(
            pred,
            world_obj=world_obj,
            trace=trace,
            world_data=world_data,
            object_name=other_name,
        )
        if other_bbox is None:
            continue
        ox0, oy0, ox1, oy1 = other_bbox
        other_label = canonical_contact_label(other_name, world_objects, label_map)
        if role == "goal":
            interior_poly = pred.final_goal_interior_polygon(trace, world_data, raw_name)
            if interior_poly and final_object_overlaps_polygon(
                pred,
                trace=trace,
                world_data=world_data,
                object_name=other_name,
                polygon=interior_poly,
            ):
                continue

        vertical_overlap = axis_overlap_amount(y0, y1, oy0, oy1)
        horizontal_overlap = axis_overlap_amount(x0, x1, ox0, ox1)
        center = bbox_center(bbox)
        other_center = bbox_center(other_bbox)
        if center is not None and other_center is not None:
            dx = float(center[0]) - float(other_center[0])
            if dx <= -CENTER_RELATIVE_POSITION_THRESHOLD_PX:
                predicates.append(
                    {
                        "category": "final_state",
                        "subtype": f"center_left_of_{int(CENTER_RELATIVE_POSITION_THRESHOLD_PX)}px",
                        "object": label,
                        "object_id": raw_name,
                        "role": role,
                        "other_object": other_label,
                        "threshold_px": CENTER_RELATIVE_POSITION_THRESHOLD_PX,
                        "time_s": None,
                        "display": f"{label}'s center point ends at least {int(CENTER_RELATIVE_POSITION_THRESHOLD_PX)} px to the left of {other_label}'s center point",
                    }
                )
            elif dx >= CENTER_RELATIVE_POSITION_THRESHOLD_PX:
                predicates.append(
                    {
                        "category": "final_state",
                        "subtype": f"center_right_of_{int(CENTER_RELATIVE_POSITION_THRESHOLD_PX)}px",
                        "object": label,
                        "object_id": raw_name,
                        "role": role,
                        "other_object": other_label,
                        "threshold_px": CENTER_RELATIVE_POSITION_THRESHOLD_PX,
                        "time_s": None,
                        "display": f"{label}'s center point ends at least {int(CENTER_RELATIVE_POSITION_THRESHOLD_PX)} px to the right of {other_label}'s center point",
                    }
                )

        if (
            x1 <= ox0 - RELATIVE_POSITION_EDGE_GAP_PX
            and vertical_overlap > 0.0
        ):
            predicates.append(
                {
                    "category": "final_state",
                    "subtype": "left_of",
                    "object": label,
                    "object_id": raw_name,
                    "role": role,
                    "other_object": other_label,
                    "time_s": None,
                    "display": f"{label} ends to the left of {other_label}",
                }
            )
        if (
            x0 >= ox1 + RELATIVE_POSITION_EDGE_GAP_PX
            and vertical_overlap > 0.0
        ):
            predicates.append(
                {
                    "category": "final_state",
                    "subtype": "right_of",
                    "object": label,
                    "object_id": raw_name,
                    "role": role,
                    "other_object": other_label,
                    "time_s": None,
                    "display": f"{label} ends to the right of {other_label}",
                }
            )
        if (
            y0 >= oy1 + RELATIVE_POSITION_EDGE_GAP_PX
            and horizontal_overlap > 0.0
        ):
            predicates.append(
                {
                    "category": "final_state",
                    "subtype": "above",
                    "object": label,
                    "object_id": raw_name,
                    "role": role,
                    "other_object": other_label,
                    "time_s": None,
                    "display": f"{label} ends above {other_label}",
                }
            )
        if (
            y1 <= oy0 - RELATIVE_POSITION_EDGE_GAP_PX
            and horizontal_overlap > 0.0
        ):
            predicates.append(
                {
                    "category": "final_state",
                    "subtype": "below",
                    "object": label,
                    "object_id": raw_name,
                    "role": role,
                    "other_object": other_label,
                    "time_s": None,
                    "display": f"{label} ends below {other_label}",
                }
            )
    return predicates


def ends_on_top_of(
    pred: Any,
    *,
    world_obj: Any,
    world_data: dict,
    trace: dict,
    upper_name: str,
    lower_name: str,
    final_contact_names_raw: Sequence[str],
) -> bool:
    if not final_visibly_touching_pair(pred, world_obj=world_obj, world_data=world_data, trace=trace, a_name=upper_name, b_name=lower_name):
        return False
    bbox_upper = final_object_bbox(pred, world_obj=world_obj, trace=trace, world_data=world_data, object_name=upper_name)
    bbox_lower = final_object_bbox(pred, world_obj=world_obj, trace=trace, world_data=world_data, object_name=lower_name)
    if bbox_upper is None or bbox_lower is None:
        return False
    ux0, uy0, ux1, uy1 = bbox_upper
    lx0, ly0, lx1, ly1 = bbox_lower
    upper_width = max(ux1 - ux0, 1.0)
    lower_width = max(lx1 - lx0, 1.0)
    lower_height = max(ly1 - ly0, 1.0)
    horizontal_overlap = min(ux1, lx1) - max(ux0, lx0)
    if horizontal_overlap < max(5.0, 0.2 * min(upper_width, lower_width)):
        return False
    if uy0 < (ly0 + ly1) / 2.0:
        return False
    if ((uy0 - ly1) if uy0 >= ly1 else (ly1 - uy0)) > 8.0:
        return False
    upper_center_y = (uy0 + uy1) / 2.0
    lower_center_y = (ly0 + ly1) / 2.0
    if upper_center_y <= lower_center_y + max(3.0, 0.15 * lower_height):
        return False
    supporting_contacts = []
    for other_name in sorted(set(name for name in final_contact_names_raw if name not in WALL_KEYS)):
        other_bbox = final_object_bbox(pred, world_obj=world_obj, trace=trace, world_data=world_data, object_name=other_name)
        if other_bbox is None:
            continue
        ox0, oy0, ox1, oy1 = other_bbox
        other_width = max(ox1 - ox0, 1.0)
        other_height = max(oy1 - oy0, 1.0)
        overlap = min(ux1, ox1) - max(ux0, ox0)
        if overlap < max(5.0, 0.2 * min(upper_width, other_width)):
            continue
        if uy0 < (oy0 + oy1) / 2.0:
            continue
        if ((uy0 - oy1) if uy0 >= oy1 else (oy1 - uy0)) > 8.0:
            continue
        if upper_center_y <= ((oy0 + oy1) / 2.0) + max(3.0, 0.15 * other_height):
            continue
        supporting_contacts.append(other_name)
    if sorted(set(supporting_contacts)) != [lower_name]:
        return False
    return True


def final_object_overlaps_polygon(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
    polygon: List[Tuple[float, float]],
) -> bool:
    if not polygon:
        return False
    object_poly = final_object_polygon(pred, trace=trace, world_data=world_data, object_name=object_name)
    if object_poly:
        if pred.polygons_overlap(object_poly, polygon):
            return True
        if any(pred.point_in_poly(px, py, polygon) for px, py in object_poly):
            return True
        center_x = sum(px for px, _ in object_poly) / float(len(object_poly))
        center_y = sum(py for _, py in object_poly) / float(len(object_poly))
        return bool(pred.point_in_poly(center_x, center_y, polygon))

    path_pts = (trace.get("path", {}) or {}).get(object_name) or []
    if not path_pts:
        return False
    return bool(pred.point_in_poly(float(path_pts[-1][0]), float(path_pts[-1][1]), polygon))


def point_segment_distance_px(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def polygon_distance_px(poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]]) -> Optional[float]:
    if not poly_a or not poly_b:
        return None
    if len(poly_a) < 2 or len(poly_b) < 2:
        return None
    if any(point in (None, []) for point in poly_a + poly_b):
        return None
    best = float("inf")
    edges_a = list(zip(poly_a, poly_a[1:] + poly_a[:1]))
    edges_b = list(zip(poly_b, poly_b[1:] + poly_b[:1]))
    for point in poly_a:
        for edge_start, edge_end in edges_b:
            best = min(best, point_segment_distance_px(point, edge_start, edge_end))
    for point in poly_b:
        for edge_start, edge_end in edges_a:
            best = min(best, point_segment_distance_px(point, edge_start, edge_end))
    return best if math.isfinite(best) else None


def final_object_near_polygon(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
    polygon: List[Tuple[float, float]],
    gap_threshold_px: float,
) -> bool:
    object_poly = final_object_polygon(pred, trace=trace, world_data=world_data, object_name=object_name)
    if not object_poly or not polygon:
        return False
    if final_object_overlaps_polygon(pred, trace=trace, world_data=world_data, object_name=object_name, polygon=polygon):
        return True
    distance = polygon_distance_px(object_poly, polygon)
    return distance is not None and distance <= float(gap_threshold_px)


def final_object_sample_points(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
) -> List[Tuple[float, float]]:
    object_poly = final_object_polygon(pred, trace=trace, world_data=world_data, object_name=object_name)
    if object_poly:
        return [(float(x), float(y)) for x, y in object_poly]

    path_pts = (trace.get("path", {}) or {}).get(object_name) or []
    if not path_pts:
        return []
    spec = ((world_data.get("world") or {}).get("objects") or {}).get(object_name, {}) or {}
    cx, cy = float(path_pts[-1][0]), float(path_pts[-1][1])
    radius = float(spec.get("radius", 0.0) or 0.0)
    if radius <= 0:
        return [(cx, cy)]
    points: List[Tuple[float, float]] = [(cx, cy)]
    for theta in [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0, math.pi, 5.0 * math.pi / 4.0, 3.0 * math.pi / 2.0, 7.0 * math.pi / 4.0]:
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return points


def final_object_fully_inside_polygon(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
    polygon: List[Tuple[float, float]],
) -> bool:
    sample_points = final_object_sample_points(pred, trace=trace, world_data=world_data, object_name=object_name)
    if not sample_points:
        return False
    return all(bool(pred.point_in_poly(float(x), float(y), polygon)) for x, y in sample_points)


def goal_exterior_wall_band_polygons(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    goal_name: Optional[str],
) -> List[List[Tuple[float, float]]]:
    if not goal_name:
        return []
    goal_spec = ((world_data.get("world") or {}).get("objects") or {}).get(goal_name, {}) or {}
    if str(goal_spec.get("type", "")) != "Container":
        return []
    points = goal_spec.get("points") or goal_spec.get("inner_vertices") or []
    if not isinstance(points, list) or len(points) < 3:
        return []
    width = float(goal_spec.get("width", 0.0) or 0.0)
    if width <= 0:
        return []
    outer_poly = pred.transformed_polygon_at_final_time(trace, world_data, goal_name, points)
    if not outer_poly:
        return []
    xs = [float(x) for x, _ in outer_poly]
    ys = [float(y) for _, y in outer_poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x - min_x <= 2.0 * width or max_y - min_y <= width:
        return []
    return [
        [(min_x, max_y), (min_x, min_y), (min_x + width, min_y), (min_x + width, max_y)],
        [(max_x - width, max_y), (max_x - width, min_y), (max_x, min_y), (max_x, max_y)],
        [(min_x, min_y + width), (min_x, min_y), (max_x, min_y), (max_x, min_y + width)],
    ]


def goal_top_contact_band_polygon(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    goal_name: Optional[str],
) -> List[Tuple[float, float]]:
    if not goal_name:
        return []
    goal_spec = ((world_data.get("world") or {}).get("objects") or {}).get(goal_name, {}) or {}
    if str(goal_spec.get("type", "")) != "Container":
        return []
    points = goal_spec.get("points") or goal_spec.get("inner_vertices") or []
    if not isinstance(points, list) or len(points) < 3:
        return []
    width = float(goal_spec.get("width", 0.0) or 0.0)
    if width <= 0:
        return []
    outer_poly = pred.transformed_polygon_at_final_time(trace, world_data, goal_name, points)
    if not outer_poly:
        return []
    xs = [float(x) for x, _ in outer_poly]
    ys = [float(y) for _, y in outer_poly]
    min_x, max_x = min(xs), max(xs)
    max_y = max(ys)
    if max_x - min_x <= 2.0 * width:
        return []
    inner_left = min_x + width
    inner_right = max_x - width
    band_half_h = max(2.0, min(width * 0.5, 6.0))
    return [
        (inner_left, max_y + band_half_h),
        (inner_left, max_y - band_half_h),
        (inner_right, max_y - band_half_h),
        (inner_right, max_y + band_half_h),
    ]


def final_object_touches_goal_top(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
    goal_name: Optional[str],
    final_contacts: Dict[str, Sequence[str]],
) -> bool:
    if not goal_name:
        return False
    top_band = goal_top_contact_band_polygon(
        pred,
        trace=trace,
        world_data=world_data,
        goal_name=goal_name,
    )
    if not top_band:
        return False
    interior_poly = pred.final_goal_interior_polygon(trace, world_data, goal_name)
    if interior_poly and final_object_fully_inside_polygon(
        pred,
        trace=trace,
        world_data=world_data,
        object_name=object_name,
        polygon=interior_poly,
    ):
        return False
    return final_object_overlaps_polygon(
        pred,
        trace=trace,
        world_data=world_data,
        object_name=object_name,
        polygon=top_band,
    )


def final_object_touches_goal_exterior_wall(
    pred: Any,
    *,
    trace: dict,
    world_data: dict,
    object_name: str,
    goal_name: Optional[str],
    final_contacts: Dict[str, Sequence[str]],
) -> bool:
    if not goal_name:
        return False
    if not goal_exterior_allowed(world_data):
        return False
    wall_band_polys = goal_exterior_wall_band_polygons(
        pred,
        trace=trace,
        world_data=world_data,
        goal_name=goal_name,
    )
    if not wall_band_polys:
        return False
    interior_poly = pred.final_goal_interior_polygon(trace, world_data, goal_name)
    if interior_poly and final_object_overlaps_polygon(
        pred,
        trace=trace,
        world_data=world_data,
        object_name=object_name,
        polygon=interior_poly,
    ):
        return False
    return any(
        final_object_near_polygon(
            pred,
            trace=trace,
            world_data=world_data,
            object_name=object_name,
            polygon=band_poly,
            gap_threshold_px=FINAL_GOAL_EXTERIOR_TOUCH_GAP_PX,
        )
        for band_poly in wall_band_polys
    )


def derive_final_state(
    pred: Any,
    world_obj: Any,
    world_data: dict,
    goal_name: Optional[str],
    *,
    raw_name: str,
    label: str,
    role: str,
    trace: dict,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    object_in_goal_intervals: Dict[str, Sequence[Sequence[float]]],
    initial_contact_pairs: set[Tuple[str, str]],
    include_tool_events: bool,
) -> Tuple[dict, List[dict]]:
    path = trace.get("path", {}) or {}
    final_contacts = trace.get("final_contacts", {}) or {}
    duration_s = float(trace.get("duration") or 0.0)

    if raw_name not in path or not path[raw_name]:
        return {}, []

    final_x = float(path[raw_name][-1][0])
    final_y = float(path[raw_name][-1][1])
    bbox = final_object_bbox(pred, world_obj=world_obj, trace=trace, world_data=world_data, object_name=raw_name)
    world_dims = ((world_data.get("world") or {}).get("dims") or [600, 600])
    world_objects = (world_data.get("world") or {}).get("objects") or {}
    final_contact_names = []
    for other_name in world_obj.objects.keys():
        other_name = str(other_name)
        if other_name == raw_name:
            continue
        if final_visibly_touching_pair(
            pred,
            world_obj=world_obj,
            world_data=world_data,
            trace=trace,
            a_name=raw_name,
            b_name=other_name,
        ):
            final_contact_names.append(other_name)
    final_contact_names = sorted(set(final_contact_names))
    final_contact_labels = [canonical_contact_label(name, world_objects, label_map) for name in final_contact_names]
    final_in_goal = any(interval_ends_at_final(float(end_s), duration_s) for _, end_s in object_in_goal_intervals.get(raw_name, []))
    final_partially_in_goal = False
    if role != "goal" and goal_name:
        interior_poly_for_object = pred.final_goal_interior_polygon(trace, world_data, goal_name)
        if interior_poly_for_object:
            final_partially_in_goal = (
                final_object_overlaps_polygon(
                    pred,
                    trace=trace,
                    world_data=world_data,
                    object_name=raw_name,
                    polygon=interior_poly_for_object,
                )
                and not final_object_fully_inside_polygon(
                    pred,
                    trace=trace,
                    world_data=world_data,
                    object_name=raw_name,
                    polygon=interior_poly_for_object,
                )
            )
    touching_goal_exterior = final_object_touches_goal_exterior_wall(
        pred=pred,
        trace=trace,
        world_data=world_data,
        object_name=raw_name,
        goal_name=goal_name,
        final_contacts=final_contacts,
    ) and not final_in_goal and not final_partially_in_goal
    touching_goal_top = final_object_touches_goal_top(
        pred=pred,
        trace=trace,
        world_data=world_data,
        object_name=raw_name,
        goal_name=goal_name,
        final_contacts=final_contacts,
    ) and not final_in_goal and not final_partially_in_goal and not touching_goal_exterior
    partial_goal_interior_contacts: List[str] = []
    if role == "goal":
        interior_poly = pred.final_goal_interior_polygon(trace, world_data, raw_name)
        if interior_poly:
            for other_name in final_contact_names:
                if other_name in WALL_KEYS:
                    continue
                other_label = canonical_contact_label(other_name, world_objects, label_map)
                if final_object_overlaps_polygon(
                    pred,
                    trace=trace,
                    world_data=world_data,
                    object_name=other_name,
                    polygon=interior_poly,
                ) and not final_object_fully_inside_polygon(
                    pred,
                    trace=trace,
                    world_data=world_data,
                    object_name=other_name,
                    polygon=interior_poly,
                ):
                    partial_goal_interior_contacts.append(other_label)
    non_floor_final_contacts = [
        other_name
        for other_name in final_contact_names
        if other_name != "_BottomWall"
    ]
    touching_floor = (
        "_BottomWall" in final_contact_names
        or (bbox is not None and bbox[1] <= FINAL_BOUNDARY_TOUCH_EPS_PX)
    )
    floor_only = (
        touching_floor
        and bbox is not None
        and bbox[1] <= FINAL_BOUNDARY_TOUCH_EPS_PX
        and not non_floor_final_contacts
        and not touching_goal_exterior
        and not touching_goal_top
        and not final_partially_in_goal
        and not final_in_goal
    )
    on_floor = touching_floor
    on_ceiling = bbox is not None and bbox[3] >= float(world_dims[1]) - FINAL_BOUNDARY_TOUCH_EPS_PX

    on_top_of: List[str] = []
    for other_name in final_contact_names:
        if other_name in WALL_KEYS:
            continue
        if role_map.get(other_name) == "goal":
            continue
        if (not include_tool_events) and role_map.get(other_name) == "tool":
            continue
        if ends_on_top_of(
            pred,
            world_obj=world_obj,
            world_data=world_data,
            trace=trace,
            upper_name=raw_name,
            lower_name=other_name,
            final_contact_names_raw=final_contact_names,
        ):
            on_top_of.append(canonical_contact_label(other_name, world_objects, label_map))

    rot_map = trace.get("rot", {}) or {}
    rot_values = rot_map.get(raw_name) or []
    spec = ((world_data.get("world") or {}).get("objects") or {}).get(raw_name, {}) or {}
    orientation = {}
    if orientation_eligible(spec):
        orientation = final_orientation_labels(
            spec,
            rot_values[0] if rot_values else None,
            rot_values[-1] if rot_values else None,
        )

    final_state = {
        "object": label,
        "object_id": raw_name,
        "role": role,
        "final_position_xy": [final_x, final_y],
        "in_goal": final_in_goal,
        "partially_in_goal": final_partially_in_goal,
        "touching_goal_exterior": touching_goal_exterior,
        "touching_goal_top": touching_goal_top,
        "partial_goal_interior_contacts": sorted(set(partial_goal_interior_contacts)),
        "on_floor": on_floor,
        "touching_floor": touching_floor,
        "floor_only": floor_only,
        "on_ceiling": on_ceiling,
        "final_contacts": final_contact_labels,
        "on_top_of": sorted(set(on_top_of)),
        "final_orientation": orientation,
        "final_rotation_delta_deg": (math.degrees(float(rot_values[-1]) - float(rot_values[0])) if len(rot_values) >= 2 else None),
    }

    predicates: List[dict] = []
    if final_in_goal:
        predicates.append(
            {
                "category": "final_state",
                "subtype": "in_goal",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends inside the green container",
            }
        )
    elif final_partially_in_goal:
        predicates.append(
            {
                "category": "final_state",
                "subtype": "partially_in_goal",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends partially inside the green container",
            }
        )
    if touching_goal_exterior:
        predicates.append(
            {
                "category": "final_state",
                "subtype": "touching_goal_exterior",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends touching the outside of the green container",
            }
        )
    for other_label in sorted(set(partial_goal_interior_contacts)):
        predicates.append(
            {
                "category": "final_state",
                "subtype": "goal_interior_partial_touch",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "other_object": other_label,
                "time_s": None,
                "display": f"{label} interior partially touches {other_label}",
            }
        )
    if floor_only:
        predicates.append(
            {
                "category": "final_state",
                "subtype": "on_floor",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends on the floor",
            }
        )
    if on_ceiling:
        predicates.append(
            {
                "category": "final_state",
                "subtype": "on_ceiling",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends on the ceiling",
            }
        )
    if orientation.get("horizontal"):
        predicates.append(
            {
                "category": "final_state",
                "subtype": "orientation_horizontal",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends roughly horizontal",
            }
        )
    if orientation.get("vertical"):
        predicates.append(
            {
                "category": "final_state",
                "subtype": "orientation_vertical",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends roughly vertical",
            }
        )
    if orientation.get("slanted"):
        predicates.append(
            {
                "category": "final_state",
                "subtype": "orientation_slanted",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends tilted by at least {int(FINAL_ORIENTATION_SLANTED_MIN_DEG)} degrees",
            }
        )
    if orientation.get("upside_down"):
        predicates.append(
            {
                "category": "final_state",
                "subtype": "orientation_upside_down",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "time_s": None,
                "display": f"{label} ends roughly upside down",
            }
        )
    partial_goal_interior_contact_labels = set(partial_goal_interior_contacts)
    final_contact_names_raw = final_contact_names
    single_touch_predicates: List[dict] = []
    combo_touch_labels: List[str] = []
    on_top_of_labels = set(on_top_of)
    for other_name in final_contact_names_raw:
        if other_name in WALL_KEYS:
            continue
        other_role = role_map.get(other_name, split_words(other_name))
        other_label = canonical_contact_label(other_name, world_objects, label_map)
        if other_label == "green container":
            if role != "goal" and touching_goal_top:
                other_label = "top of green container"
            else:
                continue
        if role == "goal" and other_label in partial_goal_interior_contact_labels:
            continue
        combo_touch_labels.append(other_label)
        if (not include_tool_events) and other_role == "tool":
            continue
        subject_raw, subject_label, _other_raw, other_display = choose_contact_subject(
            raw_name,
            other_name,
            label_a=label,
            label_b=other_label,
            role_map=role_map,
            dynamic_map=dynamic_map,
            include_tool_events=include_tool_events,
        )
        if subject_raw != raw_name:
            continue
        if dynamic_map.get(raw_name, False) and dynamic_map.get(other_name, False) and label > other_label:
            continue
        if other_display in on_top_of_labels:
            continue
        single_touch_predicates.append(
            {
                "category": "final_state",
                "subtype": "touching_object",
                "object": subject_label,
                "object_id": raw_name,
                "role": role,
                "other_object": other_display,
                "time_s": None,
                "display": f"{subject_label} ends touching {other_display}",
            }
        )

    if role != "goal" and final_partially_in_goal:
        combo_touch_labels.append("partially inside the green container")
    unique_combo_touch_labels = sorted(set(combo_touch_labels))
    if 2 <= len(unique_combo_touch_labels) <= 3:
        partial_inside_label = "partially inside the green container"
        object_labels_only = [item for item in unique_combo_touch_labels if item != partial_inside_label]
        if partial_inside_label in unique_combo_touch_labels and object_labels_only:
            combo_text = " and ".join(object_labels_only) + " and partially inside the green container"
        else:
            combo_text = " and ".join(unique_combo_touch_labels)
        predicates.append(
            {
                "category": "final_state",
                "subtype": "touching_multiple_objects",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "other_object": combo_text,
                "time_s": None,
                "display": f"{label} ends touching {combo_text}",
            }
        )
    else:
        predicates.extend(single_touch_predicates)

    for other_label in sorted(set(on_top_of)):
        predicates.append(
            {
                "category": "final_state",
                "subtype": "on_top_of",
                "object": label,
                "object_id": raw_name,
                "role": role,
                "other_object": other_label,
                "time_s": None,
                "display": f"{label} ends on top of {other_label}",
            }
        )

    predicates.extend(
        final_relative_position_predicates(
            pred=pred,
            world_obj=world_obj,
            trace=trace,
            raw_name=raw_name,
            label=label,
            role=role,
            bbox=bbox,
            world_data=world_data,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            include_tool_events=include_tool_events,
        )
    )

    return final_state, predicates


def normalize_signature_label(label: object) -> str:
    text = str(label)
    return (
        text.replace("grey tool ball", "big orange ball (dropped tool)")
        .replace("grey dotted ball", "big orange ball (dropped tool)")
        .replace("grey ball", "big orange ball (dropped tool)")
    )


def canonical_touch_combo_signature(other_object: object) -> str:
    labels = [
        normalize_signature_label(item.strip())
        for item in str(other_object).split(" and ")
        if item.strip()
    ]
    partial_label = "partially inside the green container"

    def sort_key(label: str) -> Tuple[int, str]:
        if label == partial_label:
            return (2, label)
        if label == "big orange ball (dropped tool)":
            return (1, label)
        return (0, label)

    return " and ".join(sorted(set(labels), key=sort_key))


def event_signature(event: dict) -> str:
    category = event["category"]
    if category == "movement":
        return f"movement|{event['object']}|axis={event['axis']}|direction={event['direction']}|threshold={int(event['threshold_px'])}"
    if category == "joint_movement":
        return f"joint_movement|{event['object']}|{event['other_object']}|axis={event['axis']}|direction={event['direction']}|threshold={int(event['threshold_px'])}|mindur_ms={int(JOINT_MOVEMENT_MIN_DURATION_S * 1000)}"
    if category == "pass_over":
        return f"pass_over|{event['object']}|{event['direction']}|{event['other_object']}|threshold={int(event['threshold_px'])}"
    if category == "rotation":
        return f"rotation|{event['object']}|direction={event['direction']}|threshold={int(event['threshold_deg'])}"
    if category == "rotation_sequence":
        return f"rotation_sequence|{event['object']}|{event['first_direction']}->{event['second_direction']}|threshold={int(event['threshold_deg'])}"
    if category == "contact":
        a, b = event["objects"]
        return f"contact|{a}|{b}|lasting_to_final={int(bool(event['lasting_to_final']))}"
    if category == "contact_change":
        return f"contact_change|{event['object']}|{event['subtype']}|{event['other_object']}"
    if category == "container":
        return f"container|{event['object']}|lasting_to_final={int(bool(event['lasting_to_final']))}"
    if category == "final_state":
        subtype = event["subtype"]
        if "other_object" in event:
            other_object = normalize_signature_label(event["other_object"])
            if subtype == "touching_multiple_objects":
                other_object = canonical_touch_combo_signature(other_object)
            return f"final_state|{normalize_signature_label(event['object'])}|{subtype}|{other_object}"
        return f"final_state|{normalize_signature_label(event['object'])}|{subtype}"
    return json.dumps(event, sort_keys=True)


def parse_final_touch_signature(signature: str) -> Optional[dict]:
    parts = str(signature).split("|")
    if len(parts) < 4 or parts[0] != "final_state":
        return None
    subject = parts[1]
    subtype = parts[2]
    if subtype == "touching_object" and len(parts) >= 4:
        return {
            "subject": subject,
            "subtype": subtype,
            "labels": [parts[3]],
        }
    if subtype == "touching_multiple_objects" and len(parts) >= 4:
        labels = [item.strip() for item in parts[3].split(" and ") if item.strip()]
        return {
            "subject": subject,
            "subtype": subtype,
            "labels": sorted(labels),
        }
    return None


def aggregate_event_bank(placement_rows: Sequence[dict]) -> List[dict]:
    valid_rows = [row for row in placement_rows if row.get("valid")]
    total_valid = len(valid_rows)
    stats: Dict[str, dict] = {}
    sig_to_placements: Dict[str, set[Tuple[int, int]]] = defaultdict(set)

    for row in valid_rows:
        seen = set()
        placement_xy = tuple(int(v) for v in row["placement_xy"])
        for event in row.get("event_graph", []):
            sig = event_signature(event)
            if sig in seen:
                continue
            seen.add(sig)
            sig_to_placements[sig].add(placement_xy)
            bucket = stats.setdefault(
                sig,
                {
                    "signature": sig,
                    "category": event["category"],
                    "display": event["display"],
                    "count": 0,
                    "times_s": [],
                    "durations_s": [],
                    "example_placements": [],
                },
            )
            bucket["count"] += 1
            if event.get("time_s") is not None:
                bucket["times_s"].append(float(event["time_s"]))
            if event.get("duration_s") is not None:
                bucket["durations_s"].append(float(event["duration_s"]))
            if len(bucket["example_placements"]) < 5:
                bucket["example_placements"].append(row["placement_xy"])

    suppressed_signatures: set[str] = set()
    multi_touch_by_subject: Dict[str, List[Tuple[str, set[str], set[Tuple[int, int]]]]] = defaultdict(list)
    for sig, placements in sig_to_placements.items():
        parsed = parse_final_touch_signature(sig)
        if not parsed or parsed["subtype"] != "touching_multiple_objects":
            continue
        multi_touch_by_subject[parsed["subject"]].append(
            (sig, set(parsed["labels"]), placements)
        )

    for sig, placements in sig_to_placements.items():
        parsed = parse_final_touch_signature(sig)
        if not parsed or parsed["subtype"] != "touching_object":
            continue
        subject = parsed["subject"]
        singleton_label = parsed["labels"][0]
        candidate_multis = [
            (multi_sig, multi_labels, multi_placements)
            for multi_sig, multi_labels, multi_placements in multi_touch_by_subject.get(subject, [])
            if singleton_label in multi_labels
        ]
        if not candidate_multis:
            continue
        placements_without_specific = {
            placement
            for placement in placements
            if not any(
                placement in multi_placements
                for _multi_sig, _multi_labels, multi_placements in candidate_multis
            )
        }
        if not placements_without_specific:
            suppressed_signatures.add(sig)

    ranked: List[dict] = []
    for item in stats.values():
        if item["signature"] in suppressed_signatures:
            continue
        entry = {
            "signature": item["signature"],
            "category": item["category"],
            "display": item["display"],
            "count": item["count"],
            "probability_valid_placements": (item["count"] / total_valid) if total_valid else None,
            "example_placements": item["example_placements"],
        }
        if item["times_s"]:
            entry["mean_time_s"] = statistics.fmean(item["times_s"])
            entry["median_time_s"] = statistics.median(item["times_s"])
        if item["durations_s"]:
            entry["mean_duration_s"] = statistics.fmean(item["durations_s"])
            entry["median_duration_s"] = statistics.median(item["durations_s"])
        ranked.append(entry)

    ranked.sort(
        key=lambda row: (
            row["probability_valid_placements"] if row["probability_valid_placements"] is not None else 1.0,
            row["category"],
            row["display"],
        )
    )
    return ranked


def interval_active_at_time(
    intervals: Sequence[Sequence[float]],
    *,
    time_s: float,
    duration_s: float,
) -> bool:
    """Return whether a native simulator interval is active at one sample.

    Intervals that close before the rollout endpoint use a half-open end so the
    first sample after an object leaves a relation is not mislabeled.  An
    interval ending at the rollout duration is active at the terminal sample.
    """
    for interval in intervals or []:
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            continue
        start_s = float(interval[0])
        end_s = float(interval[1])
        if start_s - 1e-9 <= float(time_s) < end_s - 1e-9:
            return True
        if (
            abs(float(time_s) - float(duration_s)) <= 1e-9
            and abs(end_s - float(duration_s)) <= 1e-9
            and start_s - 1e-9 <= float(time_s)
        ):
            return True
    return False


def snapshot_trace_at_sample(
    *,
    trace: dict,
    sample_idx: int,
) -> dict:
    """Build a two-pose trace whose final pose is one native trace sample."""
    duration_s = float(trace.get("duration") or 0.0)
    path_map = trace.get("path", {}) or {}
    rot_map = trace.get("rot", {}) or {}
    snapshot_path: Dict[str, List[List[float]]] = {}
    snapshot_rot: Dict[str, List[float]] = {}

    for raw_name, samples in path_map.items():
        if not samples:
            continue
        idx = max(0, min(int(sample_idx), len(samples) - 1))
        initial = [float(samples[0][0]), float(samples[0][1])]
        current = [float(samples[idx][0]), float(samples[idx][1])]
        snapshot_path[str(raw_name)] = [initial, current]
        rotations = rot_map.get(raw_name) or []
        initial_rot = float(rotations[0]) if rotations else 0.0
        current_rot = (
            float(rotations[idx])
            if idx < len(rotations)
            else (float(rotations[-1]) if rotations else initial_rot)
        )
        snapshot_rot[str(raw_name)] = [initial_rot, current_rot]

    sample_time_s = min(float(sample_idx) * TRACE_STEP_S, duration_s)
    active_goal_intervals: Dict[str, List[List[float]]] = {}
    for raw_name, intervals in (trace.get("object_in_goal_intervals", {}) or {}).items():
        if interval_active_at_time(
            intervals or [],
            time_s=sample_time_s,
            duration_s=duration_s,
        ):
            active_goal_intervals[str(raw_name)] = [
                [max(0.0, sample_time_s - TRACE_STEP_S), sample_time_s]
            ]

    return {
        "path": snapshot_path,
        "rot": snapshot_rot,
        "duration": sample_time_s,
        "object_in_goal_intervals": active_goal_intervals,
        "final_contacts": {},
    }


def apply_snapshot_poses(
    *,
    world_obj: Any,
    snapshot_trace: dict,
) -> None:
    """Move a clean world copy to one recorded native simulator state."""
    path_map = snapshot_trace.get("path", {}) or {}
    rot_map = snapshot_trace.get("rot", {}) or {}
    for raw_name, samples in path_map.items():
        obj = world_obj.objects.get(raw_name)
        if obj is None or not samples:
            continue
        current = samples[-1]
        obj.setPos((float(current[0]), float(current[1])))
        rotations = rot_map.get(raw_name) or []
        if rotations:
            obj.setRot(float(rotations[-1]))
        try:
            world_obj._cpSpace.reindex_shapes_for_body(obj._cpBody)
        except Exception:
            pass
    try:
        world_obj._cpSpace.reindex_static()
    except Exception:
        pass


def final_state_signatures_for_snapshot(
    *,
    pred: Any,
    world_obj: Any,
    world_data: dict,
    snapshot_trace: dict,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    initial_contact_pairs: set[Tuple[str, str]],
    include_tool_events: bool,
    candidate_signatures: Optional[set[str]] = None,
) -> set[str]:
    snapshot_trace["final_contacts"] = pred.final_contact_names(
        world_obj,
        world_obj.objects.keys(),
    )
    goal_name = None
    for raw_name, spec in ((world_data.get("world") or {}).get("objects") or {}).items():
        if str((spec or {}).get("type", "")) == "Container":
            goal_name = str(raw_name)
            break

    signatures: set[str] = set()
    candidate_subjects = {
        signature.split("|", 3)[1]
        for signature in (candidate_signatures or set())
        if signature.startswith("final_state|") and len(signature.split("|", 3)) >= 3
    }
    object_in_goal_intervals = (
        snapshot_trace.get("object_in_goal_intervals") or {}
    )
    for raw_name in (snapshot_trace.get("path", {}) or {}).keys():
        label = label_map.get(raw_name, split_words(raw_name))
        role = role_map.get(raw_name, split_words(raw_name))
        if candidate_subjects and normalize_signature_label(label) not in candidate_subjects:
            continue
        _state, predicates = derive_final_state(
            pred,
            world_obj,
            world_data,
            goal_name,
            raw_name=raw_name,
            label=label,
            role=role,
            trace=snapshot_trace,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            object_in_goal_intervals=object_in_goal_intervals,
            initial_contact_pairs=initial_contact_pairs,
            include_tool_events=include_tool_events,
        )
        if include_tool_events or role != "tool":
            generated = {event_signature(event) for event in predicates}
            if candidate_signatures is not None:
                generated.intersection_update(candidate_signatures)
            signatures.update(generated)
    return signatures


def terminal_persistent_final_state_signatures(
    *,
    pred: Any,
    world_data: dict,
    condition: str,
    coords: Sequence[int],
    trace: dict,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    initial_contact_pairs: set[Tuple[str, str]],
    include_tool_events: bool,
    endpoint_signatures: Sequence[str],
    persistence_s: float,
    candidate_signatures: Optional[Sequence[str]] = None,
) -> Tuple[List[str], int, Optional[float]]:
    """Intersect final-state predicates over the final persistence window."""
    duration_s = float(trace.get("duration") or 0.0)
    if persistence_s <= 0.0:
        return sorted(set(endpoint_signatures)), 0, None
    if duration_s + 1e-9 < float(persistence_s):
        return [], 0, None

    path_lengths = [
        len(samples)
        for samples in (trace.get("path", {}) or {}).values()
        if samples
    ]
    if not path_lengths:
        return [], 0, None
    final_idx = min(path_lengths) - 1
    start_time_s = duration_s - float(persistence_s)
    start_idx = max(0, int(math.ceil((start_time_s - 1e-9) / TRACE_STEP_S)))
    if final_idx < start_idx:
        return [], 0, None

    # Sleeping VTools bodies repeat exactly the same pose.  When every tracked
    # pose is constant throughout the persistence window, every geometric
    # final-state predicate is constant too, so the endpoint set is already
    # the exact native-tick intersection.  This fast path makes exhaustive
    # resweeps practical without weakening the definition.
    pose_constant = True
    for raw_name, samples in (trace.get("path", {}) or {}).items():
        if not samples:
            continue
        reference_xy = samples[start_idx]
        rotations = (trace.get("rot", {}) or {}).get(raw_name) or []
        reference_rot = (
            float(rotations[start_idx])
            if start_idx < len(rotations)
            else (float(rotations[-1]) if rotations else 0.0)
        )
        for idx in range(start_idx + 1, final_idx + 1):
            xy = samples[idx]
            rotation = (
                float(rotations[idx])
                if idx < len(rotations)
                else (float(rotations[-1]) if rotations else reference_rot)
            )
            if (
                float(xy[0]) != float(reference_xy[0])
                or float(xy[1]) != float(reference_xy[1])
                or rotation != reference_rot
            ):
                pose_constant = False
                break
        if not pose_constant:
            break
    full_sample_count = final_idx - start_idx + 1
    if pose_constant:
        constant_signatures = set(str(item) for item in endpoint_signatures)
        if candidate_signatures is not None:
            constant_signatures.intersection_update(
                str(item) for item in candidate_signatures
            )
        return (
            sorted(constant_signatures),
            full_sample_count,
            float(start_idx) * TRACE_STEP_S,
        )

    snapshot_world = pred.loadFromDict(world_data["world"]).copy()
    if not pred.place_ball_tool(
        snapshot_world,
        (float(coords[0]), float(coords[1])),
        TOOL_RADIUS_PX,
        str(condition),
        "orange",
    ):
        raise RuntimeError(
            f"Could not reconstruct valid placement at {list(coords)}"
        )

    survivors = set(str(item) for item in endpoint_signatures)
    if candidate_signatures is not None:
        survivors.intersection_update(str(item) for item in candidate_signatures)
    if not survivors:
        return [], 0, float(start_idx) * TRACE_STEP_S
    sample_count = 0
    for sample_idx in range(start_idx, final_idx + 1):
        sample_count += 1
        snapshot_trace = snapshot_trace_at_sample(
            trace=trace,
            sample_idx=sample_idx,
        )
        apply_snapshot_poses(
            world_obj=snapshot_world,
            snapshot_trace=snapshot_trace,
        )
        sample_signatures = final_state_signatures_for_snapshot(
            pred=pred,
            world_obj=snapshot_world,
            world_data=world_data,
            snapshot_trace=snapshot_trace,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            initial_contact_pairs=initial_contact_pairs,
            include_tool_events=include_tool_events,
            candidate_signatures=survivors,
        )
        survivors.intersection_update(sample_signatures)
        if not survivors:
            break
    return sorted(survivors), sample_count, float(start_idx) * TRACE_STEP_S


def simulate_valid_placement(
    *,
    pred: Any,
    world_data: dict,
    label_map: Dict[str, str],
    role_map: Dict[str, str],
    dynamic_map: Dict[str, bool],
    condition: str,
    coords: Sequence[int],
    movement_threshold_px: float,
    rotation_threshold_deg: float,
    contact_min_duration_s: float,
    include_tool_events: bool,
    save_trace_path: Optional[Path],
    terminal_persistence_s: float = 0.0,
    persistence_candidate_signatures: Optional[Sequence[str]] = None,
) -> dict:
    target, goal, blues, movable, static_black = pred.describe_objects(world_data)
    world_obj = pred.loadFromDict(world_data["world"]).copy()
    valid = pred.place_ball_tool(world_obj, (float(coords[0]), float(coords[1])), TOOL_RADIUS_PX, str(condition), "orange")
    if not valid:
        return {"placement_xy": [int(coords[0]), int(coords[1])], "valid": False}

    thr = pred.Thresholds()
    thr.contact_min_s = contact_min_duration_s
    # Keep the rollout alive long enough to evaluate the revised canonical
    # two-second dwell criterion from native simulator ticks.
    thr.goal_hold_s = EVALUATION_GOAL_HOLD_S
    if terminal_persistence_s > 0.0:
        # A terminal-persistence predicate cannot be evaluated if the simulator
        # stops after its historical 0.5-second settling window.  Waiting for
        # the full persistence interval leaves the physical state unchanged
        # once bodies sleep while preserving native 0.02-second ticks.
        thr.settle_window_s = max(
            float(thr.settle_window_s),
            float(terminal_persistence_s),
        )
    initial_contact_pairs = initial_relevant_contact_pairs(
        pred,
        world_obj=world_obj,
        world_data=world_data,
        min_gap_px=INITIAL_CONTACT_RELEVANCE_GAP_PX,
        include_walls=True,
    )
    trace = pred.run_trace(
        world_obj,
        world_dict=world_data,
        maxtime=TRACE_MAXTIME_S,
        step=TRACE_STEP_S,
        collision_slop=TRACE_COLLISION_SLOP,
        target_name=target.name,
        goal_name=(goal.name if goal else None),
        thr=thr,
    )
    object_in_goal_intervals = effective_object_in_goal_intervals(
        pred,
        trace=trace,
        world_data=world_data,
        role_map=role_map,
        include_tool_events=include_tool_events,
    )

    structured = build_structured_trace(
        pred,
        world_data=world_data,
        target=target,
        goal=goal,
        blues=blues,
        movable=movable,
        trace=trace,
        rollout_id=f"{int(coords[0])}_{int(coords[1])}",
    )
    # `run_trace()["in_goal_intervals"]` is not populated reliably when the
    # world's goal object is normalized from names such as Object1 to the
    # semantic goal role. The geometry-derived per-object intervals above are
    # the same intervals used by the final-state predicates, so persist them
    # explicitly and use the target object's intervals for canonical gcond
    # scoring.
    canonical_in_goal_intervals = merge_intervals(
        object_in_goal_intervals.get(str(target.name)) or []
    )
    structured["object_in_goal_intervals"] = {
        str(object_name): merge_intervals(intervals or [])
        for object_name, intervals in object_in_goal_intervals.items()
    }
    structured["in_goal_intervals"] = canonical_in_goal_intervals
    if save_trace_path is not None:
        pred.write_trace(structured, save_trace_path)

    raw_events: List[dict] = []
    object_summaries: Dict[str, dict] = {}
    final_state_map: Dict[str, dict] = {}
    final_partial_in_goal_object_ids: set[str] = set()

    path_map = trace.get("path", {}) or {}
    rot_map = trace.get("rot", {}) or {}

    for raw_name, samples in path_map.items():
        label = label_map.get(raw_name, split_words(raw_name))
        role = role_map.get(raw_name, split_words(raw_name))
        spec = ((world_data.get("world") or {}).get("objects") or {}).get(raw_name, {}) or {}
        movement_events, movement_kin = extract_movement_events(
            raw_name=raw_name,
            label=label,
            role=role,
            path_samples=samples,
            movement_threshold_px=movement_threshold_px,
        )
        rotation_events, rotation_kin = extract_rotation_events(
            raw_name=raw_name,
            label=label,
            role=role,
            spec=spec,
            rotation_samples=rot_map.get(raw_name, []),
            rotation_threshold_deg=rotation_threshold_deg,
        )
        if include_tool_events or role != "tool":
            raw_events.extend(movement_events)
            raw_events.extend(rotation_events)

        final_state, final_predicates = derive_final_state(
            pred,
            world_obj,
            world_data,
            goal.name if goal else None,
            raw_name=raw_name,
            label=label,
            role=role,
            trace=trace,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            object_in_goal_intervals=object_in_goal_intervals,
            initial_contact_pairs=initial_contact_pairs,
            include_tool_events=include_tool_events,
        )
        final_state_map[label] = final_state
        if final_state.get("partially_in_goal"):
            final_partial_in_goal_object_ids.add(raw_name)
        if include_tool_events or role != "tool":
            raw_events.extend(final_predicates)

        object_summaries[label] = {
            "object_id": raw_name,
            "role": role,
            **movement_kin,
            **rotation_kin,
        }

    contact_events, contact_change_events = extract_contact_events(
        pred=pred,
        collisions=trace.get("collisions", []) or [],
        duration_s=float(trace.get("duration") or 0.0),
        world_data=world_data,
        label_map=label_map,
        role_map=role_map,
        dynamic_map=dynamic_map,
        final_contacts=trace.get("final_contacts", {}) or {},
        object_in_goal_intervals=object_in_goal_intervals,
        goal_name=(goal.name if goal else None),
        initial_contact_pairs=initial_contact_pairs,
        final_world_obj=world_obj,
        world_objects=(world_data.get("world") or {}).get("objects") or {},
        contact_min_duration_s=contact_min_duration_s,
        include_tool_events=include_tool_events,
    )
    raw_events.extend(contact_events)
    raw_events.extend(contact_change_events)
    raw_events.extend(
        extract_glide_events(
            trace=trace,
            world_data=world_data,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            movement_threshold_px=movement_threshold_px,
            initial_contact_pairs=initial_contact_pairs,
            include_tool_events=include_tool_events,
        )
    )
    raw_events.extend(
        extract_joint_movement_events(
            path_map=path_map,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            movement_threshold_px=movement_threshold_px,
            include_tool_events=include_tool_events,
        )
    )
    raw_events.extend(
        extract_container_events(
            object_in_goal_intervals=object_in_goal_intervals,
            duration_s=float(trace.get("duration") or 0.0),
            label_map=label_map,
            role_map=role_map,
            final_partial_in_goal_object_ids=final_partial_in_goal_object_ids,
            include_tool_events=include_tool_events,
        )
    )
    raw_events.sort(key=lambda item: (float(item["time_s"]) if item.get("time_s") is not None else float("inf"), item["category"], item["display"]))

    endpoint_final_signatures = sorted(
        {
            event_signature(event)
            for event in raw_events
            if event.get("category") == "final_state"
        }
    )
    persistent_final_signatures = list(endpoint_final_signatures)
    persistence_sample_count = None
    persistence_start_time_s = None
    if terminal_persistence_s > 0.0:
        (
            persistent_final_signatures,
            persistence_sample_count,
            persistence_start_time_s,
        ) = terminal_persistent_final_state_signatures(
            pred=pred,
            world_data=world_data,
            condition=condition,
            coords=coords,
            trace=trace,
            label_map=label_map,
            role_map=role_map,
            dynamic_map=dynamic_map,
            initial_contact_pairs=initial_contact_pairs,
            include_tool_events=include_tool_events,
            endpoint_signatures=endpoint_final_signatures,
            persistence_s=float(terminal_persistence_s),
            candidate_signatures=persistence_candidate_signatures,
        )
        persistent_set = set(persistent_final_signatures)
        raw_events = [
            event
            for event in raw_events
            if event.get("category") != "final_state"
            or event_signature(event) in persistent_set
        ]

    return {
        "placement_xy": [int(coords[0]), int(coords[1])],
        "valid": True,
        "settled_within_horizon": bool(structured.get("settled_within_horizon")),
        "settle_time_s": structured.get("settle_time_s"),
        "duration_s": structured.get("duration_s"),
        "event_graph": raw_events,
        "final_state_by_object": final_state_map,
        "object_summaries": object_summaries,
        "structured_trace_path": str(save_trace_path) if save_trace_path else None,
        "terminal_persistence_s": (
            float(terminal_persistence_s)
            if terminal_persistence_s > 0.0
            else None
        ),
        "terminal_persistence_sample_count": persistence_sample_count,
        "terminal_persistence_start_time_s": persistence_start_time_s,
        "endpoint_final_state_signatures": endpoint_final_signatures,
        "persistent_final_state_signatures": persistent_final_signatures,
        "canonical_in_goal_dwell_2s": any(
            isinstance(interval, (list, tuple))
            and len(interval) >= 2
            and float(interval[1]) - float(interval[0]) >= 2.0 - 1e-9
            for interval in canonical_in_goal_intervals
        ),
        "canonical_in_goal_intervals": [
            [float(interval[0]), float(interval[1])]
            for interval in canonical_in_goal_intervals
            if isinstance(interval, (list, tuple)) and len(interval) >= 2
        ],
    }


def load_checkpoint_rows(checkpoint_path: Path) -> List[dict]:
    """Load JSONL rows, recovering only a torn final line in a partial file.

    Spot VM termination can interrupt one append after only part of its JSON
    object has reached disk. Archive that exact tail and truncate it before
    resuming. Corruption in a completed file, or anywhere except the final
    non-empty line of a partial checkpoint, remains a hard failure.
    """

    raw = checkpoint_path.read_bytes()
    chunks = raw.splitlines(keepends=True)
    nonempty_indices = [
        idx for idx, chunk in enumerate(chunks) if chunk.strip()
    ]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else None
    rows: List[dict] = []
    offset = 0
    recovered_torn_tail = False

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            offset += len(chunk)
            continue
        try:
            row = json.loads(chunk.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            recoverable = (
                checkpoint_path.name == "placements.partial.jsonl"
                and idx == last_nonempty
            )
            if not recoverable:
                raise RuntimeError(
                    f"Invalid JSONL checkpoint row {idx + 1}: {checkpoint_path}"
                )

            damaged_tail = raw[offset:]
            digest = hashlib.sha256(damaged_tail).hexdigest()[:12]
            archive_path = checkpoint_path.with_name(
                f"{checkpoint_path.name}.preemption-tail-{digest}.bad"
            )
            if not archive_path.exists():
                archive_path.write_bytes(damaged_tail)
            with checkpoint_path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            print(
                f"[checkpoint-recovery] archived torn final line to "
                f"{archive_path.name}; retained {len(rows)} rows",
                flush=True,
            )
            recovered_torn_tail = True
            break
        if not isinstance(row, dict):
            raise RuntimeError(
                f"Checkpoint row {idx + 1} is not an object: {checkpoint_path}"
            )
        rows.append(row)
        offset += len(chunk)

    if (
        checkpoint_path.name == "placements.partial.jsonl"
        and raw
        and not recovered_torn_tail
        and not raw.endswith((b"\n", b"\r"))
    ):
        with checkpoint_path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            "[checkpoint-recovery] restored missing final newline after a "
            f"complete JSON row; retained {len(rows)} rows",
            flush=True,
        )

    return rows


def process_environment(
    *,
    env_path: Path,
    condition: str,
    args: argparse.Namespace,
    output_root: Path,
) -> dict:
    pred = get_predicate_module()
    world_data = pred.load_world_file(env_path)
    world_data["_source_name"] = env_path.stem
    world_data["_env_set"] = env_path.parent.name
    world_data["_env_id"] = env_path.stem
    label_map, role_map, dynamic_map = build_label_maps(pred, world_data)
    dims = (world_data.get("world") or {}).get("dims") or [600, 600]
    grid = generate_grid(dims, args.grid_step, TOOL_RADIUS_PX)
    run_name = f"{env_path.parent.name}_{env_path.stem}_{condition}"
    persistence_candidates = None
    if getattr(args, "persistence_signatures_by_run", None) is not None:
        persistence_candidates = sorted(
            args.persistence_signatures_by_run.get(run_name, set())
        )

    env_dir = ensure_dir(output_root / f"{env_path.parent.name}_{env_path.stem}_{condition}")
    placements_path = env_dir / "placements.jsonl"
    partial_path = env_dir / "placements.partial.jsonl"
    rows: List[dict] = []
    checkpoint_path = (
        placements_path
        if placements_path.exists()
        else (partial_path if partial_path.exists() else None)
    )
    if checkpoint_path is not None:
        rows = load_checkpoint_rows(checkpoint_path)
        if len(rows) > len(grid):
            raise RuntimeError(
                f"Placement checkpoint has too many rows: {checkpoint_path}"
            )
        for idx, row in enumerate(rows):
            if list(row.get("placement_xy") or []) != list(grid[idx]):
                raise RuntimeError(
                    f"Placement checkpoint grid mismatch at row {idx}: "
                    f"{checkpoint_path}"
                )
    valid_count = sum(1 for row in rows if row.get("valid"))
    started_at = time.time()

    print(
        f"[progress] {run_name} start placements={len(rows)}/{len(grid)} "
        f"grid_step={args.grid_step}",
        flush=True,
    )

    write_path = placements_path if placements_path.exists() else partial_path
    append_mode = "a" if rows else "w"
    with write_path.open(append_mode, encoding="utf-8") as partial_handle:
        for idx, coords in enumerate(grid[len(rows) :], start=len(rows) + 1):
            trace_path = None
            if args.save_structured_traces:
                trace_path = env_dir / "structured_traces" / f"trace_{coords[0]}_{coords[1]}.json"
                ensure_dir(trace_path.parent)
            row = simulate_valid_placement(
                pred=pred,
                world_data=world_data,
                label_map=label_map,
                role_map=role_map,
                dynamic_map=dynamic_map,
                condition=condition,
                coords=coords,
                movement_threshold_px=args.movement_threshold_px,
                rotation_threshold_deg=args.rotation_threshold_deg,
                contact_min_duration_s=args.contact_min_duration_s,
                include_tool_events=args.include_tool_events,
                save_trace_path=trace_path,
                terminal_persistence_s=args.terminal_persistence_s,
                persistence_candidate_signatures=persistence_candidates,
            )
            rows.append(row)
            partial_handle.write(json.dumps(row) + "\n")
            if row.get("valid"):
                valid_count += 1
            if idx % 100 == 0:
                partial_handle.flush()
            if idx % 1000 == 0 or idx == len(grid):
                print(
                    f"[progress] {run_name} placements={idx}/{len(grid)} valid={valid_count}",
                    flush=True,
                )
            if args.max_valid_placements and valid_count >= args.max_valid_placements:
                break

    if write_path != placements_path:
        write_path.replace(placements_path)

    aggregate = aggregate_event_bank(rows)
    summary = {
        "environment_json": str(env_path),
        "condition": condition,
        "grid_step_px": args.grid_step,
        "movement_threshold_px": args.movement_threshold_px,
        "rotation_threshold_deg": args.rotation_threshold_deg,
        "contact_min_duration_s": args.contact_min_duration_s,
        "terminal_persistence_s": args.terminal_persistence_s,
        "persistence_signature_manifest": args.persistence_signature_manifest,
        "persistence_candidate_signature_count": (
            len(persistence_candidates)
            if persistence_candidates is not None
            else None
        ),
        "include_tool_events": bool(args.include_tool_events),
        "grid_points_considered": len(rows),
        "valid_placements": sum(1 for row in rows if row.get("valid")),
        "invalid_placements": sum(1 for row in rows if not row.get("valid")),
        "elapsed_s": time.time() - started_at,
        "ranked_goal_bank": aggregate,
    }
    (env_dir / "aggregate_goal_bank.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[progress] {run_name} done valid={summary['valid_placements']} "
        f"invalid={summary['invalid_placements']} elapsed_s={summary['elapsed_s']:.1f}",
        flush=True,
    )
    return {
        "environment_json": str(env_path),
        "condition": condition,
        "output_dir": str(env_dir),
        "grid_points_considered": len(rows),
        "valid_placements": summary["valid_placements"],
        "invalid_placements": summary["invalid_placements"],
    }


def main() -> None:
    args = parse_args()
    args.persistence_signatures_by_run = None
    if args.persistence_signature_manifest:
        manifest_path = Path(args.persistence_signature_manifest).expanduser().resolve()
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_goals = (
            manifest_payload.get("goals")
            if isinstance(manifest_payload, dict)
            else None
        )
        if not isinstance(manifest_goals, list):
            raise ValueError(
                "Persistence signature manifest does not contain a goals list: "
                f"{manifest_path}"
            )
        signatures_by_run: Dict[str, set[str]] = defaultdict(set)
        for goal in manifest_goals:
            run_name = str(goal.get("run_name") or "")
            signature = str(goal.get("signature") or "")
            if run_name and signature:
                signatures_by_run[run_name].add(signature)
        args.persistence_signatures_by_run = signatures_by_run
        args.persistence_signature_manifest = str(manifest_path)
    env_paths = resolve_environment_paths(args)
    conditions = ["upward", "downward"] if args.condition == "both" else [args.condition]
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve() if args.output_root else DEFAULT_OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S"))

    run_rows: List[dict] = []
    for env_path in env_paths:
        for condition in conditions:
            run_rows.append(
                process_environment(
                    env_path=env_path,
                    condition=condition,
                    args=args,
                    output_root=output_root,
                )
            )

    batch_summary = {
        "output_root": str(output_root),
        "runs": run_rows,
    }
    (output_root / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(json.dumps(batch_summary, indent=2))


if __name__ == "__main__":
    main()
