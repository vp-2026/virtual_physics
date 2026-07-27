#!/usr/bin/env python3
import math
from typing import Dict, List


VALID_TOOL_NAMES = ("obj1", "obj2", "obj3")
VALID_TOOL_COLORS = ("lightblue", "orange", "pink")
ALLOWED_TRIALS = (
    "Basic",
    "Catapult",
    "Collapse",
    "DoubleDown",
    "Unbox",
    "Trap",
    "Towers",
    "Remove",
)


def regular_ngon(radius: float, n: int = 32) -> List[List[List[float]]]:
    verts = []
    for i in range(n):
        theta = 2.0 * math.pi * (1.0 - (i / n))
        verts.append([float(radius * math.cos(theta)), float(radius * math.sin(theta))])
    return [verts]


def build_default_tools(color_obj1: str = "lightblue", color_obj2: str = "orange", color_obj3: str = "pink") -> Dict[str, dict]:
    return {
        "obj1": {
            "polys": regular_ngon(36, n=32),
            "density": 1.0,
            "friction": 0.5,
            "elasticity": 0.5,
            "color": color_obj1,
            "placement_radius": 36,
            "kind": "strong",
        },
        "obj2": {
            "polys": regular_ngon(36, n=32),
            "density": 1.0,
            "friction": 0.45,
            "elasticity": 0.9,
            "color": color_obj2,
            "placement_radius": 36,
            "launch_speed_x": 120.0,
            "launch_speed_y": 60.0,
            "launch_spin": 18.0,
            "body_damping": 0.82,
            "kind": "sometimes_helpful",
        },
        "obj3": {
            "polys": regular_ngon(36, n=32),
            "density": 0.01,
            "friction": 0.0,
            "elasticity": 0.1,
            "color": color_obj3,
            "placement_radius": 36,
            "body_damping": 0.99,
            "kind": "weak",
        },
    }


def normalize_trial_name(name: str) -> str:
    normalized = name.strip()
    if normalized in ALLOWED_TRIALS:
        return normalized
    raise ValueError(f"Unsupported trial '{name}'. Allowed: {', '.join(ALLOWED_TRIALS)}")
