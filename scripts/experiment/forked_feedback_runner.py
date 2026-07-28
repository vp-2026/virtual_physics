#!/usr/bin/env python3
"""Forked reviewer-response pilot for the balanced VTools benchmark.

The runner makes exactly one shared first-attempt model call for each
model x goal x seed unit.  It simulates that placement once and then forks
the text conversation into five supported conditions:

    full          latest 32 rollout frames + explicit success/failure
    frames_only   latest 32 rollout frames + no explicit status
    status_only   explicit success/failure + no rollout frames
    neither       neither rollout frames nor explicit status
    trace_status  latest 32 observable JSON states + explicit success/failure

The static initial screenshot is resent on every action request in every
condition.  All prior text prompts and model responses remain in context.
Older rollout images or JSON states are never resent; only the latest
rollout's evidence is available in the feedback conditions.

Status-visible branches stop at the first success.  Status-hidden branches
continue to the predeclared per-goal budget even after a latent success, so
continuation cannot leak the outcome.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OLD_RUNNER_ROOT = HERE
OLD_PYTHON_PACKAGES = REPO_ROOT / "vendor" / "python_pkgs"
OLD_RUNNER_PATH = HERE / "goal_probe_runner.py"
DEFAULT_BALANCED_MANIFEST = REPO_ROOT / "task_configs" / "benchmark_1692_seed2026.json"
DEFAULT_GOAL_BUILDER = (
    REPO_ROOT
    / "simulator"
    / "goal_semantics"
    / "build_goal_bank_from_placement_sweep.py"
)
DEFAULT_VTOOLS_ROOT = REPO_ROOT
DEFAULT_ASSET_ROOTS = (REPO_ROOT,)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiment"
DEFAULT_ASSET_INDEX = REPO_ROOT / "task_configs" / "asset_index_132.json"
DEFAULT_PYTHON_BIN = Path(sys.executable)
DEFAULT_SEEDS = (42, 123, 2026)
TOOL_DIAMETER_PX = 72
SIM_WIDTH = 600
SIM_HEIGHT = 600


def primary_terminal_persistence_s(goal: Dict[str, Any]) -> float:
    """Return the frozen primary evaluator's terminal-persistence window."""
    if str(goal.get("source") or "") == "canonical_world_gcond":
        gcond = goal.get("canonical_gcond") or {}
        return float(gcond.get("duration") or 0.0)
    return 0.0


def configure_python_path() -> None:
    if not OLD_PYTHON_PACKAGES.exists():
        return
    package_text = str(OLD_PYTHON_PACKAGES)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if package_text not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([package_text, *parts])


configure_python_path()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if not OLD_RUNNER_PATH.exists():
    raise FileNotFoundError(f"Recovered historical runner is missing: {OLD_RUNNER_PATH}")
base = load_module(OLD_RUNNER_PATH, "recovered_vtools_goal_probe_runner")


CONDITIONS: Dict[str, Dict[str, bool]] = {
    "full": {"frames": True, "status": True, "trace": False},
    "frames_only": {"frames": True, "status": False, "trace": False},
    "status_only": {"frames": False, "status": True, "trace": False},
    "neither": {"frames": False, "status": False, "trace": False},
    "trace_status": {"frames": False, "status": True, "trace": True},
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    temperature: Optional[float]
    reasoning_effort: Optional[str]
    label: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt5": ModelSpec(
        key="gpt5",
        model_id="openai/gpt-5",
        temperature=None,
        reasoning_effort="medium",
        label="GPT-5",
    ),
    "gemini31pro": ModelSpec(
        key="gemini31pro",
        model_id="google/gemini-3.1-pro-preview",
        temperature=0.2,
        reasoning_effort="medium",
        label="Gemini 3.1 Pro Preview",
    ),
    "qwen36plus": ModelSpec(
        key="qwen36plus",
        model_id="qwen/qwen3.6-plus",
        temperature=0.2,
        reasoning_effort="medium",
        label="Qwen 3.6 Plus",
    ),
    "gemini_robotics_er": ModelSpec(
        key="gemini_robotics_er",
        model_id="gemini-robotics-er-1.6-preview",
        temperature=0.2,
        reasoning_effort="medium",
        label="Gemini Robotics-ER 1.6 Preview",
    ),
}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Unsupported image type: {path}")


def parse_action(response_text: str) -> Optional[Tuple[int, int]]:
    """Parse an action while preserving the recovered runner's rounding rule."""
    parsed = parse_action_payload(response_text).get("parsed")
    if not isinstance(parsed, dict):
        return None
    point = parsed.get("point")
    if not (isinstance(point, list) and len(point) == 2):
        return None
    try:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
    except Exception:
        return None
    if 0 <= x < SIM_WIDTH and 0 <= y < SIM_HEIGHT:
        return x, y
    return None


def parse_action_payload(response_text: str) -> Dict[str, Any]:
    """Best-effort parse of the action-plus-prediction response."""
    raw = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidates = [fenced.group(1)]
    else:
        first = raw.find("{")
        last = raw.rfind("}")
        candidates = [raw[first:]] if first >= 0 else []
        if first >= 0 and last > first and last < len(raw) - 1:
            candidates.append(raw[first : last + 1])
    payload = None
    normalization = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            # Some OpenRouter backends occasionally return a complete JSON
            # object with only the final outer brace omitted despite
            # finish_reason=stop. Accept only that terminal repair.
            try:
                payload = json.loads(candidate + "}")
                normalization = "appended_one_terminal_object_brace"
                break
            except Exception:
                pass
            trimmed = candidate.rstrip()
            if trimmed.endswith("}"):
                try:
                    payload = json.loads(trimmed[:-1].rstrip())
                    normalization = "removed_one_extra_terminal_object_brace"
                    break
                except Exception:
                    pass
            continue
    if not isinstance(payload, dict):
        return {
            "schema_valid": False,
            "parsed": None,
            "coordinate_predictions": [],
            "normalization": None,
        }
    predictions = payload.get("coordinate_predictions")
    if not isinstance(predictions, list):
        predictions = []
    required_prediction_fields_valid = all(
        isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and (
            (
                isinstance(item.get("point"), list)
                and len(item["point"]) == 2
                and all(isinstance(value, (int, float)) for value in item["point"])
            )
            or str(item.get("state")) == "exited"
        )
        for item in predictions
    )
    return {
        "schema_valid": (
            isinstance(payload, dict)
            and isinstance(payload.get("point"), list)
            and len(payload["point"]) == 2
            and required_prediction_fields_valid
        ),
        "parsed": payload,
        "coordinate_predictions": predictions,
        "normalization": normalization,
    }


def model_facing_goal_text(value: Any) -> str:
    return base.model_facing_goal_text(value)


def world_object_is_dynamic(
    spec: Dict[str, Any], *, default_density: float = 1.0
) -> bool:
    """Match the simulator's physical static/dynamic distinction.

    The recovered goal-builder map is semantic: in particular, it labels a
    green goal container as "dynamic" even when its world density is zero.
    Model-facing prompts and prediction targets must instead follow the
    physics engine, where density-zero objects are fixed.
    """
    raw_density = spec.get("density", default_density)
    try:
        return float(raw_density) > 0.0
    except (TypeError, ValueError):
        return float(default_density) > 0.0


def prediction_targets_for_goal(goal: Dict[str, Any], goal_bank: Any) -> List[Dict[str, str]]:
    ctx = goal_bank.context_for(goal)
    world = ctx["world_data"].get("world") or {}
    world_objects = world.get("objects") or {}
    default_density = float((world.get("defaults") or {}).get("density") or 1.0)
    targets: List[Dict[str, str]] = []
    for raw_id in sorted(world_objects):
        if str(raw_id).startswith("_"):
            continue
        spec = world_objects[raw_id] or {}
        if not world_object_is_dynamic(spec, default_density=default_density):
            continue
        targets.append(
            {
                "id": str(raw_id),
                "label": str(ctx["label_map"].get(raw_id) or raw_id),
                "shape": str(spec.get("type") or "object"),
            }
        )
    targets.append({"id": "PLACED", "label": "orange ball tool", "shape": "Ball"})
    return targets


def response_schema_text(prediction_targets: Sequence[Dict[str, str]]) -> str:
    target_lines = "\n".join(
        f'- id "{target["id"]}": {target["label"]} ({target["shape"]})'
        for target in prediction_targets
    )
    return f"""Before seeing the rollout, predict the terminal state of every listed movable object:
{target_lines}

For an object remaining in the scene, use {{"id":"...","state":"in_scene","point":[x,y],"orientation_deg":number_or_null}}.
For an object that exits, use {{"id":"...","state":"exited","exit_side":"left|right|top|bottom","point":null,"orientation_deg":null}}.
Use orientation_deg only when meaningful for a non-circular object; otherwise use null.

Return exactly one JSON object:
{{"action":"drop","point":[x,y],"coordinate_predictions":[...one entry for every id above...],"reasoning":"<brief reasoning>"}}"""


def build_goal_reference_map(data_root: Path) -> Dict[str, Dict[str, Any]]:
    """Recover normalized puzzle/index/signature fields for the 861 new goals."""
    references: Dict[str, Dict[str, Any]] = {}
    for path in data_root.rglob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            entries = payload.get("selected_goals") or []
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            balanced_id = str(entry.get("balanced_goal_id") or "")
            if balanced_id and entry.get("puzzle_key") and entry.get("goal_index") is not None:
                references[balanced_id] = dict(entry)
    return references


def parse_goal_identity(
    goal: Dict[str, Any],
    reference_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, int]:
    run_name = str(goal.get("run_name") or "")
    goal_dir = str(goal.get("goal_dir") or goal.get("balanced_goal_id") or "")
    match = re.match(r"^\d+_(.+)__goal_(\d+)$", goal_dir)
    if match:
        puzzle_key = run_name or match.group(1)
        return puzzle_key, int(match.group(2))
    if str(goal.get("source")) == "canonical_world_gcond" and run_name:
        return run_name, -1
    balanced_id = str(goal.get("balanced_goal_id") or "")
    reference = (reference_map or {}).get(balanced_id)
    if reference:
        return str(reference["puzzle_key"]), int(reference["goal_index"])
    if run_name:
        # Newly sampled and canonical manifest rows carry the normalized
        # signature directly, so the historical goal-bank ordinal is not
        # needed for simulation or scoring.
        return run_name, -1
    raise ValueError(f"Could not parse goal identity from {goal_dir!r}")


def parse_puzzle_key(puzzle_key: str) -> Tuple[str, str, str]:
    parts = puzzle_key.rsplit("_", 2)
    if len(parts) != 3 or parts[2] not in {"upward", "downward"}:
        raise ValueError(f"Invalid puzzle key: {puzzle_key}")
    return parts[0], parts[1], parts[2]


def build_asset_index(
    *,
    asset_roots: Sequence[Path],
    needed_puzzles: Iterable[str],
    output_path: Path,
) -> Dict[str, Dict[str, str]]:
    needed = set(needed_puzzles)
    candidates: Dict[str, List[Tuple[Path, Path]]] = {key: [] for key in needed}
    pattern = re.compile(r"^\d+_(.+)__goal_\d+$")
    for root in asset_roots:
        if not root.exists():
            continue
        for world_path in root.rglob("simulation_world.json"):
            match = pattern.match(world_path.parent.name)
            if not match:
                continue
            puzzle_key = match.group(1)
            if puzzle_key not in needed:
                continue
            screenshot_path = world_path.parent / "initial_observation.png"
            if screenshot_path.exists():
                candidates[puzzle_key].append((world_path, screenshot_path))

    index: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []
    inconsistent: Dict[str, Dict[str, int]] = {}
    for puzzle_key in sorted(needed):
        usable: List[Tuple[Path, Path, str, str]] = []
        for world_path, screenshot_path in candidates.get(puzzle_key, []):
            try:
                load_json(world_path)
                screenshot_path.open("rb").read(16)
                usable.append(
                    (
                        world_path.resolve(),
                        screenshot_path.resolve(),
                        sha256_file(world_path),
                        sha256_file(screenshot_path),
                    )
                )
            except Exception:
                continue
        if not usable:
            missing.append(puzzle_key)
            continue
        world_hashes = {item[2] for item in usable}
        screenshot_hashes = {item[3] for item in usable}
        if len(world_hashes) > 1 or len(screenshot_hashes) > 1:
            inconsistent[puzzle_key] = {
                "world_variants": len(world_hashes),
                "screenshot_variants": len(screenshot_hashes),
            }
        world_path, screenshot_path, world_hash, screenshot_hash = usable[0]
        index[puzzle_key] = {
            "world_path": str(world_path),
            "screenshot_path": str(screenshot_path),
            "world_sha256": world_hash,
            "screenshot_sha256": screenshot_hash,
            "candidate_copies": str(len(usable)),
        }
    if missing:
        raise RuntimeError(f"Missing readable cached assets for {len(missing)} puzzles: {missing[:10]}")
    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "asset_roots": [str(path.resolve()) for path in asset_roots],
        "puzzle_count": len(index),
        "inconsistent_assets": inconsistent,
        "puzzles": index,
    }
    atomic_write_json(output_path, payload)
    return index


def load_or_build_asset_index(
    manifest_goals: Sequence[Dict[str, Any]],
    *,
    asset_index_path: Path,
    asset_roots: Sequence[Path],
    rebuild: bool,
) -> Dict[str, Dict[str, str]]:
    needed = {str(goal.get("run_name") or parse_goal_identity(goal)[0]) for goal in manifest_goals}
    if asset_index_path.exists() and not rebuild:
        payload = load_json(asset_index_path)
        index = dict(payload.get("puzzles") or {})
        for assets in index.values():
            for field in ("world_path", "screenshot_path", "tool_path"):
                raw_path = assets.get(field)
                if not raw_path:
                    continue
                path = Path(str(raw_path)).expanduser()
                if not path.is_absolute():
                    path = asset_index_path.parent / path
                assets[field] = str(path.resolve())
        raw_by_puzzle = {
            str(goal.get("run_name") or parse_goal_identity(goal)[0]): goal
            for goal in manifest_goals
        }
        for puzzle_key in sorted(needed - set(index)):
            family, env_id, condition = parse_puzzle_key(puzzle_key)
            counterpart = f"{family}_{env_id}_{'downward' if condition == 'upward' else 'upward'}"
            counterpart_assets = index.get(counterpart)
            raw = raw_by_puzzle.get(puzzle_key) or {}
            world_source = raw.get("environment_json_source")
            if counterpart_assets and world_source and Path(str(world_source)).exists():
                index[puzzle_key] = {
                    **counterpart_assets,
                    "world_path": str(Path(str(world_source)).resolve()),
                    "world_sha256": sha256_file(Path(str(world_source))),
                    "candidate_copies": "canonical_counterpart_screenshot_alias",
                }
        if needed.issubset(index):
            return index
    return build_asset_index(
        asset_roots=asset_roots,
        needed_puzzles=needed,
        output_path=asset_index_path,
    )


def load_balanced_goals(
    manifest_path: Path,
    *,
    asset_index: Dict[str, Dict[str, str]],
    budget_mode: str,
    fixed_budget: int,
) -> List[Dict[str, Any]]:
    payload = load_json(manifest_path)
    raw_goals = payload.get("goals") if isinstance(payload, dict) else None
    if not isinstance(raw_goals, list):
        raise ValueError(f"Balanced manifest does not contain a goals list: {manifest_path}")
    reference_map = build_goal_reference_map(manifest_path.parent)
    goals: List[Dict[str, Any]] = []
    for benchmark_index, raw in enumerate(raw_goals):
        puzzle_key, goal_index = parse_goal_identity(raw, reference_map)
        family, env_id, condition = parse_puzzle_key(puzzle_key)
        assets = asset_index[puzzle_key]
        reference = reference_map.get(str(raw.get("balanced_goal_id") or ""), {})
        signature = str(reference.get("signature") or raw["signature"])
        goal_text = (
            reference.get("goal_text")
            or raw.get("goal_text_cleaned")
            or raw.get("goal_text_original")
            or signature
        )
        manifest_budget = int(float(raw["attempt_limit_dynamic"]))
        budget = manifest_budget if budget_mode == "manifest" else fixed_budget
        goals.append(
            {
                "benchmark_index": benchmark_index,
                "balanced_goal_id": str(raw["balanced_goal_id"]),
                "puzzle_key": puzzle_key,
                "family": family,
                "env_id": env_id,
                "condition": condition,
                "goal_index": goal_index,
                "signature": signature,
                "goal_text": model_facing_goal_text(goal_text),
                "category": raw.get("raw_category") or raw.get("category"),
                "category_5": str(raw["category_5"]),
                "internal_subtype": str(raw["internal_subtype"]),
                "probability_valid_placements": (
                    float(raw["p_success"]) if raw.get("p_success") is not None else None
                ),
                "attempt_limit_manifest": manifest_budget,
                "attempt_budget": budget,
                "source": str(raw.get("source") or ""),
                "canonical_gcond": raw.get("canonical_gcond"),
                "temporal_category": str(raw.get("temporal_category") or ""),
                "environment_json": assets["world_path"],
                "initial_screenshot": assets["screenshot_path"],
                "asset_world_sha256": assets["world_sha256"],
                "asset_screenshot_sha256": assets["screenshot_sha256"],
            }
        )
    return goals


def validate_balanced_goals(goals: Sequence[Dict[str, Any]]) -> None:
    if not goals:
        raise RuntimeError("Manifest contains no goals")
    counts: Dict[str, int] = {}
    gravity_counts: Dict[str, int] = {}
    for goal in goals:
        category = str(goal["category_5"])
        counts[category] = counts.get(category, 0) + 1
        gravity = str(goal["condition"])
        gravity_counts[gravity] = gravity_counts.get(gravity, 0) + 1
        budget = int(goal["attempt_budget"])
        if not 1 <= budget <= 8:
            raise RuntimeError(f"Invalid attempt budget {budget}: {goal['balanced_goal_id']}")
    if len(goals) in {152, 1514} and len(set(gravity_counts.values())) != 1:
        raise RuntimeError(f"Expected gravity balance, found {gravity_counts}")


def build_system_prompt(prompt_variant: str = "full") -> str:
    if prompt_variant == "compact":
        role = "You solve a visual tool-placement task."
    elif prompt_variant == "full":
        role = (
            "You are an expert physical reasoning agent solving deterministic "
            "2D rigid-body physics puzzles."
        )
    else:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant}")
    return f"""{role}
Use the static scene image and any rollout evidence supplied on the current turn.
The experimenter controls continuation; being asked for another candidate does not by itself imply success or failure.
Do not decide when the experiment should stop.
Return only the exact JSON schema requested in each prompt."""


def scene_and_tool_rules(goal: Dict[str, Any], goal_bank: Any) -> Tuple[str, str]:
    ctx = goal_bank.context_for(goal)
    world = ctx["world_data"].get("world") or {}
    world_objects = world.get("objects") or {}
    default_density = float((world.get("defaults") or {}).get("density") or 1.0)
    green_is_movable: Optional[bool] = None
    for raw_id, spec in world_objects.items():
        if str(raw_id).startswith("_"):
            continue
        color = str(
            (spec or {}).get("color")
            or (spec or {}).get("innerColor")
            or "black"
        ).lower()
        if color == "green" or str((spec or {}).get("type")) == "Container":
            green_is_movable = world_object_is_dynamic(
                spec or {}, default_density=default_density
            )
            break
    if green_is_movable is None:
        raise RuntimeError(
            f"Could not identify the green goal container for {goal['balanced_goal_id']}"
        )
    scene_rules = "\n".join(
        (
            "- Black objects are fixed and cannot be moved.",
            "- The red ball and all blue objects are movable.",
            (
                "- The green goal container is movable."
                if green_is_movable
                else "- The green goal container is fixed and cannot be moved."
            ),
        )
    )
    if str(goal["condition"]) == "upward":
        tool_dynamics = (
            "After placement, the orange tool is released and accelerates upward, "
            "while every other movable object accelerates downward."
        )
    elif str(goal["condition"]) == "downward":
        tool_dynamics = (
            "After placement, the orange tool is released and accelerates downward, "
            "as do the other movable objects."
        )
    else:
        raise RuntimeError(
            f"Unsupported dynamics condition for {goal['balanced_goal_id']}: "
            f"{goal['condition']}"
        )
    return scene_rules, tool_dynamics


def build_intro_prompt(
    goal_text: str,
    attempt_budget: int,
    prediction_targets: Sequence[Dict[str, str]],
    *,
    representation_note: str = "The first attached image is the static initial scene.",
    scene_rules: str = "",
    tool_dynamics: str = "",
) -> str:
    return f"""Goal: {goal_text}

You may be asked for up to {attempt_budget} valid candidate placements. Each candidate is simulated independently from the same initial scene. The experimenter controls whether another candidate is requested and what evidence is supplied, so continuation alone does not reveal the preceding outcome.

{representation_note}

The environment uses a 600 x 600 coordinate space:
- Coordinates are integer pixels in the range 0-599.
- The origin (0,0) is at the bottom-left corner.
- The x coordinate increases from left to right.
- The y coordinate increases from bottom to top.
{scene_rules}
- You may drop one orange ball tool with diameter {TOOL_DIAMETER_PX} pixels.
- {tool_dynamics}
- Choose a center point where the whole orange ball fits.
- Do not choose a placement that overlaps an existing object or scene boundary.

Interaction context:
- Your earlier action coordinates and coordinate predictions remain in the text conversation.
- Any success/failure labels revealed by this condition remain in the text conversation.
- Rollout media do not persist. On a continuation turn, only the latest rollout's 32 images or 32 JSON states are supplied.

{response_schema_text(prediction_targets)}"""


def build_continuation_prompt(
    *,
    condition: str,
    goal_text: str,
    previous_coords: Sequence[int],
    previous_solved: bool,
    completed_attempt: int,
    attempt_budget: int,
    frame_count: int,
    prediction_targets: Sequence[Dict[str, str]],
    trace_payload: Optional[Dict[str, Any]] = None,
) -> str:
    spec = CONDITIONS[condition]
    remaining = attempt_budget - completed_attempt
    lines = [
        f"Attempt {completed_attempt} was recorded at placement {list(previous_coords)}.",
        f"Goal: {goal_text}",
        (
            "The unchanged static initial scene is supplied below as observable JSON."
            if spec["trace"]
            else "The first attached image is the unchanged static initial scene."
        ),
    ]
    if spec["frames"]:
        lines.append(
            f"The next {frame_count} attached images are uniformly sampled frames from only the latest rollout, ordered early to late."
        )
    elif spec["trace"]:
        lines.append(
            "The JSON below contains 32 uniformly sampled observable states from only the latest rollout."
        )
    else:
        lines.append("Rollout frames are withheld in this condition.")
    if spec["status"]:
        result = "satisfied the goal" if previous_solved else "did not satisfy the goal"
        lines.append(f"Simulator result: the previous placement {result}.")
    else:
        lines.append("The simulator success/failure result is withheld in this condition.")
    details = "\n".join(f"- {line}" for line in lines)
    trace_text = ""
    if trace_payload is not None:
        trace_text = "\n\nLatest observable rollout JSON:\n" + json.dumps(
            trace_payload, separators=(",", ":"), sort_keys=True
        )
    return f"""Latest recorded candidate:
{details}
{trace_text}

You have {remaining} valid candidate placements remaining. Continuation does not by itself reveal the previous result. Use the information available in the conversation to submit a distinct next candidate.

{response_schema_text(prediction_targets)}"""


def build_invalid_response_prompt(
    prediction_targets: Sequence[Dict[str, str]],
) -> str:
    return """Your response did not contain a valid in-bounds placement.
This formatting repair conveys no simulator outcome.
""" + response_schema_text(prediction_targets)


def build_blocked_prompt(
    coords: Sequence[int],
    prediction_targets: Sequence[Dict[str, str]],
) -> str:
    return f"""The proposed placement {list(coords)} is geometrically invalid because the orange ball overlaps an existing object or scene boundary. This validity repair is not a physics-goal outcome and does not count against the valid-attempt budget.

Choose a different in-bounds, non-overlapping placement.
{response_schema_text(prediction_targets)}"""


def build_duplicate_prompt(
    coords: Sequence[int],
    prediction_targets: Sequence[Dict[str, str]],
) -> str:
    return f"""The proposed placement {list(coords)} duplicates an earlier candidate in this branch. This distinct-action repair conveys no simulator outcome and does not count against the valid-attempt budget.

Choose a different in-bounds, non-overlapping placement.
{response_schema_text(prediction_targets)}"""


class Provider:
    def __init__(self, model_spec: ModelSpec, seed: int, system_prompt: str) -> None:
        self.model_spec = model_spec
        self.seed = seed
        self.system_prompt = system_prompt
        self.last_call_metadata: Dict[str, Any] = {}
        self.cumulative_cost_usd = 0.0

    def generate(
        self,
        *,
        history: Sequence[Dict[str, str]],
        prompt: str,
        image_paths: Sequence[Path],
    ) -> str:
        raise NotImplementedError


def provider_cost_from_call_logs(paths: Iterable[Path]) -> float:
    """Recover prior billed/estimated cost so restart guards stay cumulative."""
    total = 0.0
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = record.get("provider_metadata") or {}
                usage = metadata.get("usage") or {}
                value = (
                    usage.get("cost")
                    if isinstance(usage, dict)
                    else None
                )
                if value is None:
                    value = metadata.get("estimated_cost_usd")
                try:
                    total += float(value or 0.0)
                except (TypeError, ValueError):
                    continue
    return total


class MockProvider(Provider):
    def __init__(self, model_spec: ModelSpec, seed: int, system_prompt: str) -> None:
        super().__init__(model_spec, seed, system_prompt)
        self.call_count = 0

    def generate(
        self,
        *,
        history: Sequence[Dict[str, str]],
        prompt: str,
        image_paths: Sequence[Path],
    ) -> str:
        del image_paths
        self.call_count += 1
        prior = sum(1 for message in history if message.get("role") == "assistant")
        placements = [
            (180, 420),
            (300, 420),
            (420, 300),
            (250, 520),
            (500, 220),
            (120, 360),
            (360, 350),
            (220, 280),
            (460, 460),
            (80, 500),
        ]
        x, y = placements[(prior + self.call_count + self.seed) % len(placements)]
        text = json.dumps(
            {
                "action": "drop",
                "point": [x, y],
                "coordinate_predictions": [],
                "reasoning": f"deterministic mock action {self.call_count}",
            }
        )
        self.last_call_metadata = {
            "provider": "mock",
            "call_count": self.call_count,
            "prompt_sha256": sha256_text(prompt),
        }
        return text


class OpenRouterProvider(Provider):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        model_spec: ModelSpec,
        seed: int,
        system_prompt: str,
        base_url: str,
        max_tokens: int,
        request_timeout_s: float,
        max_provider_cost_usd: Optional[float],
    ) -> None:
        if base.OpenAI is None:
            raise ImportError("The recovered Python environment does not contain the OpenAI client.")
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not visible to this process. Export it before launching Codex/this runner, "
                "or pass --api-key for a one-off smoke test."
            )
        super().__init__(model_spec, seed, system_prompt)
        self.client = base.OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=request_timeout_s,
            default_headers={
                "HTTP-Referer": "https://openai.com/codex",
                "X-Title": "VTools forked feedback baseline",
            },
        )
        self.max_tokens = max_tokens
        self.max_provider_cost_usd = max_provider_cost_usd

    def image_data_url(self, path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type(path)};base64,{encoded}"

    def _request_with_retries(self, kwargs: Dict[str, Any]) -> Any:
        last_exc: Optional[Exception] = None
        mutable = copy.deepcopy(kwargs)
        for attempt in range(1, 5):
            try:
                response = self.client.chat.completions.create(**mutable)
                if not getattr(response, "choices", None):
                    raise RuntimeError("provider response omitted choices")
                return response
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                status = getattr(exc, "status_code", None)
                retryable = status in {408, 409, 425, 429, 500, 502, 503, 504} or any(
                    token in message
                    for token in (
                        "timeout",
                        "timed out",
                        "connection",
                        "rate limit",
                        "temporarily unavailable",
                        "internal error",
                        "expecting value",
                        "omitted choices",
                        "data_inspection_failed",
                        "inappropriate content",
                    )
                )
                if attempt >= 4 or not retryable:
                    raise
                delay = min(45.0, 4.0 * (2 ** (attempt - 1)))
                delay += random.Random(f"{self.seed}:{attempt}:{self.model_spec.key}").random() * 2.0
                print(
                    f"[OpenRouter retry] {self.model_spec.key} attempt {attempt}/4: {type(exc).__name__}; "
                    f"waiting {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenRouter request failed without an exception")

    def generate(
        self,
        *,
        history: Sequence[Dict[str, str]],
        prompt: str,
        image_paths: Sequence[Path],
    ) -> str:
        if (
            self.max_provider_cost_usd is not None
            and self.cumulative_cost_usd >= self.max_provider_cost_usd
        ):
            raise RuntimeError(
                "Provider-cost guard reached: "
                f"${self.cumulative_cost_usd:.4f} >= ${self.max_provider_cost_usd:.4f}"
            )
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for message in history:
            role = "assistant" if message.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": str(message.get("content", ""))})
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self.image_data_url(path)},
                }
            )
        messages.append({"role": "user", "content": content})
        request_max_tokens = (
            self.max_tokens * 2
            if "Latest observable rollout JSON:" in prompt
            else self.max_tokens
        )
        kwargs: Dict[str, Any] = {
            "model": self.model_spec.model_id,
            "messages": messages,
            "max_tokens": request_max_tokens,
            "seed": self.seed,
            "response_format": {"type": "json_object"},
        }
        if self.model_spec.temperature is not None:
            kwargs["temperature"] = self.model_spec.temperature
        if self.model_spec.reasoning_effort:
            kwargs["extra_body"] = {
                "reasoning": {
                    "effort": self.model_spec.reasoning_effort,
                    "exclude": True,
                }
            }
        started = time.time()
        response = self._request_with_retries(kwargs)
        duration = time.time() - started
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = None
        if getattr(response, "usage", None) is not None:
            usage_obj = response.usage
            usage = usage_obj.model_dump(mode="json") if hasattr(usage_obj, "model_dump") else str(usage_obj)
        if isinstance(usage, dict):
            self.cumulative_cost_usd += float(usage.get("cost") or 0.0)
        response_extra = getattr(response, "model_extra", None) or {}
        self.last_call_metadata = {
            "provider": "openrouter",
            "requested_model": self.model_spec.model_id,
            "returned_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "duration_seconds": duration,
            "usage": usage,
            "response_extra": response_extra,
            "image_count": len(image_paths),
            "request_parameters": {
                "seed": kwargs["seed"],
                "max_tokens": kwargs["max_tokens"],
                "temperature": kwargs.get("temperature"),
                "reasoning": kwargs.get("extra_body"),
                "response_format": kwargs.get("response_format"),
            },
            "cumulative_cost_usd": self.cumulative_cost_usd,
        }
        return text


class GeminiDirectProvider(Provider):
    INPUT_USD_PER_MILLION = 1.0
    OUTPUT_USD_PER_MILLION = 5.0

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model_spec: ModelSpec,
        seed: int,
        system_prompt: str,
        max_tokens: int,
        request_timeout_s: float,
        max_provider_cost_usd: Optional[float],
        thinking_budget: int,
        keychain_service: Optional[str],
        keychain_account: Optional[str],
    ) -> None:
        if base.genai is None or base.gemini_types is None:
            raise ImportError(
                "The recovered Python environment does not contain google-genai."
            )
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key and keychain_service:
            command = [
                "security",
                "find-generic-password",
                "-s",
                keychain_service,
                "-w",
            ]
            if keychain_account:
                command[2:2] = ["-a", keychain_account]
            try:
                resolved_key = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                raise ValueError(
                    "Gemini API key is unavailable. Set GEMINI_API_KEY, pass a "
                    "credential file, or store the key in the configured macOS "
                    "Keychain service."
                ) from exc
        if not resolved_key:
            raise ValueError(
                "Gemini API key is unavailable. Set GEMINI_API_KEY, pass a "
                "credential file, or configure --gemini-keychain-service."
            )
        super().__init__(model_spec, seed, system_prompt)
        self.client = base.genai.Client(
            api_key=resolved_key,
            http_options=base.gemini_types.HttpOptions(
                timeout=int(request_timeout_s * 1000)
            ),
        )
        self.max_tokens = max_tokens
        self.max_provider_cost_usd = max_provider_cost_usd
        self.thinking_budget = thinking_budget

    @staticmethod
    def _model_dump(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        if hasattr(value, "to_json_dict"):
            return value.to_json_dict()
        return str(value)

    def _request_with_retries(self, *, contents: Any, config: Any) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                response = self.client.models.generate_content(
                    model=self.model_spec.model_id,
                    contents=contents,
                    config=config,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Gemini response omitted text")
                return response
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                status = getattr(exc, "status_code", None)
                retryable = status in {408, 409, 425, 429, 500, 502, 503, 504} or any(
                    token in message
                    for token in (
                        "timeout",
                        "timed out",
                        "connection",
                        "rate limit",
                        "temporarily unavailable",
                        "internal error",
                        "resource exhausted",
                        "omitted text",
                    )
                )
                if attempt >= 4 or not retryable:
                    raise
                delay = min(45.0, 4.0 * (2 ** (attempt - 1)))
                delay += (
                    random.Random(
                        f"gemini:{self.seed}:{attempt}:{self.model_spec.key}"
                    ).random()
                    * 2.0
                )
                print(
                    f"[Gemini retry] {self.model_spec.key} attempt {attempt}/4: "
                    f"{type(exc).__name__}; waiting {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini request failed without an exception")

    def generate(
        self,
        *,
        history: Sequence[Dict[str, str]],
        prompt: str,
        image_paths: Sequence[Path],
    ) -> str:
        if (
            self.max_provider_cost_usd is not None
            and self.cumulative_cost_usd >= self.max_provider_cost_usd
        ):
            raise RuntimeError(
                "Provider-cost guard reached: "
                f"${self.cumulative_cost_usd:.4f} >= "
                f"${self.max_provider_cost_usd:.4f}"
            )
        contents = []
        for message in history:
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append(
                base.gemini_types.Content(
                    role=role,
                    parts=[
                        base.gemini_types.Part(
                            text=str(message.get("content", ""))
                        )
                    ],
                )
            )
        parts = [base.gemini_types.Part(text=prompt)]
        for image_path in image_paths:
            parts.append(
                base.gemini_types.Part.from_bytes(
                    data=image_path.read_bytes(),
                    mime_type=mime_type(image_path),
                )
            )
        contents.append(base.gemini_types.Content(role="user", parts=parts))
        request_max_tokens = (
            self.max_tokens * 2
            if "Latest observable rollout JSON:" in prompt
            else self.max_tokens
        )
        config = base.gemini_types.GenerateContentConfig(
            temperature=self.model_spec.temperature,
            max_output_tokens=request_max_tokens,
            seed=self.seed,
            response_mime_type="application/json",
            system_instruction=self.system_prompt,
            thinking_config=base.gemini_types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=False,
            ),
        )
        started = time.time()
        response = self._request_with_retries(contents=contents, config=config)
        duration = time.time() - started
        usage_obj = getattr(response, "usage_metadata", None)
        usage = self._model_dump(usage_obj)
        prompt_tokens = int(getattr(usage_obj, "prompt_token_count", 0) or 0)
        candidate_tokens = int(
            getattr(usage_obj, "candidates_token_count", 0) or 0
        )
        thought_tokens = int(getattr(usage_obj, "thoughts_token_count", 0) or 0)
        estimated_cost = (
            prompt_tokens * self.INPUT_USD_PER_MILLION
            + (candidate_tokens + thought_tokens) * self.OUTPUT_USD_PER_MILLION
        ) / 1_000_000
        self.cumulative_cost_usd += estimated_cost
        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        self.last_call_metadata = {
            "provider": "google_gemini_api",
            "requested_model": self.model_spec.model_id,
            "returned_model": getattr(response, "model_version", None),
            "response_id": getattr(response, "response_id", None),
            "finish_reason": (
                str(getattr(candidate, "finish_reason", None))
                if candidate is not None
                else None
            ),
            "duration_seconds": duration,
            "usage": usage,
            "estimated_cost_usd": estimated_cost,
            "price_assumption_usd_per_million": {
                "input": self.INPUT_USD_PER_MILLION,
                "output_including_thoughts": self.OUTPUT_USD_PER_MILLION,
            },
            "image_count": len(image_paths),
            "request_parameters": {
                "seed": self.seed,
                "max_output_tokens": request_max_tokens,
                "temperature": self.model_spec.temperature,
                "thinking_budget": self.thinking_budget,
                "response_mime_type": "application/json",
            },
            "cumulative_cost_usd": self.cumulative_cost_usd,
        }
        return getattr(response, "text", None) or ""


def load_credential_file(path: Path, variable_name: str) -> str:
    credential_text = path.read_text(encoding="utf-8").strip()
    if "\n" in credential_text or credential_text.startswith(
        ("export ", f"{variable_name}=")
    ):
        candidates = []
        for raw_line in credential_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if line.startswith(f"{variable_name}="):
                candidates.append(line.split("=", 1)[1].strip().strip("'\""))
        if len(candidates) != 1:
            raise ValueError(
                "--api-key-file must contain either the raw key or exactly one "
                f"{variable_name} assignment"
            )
        credential = candidates[0]
    else:
        credential = credential_text.strip("'\"")
    if not credential:
        raise ValueError(f"Credential file is empty: {path}")
    return credential


def make_provider(args: argparse.Namespace, model_spec: ModelSpec) -> Provider:
    prompt_variant = str(getattr(args, "prompt_variant", "full"))
    if args.provider == "mock":
        return MockProvider(
            model_spec,
            args.seed,
            build_system_prompt(prompt_variant),
        )
    api_key = args.api_key
    if args.api_key_file:
        variable_name = (
            "GEMINI_API_KEY" if args.provider == "gemini" else "OPENROUTER_API_KEY"
        )
        api_key = load_credential_file(args.api_key_file, variable_name)
    if args.provider == "gemini":
        return GeminiDirectProvider(
            api_key=api_key,
            model_spec=model_spec,
            seed=args.seed,
            system_prompt=build_system_prompt(prompt_variant),
            max_tokens=args.max_tokens,
            request_timeout_s=args.request_timeout_s,
            max_provider_cost_usd=args.max_provider_cost_usd,
            thinking_budget=args.gemini_thinking_budget,
            keychain_service=args.gemini_keychain_service,
            keychain_account=args.gemini_keychain_account,
        )
    return OpenRouterProvider(
        api_key=api_key,
        model_spec=model_spec,
        seed=args.seed,
        system_prompt=build_system_prompt(prompt_variant),
        base_url=args.openrouter_base_url,
        max_tokens=args.max_tokens,
        request_timeout_s=args.request_timeout_s,
        max_provider_cost_usd=args.max_provider_cost_usd,
    )


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def request_valid_action(
    *,
    provider: Provider,
    history: List[Dict[str, str]],
    prompt: str,
    image_paths: Sequence[Path],
    call_log_path: Path,
    call_label: str,
    max_format_repairs: int,
    prediction_targets: Sequence[Dict[str, str]],
    history_prompt: Optional[str] = None,
) -> Tuple[Tuple[int, int], List[Dict[str, str]], List[Dict[str, Any]]]:
    updated = copy.deepcopy(history)
    records: List[Dict[str, Any]] = []
    current_prompt = prompt
    current_images = list(image_paths)
    for repair_index in range(max_format_repairs + 1):
        started = time.time()
        response_text = provider.generate(
            history=updated,
            prompt=current_prompt,
            image_paths=current_images,
        )
        record = {
            "timestamp_unix": time.time(),
            "call_label": call_label,
            "format_repair_index": repair_index,
            "prompt": current_prompt,
            "prompt_sha256": sha256_text(current_prompt),
            "image_paths": [str(path) for path in current_images],
            "image_sha256": [sha256_file(path) for path in current_images],
            "response_text": response_text,
            "response_sha256": sha256_text(response_text),
            "parsed_action_payload": parse_action_payload(response_text),
            "provider_metadata": copy.deepcopy(provider.last_call_metadata),
            "wall_seconds": time.time() - started,
        }
        append_jsonl(call_log_path, record)
        records.append(record)
        persisted_prompt = (
            history_prompt
            if repair_index == 0 and history_prompt is not None
            else current_prompt
        )
        updated.append({"role": "user", "content": persisted_prompt})
        updated.append({"role": "assistant", "content": response_text})
        coords = parse_action(response_text)
        if coords is not None:
            return coords, updated, records
        current_prompt = build_invalid_response_prompt(prediction_targets)
        current_images = [Path(path) for path in image_paths[:1]]
    raise RuntimeError(f"Model did not return a valid placement after {max_format_repairs} repairs")


def copy_unit_assets(goal: Dict[str, Any], unit_dir: Path) -> Tuple[Path, Path]:
    assets_dir = unit_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    source_world = Path(goal["environment_json"])
    source_screenshot = Path(goal["initial_screenshot"])
    world_path = assets_dir / "simulation_world.json"
    screenshot_path = assets_dir / "initial_observation.png"
    if not world_path.exists():
        shutil.copy2(source_world, world_path)
    if not screenshot_path.exists():
        shutil.copy2(source_screenshot, screenshot_path)
    if sha256_file(world_path) != goal["asset_world_sha256"]:
        raise RuntimeError(f"World asset hash mismatch for {goal['puzzle_key']}")
    if sha256_file(screenshot_path) != goal["asset_screenshot_sha256"]:
        raise RuntimeError(f"Screenshot asset hash mismatch for {goal['puzzle_key']}")
    return world_path, screenshot_path


def cleanup_rendered_rollout_frames(unit_dir: Path) -> Dict[str, Any]:
    removed = 0
    removed_bytes = 0
    for frame_path in unit_dir.rglob("rollout_frames/*.png"):
        try:
            removed_bytes += frame_path.stat().st_size
            frame_path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    record = {
        "rendered_pngs_retained": False,
        "removed_frame_count": removed,
        "removed_bytes": removed_bytes,
        "regeneration_source": (
            "retained structured truth trace plus recorded sample indices "
            "for canonical transfer sources; otherwise the released "
            "simulation config, action, and deterministic trace renderer"
        ),
    }
    atomic_write_json(unit_dir / "frame_cleanup.json", record)
    return record


def cleanup_truth_traces(unit_dir: Path) -> Dict[str, Any]:
    removed = 0
    removed_bytes = 0
    for trace_path in unit_dir.rglob("truth_trace/*.json"):
        try:
            removed_bytes += trace_path.stat().st_size
            trace_path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    record = {
        "full_truth_traces_retained": False,
        "removed_trace_count": removed,
        "removed_bytes": removed_bytes,
        "retained_prediction_truth": (
            "truth.prediction_endpoints in every accepted attempt record"
        ),
        "regeneration_source": (
            "simulation_world.json, action coordinates, gravity, seed, and "
            "the released deterministic simulator/evaluator"
        ),
    }
    atomic_write_json(unit_dir / "trace_cleanup.json", record)
    return record


def compact_prediction_endpoints(
    trace_path: Path,
    prediction_targets: Sequence[Dict[str, str]],
) -> Dict[str, Any] | None:
    if not trace_path.exists():
        return None
    trace = load_json(trace_path)
    target_shapes = {
        str(target["id"]): str(target.get("shape") or "")
        for target in prediction_targets
    }
    objects = {}
    for object_id, spec in (trace.get("objects") or {}).items():
        if not bool((spec or {}).get("is_dynamic")):
            continue
        samples = (trace.get("pose_samples") or {}).get(object_id) or []
        if not samples:
            continue
        final = samples[-1]
        objects[str(object_id)] = {
            "id": str(object_id),
            "label": str((spec or {}).get("label") or object_id),
            "role": str((spec or {}).get("role") or ""),
            "shape": target_shapes.get(str(object_id), ""),
            "total_displacement_px": float(
                (spec or {}).get("total_displacement_px") or 0.0
            ),
            "final_x": float(final["x"]),
            "final_y": float(final["y"]),
            "final_angle_rad": float(final.get("angle_rad") or 0.0),
        }
    return {
        "schema_version": 1,
        "source_trace_sha256": sha256_file(trace_path),
        "objects": objects,
    }


def render_structured_trace_frames(
    *,
    goal: Dict[str, Any],
    goal_bank: Any,
    coords: Sequence[int],
    trace_path: Path,
    out_dir: Path,
    frame_count: int,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Render uniformly sampled states from the exact rollout used for scoring.

    The historical runner simulated once for semantic scoring and a second time
    for a real-time MP4.  Replaying the scored trace avoids both that duplicate
    simulation and the renderer's real-time throttle.  It also guarantees that
    every feedback image depicts the same trajectory that produced the success
    label and event graph.
    """
    trace = load_json(trace_path)
    pose_samples = trace.get("pose_samples") or {}
    nonempty = {
        str(name): list(samples)
        for name, samples in pose_samples.items()
        if isinstance(samples, list) and samples
    }
    if not nonempty:
        raise RuntimeError(f"Structured trace has no pose samples: {trace_path}")
    sample_total = max(len(samples) for samples in nonempty.values())
    indices = [
        int(round(i * (sample_total - 1) / max(frame_count - 1, 1)))
        for i in range(frame_count)
    ]

    ctx = goal_bank.context_for(goal)
    pred = ctx["pred"]
    world_obj = pred.loadFromDict(ctx["world_data"]["world"]).copy()
    valid = pred.place_ball_tool(
        world_obj,
        (float(coords[0]), float(coords[1])),
        float(goal_bank.goal_builder.TOOL_RADIUS_PX),
        str(goal["condition"]),
        "orange",
    )
    if not valid:
        raise RuntimeError(
            f"Trace replay placement unexpectedly overlaps: {goal['balanced_goal_id']} {list(coords)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    sampled_times: List[float] = []
    try:
        pred.pg.init()
        for output_index, trace_index in enumerate(indices, start=1):
            frame_time: Optional[float] = None
            for object_name, samples in nonempty.items():
                obj = world_obj.objects.get(object_name)
                if obj is None or obj.isStatic():
                    continue
                sample = samples[min(trace_index, len(samples) - 1)]
                obj.setPos((float(sample["x"]), float(sample["y"])))
                obj.setRot(float(sample.get("angle_rad") or 0.0))
                if frame_time is None:
                    frame_time = float(sample.get("time_s") or 0.0)
            surface = pred.drawWorld(world_obj)
            pixels = pred.pg.surfarray.array3d(surface).swapaxes(0, 1)
            out_path = out_dir / f"frame_{output_index:02d}.png"
            pred.imageio.imwrite(str(out_path), pixels)
            paths.append(out_path.resolve())
            sampled_times.append(float(frame_time or 0.0))
    finally:
        pred.pg.quit()

    metadata = {
        "method": "structured_trace_replay",
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha256_file(trace_path),
        "source_pose_sample_count": sample_total,
        "sample_indices": indices,
        "sample_times_s": sampled_times,
        "frame_count": len(paths),
        "frame_dimensions_px": [SIM_WIDTH, SIM_HEIGHT],
    }
    return paths, metadata


def sanitized_initial_scene(goal: Dict[str, Any], goal_bank: Any) -> Dict[str, Any]:
    ctx = goal_bank.context_for(goal)
    world = ctx["world_data"].get("world") or {}
    default_density = float((world.get("defaults") or {}).get("density") or 1.0)
    visible = []
    for raw_id, raw_spec in (world.get("objects") or {}).items():
        if str(raw_id).startswith("_"):
            continue
        spec = raw_spec or {}
        geometry = {"type": spec.get("type")}
        for key in (
            "position",
            "radius",
            "points",
            "vertices",
            "polygons",
            "polylist",
            "polys",
            "width",
        ):
            if key in spec:
                geometry[key] = spec[key]
        visible.append(
            {
                "id": str(raw_id),
                "label": str(ctx["label_map"].get(raw_id) or raw_id),
                "color": str(spec.get("color") or spec.get("innerColor") or "black"),
                "movable": world_object_is_dynamic(
                    spec, default_density=default_density
                ),
                "geometry": geometry,
            }
        )
    dims = world.get("dims") or [SIM_WIDTH, SIM_HEIGHT]
    return {
        "canvas": {
            "width": int(dims[0]),
            "height": int(dims[1]),
            "coordinate_origin": "bottom-left",
            "x_direction": "right",
            "y_direction": "up",
            "valid_integer_coordinate_range": [0, 599],
        },
        "visible_objects": visible,
        "tool": {
            "id": "PLACED",
            "label": "orange ball tool",
            "shape": "Ball",
            "diameter_px": TOOL_DIAMETER_PX,
        },
        "note": "Observable initial geometry only; no hidden physical parameters.",
    }


def serialize_observable_trace(
    trace_path: Path,
    *,
    state_count: int,
) -> Dict[str, Any]:
    trace = load_json(trace_path)
    pose_samples = trace.get("pose_samples") or {}
    objects = trace.get("objects") or {}
    nonempty = {
        str(name): list(samples)
        for name, samples in pose_samples.items()
        if isinstance(samples, list)
        and samples
        and bool((objects.get(name) or {}).get("is_dynamic"))
    }
    if not nonempty:
        raise RuntimeError(f"Structured trace has no dynamic pose samples: {trace_path}")
    sample_total = max(len(samples) for samples in nonempty.values())
    indices = [
        int(round(i * (sample_total - 1) / max(state_count - 1, 1)))
        for i in range(state_count)
    ]
    states = []
    for state_index, trace_index in enumerate(indices):
        state_objects = []
        state_time = 0.0
        for raw_id in sorted(nonempty):
            samples = nonempty[raw_id]
            sample = samples[min(trace_index, len(samples) - 1)]
            state_time = max(state_time, float(sample.get("time_s") or 0.0))
            state_objects.append(
                {
                    "id": raw_id,
                    "label": str((objects.get(raw_id) or {}).get("label") or raw_id),
                    "center": [
                        round(float(sample["x"]), 3),
                        round(float(sample["y"]), 3),
                    ],
                    "orientation_deg": round(
                        math.degrees(float(sample.get("angle_rad") or 0.0)), 3
                    ),
                }
            )
        states.append(
            {
                "state_index": state_index,
                "time_s": round(state_time, 3),
                "objects": state_objects,
            }
        )
    return {
        "schema_version": 1,
        "sampling": "uniform_over_native_trace",
        "source_pose_sample_count": sample_total,
        "sample_indices": indices,
        "state_count": len(states),
        "states": states,
    }


def evaluate_goal_with_primary_semantics(
    *,
    goal_bank: Any,
    goal: Dict[str, Any],
    coords: Sequence[int],
    trace_dir: Path,
    terminal_persistence_s: float,
) -> Dict[str, Any]:
    """Evaluate with compatibility for the recovered pre-persistence runner."""
    try:
        return goal_bank.evaluate_goal_at(
            goal,
            coords,
            trace_dir=trace_dir,
            terminal_persistence_s=terminal_persistence_s,
        )
    except TypeError as error:
        if "terminal_persistence_s" not in str(error):
            raise
        if terminal_persistence_s <= 0.0:
            return goal_bank.evaluate_goal_at(
                goal,
                coords,
                trace_dir=trace_dir,
            )

    # The cloud VM's recovered GoalBank predates the persistence keyword. Its
    # loaded goal-builder is current, so reproduce that small adapter only for
    # canonical goals. Submitted goals stay on the historical path above.
    ctx = goal_bank.context_for(goal)
    trace_dir.mkdir(parents=True, exist_ok=True)
    save_trace_path = (
        trace_dir / f"trace_{int(coords[0])}_{int(coords[1])}.json"
    )
    simulation_kwargs = {
        "pred": ctx["pred"],
        "world_data": ctx["world_data"],
        "label_map": ctx["label_map"],
        "role_map": ctx["role_map"],
        "dynamic_map": ctx["dynamic_map"],
        "condition": str(goal["condition"]),
        "coords": coords,
        "movement_threshold_px": float(
            goal_bank.goal_builder.CENTER_RELATIVE_POSITION_THRESHOLD_PX
        ),
        "rotation_threshold_deg": float(
            goal_bank.goal_builder.ROTATION_FIRST_DIRECTION_EPS_DEG * 6.0
        ),
        "contact_min_duration_s": float(
            goal_bank.goal_builder.CONTAINER_EVENT_MIN_DURATION_S
        ),
        "include_tool_events": False,
        "save_trace_path": save_trace_path,
    }
    try:
        row = goal_bank.goal_builder.simulate_valid_placement(
            **simulation_kwargs,
            terminal_persistence_s=terminal_persistence_s,
            persistence_candidate_signatures=[str(goal["signature"])],
        )
    except TypeError as error:
        if "terminal_persistence_s" not in str(error):
            raise
        # Some recovered paper-era simulator modules predate the persistence
        # keyword. They still save the complete trace and in-goal intervals.
        # simulate_attempt applies the saved canonical gcond duration directly
        # to those intervals immediately below, so no evaluator semantics are
        # lost by using this compatibility call.
        row = goal_bank.goal_builder.simulate_valid_placement(
            **simulation_kwargs
        )
    signatures = (
        {
            goal_bank.goal_builder.event_signature(event)
            for event in row.get("event_graph", [])
        }
        if row.get("valid")
        else set()
    )
    matched = str(goal["signature"]) in signatures
    return {
        "valid": bool(row.get("valid")),
        "answer": "yes" if matched else "no",
        "matched_signature": matched,
        "event_graph": row.get("event_graph", []),
        "placement_row": row,
    }


def simulate_attempt(
    *,
    goal: Dict[str, Any],
    goal_bank: Any,
    world_path: Path,
    coords: Sequence[int],
    attempt_dir: Path,
    attempt_number: int,
    frame_count: int,
    attempt_budget: int,
    frames_condition: bool,
    stop_on_success: bool,
    trace_condition: bool = False,
    trace_state_count: int = 32,
) -> Dict[str, Any]:
    checkpoint = attempt_dir / "attempt_record.json"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        record = load_json(checkpoint)
    else:
        # Preserve the submitted evaluator for the original 1,560 diverse
        # goals.  The reviewer-response addition is deliberately narrower:
        # only the 132 canonical red-ball-in-goal goals require a continuous
        # two-second dwell.  Passing 2.0 for every goal would silently replace
        # endpoint final-state predicates with a stricter persistence test.
        terminal_persistence_s = primary_terminal_persistence_s(goal)
        truth = evaluate_goal_with_primary_semantics(
            goal_bank=goal_bank,
            goal=goal,
            coords=coords,
            trace_dir=attempt_dir / "truth_trace",
            terminal_persistence_s=terminal_persistence_s,
        )
        trace_path = attempt_dir / "truth_trace" / f"trace_{int(coords[0])}_{int(coords[1])}.json"
        placement_row = truth.get("placement_row") or {}
        endpoint_final_signatures = (
            placement_row.get("endpoint_final_state_signatures") or []
        )
        endpoint_goal_succeeded = bool(truth.get("matched_signature"))
        if str(goal.get("signature") or "").startswith("final_state|"):
            endpoint_goal_succeeded = str(goal["signature"]) in set(
                str(item) for item in endpoint_final_signatures
            )
        if goal.get("source") == "canonical_world_gcond" and trace_path.exists():
            trace_payload = load_json(trace_path)
            canonical_gcond = goal.get("canonical_gcond") or {}
            target_id = str(canonical_gcond.get("obj") or "Ball")
            object_intervals = (
                trace_payload.get("object_in_goal_intervals") or {}
            )
            intervals = (
                object_intervals.get(target_id)
                or trace_payload.get("goal_intervals")
                or trace_payload.get("in_goal_intervals")
                or []
            )
            dwell_s = float(canonical_gcond.get("duration") or 0.0)
            matched = any(
                isinstance(interval, (list, tuple))
                and len(interval) >= 2
                and float(interval[1]) - float(interval[0]) >= dwell_s - 1e-9
                for interval in intervals
            )
            truth["answer"] = "yes" if matched else "no"
            truth["matched_signature"] = matched
            truth["canonical_dwell_s"] = dwell_s
            truth["canonical_goal_intervals"] = intervals
            duration_s = float(trace_payload.get("duration_s") or 0.0)
            endpoint_goal_succeeded = any(
                isinstance(interval, (list, tuple))
                and len(interval) >= 2
                and abs(float(interval[1]) - duration_s) <= 0.05
                for interval in intervals
            )
        truth_summary: Dict[str, Any] = {
            "valid": bool(truth.get("valid")),
            "answer": str(truth.get("answer")),
            "matched_signature": bool(truth.get("matched_signature")),
            "endpoint_goal_succeeded_original_definition": (
                endpoint_goal_succeeded
            ),
            "event_graph": truth.get("event_graph") or [],
            "canonical_dwell_s": truth.get("canonical_dwell_s"),
            "canonical_goal_intervals": truth.get("canonical_goal_intervals"),
            "terminal_persistence_s": terminal_persistence_s,
            "terminal_persistence_sample_count": (
                placement_row.get(
                    "terminal_persistence_sample_count"
                )
            ),
            "endpoint_final_state_signatures": (
                placement_row.get(
                    "endpoint_final_state_signatures"
                )
            ),
            "persistent_final_state_signatures": (
                placement_row.get(
                    "persistent_final_state_signatures"
                )
            ),
            "structured_trace": str(trace_path.resolve()) if trace_path.exists() else None,
            "prediction_endpoints": compact_prediction_endpoints(
                trace_path,
                prediction_targets_for_goal(goal, goal_bank),
            ),
        }
        record = {
            "attempt": attempt_number,
            "coords": [int(coords[0]), int(coords[1])],
            "obstruction_detected": not truth_summary["valid"],
            "goal_succeeded": truth_summary["answer"] == "yes",
            "goal_succeeded_original_endpoint_definition": truth_summary[
                "endpoint_goal_succeeded_original_definition"
            ],
            "goal_signature": goal["signature"],
            "rollout_video": None,
            "sampled_frames": [],
            "frame_count": 0,
            "rollout_render": None,
            "observable_trace": None,
            "truth": truth_summary,
        }
        atomic_write_json(checkpoint, record)

    needs_followup = attempt_number < attempt_budget and not (
        bool(record["goal_succeeded"]) and stop_on_success
    )
    should_render = (
        frames_condition
        and needs_followup
        and not bool(record["obstruction_detected"])
        and int(record.get("frame_count") or 0) != frame_count
    )
    if should_render:
        del world_path
        trace_value = record.get("truth", {}).get("structured_trace")
        if trace_value:
            trace_path = Path(trace_value)
        else:
            trace_path = attempt_dir / "truth_trace" / f"trace_{int(coords[0])}_{int(coords[1])}.json"
        if not trace_path.exists():
            raise RuntimeError(f"Structured trace is missing for {goal['balanced_goal_id']}: {trace_path}")
        frames, render_metadata = render_structured_trace_frames(
            goal=goal,
            goal_bank=goal_bank,
            coords=coords,
            trace_path=trace_path,
            out_dir=attempt_dir / "rollout_frames",
            frame_count=frame_count,
        )
        record["sampled_frames"] = [str(path.resolve()) for path in frames]
        record["frame_count"] = len(frames)
        record["rollout_render"] = render_metadata
        atomic_write_json(checkpoint, record)
    should_serialize_trace = (
        trace_condition
        and needs_followup
        and not bool(record["obstruction_detected"])
        and not record.get("observable_trace")
    )
    if should_serialize_trace:
        trace_value = record.get("truth", {}).get("structured_trace")
        if not trace_value or not Path(trace_value).exists():
            raise RuntimeError(
                f"Structured trace is missing for {goal['balanced_goal_id']}: {trace_value}"
            )
        observable_path = attempt_dir / "observable_trace_32.json"
        observable = serialize_observable_trace(
            Path(trace_value), state_count=trace_state_count
        )
        atomic_write_json(observable_path, observable)
        record["observable_trace"] = str(observable_path.resolve())
        record["observable_trace_sha256"] = sha256_file(observable_path)
        record["observable_trace_state_count"] = trace_state_count
        atomic_write_json(checkpoint, record)
    return record


def initial_branch_state(
    *,
    condition: str,
    shared: Dict[str, Any],
    attempt_budget: int,
) -> Dict[str, Any]:
    shared_attempt = copy.deepcopy(shared["attempt"])
    succeeded = bool(shared_attempt["goal_succeeded"])
    status_visible = CONDITIONS[condition]["status"]
    completed = succeeded and status_visible
    return {
        "schema_version": 2,
        "condition": condition,
        "history": copy.deepcopy(shared["history"]),
        "attempt_budget": attempt_budget,
        "attempts": [shared_attempt],
        "blocked_actions": [],
        "duplicate_actions": [],
        "provider_calls": [],
        "next_attempt": 2,
        "first_success_attempt": 1 if succeeded else None,
        "completed": completed,
        "completion_reason": "shared_attempt_success_status_visible" if completed else None,
    }


def run_shared_attempt(
    *,
    args: argparse.Namespace,
    provider: Provider,
    goal: Dict[str, Any],
    goal_bank: Any,
    unit_dir: Path,
    world_path: Path,
    screenshot_path: Path,
    condition_scope: Optional[Sequence[str]] = None,
    shared_dir_name: str = "shared_attempt_1",
) -> Dict[str, Any]:
    scoped_conditions = list(
        args.conditions if condition_scope is None else condition_scope
    )
    trace_initial = scoped_conditions == ["trace_status"]
    shared_dir = unit_dir / shared_dir_name
    checkpoint = shared_dir / "state.json"
    if checkpoint.exists():
        return load_json(checkpoint)
    shared_dir.mkdir(parents=True, exist_ok=True)
    prediction_targets = prediction_targets_for_goal(goal, goal_bank)
    scene_rules, tool_dynamics = scene_and_tool_rules(goal, goal_bank)
    history: List[Dict[str, str]] = []
    persistent_prompt = build_intro_prompt(
        goal["goal_text"],
        int(goal["attempt_budget"]),
        prediction_targets,
        representation_note=(
            "The observable initial scene is supplied below as JSON; no image is provided."
            if trace_initial
            else "The first attached image is the static initial scene."
        ),
        scene_rules=scene_rules,
        tool_dynamics=tool_dynamics,
    )
    prompt = persistent_prompt
    if trace_initial:
        prompt += "\n\nInitial observable scene JSON:\n" + json.dumps(
            sanitized_initial_scene(goal, goal_bank),
            separators=(",", ":"),
            sort_keys=True,
        )
    blocked: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    call_records: List[Dict[str, Any]] = []
    current_prompt = prompt
    max_repairs = args.max_blocked_repairs
    proposal_coords: set[Tuple[int, int]] = set()
    proposal_index = 0
    repair_count = 0
    while True:
        proposal_index += 1
        coords, history, calls = request_valid_action(
            provider=provider,
            history=history,
            prompt=current_prompt,
            image_paths=[] if trace_initial else [screenshot_path],
            call_log_path=shared_dir / "provider_calls.jsonl",
            call_label="shared_attempt_1",
            max_format_repairs=args.max_format_repairs,
            prediction_targets=prediction_targets,
            history_prompt=persistent_prompt if trace_initial else None,
        )
        call_records.extend(calls)
        if coords in proposal_coords:
            repair_count += 1
            duplicates.append(
                {
                    "coords": list(coords),
                    "candidate_index": proposal_index,
                }
            )
            if repair_count > max_repairs:
                raise RuntimeError(f"Exceeded {max_repairs} procedural repairs in shared attempt")
            current_prompt = build_duplicate_prompt(coords, prediction_targets)
            continue
        proposal_coords.add(coords)
        attempt = simulate_attempt(
            goal=goal,
            goal_bank=goal_bank,
            world_path=world_path,
            coords=coords,
            attempt_dir=shared_dir / f"candidate_{proposal_index:02d}",
            attempt_number=1,
            frame_count=args.frame_count,
            attempt_budget=int(goal["attempt_budget"]),
            frames_condition=any(CONDITIONS[name]["frames"] for name in scoped_conditions),
            stop_on_success=all(CONDITIONS[name]["status"] for name in scoped_conditions),
            trace_condition=any(CONDITIONS[name]["trace"] for name in scoped_conditions),
            trace_state_count=args.trace_state_count,
        )
        attempt["model_response"] = copy.deepcopy(
            calls[-1].get("parsed_action_payload") if calls else None
        )
        if not attempt["obstruction_detected"]:
            state = {
                "schema_version": 2,
                "initial_prompt": prompt,
                "history": history,
                "attempt": attempt,
                "blocked_actions": blocked,
                "duplicate_actions": duplicates,
                "provider_calls": call_records,
            }
            atomic_write_json(checkpoint, state)
            return state
        blocked.append(
            {
                "coords": list(coords),
                "candidate_index": proposal_index,
                "candidate_dir": str(shared_dir / f"candidate_{proposal_index:02d}"),
            }
        )
        repair_count += 1
        if repair_count > max_repairs:
            raise RuntimeError(f"Exceeded {max_repairs} procedural repairs in shared attempt")
        current_prompt = build_blocked_prompt(coords, prediction_targets)


def run_branch(
    *,
    args: argparse.Namespace,
    provider: Provider,
    condition: str,
    goal: Dict[str, Any],
    goal_bank: Any,
    unit_dir: Path,
    world_path: Path,
    screenshot_path: Path,
    shared: Dict[str, Any],
) -> Dict[str, Any]:
    branch_dir = unit_dir / "branches" / condition
    state_path = branch_dir / "state.json"
    branch_dir.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = load_json(state_path)
        state.setdefault("duplicate_actions", [])
    else:
        state = initial_branch_state(
            condition=condition,
            shared=shared,
            attempt_budget=int(goal["attempt_budget"]),
        )
        atomic_write_json(state_path, state)

    if state["completed"]:
        return state

    budget = int(state["attempt_budget"])
    prediction_targets = prediction_targets_for_goal(goal, goal_bank)
    initial_scene_json = (
        sanitized_initial_scene(goal, goal_bank)
        if CONDITIONS[condition]["trace"]
        else None
    )
    while int(state["next_attempt"]) <= budget:
        attempt_number = int(state["next_attempt"])
        previous = state["attempts"][-1]
        prompt = build_continuation_prompt(
            condition=condition,
            goal_text=goal["goal_text"],
            previous_coords=previous["coords"],
            previous_solved=bool(previous["goal_succeeded"]),
            completed_attempt=attempt_number - 1,
            attempt_budget=budget,
            frame_count=args.frame_count,
            prediction_targets=prediction_targets,
            trace_payload=(
                load_json(Path(previous["observable_trace"]))
                if CONDITIONS[condition]["trace"]
                and previous.get("observable_trace")
                else None
            ),
        )
        persistent_prompt = build_continuation_prompt(
            condition=condition,
            goal_text=goal["goal_text"],
            previous_coords=previous["coords"],
            previous_solved=bool(previous["goal_succeeded"]),
            completed_attempt=attempt_number - 1,
            attempt_budget=budget,
            frame_count=args.frame_count,
            prediction_targets=prediction_targets,
            trace_payload=None,
        )
        if initial_scene_json is not None:
            prompt += "\n\nInitial observable scene JSON:\n" + json.dumps(
                initial_scene_json, separators=(",", ":"), sort_keys=True
            )
        images: List[Path] = (
            [] if CONDITIONS[condition]["trace"] else [screenshot_path]
        )
        if CONDITIONS[condition]["frames"]:
            frames = [Path(path) for path in previous.get("sampled_frames") or []]
            if len(frames) != args.frame_count:
                raise RuntimeError(
                    f"{condition} requires {args.frame_count} latest frames, found {len(frames)} "
                    f"for {goal['balanced_goal_id']} attempt {attempt_number - 1}"
                )
            images.extend(frames)

        current_prompt = prompt
        proposal_index = 0
        repair_count = 0
        tried_coords = {tuple(int(value) for value in attempt["coords"]) for attempt in state["attempts"]}
        while True:
            proposal_index += 1
            coords, updated_history, calls = request_valid_action(
                provider=provider,
                history=state["history"],
                prompt=current_prompt,
                image_paths=(
                    images
                    if proposal_index == 1
                    else ([] if CONDITIONS[condition]["trace"] else [screenshot_path])
                ),
                call_log_path=branch_dir / "provider_calls.jsonl",
                call_label=f"{condition}_attempt_{attempt_number}",
                max_format_repairs=args.max_format_repairs,
                prediction_targets=prediction_targets,
                history_prompt=(
                    persistent_prompt if CONDITIONS[condition]["trace"] else None
                ),
            )
            state["history"] = updated_history
            state["provider_calls"].extend(calls)
            if coords in tried_coords:
                repair_count += 1
                state["duplicate_actions"].append(
                    {
                        "attempt": attempt_number,
                        "coords": list(coords),
                        "candidate_index": proposal_index,
                    }
                )
                atomic_write_json(state_path, state)
                if repair_count > args.max_blocked_repairs:
                    raise RuntimeError(
                        f"Exceeded procedural repair limit in {condition} attempt {attempt_number}"
                    )
                current_prompt = build_duplicate_prompt(coords, prediction_targets)
                continue
            attempt = simulate_attempt(
                goal=goal,
                goal_bank=goal_bank,
                world_path=world_path,
                coords=coords,
                attempt_dir=branch_dir / f"attempt_{attempt_number:02d}_candidate_{proposal_index:02d}",
                attempt_number=attempt_number,
                frame_count=args.frame_count,
                attempt_budget=budget,
                frames_condition=CONDITIONS[condition]["frames"],
                stop_on_success=CONDITIONS[condition]["status"],
                trace_condition=CONDITIONS[condition]["trace"],
                trace_state_count=args.trace_state_count,
            )
            attempt["model_response"] = copy.deepcopy(
                calls[-1].get("parsed_action_payload") if calls else None
            )
            if not attempt["obstruction_detected"]:
                break
            repair_count += 1
            state["blocked_actions"].append(
                {
                    "attempt": attempt_number,
                    "coords": list(coords),
                    "candidate_index": proposal_index,
                }
            )
            atomic_write_json(state_path, state)
            if repair_count > args.max_blocked_repairs:
                raise RuntimeError(
                    f"Exceeded procedural repair limit in {condition} attempt {attempt_number}"
                )
            current_prompt = build_blocked_prompt(coords, prediction_targets)
            images = [] if CONDITIONS[condition]["trace"] else [screenshot_path]

        state["attempts"].append(attempt)
        if attempt["goal_succeeded"] and state["first_success_attempt"] is None:
            state["first_success_attempt"] = attempt_number
        state["next_attempt"] = attempt_number + 1
        if attempt["goal_succeeded"] and CONDITIONS[condition]["status"]:
            state["completed"] = True
            state["completion_reason"] = "first_success_status_visible"
        elif state["next_attempt"] > budget:
            state["completed"] = True
            state["completion_reason"] = "attempt_budget_exhausted"
        atomic_write_json(state_path, state)
        if state["completed"]:
            break
    return state


def goal_unit_dir(output_root: Path, model_key: str, seed: int, goal: Dict[str, Any]) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(goal["balanced_goal_id"]))
    return output_root / model_key / f"seed_{seed}" / f"{int(goal['benchmark_index']):04d}_{safe_id}"


def summarize_unit(
    *,
    model_spec: ModelSpec,
    seed: int,
    goal: Dict[str, Any],
    unit_dir: Path,
    shared: Dict[str, Any],
    branches: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "model_key": model_spec.key,
        "model_id": model_spec.model_id,
        "model_label": model_spec.label,
        "seed": seed,
        "goal": goal,
        "shared_attempt_1": {
            "coords": shared["attempt"]["coords"],
            "goal_succeeded": shared["attempt"]["goal_succeeded"],
            "history_sha256": sha256_text(json.dumps(shared["history"], sort_keys=True)),
        },
        "conditions": {
            name: {
                "attempt_count": len(state["attempts"]),
                "attempt_budget": state["attempt_budget"],
                "first_success_attempt": state["first_success_attempt"],
                "solved_by_budget": state["first_success_attempt"] is not None,
                "completion_reason": state["completion_reason"],
                "coords": [attempt["coords"] for attempt in state["attempts"]],
                "blocked_action_count": len(state["blocked_actions"]),
                "duplicate_action_count": len(state.get("duplicate_actions") or []),
            }
            for name, state in branches.items()
        },
        "unit_dir": str(unit_dir),
    }


def run_unit(
    *,
    args: argparse.Namespace,
    provider: Provider,
    model_spec: ModelSpec,
    goal: Dict[str, Any],
    goal_bank: Any,
    output_root: Path,
) -> Dict[str, Any]:
    unit_dir = goal_unit_dir(output_root, model_spec.key, args.seed, goal)
    summary_path = unit_dir / "unit_summary.json"
    if args.skip_existing and summary_path.exists():
        if not args.retain_rollout_frames:
            cleanup_rendered_rollout_frames(unit_dir)
        if (
            not args.retain_truth_traces
            and str(goal.get("source") or "") != "canonical_world_gcond"
        ):
            cleanup_truth_traces(unit_dir)
        return load_json(summary_path)
    unit_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        unit_dir / "unit_manifest.json",
        {
            "model": model_spec.__dict__,
            "seed": args.seed,
            "prompt_variant": args.prompt_variant,
            "goal": goal,
            "conditions": CONDITIONS,
            "frame_count": args.frame_count,
            "trace_state_count": args.trace_state_count,
            "static_initial_scene_resent_each_turn": True,
            "initial_scene_representation": (
                "image for full/frames_only/status_only/neither; "
                "observable JSON for trace_status"
            ),
            "older_rollout_frames_retained": False,
            "rendered_rollout_pngs_retained_after_unit": args.retain_rollout_frames,
            "full_truth_traces_retained_after_unit": (
                args.retain_truth_traces
                or str(goal.get("source") or "")
                == "canonical_world_gcond"
            ),
            "canonical_traces_temporarily_retained_for_transfer_sidecars": True,
            "all_prior_text_retained": True,
        },
    )
    world_path, screenshot_path = copy_unit_assets(goal, unit_dir)
    primary_conditions = [
        condition for condition in args.conditions if condition != "trace_status"
    ]
    shared = run_shared_attempt(
        args=args,
        provider=provider,
        goal=goal,
        goal_bank=goal_bank,
        unit_dir=unit_dir,
        world_path=world_path,
        screenshot_path=screenshot_path,
        condition_scope=primary_conditions,
    )
    branches: Dict[str, Dict[str, Any]] = {}
    if shared["attempt"]["goal_succeeded"]:
        for condition in primary_conditions:
            state_path = unit_dir / "branches" / condition / "state.json"
            if state_path.exists():
                state = load_json(state_path)
            else:
                state = initial_branch_state(
                    condition=condition,
                    shared=shared,
                    attempt_budget=int(goal["attempt_budget"]),
                )
                state["completed"] = True
                state["completion_reason"] = "shared_attempt_success_absorbed"
                atomic_write_json(state_path, state)
            branches[condition] = state
    else:
        for condition in primary_conditions:
            branches[condition] = run_branch(
                args=args,
                provider=provider,
                condition=condition,
                goal=goal,
                goal_bank=goal_bank,
                unit_dir=unit_dir,
                world_path=world_path,
                screenshot_path=screenshot_path,
                shared=shared,
            )
    if "trace_status" in args.conditions:
        trace_shared = run_shared_attempt(
            args=args,
            provider=provider,
            goal=goal,
            goal_bank=goal_bank,
            unit_dir=unit_dir,
            world_path=world_path,
            screenshot_path=screenshot_path,
            condition_scope=["trace_status"],
            shared_dir_name="branches/trace_status/independent_attempt_1",
        )
        if trace_shared["attempt"]["goal_succeeded"]:
            trace_state_path = unit_dir / "branches" / "trace_status" / "state.json"
            trace_state = initial_branch_state(
                condition="trace_status",
                shared=trace_shared,
                attempt_budget=int(goal["attempt_budget"]),
            )
            trace_state["completed"] = True
            trace_state["completion_reason"] = "independent_attempt_1_success"
            atomic_write_json(trace_state_path, trace_state)
            branches["trace_status"] = trace_state
        else:
            branches["trace_status"] = run_branch(
                args=args,
                provider=provider,
                condition="trace_status",
                goal=goal,
                goal_bank=goal_bank,
                unit_dir=unit_dir,
                world_path=world_path,
                screenshot_path=screenshot_path,
                shared=trace_shared,
            )
    summary = summarize_unit(
        model_spec=model_spec,
        seed=args.seed,
        goal=goal,
        unit_dir=unit_dir,
        shared=shared,
        branches=branches,
    )
    atomic_write_json(summary_path, summary)
    if not args.retain_rollout_frames:
        summary["frame_cleanup"] = cleanup_rendered_rollout_frames(unit_dir)
    if (
        not args.retain_truth_traces
        and str(goal.get("source") or "") != "canonical_world_gcond"
    ):
        summary["trace_cleanup"] = cleanup_truth_traces(unit_dir)
    atomic_write_json(summary_path, summary)
    return summary


def select_goals(goals: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = list(goals)
    if args.goal_index is not None:
        selected = [selected[args.goal_index]]
    else:
        start = max(0, args.start_index)
        selected = selected[start:]
        if args.limit is not None:
            selected = selected[: args.limit]
    if args.gravity_scope != "all":
        selected = [
            goal
            for goal in selected
            if str(goal.get("condition") or "") == args.gravity_scope
        ]
    if args.shard_count > 1:
        selected = [
            goal
            for goal in selected
            if int(goal["benchmark_index"]) % args.shard_count == args.shard_index
        ]
    return selected


def call_scope(goals: Sequence[Dict[str, Any]], condition_names: Sequence[str]) -> Dict[str, int]:
    maximum = 0
    no_status_forced_if_attempt1_fails = 0
    trace_independent = "trace_status" in condition_names
    primary_names = [name for name in condition_names if name != "trace_status"]
    for goal in goals:
        budget = int(goal["attempt_budget"])
        maximum += (
            1
            + len(primary_names) * (budget - 1)
            + (budget if trace_independent else 0)
        )
        hidden_count = sum(
            1 for name in primary_names if not CONDITIONS[name]["status"]
        )
        no_status_forced_if_attempt1_fails += (
            1
            + hidden_count * (budget - 1)
            + (1 if trace_independent else 0)
        )
    return {
        "goal_units": len(goals),
        "absolute_minimum_calls_all_attempt1_succeed": len(goals)
        * (2 if trace_independent else 1),
        "calls_if_only_status_hidden_branches_continue_after_attempt1_failures": no_status_forced_if_attempt1_fails,
        "maximum_action_calls": maximum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("openrouter", "gemini", "mock"),
        default="openrouter",
    )
    parser.add_argument("--api-key", default=None, help="Prefer OPENROUTER_API_KEY; this value is never logged.")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Local file containing a raw key or OPENROUTER_API_KEY assignment; the value is never logged.",
    )
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--gemini-keychain-service",
        default="codex-gemini-api-key",
        help="macOS Keychain service used when GEMINI_API_KEY is unset.",
    )
    parser.add_argument(
        "--gemini-keychain-account",
        default=None,
        help=(
            "Optional macOS Keychain account paired with "
            "--gemini-keychain-service."
        ),
    )
    parser.add_argument(
        "--gemini-thinking-budget",
        type=int,
        default=4096,
        help="Fixed hidden-thinking token budget for direct Gemini calls.",
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="gpt5")
    parser.add_argument("--model-id", default=None, help="Explicit override; keep defaults for paper-comparable runs.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional override.")
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"), default=None)
    parser.add_argument(
        "--prompt-variant",
        choices=("full", "compact"),
        default="full",
        help=(
            "Full deterministic-2D-rigid-body framing or a compact role line. "
            "Task rules, coordinates, dynamics, context policy, and response "
            "schema remain identical."
        ),
    )
    parser.add_argument("--seed", type=int, choices=DEFAULT_SEEDS, default=42)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--trace-state-count", type=int, default=32)
    parser.add_argument("--budget-mode", choices=("manifest", "fixed"), default="fixed")
    parser.add_argument("--fixed-budget", type=int, default=8)
    parser.add_argument("--goal-index", type=int, default=None, help="Zero-based index in the selected manifest.")
    parser.add_argument(
        "--gravity-scope",
        choices=("all", "upward", "downward"),
        default="all",
        help="Run only one tool-gravity direction while retaining original benchmark indices.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_BALANCED_MANIFEST)
    parser.add_argument("--asset-index", type=Path, default=DEFAULT_ASSET_INDEX)
    parser.add_argument("--rebuild-asset-index", action="store_true")
    parser.add_argument("--asset-root", type=Path, action="append", default=None)
    parser.add_argument("--goal-builder", type=Path, default=DEFAULT_GOAL_BUILDER)
    parser.add_argument("--vtools-root", type=Path, default=DEFAULT_VTOOLS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON_BIN))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-provider-cost-usd", type=float, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-format-repairs", type=int, default=1)
    parser.add_argument("--max-blocked-repairs", type=int, default=12)
    parser.add_argument(
        "--retain-rollout-frames",
        action="store_true",
        help="Keep rendered PNGs after a unit completes; default is trace-only retention.",
    )
    parser.add_argument(
        "--retain-truth-traces",
        action="store_true",
        help=(
            "Keep full native truth traces after a unit. By default they are "
            "compacted and removed for noncanonical units; canonical source "
            "traces remain temporarily for transfer sidecars."
        ),
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.model_id or args.temperature is not None or args.reasoning_effort is not None:
        original = MODEL_SPECS[args.model]
        MODEL_SPECS[args.model] = ModelSpec(
            key=original.key,
            model_id=args.model_id or original.model_id,
            temperature=args.temperature if args.temperature is not None else original.temperature,
            reasoning_effort=args.reasoning_effort or original.reasoning_effort,
            label=original.label,
        )
    if not 1 <= args.fixed_budget <= 8:
        parser.error("--fixed-budget must be between 1 and 8")
    if args.api_key and args.api_key_file:
        parser.error("Use only one of --api-key or --api-key-file")
    if args.provider == "gemini" and args.model != "gemini_robotics_er":
        parser.error(
            "The direct Gemini provider is reserved for --model "
            "gemini_robotics_er in the reviewer-response run."
        )
    if args.provider != "gemini" and args.model == "gemini_robotics_er":
        parser.error(
            "Gemini Robotics-ER is not available through OpenRouter; use "
            "--provider gemini."
        )
    if args.gemini_thinking_budget < 0:
        parser.error("--gemini-thinking-budget must be non-negative")
    if args.frame_count != 32:
        parser.error("The revised visual protocol requires --frame-count 32")
    if args.trace_state_count != 32:
        parser.error("The symbolic control requires --trace-state-count 32")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    if args.goal_index is not None and args.goal_index < 0:
        parser.error("--goal-index must be non-negative")
    args.manifest = args.manifest.expanduser().resolve()
    if args.api_key_file:
        args.api_key_file = args.api_key_file.expanduser().resolve()
    args.asset_index = args.asset_index.expanduser().resolve()
    args.goal_builder = args.goal_builder.expanduser().resolve()
    args.vtools_root = args.vtools_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.conditions = list(dict.fromkeys(args.conditions))
    return args


def main() -> None:
    args = parse_args()
    raw_manifest = load_json(args.manifest)
    raw_goals = raw_manifest.get("goals") or []
    asset_roots = tuple(path.expanduser().resolve() for path in (args.asset_root or DEFAULT_ASSET_ROOTS))
    asset_index = load_or_build_asset_index(
        raw_goals,
        asset_index_path=args.asset_index,
        asset_roots=asset_roots,
        rebuild=args.rebuild_asset_index,
    )
    goals = load_balanced_goals(
        args.manifest,
        asset_index=asset_index,
        budget_mode=args.budget_mode,
        fixed_budget=args.fixed_budget,
    )
    validate_balanced_goals(goals)
    selected = select_goals(goals, args)
    model_spec = MODEL_SPECS[args.model]
    scope = call_scope(selected, args.conditions)
    run_manifest = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "provider": args.provider,
        "model": model_spec.__dict__,
        "seed": args.seed,
        "prompt_variant": args.prompt_variant,
        "conditions": args.conditions,
        "gravity_scope": args.gravity_scope,
        "condition_definitions": CONDITIONS,
        "frame_count": args.frame_count,
        "trace_state_count": args.trace_state_count,
        "base_max_tokens": args.max_tokens,
        "json_trace_continuation_max_tokens": args.max_tokens * 2,
        "budget_mode": args.budget_mode,
        "endpoint_definition": (
            "canonical: the saved world gcond uninterrupted in-goal dwell "
            "(2 seconds in the released expanded manifest); "
            "original final_state goals: submitted endpoint predicate; "
            "during-rollout and transient goals: submitted event semantics"
        ),
        "selected_goal_count": len(selected),
        "selected_goal_indices": [goal["benchmark_index"] for goal in selected],
        "scope": scope,
        "manifest": str(args.manifest),
        "asset_index": str(args.asset_index),
        "vtools_root": str(args.vtools_root),
        "output_root": str(args.output_root),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    console_manifest = copy.deepcopy(run_manifest)
    selected_indices = list(run_manifest["selected_goal_indices"])
    if len(selected_indices) > 20:
        console_manifest["selected_goal_indices"] = {
            "count": len(selected_indices),
            "first_10": selected_indices[:10],
            "last_10": selected_indices[-10:],
            "sha256": sha256_text(json.dumps(selected_indices, separators=(",", ":"))),
        }
    print(json.dumps(console_manifest, indent=2), flush=True)
    if args.dry_run:
        return
    scope_suffix = "" if args.gravity_scope == "all" else f"_{args.gravity_scope}"
    run_manifest_path = (
        args.output_root
        / "_run_manifests"
        / (
            f"{model_spec.key}_seed{args.seed}{scope_suffix}_"
            f"shard{args.shard_index:03d}of{args.shard_count:03d}.json"
        )
    )
    atomic_write_json(run_manifest_path, run_manifest)
    provider = make_provider(args, model_spec)
    prior_call_logs: List[Path] = []
    for goal in selected:
        unit_dir = goal_unit_dir(
            args.output_root,
            model_spec.key,
            args.seed,
            goal,
        )
        prior_call_logs.extend(
            path
            for path in unit_dir.rglob("provider_calls.jsonl")
            if "transfer_sidecars" not in path.parts
        )
    provider.cumulative_cost_usd = provider_cost_from_call_logs(
        prior_call_logs
    )
    if provider.cumulative_cost_usd:
        print(
            "Recovered prior provider cost for this restart/shard: "
            f"${provider.cumulative_cost_usd:.4f}",
            flush=True,
        )
    goal_bank = base.GoalBank(REPO_ROOT / "task_configs", args.goal_builder)
    summaries: List[Dict[str, Any]] = []
    unit_errors: List[Dict[str, str]] = []
    for ordinal, goal in enumerate(selected, start=1):
        print(
            f"[{ordinal}/{len(selected)}] {model_spec.key} seed={args.seed} "
            f"goal={goal['benchmark_index']:04d} {goal['balanced_goal_id']}",
            flush=True,
        )
        try:
            summaries.append(
                run_unit(
                    args=args,
                    provider=provider,
                    model_spec=model_spec,
                    goal=goal,
                    goal_bank=goal_bank,
                    output_root=args.output_root,
                )
            )
        except Exception as exc:
            error = {
                "goal_id": str(goal["balanced_goal_id"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            unit_errors.append(error)
            print(
                f"[unit error] {error['goal_id']}: {error['error']}",
                flush=True,
            )
    shard_summary = {
        **run_manifest,
        "completed_unit_count": len(summaries),
        "failed_unit_count": len(unit_errors),
        "unit_errors": unit_errors,
        "unit_summary_paths": [str(Path(item["unit_dir"]) / "unit_summary.json") for item in summaries],
    }
    atomic_write_json(run_manifest_path.with_name(run_manifest_path.stem + "_summary.json"), shard_summary)


if __name__ == "__main__":
    main()
