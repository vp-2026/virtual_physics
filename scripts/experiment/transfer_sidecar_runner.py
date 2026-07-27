#!/usr/bin/env python3
"""Run isolated early, terminal, and same-scene goal-transfer sidecars.

This program consumes completed main-run units.  Sidecar responses never enter
the solver history.  Probe coordinates are selected jointly before feedback,
using only scene geometry, simulator rollout features, and the two independent
attempt-1 coordinates (visual and JSON); selection never uses goal success or
model prediction accuracy.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNNER_PATH = HERE / "forked_feedback_runner.py"
DEFAULT_MANIFEST = REPO_ROOT / "task_configs" / "benchmark_1692_seed2026.json"
DEFAULT_PROBE_CACHE = REPO_ROOT / "outputs" / "probe_bank_seed2026"
DEFAULT_GOAL_BUILDER = (
    REPO_ROOT
    / "simulator"
    / "goal_semantics"
    / "build_goal_bank_from_placement_sweep.py"
)
PRIMARY_CONDITIONS = ("full", "frames_only", "status_only", "neither")
ALL_CONDITIONS = (*PRIMARY_CONDITIONS, "trace_status")
GRID_STEP_PX = 30
GRID_OFFSET_PX = 42
PROBE_BANK_SIZE = 32
MIN_PROBE_ACTION_DISTANCE_PX = 60.0
MIN_PROBE_PAIR_DISTANCE_PX = 60.0
CONTAMINATION_DISTANCE_PX = 40.0
PREDICTION_SIGMA_PX = 150.0


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module(RUNNER_PATH, "vtools_transfer_base_runner")


def stable_hash(value: str, seed: int = 2026) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json_object(response_text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    raw = response_text.strip()
    first = raw.find("{")
    last = raw.rfind("}")
    candidates = []
    if first >= 0:
        candidates.append(raw[first:])
    if first >= 0 and last > first:
        candidates.append(raw[first : last + 1])
    for candidate in candidates:
        for variant, normalization in (
            (candidate, None),
            (candidate + "}", "appended_one_terminal_object_brace"),
            (
                candidate[:-1].rstrip() if candidate.rstrip().endswith("}") else "",
                "removed_one_extra_terminal_object_brace",
            ),
        ):
            if not variant:
                continue
            try:
                payload = json.loads(variant)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload, normalization
    return None, None


def validate_predictions(
    payload: Optional[dict[str, Any]],
    *,
    targets: Sequence[dict[str, str]],
    required_placement: Sequence[int],
) -> dict[str, Any]:
    required_ids = [str(target["id"]) for target in targets]
    required_set = set(required_ids)
    if not isinstance(payload, dict):
        return {
            "schema_valid": False,
            "payload": payload,
            "predictions": [],
            "missing_ids": required_ids,
            "duplicate_ids": [],
            "unexpected_ids": [],
            "placement_matches": False,
        }
    placement = payload.get("placement")
    placement_matches = (
        isinstance(placement, list)
        and len(placement) == 2
        and all(isinstance(value, (int, float)) for value in placement)
        and [int(round(float(value))) for value in placement]
        == [int(required_placement[0]), int(required_placement[1])]
    )
    predictions = payload.get("coordinate_predictions")
    if not isinstance(predictions, list):
        predictions = []
    ids = [
        str(item.get("id"))
        for item in predictions
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    unexpected_ids = sorted(set(ids) - required_set)
    missing_ids = [item for item in required_ids if item not in ids]
    entries_valid = True
    for item in predictions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            entries_valid = False
            continue
        state = str(item.get("state") or "")
        if state == "in_scene":
            point = item.get("point")
            if not (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, (int, float)) for value in point)
                and all(0 <= float(value) <= 599 for value in point)
            ):
                entries_valid = False
        elif state == "exited":
            if str(item.get("exit_side") or "") not in {
                "left",
                "right",
                "top",
                "bottom",
            }:
                entries_valid = False
        else:
            entries_valid = False
    return {
        "schema_valid": bool(
            placement_matches
            and not missing_ids
            and not duplicate_ids
            and not unexpected_ids
            and entries_valid
            and len(predictions) == len(required_ids)
        ),
        "payload": payload,
        "predictions": predictions,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "unexpected_ids": unexpected_ids,
        "placement_matches": placement_matches,
    }


def prediction_schema_text(
    targets: Sequence[dict[str, str]], coords: Sequence[int]
) -> str:
    target_lines = "\n".join(
        f'- id "{target["id"]}": {target["label"]} ({target["shape"]})'
        for target in targets
    )
    return f"""The orange tool placement is fixed at {[int(coords[0]), int(coords[1])]}. Do not choose or change the placement.

Predict the terminal state of every listed movable object:
{target_lines}

For an object remaining in the scene, use {{"id":"...","state":"in_scene","point":[x,y],"orientation_deg":number_or_null}}.
For an object that exits, use {{"id":"...","state":"exited","exit_side":"left|right|top|bottom","point":null,"orientation_deg":null}}.
Use orientation_deg only when meaningful for a non-circular object; otherwise use null.

Return exactly one JSON object:
{{"placement":{[int(coords[0]), int(coords[1])]},"coordinate_predictions":[...exactly one entry for every id above...],"reasoning":"<brief reasoning>"}}"""


def request_prediction(
    *,
    provider: Any,
    history: Sequence[dict[str, str]],
    prompt: str,
    image_paths: Sequence[Path],
    targets: Sequence[dict[str, str]],
    coords: Sequence[int],
    call_log_path: Path,
    call_label: str,
    max_format_repairs: int,
) -> dict[str, Any]:
    records = []
    current_prompt = prompt
    current_images = list(image_paths)
    final_validation: dict[str, Any] = {}
    for repair_index in range(max_format_repairs + 1):
        started = time.time()
        response_text = provider.generate(
            history=history,
            prompt=current_prompt,
            image_paths=current_images,
        )
        payload, normalization = extract_json_object(response_text)
        validation = validate_predictions(
            payload,
            targets=targets,
            required_placement=coords,
        )
        final_validation = validation
        record = {
            "timestamp_unix": time.time(),
            "call_label": call_label,
            "format_repair_index": repair_index,
            "prompt": current_prompt,
            "prompt_sha256": runner.sha256_text(current_prompt),
            "image_paths": [str(path) for path in current_images],
            "image_sha256": [
                runner.sha256_file(path) for path in current_images
            ],
            "response_text": response_text,
            "response_sha256": runner.sha256_text(response_text),
            "normalization": normalization,
            "validation": validation,
            "provider_metadata": copy.deepcopy(provider.last_call_metadata),
            "wall_seconds": time.time() - started,
        }
        runner.append_jsonl(call_log_path, record)
        records.append(record)
        if validation["schema_valid"]:
            break
        current_prompt = (
            "Your previous response did not match the required prediction-only "
            "JSON schema. This is a format-only repair and conveys no simulator "
            "outcome.\n\n"
            + prediction_schema_text(targets, coords)
        )
        current_images = list(image_paths[:1])
    return {
        "schema_valid": bool(final_validation.get("schema_valid")),
        "parsed_payload": final_validation.get("payload"),
        "validation": final_validation,
        "provider_call_count": len(records),
        "last_provider_metadata": (
            copy.deepcopy(records[-1]["provider_metadata"])
            if records
            else None
        ),
    }


def geometry_valid(goal: dict[str, Any], goal_bank: Any, coords: Sequence[int]) -> bool:
    ctx = goal_bank.context_for(goal)
    pred = ctx["pred"]
    world_obj = pred.loadFromDict(ctx["world_data"]["world"]).copy()
    return bool(
        pred.place_ball_tool(
            world_obj,
            (float(coords[0]), float(coords[1])),
            float(goal_bank.goal_builder.TOOL_RADIUS_PX),
            str(goal["condition"]),
            "orange",
        )
    )


def farthest_point_sample(
    points: Sequence[tuple[int, int]],
    count: int,
    *,
    key: str,
    seed: int,
) -> list[tuple[int, int]]:
    remaining = list(points)
    if len(remaining) <= count:
        return sorted(remaining)
    first = min(
        remaining,
        key=lambda point: stable_hash(f"{key}|{point[0]}|{point[1]}", seed),
    )
    selected = [first]
    remaining.remove(first)
    while len(selected) < count and remaining:
        chosen = min(
            remaining,
            key=lambda point: (
                -min(math.dist(point, other) for other in selected),
                stable_hash(f"{key}|{point[0]}|{point[1]}", seed),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def exit_side_for_point(x: float, y: float) -> str:
    violations = (
        (max(0.0, -x), "left"),
        (max(0.0, x - 599.0), "right"),
        (max(0.0, -y), "bottom"),
        (max(0.0, y - 599.0), "top"),
    )
    return max(violations, key=lambda item: item[0])[1]


def endpoint_truth(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects = trace.get("objects") or {}
    output: dict[str, dict[str, Any]] = {}
    for object_id, spec in objects.items():
        if not bool((spec or {}).get("is_dynamic")):
            continue
        samples = (trace.get("pose_samples") or {}).get(object_id) or []
        if not samples:
            continue
        final = samples[-1]
        x, y = float(final["x"]), float(final["y"])
        state = "in_scene" if 0 <= x <= 599 and 0 <= y <= 599 else "exited"
        output[str(object_id)] = {
            "state": state,
            "exit_side": (
                exit_side_for_point(x, y) if state == "exited" else None
            ),
            "point": [x, y],
            "orientation_deg": math.degrees(
                float(final.get("angle_rad") or 0.0)
            ),
        }
    return output


def rollout_features(
    trace: dict[str, Any], coords: Sequence[int]
) -> dict[str, Any]:
    objects = trace.get("objects") or {}
    affected = 0
    total_displacement = 0.0
    for object_id, spec in objects.items():
        if (
            str(object_id) == "PLACED"
            or not bool((spec or {}).get("is_dynamic"))
        ):
            continue
        displacement = float((spec or {}).get("total_displacement_px") or 0.0)
        total_displacement += displacement
        if displacement > 30.0:
            affected += 1
    endpoints = endpoint_truth(trace)
    tool_exit = (endpoints.get("PLACED") or {}).get("state") == "exited"
    contact_pairs = trace.get("contact_pairs") or []
    unique_contacts = {
        tuple(sorted((str(item.get("a")), str(item.get("b")))))
        for item in contact_pairs
        if isinstance(item, dict)
    }
    tool_contact = any("PLACED" in pair for pair in unique_contacts)
    x, y = float(coords[0]), float(coords[1])
    boundary_clearance = min(x - 36.0, 564.0 - x, y - 36.0, 564.0 - y)
    return {
        "tool_exit": bool(tool_exit),
        "tool_contact": bool(tool_contact),
        "affected_object_count": affected,
        "unique_contact_count": len(unique_contacts),
        "total_non_tool_displacement_px": total_displacement,
        "settle_time_s": float(
            trace.get("settle_time_s")
            if trace.get("settle_time_s") is not None
            else trace.get("duration_s")
            or 0.0
        ),
        "boundary_clearance_px": boundary_clearance,
    }


def prepare_probe_bank(
    *,
    goal: dict[str, Any],
    goal_bank: Any,
    probe_cache: Path,
    seed: int,
) -> dict[str, Any]:
    bank_path = probe_cache / str(goal["puzzle_key"]) / "candidate_bank.json"
    if bank_path.exists():
        return read_json(bank_path)
    all_points = [
        (x, y)
        for x in range(GRID_OFFSET_PX, 565, GRID_STEP_PX)
        for y in range(GRID_OFFSET_PX, 565, GRID_STEP_PX)
    ]
    valid_points = [
        point for point in all_points if geometry_valid(goal, goal_bank, point)
    ]
    sampled = farthest_point_sample(
        valid_points,
        PROBE_BANK_SIZE,
        key=str(goal["puzzle_key"]),
        seed=seed,
    )
    candidates = []
    trace_dir = bank_path.parent / "_trace_build"
    for coords in sampled:
        result = goal_bank.evaluate_goal_at(
            goal,
            coords,
            trace_dir=trace_dir,
        )
        trace_path = trace_dir / f"trace_{coords[0]}_{coords[1]}.json"
        if not result.get("valid") or not trace_path.exists():
            continue
        trace = read_json(trace_path)
        candidates.append(
            {
                "coords": list(coords),
                "features": rollout_features(trace, coords),
                "endpoints": endpoint_truth(trace),
                "trace_sha256": runner.sha256_file(trace_path),
            }
        )
        trace_path.unlink()
    if trace_dir.exists() and not any(trace_dir.iterdir()):
        trace_dir.rmdir()
    payload = {
        "schema_version": 1,
        "puzzle_key": goal["puzzle_key"],
        "selection_seed": seed,
        "grid_step_px": GRID_STEP_PX,
        "grid_offset_px": GRID_OFFSET_PX,
        "geometrically_valid_grid_count": len(valid_points),
        "candidate_count": len(candidates),
        "candidate_selection": "deterministic spatial farthest-point sample",
        "selection_uses_goal_success": False,
        "candidates": candidates,
    }
    if len(candidates) < 2:
        raise RuntimeError(
            f"Too few valid probe candidates for {goal['puzzle_key']}"
        )
    runner.atomic_write_json(bank_path, payload)
    return payload


def standardized_pair_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    candidates: Sequence[dict[str, Any]],
    avoid_coords: Sequence[Sequence[int]],
) -> tuple[int, float]:
    left_features = left["features"]
    right_features = right["features"]
    categorical_mismatches = sum(
        left_features[name] != right_features[name]
        for name in ("tool_exit", "tool_contact", "affected_object_count")
    )
    continuous_names = (
        "unique_contact_count",
        "total_non_tool_displacement_px",
        "settle_time_s",
        "boundary_clearance_px",
    )
    distance = 0.0
    for name in continuous_names:
        values = [float(item["features"][name]) for item in candidates]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance) or 1.0
        distance += (
            (float(left_features[name]) - float(right_features[name])) / scale
        ) ** 2
    for action in avoid_coords:
        left_distance = math.dist(left["coords"], action)
        right_distance = math.dist(right["coords"], action)
        distance += ((left_distance - right_distance) / 100.0) ** 2
    return categorical_mismatches, math.sqrt(distance)


def select_probe_pair(
    *,
    bank: dict[str, Any],
    avoid_coords: Sequence[Sequence[int]],
    unit_key: str,
    seed: int,
) -> dict[str, Any]:
    eligible = [
        item
        for item in bank["candidates"]
        if all(
            math.dist(item["coords"], coords)
            >= MIN_PROBE_ACTION_DISTANCE_PX
            for coords in avoid_coords
        )
    ]
    if len(eligible) < 2:
        eligible = list(bank["candidates"])
    pairs = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            if (
                math.dist(left["coords"], right["coords"])
                < MIN_PROBE_PAIR_DISTANCE_PX
            ):
                continue
            mismatch_count, feature_distance = standardized_pair_distance(
                left,
                right,
                candidates=eligible,
                avoid_coords=avoid_coords,
            )
            pairs.append(
                (
                    mismatch_count,
                    feature_distance,
                    stable_hash(
                        f"{unit_key}|{left['coords']}|{right['coords']}", seed
                    ),
                    left,
                    right,
                )
            )
    if not pairs:
        raise RuntimeError(f"No separated probe pair for {unit_key}")
    mismatch_count, feature_distance, _, left, right = min(pairs)
    if stable_hash(f"{unit_key}|early|{left['coords']}", seed) <= stable_hash(
        f"{unit_key}|early|{right['coords']}", seed
    ):
        early, terminal = left, right
    else:
        early, terminal = right, left
    return {
        "schema_version": 1,
        "selected_before_feedback": True,
        "selection_uses_goal_success": False,
        "selection_uses_model_predictions_or_performance": False,
        "avoid_attempt_1_coords": [
            [int(coords[0]), int(coords[1])] for coords in avoid_coords
        ],
        "minimum_action_distance_px": MIN_PROBE_ACTION_DISTANCE_PX,
        "minimum_pair_distance_px": MIN_PROBE_PAIR_DISTANCE_PX,
        "categorical_feature_mismatch_count": mismatch_count,
        "standardized_feature_distance": feature_distance,
        "early_probe": early,
        "terminal_probe": terminal,
    }


def select_clear_path_probe(
    *,
    bank: dict[str, Any],
    gravity: str,
    unit_key: str,
    seed: int,
) -> dict[str, Any]:
    """Select one prospective calibration whose placed tool has no contact."""
    if gravity not in {"upward", "downward"}:
        raise ValueError(f"Unsupported gravity for clear-path probe: {gravity}")
    expected_exit_side = "top" if gravity == "upward" else "bottom"
    candidates = []
    for candidate in bank["candidates"]:
        features = candidate.get("features") or {}
        tool_truth = (candidate.get("endpoints") or {}).get("PLACED") or {}
        if bool(features.get("tool_contact")):
            continue
        if str(tool_truth.get("state") or "") != "exited":
            continue
        if str(tool_truth.get("exit_side") or "") != expected_exit_side:
            continue
        candidates.append(candidate)
    if not candidates:
        return {
            "available": False,
            "reason": (
                "no_collision_free_expected_direction_tool_exit_in_"
                "deterministic_probe_bank"
            ),
            "gravity": gravity,
            "expected_tool_exit_side": expected_exit_side,
            "selection_uses_goal_success": False,
            "selection_uses_model_predictions_or_performance": False,
        }
    selected = min(
        candidates,
        key=lambda candidate: (
            -float(
                (candidate.get("features") or {}).get(
                    "boundary_clearance_px"
                )
                or 0.0
            ),
            stable_hash(
                f"{unit_key}|clear_path|{candidate['coords']}", seed
            ),
        ),
    )
    return {
        **copy.deepcopy(selected),
        "available": True,
        "gravity": gravity,
        "expected_tool_exit_side": expected_exit_side,
        "selection_rule": (
            "no_placed_tool_contact_and_expected_gravity_direction_exit_"
            "then_maximum_initial_boundary_clearance"
        ),
        "selection_uses_goal_success": False,
        "selection_uses_model_predictions_or_performance": False,
    }


def build_pre_feedback_prompt(
    *,
    goal: dict[str, Any],
    coords: Sequence[int],
    targets: Sequence[dict[str, str]],
    trace_initial_scene: Optional[dict[str, Any]],
) -> str:
    prompt = f"""Isolated counterfactual-prediction sidecar.

Goal context: {goal["goal_text"]}

This query occurs after your first candidate was proposed but before any simulator rollout or success/failure evidence is supplied. The fixed placement below is unexecuted, your prediction will receive no feedback, and this sidecar never returns to the solver conversation.

{prediction_schema_text(targets, coords)}"""
    if trace_initial_scene is not None:
        prompt += "\n\nInitial observable scene JSON:\n" + json.dumps(
            trace_initial_scene, separators=(",", ":"), sort_keys=True
        )
    return prompt


def checkpoint_material(
    *,
    condition: str,
    goal: dict[str, Any],
    goal_bank: Any,
    attempt: dict[str, Any],
    completed_attempts: int,
    frame_count: int,
    screenshot_path: Path,
    temporary_root: Path,
) -> tuple[str, list[Path]]:
    spec = runner.CONDITIONS[condition]
    lines = [
        f"The latest recorded source-goal candidate was attempt {completed_attempts} at placement {list(attempt['coords'])}.",
        f"Source goal: {goal['goal_text']}",
    ]
    images: list[Path] = [] if spec["trace"] else [screenshot_path]
    trace_value = (attempt.get("truth") or {}).get("structured_trace")
    if spec["frames"]:
        if not trace_value or not Path(trace_value).exists():
            raise RuntimeError("Missing structured trace for visual sidecar")
        frames, _ = runner.render_structured_trace_frames(
            goal=goal,
            goal_bank=goal_bank,
            coords=attempt["coords"],
            trace_path=Path(trace_value),
            out_dir=temporary_root / f"{condition}_frames",
            frame_count=frame_count,
        )
        images.extend(frames)
        lines.append(
            f"The next {frame_count} attached images are uniformly sampled frames from only that latest rollout, ordered early to late."
        )
    elif spec["trace"]:
        if not trace_value or not Path(trace_value).exists():
            raise RuntimeError("Missing structured trace for JSON sidecar")
        trace_payload = runner.serialize_observable_trace(
            Path(trace_value), state_count=frame_count
        )
        lines.append(
            f"The JSON below contains {frame_count} uniformly sampled observable states from only that latest rollout."
        )
        lines.append(
            "Latest observable rollout JSON:\n"
            + json.dumps(trace_payload, separators=(",", ":"), sort_keys=True)
        )
        lines.append(
            "Initial observable scene JSON:\n"
            + json.dumps(
                runner.sanitized_initial_scene(goal, goal_bank),
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        lines.append("Rollout frames are withheld in this condition.")
    if spec["status"]:
        result = (
            "satisfied the source goal"
            if bool(attempt["goal_succeeded"])
            else "did not satisfy the source goal"
        )
        lines.append(f"Simulator result: the latest placement {result}.")
    else:
        lines.append(
            "The simulator success/failure result is withheld in this condition."
        )
    lines.append(
        "The timing of this isolated evaluation query does not itself reveal why the solver interaction was continued or stopped."
    )
    return "\n".join(lines), images


def clean_temporary_media(temporary_root: Path) -> None:
    if not temporary_root.exists():
        return
    for path in sorted(temporary_root.rglob("*.png")):
        path.unlink()
    for directory in sorted(
        (path for path in temporary_root.rglob("*") if path.is_dir()),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    if not any(temporary_root.iterdir()):
        temporary_root.rmdir()


def prediction_score_rows(
    *,
    response: dict[str, Any],
    probe: dict[str, Any],
) -> list[dict[str, Any]]:
    payload = response.get("parsed_payload") or {}
    predictions = payload.get("coordinate_predictions") or []
    prediction_map = {
        str(item["id"]): item
        for item in predictions
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    rows = []
    for object_id, truth in (probe.get("endpoints") or {}).items():
        prediction = prediction_map.get(str(object_id))
        predicted_state = str((prediction or {}).get("state") or "missing")
        actual_state = str(truth["state"])
        point = (prediction or {}).get("point")
        has_point = (
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) for value in point)
        )
        error: Optional[float] = None
        quality = 0.0
        predicted_exit_side = str(
            (prediction or {}).get("exit_side") or ""
        )
        actual_exit_side = str(truth.get("exit_side") or "")
        exit_side_correct: Optional[bool] = None
        if (
            predicted_state == "in_scene"
            and actual_state == "in_scene"
            and has_point
        ):
            error = math.dist(
                [float(point[0]), float(point[1])],
                [float(truth["point"][0]), float(truth["point"][1])],
            )
            quality = math.exp(
                -(error**2) / (2 * PREDICTION_SIGMA_PX**2)
            )
        elif predicted_state == "exited" and actual_state == "exited":
            exit_side_correct = predicted_exit_side == actual_exit_side
            quality = 1.0 if exit_side_correct else 0.0
        rows.append(
            {
                "object_id": str(object_id),
                "prediction_present": prediction is not None,
                "predicted_state": predicted_state,
                "actual_state": actual_state,
                "state_correct": predicted_state == actual_state,
                "predicted_exit_side": predicted_exit_side or None,
                "actual_exit_side": actual_exit_side or None,
                "exit_side_correct": exit_side_correct,
                "endpoint_error_px": error,
                "prediction_quality_sigma150": quality,
                "within_50px": error is not None and error <= 50,
                "within_100px": error is not None and error <= 100,
            }
        )
    return rows


def validate_probe_truth_targets(
    probe: dict[str, Any],
    targets: Sequence[dict[str, str]],
    *,
    probe_name: str,
) -> None:
    target_ids = {str(item["id"]) for item in targets}
    truth_ids = {
        str(object_id) for object_id in (probe.get("endpoints") or {})
    }
    missing = sorted(target_ids - truth_ids)
    unexpected = sorted(truth_ids - target_ids)
    if missing or unexpected:
        raise RuntimeError(
            f"{probe_name} prediction/truth target mismatch: "
            f"missing_truth={missing}, unexpected_truth={unexpected}"
        )


def branch_state(
    unit_dir: Path, condition: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition == "trace_status":
        shared = read_json(
            unit_dir
            / "branches"
            / "trace_status"
            / "independent_attempt_1"
            / "state.json"
        )
    else:
        shared = read_json(unit_dir / "shared_attempt_1" / "state.json")
    state = read_json(unit_dir / "branches" / condition / "state.json")
    return shared, state


def run_prediction_sidecar(
    *,
    provider: Any,
    history: Sequence[dict[str, str]],
    prompt: str,
    images: Sequence[Path],
    targets: Sequence[dict[str, str]],
    probe: dict[str, Any],
    sidecar_dir: Path,
    call_label: str,
    max_format_repairs: int,
) -> dict[str, Any]:
    result_path = sidecar_dir / "result.json"
    if result_path.exists():
        existing = read_json(result_path)
        if bool(existing.get("protocol_finalized")) or bool(
            existing.get("schema_valid")
        ):
            return existing
        archive = sidecar_dir / "result.incomplete_before_retry.json"
        if not archive.exists():
            shutil.copy2(result_path, archive)
    result = request_prediction(
        provider=provider,
        history=history,
        prompt=prompt,
        image_paths=images,
        targets=targets,
        coords=probe["coords"],
        call_log_path=sidecar_dir / "provider_calls.jsonl",
        call_label=call_label,
        max_format_repairs=max_format_repairs,
    )
    result["protocol_finalized"] = True
    result["permitted_format_repairs"] = int(max_format_repairs)
    result["probe"] = probe
    result["score_rows"] = prediction_score_rows(
        response=result,
        probe=probe,
    )
    runner.atomic_write_json(result_path, result)
    return result


def compact_prediction_result(
    result: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    return {
        "result_path": str(result_path.resolve()),
        "schema_valid": bool(result.get("schema_valid")),
        "protocol_finalized": bool(
            result.get("protocol_finalized")
            or result.get("schema_valid")
        ),
        "provider_call_count": int(result.get("provider_call_count") or 0),
        "score_rows": result.get("score_rows") or [],
    }


def compact_goal_transfer_result(
    result: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    attempt = result.get("attempt") or {}
    return {
        "result_path": str(result_path.resolve()),
        "error": result.get("error"),
        "protocol_finalized": bool(result.get("protocol_finalized")),
        "provider_call_count": int(result.get("provider_call_count") or 0),
        "coords": attempt.get("coords"),
        "goal_succeeded": (
            bool(attempt.get("goal_succeeded")) if attempt else None
        ),
    }


def goal_transfer_prompt(
    *,
    source_goal: dict[str, Any],
    target_goal: dict[str, Any],
    targets: Sequence[dict[str, str]],
    feedback_text: str,
) -> str:
    return f"""Isolated same-scene goal-transfer branch.

{feedback_text}

The source-goal episode is now set aside. This is the same environment with the same initial scene, the same gravity, the same orange ball tool, and the same physical dynamics as before. Only the objective is replaced with this different target:

New target goal: {target_goal["goal_text"]}

Submit exactly one placement for the new target goal and prospectively predict the terminal state caused by your own placement. This branch does not return to the source solver.

{runner.response_schema_text(targets)}"""


def run_goal_transfer(
    *,
    args: argparse.Namespace,
    provider: Any,
    source_goal: dict[str, Any],
    target_goal: dict[str, Any],
    goal_bank: Any,
    history: Sequence[dict[str, str]],
    feedback_text: str,
    images: Sequence[Path],
    unit_dir: Path,
    condition: str,
    world_path: Path,
) -> dict[str, Any]:
    sidecar_dir = unit_dir / "transfer_sidecars" / condition / "goal_transfer"
    result_path = sidecar_dir / "result.json"
    if result_path.exists():
        existing = read_json(result_path)
        if bool(existing.get("protocol_finalized")):
            return existing
        if not existing.get("error") and isinstance(
            existing.get("attempt"), dict
        ):
            return existing
        archive = sidecar_dir / "result.incomplete_before_retry.json"
        if not archive.exists():
            shutil.copy2(result_path, archive)
    targets = runner.prediction_targets_for_goal(target_goal, goal_bank)
    prompt = goal_transfer_prompt(
        source_goal=source_goal,
        target_goal=target_goal,
        targets=targets,
        feedback_text=feedback_text,
    )
    try:
        working_history = list(copy.deepcopy(history))
        current_prompt = prompt
        current_images = list(images)
        all_calls = []
        blocked_actions = []
        attempt = None
        calls = []
        for candidate_index in range(1, args.max_blocked_repairs + 2):
            coords, working_history, calls = runner.request_valid_action(
                provider=provider,
                history=working_history,
                prompt=current_prompt,
                image_paths=current_images,
                call_log_path=sidecar_dir / "provider_calls.jsonl",
                call_label=(
                    f"{condition}_goal_transfer_candidate_{candidate_index:02d}"
                ),
                max_format_repairs=args.max_format_repairs,
                prediction_targets=targets,
                history_prompt=None,
            )
            all_calls.extend(calls)
            attempt = runner.simulate_attempt(
                goal=target_goal,
                goal_bank=goal_bank,
                world_path=world_path,
                coords=coords,
                attempt_dir=(
                    sidecar_dir
                    / f"target_attempt_1_candidate_{candidate_index:02d}"
                ),
                attempt_number=1,
                frame_count=args.frame_count,
                attempt_budget=1,
                frames_condition=False,
                stop_on_success=True,
                trace_condition=False,
                trace_state_count=args.frame_count,
            )
            if not bool(attempt["obstruction_detected"]):
                break
            blocked_actions.append(list(coords))
            if candidate_index > args.max_blocked_repairs:
                raise RuntimeError(
                    "Exceeded goal-transfer geometric repair limit"
                )
            current_prompt = runner.build_blocked_prompt(coords, targets)
            current_images = list(images[:1])
        if attempt is None:
            raise RuntimeError("Goal-transfer action loop produced no attempt")
        attempt["model_response"] = copy.deepcopy(
            calls[-1].get("parsed_action_payload") if calls else None
        )
        result = {
            "schema_version": 1,
            "source_goal_id": source_goal["balanced_goal_id"],
            "target_goal_id": target_goal["balanced_goal_id"],
            "condition": condition,
            "attempt": attempt,
            "blocked_actions": blocked_actions,
            "provider_call_count": len(all_calls),
            "last_provider_metadata": (
                copy.deepcopy(all_calls[-1].get("provider_metadata"))
                if all_calls
                else None
            ),
            "error": None,
            "protocol_finalized": True,
        }
    except Exception as exc:
        result = {
            "schema_version": 1,
            "source_goal_id": source_goal["balanced_goal_id"],
            "target_goal_id": target_goal["balanced_goal_id"],
            "condition": condition,
            "attempt": None,
            "provider_call_count": 0,
            "last_provider_metadata": None,
            "error": f"{type(exc).__name__}: {exc}",
            "protocol_finalized": True,
        }
    runner.atomic_write_json(result_path, result)
    return result


def probe_contaminated(
    probe_coords: Sequence[int], attempts: Iterable[dict[str, Any]]
) -> bool:
    return any(
        math.dist(probe_coords, attempt["coords"])
        < CONTAMINATION_DISTANCE_PX
        for attempt in attempts
    )


def sidecar_summary_complete(
    summary: dict[str, Any],
    expected_conditions: Optional[Sequence[str]] = None,
) -> bool:
    results = summary.get("results") or {}
    pre_feedback = results.get("pre_feedback") or {}
    required_pre = {
        "visual_early_probe",
        "visual_terminal_probe",
        "trace_early_probe",
        "trace_terminal_probe",
        "visual_clear_path_probe",
    }
    if set(pre_feedback) != required_pre:
        return False
    for name, item in pre_feedback.items():
        if name == "visual_clear_path_probe" and item.get("available") is False:
            continue
        if not bool(
            item.get("protocol_finalized") or item.get("schema_valid")
        ):
            return False
    conditions = results.get("conditions") or {}
    expected = set(
        expected_conditions
        or summary.get("expected_conditions")
        or ALL_CONDITIONS
    )
    if set(conditions) != expected:
        return False
    for branch in conditions.values():
        early = branch.get("early_prediction") or {}
        if not bool(
            early.get("protocol_finalized") or early.get("schema_valid")
        ):
            return False
        terminal = branch.get("terminal_prediction") or {}
        if not bool(
            terminal.get("protocol_finalized")
            or terminal.get("schema_valid")
        ):
            return False
        goal_transfer = branch.get("goal_transfer") or {}
        if goal_transfer.get("eligible") is False:
            continue
        if goal_transfer.get("protocol_finalized") is True:
            continue
        if goal_transfer.get("error") or goal_transfer.get("coords") is None:
            return False
    return True


def run_unit_sidecars(
    *,
    args: argparse.Namespace,
    provider: Any,
    goal: dict[str, Any],
    target_goal: Optional[dict[str, Any]],
    goal_bank: Any,
    unit_dir: Path,
    probe_cache: Path,
) -> dict[str, Any]:
    output_path = unit_dir / "transfer_sidecars" / "summary.json"
    main_summary = read_json(unit_dir / "unit_summary.json")
    active_conditions = tuple(
        condition
        for condition in ALL_CONDITIONS
        if condition in (main_summary.get("conditions") or {})
    )
    if "full" not in active_conditions or "trace_status" not in active_conditions:
        raise RuntimeError(
            f"Source unit must contain full and trace_status branches: {unit_dir}"
        )
    if args.skip_existing and output_path.exists():
        existing = read_json(output_path)
        if sidecar_summary_complete(
            existing, expected_conditions=active_conditions
        ):
            return existing
        archive = (
            unit_dir
            / "transfer_sidecars"
            / "summary.incomplete_before_retry.json"
        )
        if not archive.exists():
            shutil.copy2(output_path, archive)
    screenshot_path = unit_dir / "assets" / "initial_observation.png"
    world_path = unit_dir / "assets" / "simulation_world.json"
    visual_shared, _ = branch_state(unit_dir, "full")
    trace_shared, _ = branch_state(unit_dir, "trace_status")
    avoid_coords = [
        visual_shared["attempt"]["coords"],
        trace_shared["attempt"]["coords"],
    ]
    bank = prepare_probe_bank(
        goal=goal,
        goal_bank=goal_bank,
        probe_cache=probe_cache,
        seed=args.seed,
    )
    selection_path = unit_dir / "transfer_sidecars" / "probe_selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
    else:
        selection = select_probe_pair(
            bank=bank,
            avoid_coords=avoid_coords,
            unit_key=f"{args.model}|{goal['balanced_goal_id']}",
            seed=args.seed,
        )
        runner.atomic_write_json(selection_path, selection)
    if "clear_path_probe" not in selection:
        selection["clear_path_probe"] = select_clear_path_probe(
            bank=bank,
            gravity=str(goal["condition"]),
            unit_key=str(goal["balanced_goal_id"]),
            seed=args.seed,
        )
        runner.atomic_write_json(selection_path, selection)
    early_probe = selection["early_probe"]
    terminal_probe = selection["terminal_probe"]
    clear_path_probe = selection["clear_path_probe"]
    targets = runner.prediction_targets_for_goal(goal, goal_bank)
    validate_probe_truth_targets(
        early_probe, targets, probe_name="early_probe"
    )
    validate_probe_truth_targets(
        terminal_probe, targets, probe_name="terminal_probe"
    )
    initial_scene_json = runner.sanitized_initial_scene(goal, goal_bank)

    results: dict[str, Any] = {"pre_feedback": {}, "conditions": {}}
    for representation, shared, trace_initial in (
        ("visual", visual_shared, None),
        ("trace", trace_shared, initial_scene_json),
    ):
        for probe_name, probe in (
            ("early_probe", early_probe),
            ("terminal_probe", terminal_probe),
        ):
            prompt = build_pre_feedback_prompt(
                goal=goal,
                coords=probe["coords"],
                targets=targets,
                trace_initial_scene=trace_initial,
            )
            images = [] if trace_initial is not None else [screenshot_path]
            sidecar_dir = (
                unit_dir
                / "transfer_sidecars"
                / "pre_feedback"
                / representation
                / probe_name
            )
            result = run_prediction_sidecar(
                    provider=provider,
                    history=shared["history"],
                    prompt=prompt,
                    images=images,
                    targets=targets,
                    probe=probe,
                    sidecar_dir=sidecar_dir,
                    call_label=f"pre_feedback_{representation}_{probe_name}",
                    max_format_repairs=args.max_format_repairs,
                )
            results["pre_feedback"][f"{representation}_{probe_name}"] = (
                compact_prediction_result(result, sidecar_dir / "result.json")
            )

    if clear_path_probe.get("available") is False:
        results["pre_feedback"]["visual_clear_path_probe"] = {
            "available": False,
            "reason": clear_path_probe.get("reason"),
            "schema_valid": None,
            "provider_call_count": 0,
            "score_rows": [],
        }
    else:
        validate_probe_truth_targets(
            clear_path_probe,
            targets,
            probe_name="clear_path_probe",
        )
        clear_prompt = build_pre_feedback_prompt(
            goal=goal,
            coords=clear_path_probe["coords"],
            targets=targets,
            trace_initial_scene=None,
        )
        clear_sidecar_dir = (
            unit_dir
            / "transfer_sidecars"
            / "pre_feedback"
            / "visual"
            / "clear_path_probe"
        )
        clear_result = run_prediction_sidecar(
            provider=provider,
            history=visual_shared["history"],
            prompt=clear_prompt,
            images=[screenshot_path],
            targets=targets,
            probe=clear_path_probe,
            sidecar_dir=clear_sidecar_dir,
            call_label="pre_feedback_visual_clear_path_probe",
            max_format_repairs=args.max_format_repairs,
        )
        results["pre_feedback"]["visual_clear_path_probe"] = {
            "available": True,
            **compact_prediction_result(
                clear_result,
                clear_sidecar_dir / "result.json",
            ),
        }

    for condition in active_conditions:
        shared, state = branch_state(unit_dir, condition)
        attempts = state.get("attempts") or []
        if not attempts:
            raise RuntimeError(f"No attempts in {unit_dir} {condition}")
        first_attempt = attempts[0]
        terminal_attempt = attempts[-1]
        temporary_root = (
            unit_dir / "transfer_sidecars" / condition / "_temporary_media"
        )
        condition_results: dict[str, Any] = {}
        try:
            early_feedback, early_images = checkpoint_material(
                condition=condition,
                goal=goal,
                goal_bank=goal_bank,
                attempt=first_attempt,
                completed_attempts=1,
                frame_count=args.frame_count,
                screenshot_path=screenshot_path,
                temporary_root=temporary_root / "early",
            )
            early_prompt = f"""Isolated counterfactual-prediction sidecar after exactly one source-goal evidence exposure.

{early_feedback}

The fixed placement below is unexecuted, your prediction will receive no feedback, and this sidecar never returns to the solver conversation.

{prediction_schema_text(targets, early_probe["coords"])}"""
            early_sidecar_dir = (
                unit_dir
                / "transfer_sidecars"
                / condition
                / "early_prediction"
            )
            early_result = run_prediction_sidecar(
                provider=provider,
                history=shared["history"],
                prompt=early_prompt,
                images=early_images,
                targets=targets,
                probe=early_probe,
                sidecar_dir=early_sidecar_dir,
                call_label=f"{condition}_early_prediction",
                max_format_repairs=args.max_format_repairs,
            )
            condition_results["early_prediction"] = (
                compact_prediction_result(
                    early_result, early_sidecar_dir / "result.json"
                )
            )
            clean_temporary_media(temporary_root / "early")

            terminal_feedback, terminal_images = checkpoint_material(
                condition=condition,
                goal=goal,
                goal_bank=goal_bank,
                attempt=terminal_attempt,
                completed_attempts=len(attempts),
                frame_count=args.frame_count,
                screenshot_path=screenshot_path,
                temporary_root=temporary_root / "terminal",
            )
            terminal_prompt = f"""Isolated late counterfactual-prediction sidecar after the completed source-goal interaction history.

{terminal_feedback}

The fixed placement below was selected before feedback and has not been executed in the solver conversation. Your prediction receives no feedback and never returns to the solver.

{prediction_schema_text(targets, terminal_probe["coords"])}"""
            terminal_sidecar_dir = (
                unit_dir
                / "transfer_sidecars"
                / condition
                / "terminal_prediction"
            )
            terminal_result = run_prediction_sidecar(
                    provider=provider,
                    history=state["history"],
                    prompt=terminal_prompt,
                    images=terminal_images,
                    targets=targets,
                    probe=terminal_probe,
                    sidecar_dir=terminal_sidecar_dir,
                    call_label=f"{condition}_terminal_prediction",
                    max_format_repairs=args.max_format_repairs,
                )
            condition_results["terminal_prediction"] = (
                compact_prediction_result(
                    terminal_result, terminal_sidecar_dir / "result.json"
                )
            )
            condition_results["terminal_probe_contaminated"] = (
                probe_contaminated(terminal_probe["coords"], attempts)
            )
            condition_results["source_attempt_count"] = len(attempts)
            condition_results["source_first_success_attempt"] = state.get(
                "first_success_attempt"
            )
            if target_goal is None:
                condition_results["goal_transfer"] = {
                    "eligible": False,
                    "reason": "no_noncanonical_goal_in_main_manifest_cell",
                    "error": None,
                    "provider_call_count": 0,
                    "coords": None,
                    "goal_succeeded": None,
                }
            else:
                goal_transfer_result = run_goal_transfer(
                    args=args,
                    provider=provider,
                    source_goal=goal,
                    target_goal=target_goal,
                    goal_bank=goal_bank,
                    history=state["history"],
                    feedback_text=terminal_feedback,
                    images=terminal_images,
                    unit_dir=unit_dir,
                    condition=condition,
                    world_path=world_path,
                )
                condition_results["goal_transfer"] = {
                    "eligible": True,
                    **compact_goal_transfer_result(
                        goal_transfer_result,
                        unit_dir
                        / "transfer_sidecars"
                        / condition
                        / "goal_transfer"
                        / "result.json",
                    ),
                }
        finally:
            clean_temporary_media(temporary_root)
        results["conditions"][condition] = condition_results

    summary = {
        "schema_version": 2,
        "model_key": args.model,
        "seed": args.seed,
        "expected_conditions": list(active_conditions),
        "source_goal_id": goal["balanced_goal_id"],
        "target_goal_id": (
            target_goal["balanced_goal_id"] if target_goal is not None else None
        ),
        "puzzle_key": goal["puzzle_key"],
        "probe_selection": selection,
        "results": results,
    }
    runner.atomic_write_json(output_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openrouter", "mock"), default="openrouter")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", choices=tuple(runner.MODEL_SPECS), default="gpt5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-index", type=Path, default=runner.DEFAULT_ASSET_INDEX)
    parser.add_argument("--asset-root", type=Path, action="append", default=None)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--probe-cache", type=Path, default=DEFAULT_PROBE_CACHE)
    parser.add_argument("--goal-builder", type=Path, default=DEFAULT_GOAL_BUILDER)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-provider-cost-usd", type=float, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-format-repairs", type=int, default=1)
    parser.add_argument("--max-blocked-repairs", type=int, default=12)
    parser.add_argument("--goal-index", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--prepare-probes-only", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.frame_count != 32:
        parser.error("The frozen transfer protocol requires exactly 32 frames/states")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    if args.api_key and args.api_key_file:
        parser.error("Use only one of --api-key or --api-key-file")
    for name in (
        "manifest",
        "asset_index",
        "result_root",
        "probe_cache",
        "goal_builder",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.api_key_file:
        args.api_key_file = args.api_key_file.expanduser().resolve()
    return args


def select_goals(
    goals: Sequence[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    selected = list(goals)
    if args.goal_index is not None:
        selected = [selected[args.goal_index]]
    else:
        selected = selected[max(0, args.start_index) :]
        if args.limit is not None:
            selected = selected[: args.limit]
    if args.shard_count > 1:
        selected = [
            goal
            for goal in selected
            if int(goal["benchmark_index"]) % args.shard_count
            == args.shard_index
        ]
    return selected


def main() -> None:
    args = parse_args()
    raw_manifest = read_json(args.manifest)
    raw_goals = raw_manifest.get("goals") or []
    assets = runner.load_or_build_asset_index(
        raw_goals,
        asset_index_path=args.asset_index,
        asset_roots=tuple(
            path.expanduser().resolve()
            for path in (args.asset_root or runner.DEFAULT_ASSET_ROOTS)
        ),
        rebuild=False,
    )
    goals = runner.load_balanced_goals(
        args.manifest,
        asset_index=assets,
        budget_mode="fixed",
        fixed_budget=8,
    )
    raw_by_id = {
        str(goal["balanced_goal_id"]): goal for goal in raw_goals
    }
    goal_by_id = {
        str(goal["balanced_goal_id"]): goal for goal in goals
    }
    source_goals = [
        goal
        for goal in goals
        if bool(
            raw_by_id[str(goal["balanced_goal_id"])].get("transfer_source")
        )
    ]
    if len(source_goals) != 132:
        raise RuntimeError(
            f"Expected 132 annotated transfer sources; found {len(source_goals)}"
        )
    selected = select_goals(source_goals, args)
    goal_bank = runner.base.GoalBank(
        REPO_ROOT / "task_configs", args.goal_builder
    )
    if args.prepare_probes_only:
        first_by_cell = {}
        for goal in selected:
            first_by_cell.setdefault(str(goal["puzzle_key"]), goal)
        for ordinal, goal in enumerate(first_by_cell.values(), start=1):
            print(
                f"[probe bank {ordinal}/{len(first_by_cell)}] {goal['puzzle_key']}",
                flush=True,
            )
            prepare_probe_bank(
                goal=goal,
                goal_bank=goal_bank,
                probe_cache=args.probe_cache,
                seed=args.seed,
            )
        return

    provider = runner.make_provider(args, runner.MODEL_SPECS[args.model])
    sidecar_call_logs = []
    for goal in selected:
        unit_dir = runner.goal_unit_dir(
            args.result_root, args.model, args.seed, goal
        )
        transfer_root = unit_dir / "transfer_sidecars"
        if transfer_root.exists():
            sidecar_call_logs.extend(
                transfer_root.rglob("provider_calls.jsonl")
            )
    provider.cumulative_cost_usd = runner.provider_cost_from_call_logs(
        sidecar_call_logs
    )
    if provider.cumulative_cost_usd:
        print(
            "Recovered prior sidecar provider cost for this restart/shard: "
            f"${provider.cumulative_cost_usd:.4f}",
            flush=True,
        )
    summaries = []
    for ordinal, goal in enumerate(selected, start=1):
        raw = raw_by_id[str(goal["balanced_goal_id"])]
        target_id = str(raw.get("transfer_target_goal_id") or "")
        target_goal = goal_by_id.get(target_id) if target_id else None
        if bool(raw.get("goal_transfer_eligible")) and target_goal is None:
            raise RuntimeError(
                f"Missing transfer target for {goal['balanced_goal_id']}: {target_id}"
            )
        unit_dir = runner.goal_unit_dir(
            args.result_root, args.model, args.seed, goal
        )
        if not (unit_dir / "unit_summary.json").exists():
            raise RuntimeError(f"Main-run unit is incomplete: {unit_dir}")
        print(
            f"[{ordinal}/{len(selected)}] {args.model} "
            f"{goal['balanced_goal_id']} -> {target_id}",
            flush=True,
        )
        lock_path = unit_dir / "transfer_sidecars" / ".unit.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            summaries.append(
                run_unit_sidecars(
                    args=args,
                    provider=provider,
                    goal=goal,
                    target_goal=target_goal,
                    goal_bank=goal_bank,
                    unit_dir=unit_dir,
                    probe_cache=args.probe_cache,
                )
            )
    incomplete_ids = [
        str(summary.get("source_goal_id"))
        for summary in summaries
        if not sidecar_summary_complete(summary)
    ]
    manifest_path = (
        args.result_root
        / "_transfer_manifests"
        / f"{args.model}_seed{args.seed}_shard{args.shard_index:03d}of{args.shard_count:03d}.json"
    )
    runner.atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "model": runner.MODEL_SPECS[args.model].__dict__,
            "seed": args.seed,
            "manifest": str(args.manifest),
            "selected_goal_count": len(selected),
            "completed_unit_count": len(summaries),
            "protocol_complete_unit_count": (
                len(summaries) - len(incomplete_ids)
            ),
            "incomplete_source_goal_ids": incomplete_ids,
            "frame_count": args.frame_count,
            "probe_cache": str(args.probe_cache),
            "cumulative_provider_cost_usd": provider.cumulative_cost_usd,
        },
    )
    if len(summaries) != len(selected) or incomplete_ids:
        raise RuntimeError(
            "Transfer shard coverage incomplete: "
            f"selected={len(selected)} summaries={len(summaries)} "
            f"incomplete={incomplete_ids[:10]}"
        )


if __name__ == "__main__":
    main()
