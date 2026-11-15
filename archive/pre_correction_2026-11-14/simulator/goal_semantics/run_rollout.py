#!/usr/bin/env python
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SIM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SIM_DIR.parents[1]
RUNTIME_ROOT = SIM_DIR.parent / "runtime"
SCRIPT_ROOT = RUNTIME_ROOT
ENV_SET_ROOT = PROJECT_ROOT / "132_base_environments" / "cells"
PUBLIC_ROOT = PROJECT_ROOT

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import pygame as pg  # noqa: E402
import make_trial_onetool_3 as mto3  # noqa: E402


def load_request() -> dict:
    raw = sys.stdin.read()
    return json.loads(raw or "{}")


def resolve_world_path(payload: dict) -> Path:
    explicit = payload.get("worldPath")
    if explicit:
        explicit_path = str(explicit).strip()
        if explicit_path.startswith("./"):
            return (PUBLIC_ROOT / explicit_path[2:]).resolve()
        if explicit_path.startswith("/"):
            return (PROJECT_ROOT / explicit_path.lstrip("/")).resolve()
        explicit_candidate = Path(explicit_path)
        if explicit_candidate.is_absolute():
            return explicit_candidate
        return (PROJECT_ROOT / explicit_candidate).resolve()
    env_set = str(payload["environmentSet"])
    env_id = str(payload["envId"])
    return ENV_SET_ROOT / env_set / f"{env_id}.json"


def simulate_rollout(payload: dict) -> dict:
    return simulate_rollout_with_callback(payload)


def simulate_rollout_with_callback(payload: dict, accepted_callback=None, frame_callback=None) -> dict:
    world_path = resolve_world_path(payload)
    if not world_path.exists():
        raise FileNotFoundError(f"Environment JSON not found: {world_path}")

    fps = float(payload.get("fps", 60.0))
    stillness_duration = float(payload.get("stillnessDuration", 0.5))
    linear_threshold = float(payload.get("linearThreshold", 1.0))
    angular_threshold = float(payload.get("angularThreshold", 0.1))
    max_sim_seconds = float(payload.get("maxSimSeconds", 10.0))
    record_window_seconds = float(payload.get("recordWindowSeconds", 10.0))

    world_dict = mto3.load_world_from_path(str(world_path))
    requested_goal_duration = payload.get("goalDurationSeconds")
    if requested_goal_duration is not None and world_dict.get("world", {}).get("gcond"):
        world_dict["world"]["gcond"]["duration"] = float(requested_goal_duration)
    sim_width, sim_height = [int(v) for v in world_dict["world"]["dims"]]

    tool_name = payload.get("selectedTool") or "obj1"
    no_tool = tool_name in (None, "", "no_tool", mto3.NO_TOOL_LABEL)
    drop_xy = payload.get("dropCoords")
    if not no_tool and (not isinstance(drop_xy, list) or len(drop_xy) < 2):
        raise ValueError("dropCoords must be provided for tool rollouts.")

    gravity_mode = str(payload.get("gravityMode") or "downward").lower()
    if gravity_mode not in {"downward", "upward"}:
        raise ValueError("gravityMode must be 'downward' or 'upward'.")

    tool_color = str(payload.get("toolColor") or "orange")
    tools_dict = mto3.build_tools(tool_color, tool_color, tool_color)
    tools_dict = {"obj1": dict(tools_dict["obj1"])}
    if gravity_mode == "upward":
        tools_dict["obj1"]["inverse_gravity"] = True

    video_output_dir = payload.get("videoOutputDir")
    video_basename = payload.get("videoBasename")
    old_sim = float(mto3.MAX_SIM_SECONDS)
    old_rec = float(mto3.MAX_RECORDED_SECONDS)
    mto3.MAX_SIM_SECONDS = max_sim_seconds
    mto3.MAX_RECORDED_SECONDS = record_window_seconds
    try:
        raw_payload, saved_video_path = mto3.run_headless_episode(
            world_dict,
            tool_name=(None if no_tool else tool_name),
            tools_dict=tools_dict,
            drop_xy=(None if no_tool else (int(drop_xy[0]), int(drop_xy[1]))),
            no_tool=no_tool,
            fps=fps,
            stillness_duration=stillness_duration,
            linear_threshold=linear_threshold,
            angular_threshold=angular_threshold,
            record_video=bool(video_output_dir),
            video_dir=video_output_dir,
            video_basename=video_basename,
            accepted_callback=accepted_callback,
            frame_callback=frame_callback,
        )
    finally:
        mto3.MAX_SIM_SECONDS = old_sim
        mto3.MAX_RECORDED_SECONDS = old_rec

    placement = deepcopy(raw_payload["placements"][0])
    placement["trajectory_frame_count"] = len(placement.get("trajectory_data", {}).get("PLACED", []))
    # Persist the target object's final location explicitly for downstream logging.
    # Fallback to "Ball" when the world goal condition omits "obj".
    target_name = (
        payload.get("targetObjectName")
        or world_dict.get("world", {}).get("gcond", {}).get("obj")
        or "Ball"
    )
    landing_positions = placement.get("landing_positions") or {}
    placement["target_final_position"] = landing_positions.get(target_name)
    placement.pop("trajectory_data", None)

    return {
        "placement": placement,
        "worldDims": [sim_width, sim_height],
        "goalText": world_dict.get("sucText"),
        "gravityMode": gravity_mode,
        "videoGenerated": bool(saved_video_path),
        "rolloutVideoPath": saved_video_path,
    }


def stream_rollout(payload: dict) -> None:
    def emit(event_type: str, body: dict) -> None:
        print(json.dumps({"type": event_type, **body}), flush=True)

    result = simulate_rollout_with_callback(
        payload,
        accepted_callback=lambda accepted: emit("accepted", {"accepted": accepted}),
        frame_callback=lambda frame: emit("frame", {"frame": frame}),
    )
    emit("result", {"result": result})


def main() -> None:
    payload = load_request()
    try:
        if payload.get("streamMode"):
            stream_rollout(payload)
            return
        result = simulate_rollout(payload)
        print(json.dumps({"ok": True, "result": result}))
    except Exception as exc:
        if payload.get("streamMode"):
            print(json.dumps({"type": "error", "error": str(exc)}), flush=True)
            return
        print(json.dumps({"ok": False, "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
