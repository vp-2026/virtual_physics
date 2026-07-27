#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from google import genai
    from google.genai import types as gemini_types
except Exception:
    genai = None
    gemini_types = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CACHE_ROOT = REPO_ROOT / "outputs" / "goal_probe"
BUNDLED_NODE_BIN = REPO_ROOT / "vendor" / "node" / "bin"
DEFAULT_OUTPUT_ROOT = CACHE_ROOT / "outputs"
DEFAULT_VTOOLS_ROOT = REPO_ROOT
DEFAULT_GOAL_BANK_ROOT = REPO_ROOT / "task_configs"
DEFAULT_GOAL_BUILDER = (
    REPO_ROOT
    / "simulator"
    / "goal_semantics"
    / "build_goal_bank_from_placement_sweep.py"
)
SIM_WIDTH = 600
SIM_HEIGHT = 600
TOOL_RADIUS_PX = 36.0
TOOL_DIAMETER_PX = 72
DEFAULT_MODEL_ID = "gemini-3.1-pro-preview"
DEFAULT_SEED = 42
ORANGE_BALL_POLYS = [[
    [
        TOOL_RADIUS_PX * math.cos(2.0 * math.pi * (1.0 - idx / 32.0)),
        TOOL_RADIUS_PX * math.sin(2.0 * math.pi * (1.0 - idx / 32.0)),
    ]
    for idx in range(32)
]]


def ensure_runtime_environment() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if (BUNDLED_NODE_BIN / "node").exists():
        node_bin = str(BUNDLED_NODE_BIN)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if node_bin not in path_parts:
            os.environ["PATH"] = node_bin + os.pathsep + os.environ.get("PATH", "")


ensure_runtime_environment()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".mp4":
        return "video/mp4"
    return "application/octet-stream"


def remap_user_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    return path.resolve()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_json_object(text: str) -> Optional[dict]:
    candidates = []
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_match:
        candidates.append(code_match.group(1))
    dict_match = re.search(r"\{.*\}", text, re.DOTALL)
    if dict_match:
        candidates.append(dict_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_action(response_text: str) -> Optional[Tuple[int, int]]:
    parsed = parse_json_object(response_text)
    if not isinstance(parsed, dict):
        return None
    part1 = parsed.get("part1")
    if not isinstance(part1, dict):
        part1 = parsed
    point = part1.get("point")
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


def parse_probe_answers(response_text: str, question_ids: Sequence[str]) -> Optional[dict]:
    expected = {str(qid) for qid in question_ids}
    parsed = parse_json_object(response_text)
    out: Dict[str, str] = {}
    if isinstance(parsed, dict):
        predictions = parsed.get("answers", parsed.get("predictions"))
        if isinstance(predictions, list):
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                qid = str(item.get("id", "")).strip()
                answer = str(item.get("answer", "")).strip().lower()
                if qid in expected and answer in {"yes", "no"}:
                    out[qid] = answer
        else:
            for qid in expected:
                answer = str(parsed.get(qid, "")).strip().lower()
                if answer in {"yes", "no"}:
                    out[qid] = answer
    if set(out) == expected:
        return {"answers": [{"id": qid, "answer": out[qid]} for qid in question_ids]}
    return None


def model_facing_goal_text(text: Any) -> str:
    value = str(text)
    replacements = [
        ("big orange ball (dropped tool)", "orange ball tool"),
        ("grey tool ball", "orange ball tool"),
        ("grey dotted ball", "orange ball tool"),
        ("grey ball", "orange ball tool"),
    ]
    for src, dst in replacements:
        value = value.replace(src, dst)
    return value


def broad_goal_category(goal: dict) -> str:
    sig = str(goal.get("signature", "")).lower()
    cat = str(goal.get("category", "")).lower()
    text = str(goal.get("goal_text", goal.get("display", ""))).lower()
    if any(token in sig or token in text for token in ("on_floor", "on_ceiling", "on_wall", "on the floor", "on the ceiling", "on the wall")):
        return "contact"
    if any(token in sig or token in text for token in ("container", "inside", "contain", "partially inside")):
        if "touching" not in sig and "touching" not in text and "contact_change" not in sig:
            return "containment"
        if "inside" in sig or "inside" in text or "partially inside" in text:
            return "containment"
    if cat in {"contact", "contact_change"} or any(
        token in sig or token in text
        for token in ("touching", "contact", "no_longer_touching", "on_floor", "on the floor")
    ):
        return "contact"
    if cat == "rotation" or any(
        token in sig or token in text
        for token in ("rotation", "rotate", "rotates", "clockwise", "counterclockwise", "orientation")
    ):
        return "orientation"
    if cat in {"movement", "joint_movement", "pass_over"} or any(
        token in sig or token in text
        for token in ("moves", "movement", "moved", "pass_over", "passes over", "travels", "displacement")
    ):
        return "movement"
    if any(
        token in sig or token in text
        for token in (
            "left_of",
            "right_of",
            "above",
            "below",
            "on_top_of",
            "center_left",
            "center_right",
            "center_above",
            "center_below",
            "ends to the left",
            "ends to the right",
            "ends above",
            "ends below",
            "on top of",
            "center point ends",
        )
    ):
        return "position"
    return "other"


def extract_sampled_frames(video_path: Path, out_dir: Path, prefix: str, frame_count: int) -> List[Path]:
    if imageio is None:
        raise RuntimeError("imageio is required to sample rollout frames.")
    ensure_dir(out_dir)
    reader = imageio.get_reader(str(video_path))
    try:
        total_frames = int(reader.count_frames())
    except Exception:
        total_frames = 0
    indices = [int(round(i * (total_frames - 1) / max(frame_count - 1, 1))) for i in range(frame_count)]
    if total_frames <= 0:
        frames = list(reader)
        if not frames:
            raise RuntimeError(f"No frames found in rollout video: {video_path}")
        indices = [int(round(i * (len(frames) - 1) / max(frame_count - 1, 1))) for i in range(frame_count)]
    else:
        frames = []
    paths = []
    for out_idx, frame_idx in enumerate(indices, start=1):
        frame = frames[min(frame_idx, len(frames) - 1)] if frames else reader.get_data(frame_idx)
        out_path = out_dir / f"{prefix}_{out_idx:02d}.png"
        imageio.imwrite(str(out_path), frame)
        paths.append(out_path)
    reader.close()
    return paths


def video_duration_seconds(video_path: Optional[Path]) -> Optional[float]:
    if video_path is None or imageio is None or not video_path.exists():
        return None
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data() or {}
        duration = meta.get("duration")
        if isinstance(duration, (int, float)) and math.isfinite(float(duration)) and float(duration) > 0:
            return round(float(duration), 3)
        fps = meta.get("fps")
        frame_count = reader.count_frames()
        if fps and frame_count:
            return round(float(frame_count) / float(fps), 3)
    except Exception:
        return None
    finally:
        try:
            reader.close()
        except Exception:
            pass
    return None


class Provider:
    def __init__(self, model_id: str, temperature: float, system_prompt: str, seed: int) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.seed = seed

    def is_retryable_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        exc_name = type(exc).__name__.lower()
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        retry_tokens = (
            "timeout",
            "timed out",
            "connecttimeout",
            "readtimeout",
            "connection error",
            "connection reset",
            "temporarily unavailable",
            "rate limit",
            "server error",
            "internal error",
            "failed to download multimodal content",
        )
        return any(token in message or token in exc_name for token in retry_tokens)

    def call_with_retries(self, label: str, call) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                return call()
            except Exception as exc:
                last_exc = exc
                if attempt >= 4 or not self.is_retryable_error(exc):
                    raise
                delay = min(45.0, 4.0 * (2 ** (attempt - 1)))
                delay += random.Random(f"{self.seed}:{label}:{attempt}").random() * 2.0
                print(
                    f"[provider retry] {label} attempt {attempt}/4 failed with {type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{label} failed without an exception")

    def generate(
        self,
        *,
        history: Sequence[dict],
        prompt: str,
        image_paths: Sequence[Path],
    ) -> str:
        raise NotImplementedError


class MockProvider(Provider):
    def __init__(self, model_id: str, temperature: float, system_prompt: str, seed: int) -> None:
        super().__init__(model_id, temperature, system_prompt, seed)
        self.call_count = 0

    def generate(self, *, history: Sequence[dict], prompt: str, image_paths: Sequence[Path]) -> str:
        del history, image_paths
        self.call_count += 1
        question_ids = re.findall(r"-\s*(q[12])\s*:", prompt)
        if question_ids:
            return json.dumps(
                {
                    "answers": [
                        {"id": qid, "answer": "yes" if (idx + self.call_count) % 2 == 0 else "no"}
                        for idx, qid in enumerate(question_ids)
                    ]
                }
            )
        placements = [(180, 420), (300, 420), (420, 300), (250, 520), (500, 220), (120, 360)]
        x, y = placements[(self.call_count - 1) % len(placements)]
        return json.dumps({"part1": {"point": [x, y], "reasoning": "mock placement for smoke testing"}})


class GeminiProvider(Provider):
    def __init__(self, *, api_key: Optional[str], model_id: str, temperature: float, system_prompt: str, seed: int) -> None:
        if genai is None or gemini_types is None:
            raise ImportError("google-genai is required for Gemini runs.")
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("Set GEMINI_API_KEY or pass --api-key.")
        super().__init__(model_id, temperature, system_prompt, seed)
        self.client = genai.Client(api_key=resolved_api_key)

    def generate(self, *, history: Sequence[dict], prompt: str, image_paths: Sequence[Path]) -> str:
        contents = []
        for message in history:
            role = "user" if message.get("role") == "user" else "model"
            contents.append(gemini_types.Content(role=role, parts=[gemini_types.Part(text=str(message.get("content", "")))]))
        parts = [gemini_types.Part(text=prompt)]
        for image_path in image_paths:
            parts.append(
                gemini_types.Part.from_bytes(
                    data=image_path.read_bytes(),
                    mime_type=get_mime_type(image_path),
                )
            )
        contents.append(gemini_types.Content(role="user", parts=parts))
        response = self.call_with_retries(
            "gemini_generate_content",
            lambda: self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=gemini_types.GenerateContentConfig(
                    temperature=self.temperature,
                    system_instruction=self.system_prompt,
                ),
            ),
        )
        return getattr(response, "text", None) or ""


class OpenAIResponsesProvider(Provider):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        model_id: str,
        temperature: float,
        system_prompt: str,
        seed: int,
        reasoning_effort: Optional[str],
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai is required for GPT runs.")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("Set OPENAI_API_KEY or pass --api-key.")
        super().__init__(model_id, temperature, system_prompt, seed)
        self.client = OpenAI(api_key=resolved_api_key)
        self.reasoning_effort = reasoning_effort

    def data_url(self, path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{get_mime_type(path)};base64,{encoded}"

    def extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
        return ""

    def generate(self, *, history: Sequence[dict], prompt: str, image_paths: Sequence[Path]) -> str:
        input_messages = []
        for message in history:
            if message.get("role") == "model":
                input_messages.append(
                    {"role": "assistant", "content": [{"type": "output_text", "text": str(message.get("content", ""))}]}
                )
            else:
                input_messages.append(
                    {"role": "user", "content": [{"type": "input_text", "text": str(message.get("content", ""))}]}
                )
        current_content: List[dict] = [{"type": "input_text", "text": prompt}]
        for image_path in image_paths:
            current_content.append({"type": "input_image", "image_url": self.data_url(image_path)})
        input_messages.append({"role": "user", "content": current_content})
        request_kwargs = {
            "model": self.model_id,
            "input": input_messages,
            "instructions": self.system_prompt,
        }
        if not self.model_id.startswith("gpt-5"):
            request_kwargs["temperature"] = self.temperature
        elif self.reasoning_effort:
            request_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        def do_create():
            try:
                return self.client.responses.create(**request_kwargs)
            except Exception as exc:
                message = str(exc).lower()
                if "seed" in request_kwargs and "seed" in message and ("unsupported" in message or "unexpected" in message):
                    request_kwargs.pop("seed", None)
                    return self.client.responses.create(**request_kwargs)
                if "temperature" in request_kwargs and "temperature" in message and ("unsupported" in message or "unexpected" in message):
                    request_kwargs.pop("temperature", None)
                    return self.client.responses.create(**request_kwargs)
                raise

        response = self.call_with_retries("openai_responses_create", do_create)
        return self.extract_text(response)


class QwenChatProvider(Provider):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        model_id: str,
        temperature: float,
        system_prompt: str,
        seed: int,
        enable_thinking: bool,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai is required for Qwen OpenAI-compatible runs.")
        resolved_api_key = (
            api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError("Set DASHSCOPE_API_KEY or QWEN_API_KEY, or pass --api-key.")
        resolved_base_url = (
            base_url
            or os.environ.get("DASHSCOPE_BASE_URL")
            or os.environ.get("QWEN_BASE_URL")
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        super().__init__(model_id, temperature, system_prompt, seed)
        self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
        self.enable_thinking = enable_thinking

    def generate(self, *, history: Sequence[dict], prompt: str, image_paths: Sequence[Path]) -> str:
        messages: List[dict] = [{"role": "system", "content": self.system_prompt}]
        for message in history:
            role = "assistant" if message.get("role") == "model" else "user"
            messages.append({"role": role, "content": str(message.get("content", ""))})
        content: List[dict] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{get_mime_type(image_path)};base64,{encoded}"}})
        messages.append({"role": "user", "content": content})
        request_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.enable_thinking:
            request_kwargs["extra_body"] = {"enable_thinking": True}
        def do_create():
            try:
                return self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                message = str(exc).lower()
                if "seed" in request_kwargs and "seed" in message and "unsupported" in message:
                    request_kwargs.pop("seed", None)
                    return self.client.chat.completions.create(**request_kwargs)
                raise

        response = self.call_with_retries("qwen_chat_completions_create", do_create)
        return response.choices[0].message.content or ""


def build_system_prompt() -> str:
    return """You are an expert physical reasoning agent solving virtual physics puzzles.
Use the provided screenshot and rollout frames as evidence.
Only return the exact JSON schema requested in each prompt."""


def build_intro_prompt(goal_text: str, max_attempts: int) -> str:
    return f"""Goal: {goal_text}

You have up to {max_attempts} valid attempts.

The environment uses a 600 x 600 coordinate space:
- Coordinates are integer pixels in the range 0-599.
- The origin (0,0) is at the bottom-left corner.
- You may drop one orange ball tool with diameter {TOOL_DIAMETER_PX} pixels.
- The orange ball has real size, so choose a center point where the whole ball can fit.
- Do not choose placements that overlap existing objects or scene boundaries.

Return exactly one JSON object:
{{"part1": {{"point": [x, y], "reasoning": "<brief reasoning>"}}}}"""


def build_feedback_prompt(
    goal_text: str,
    coords: Sequence[int],
    solved: bool,
    attempt_number: int,
    max_attempts: int,
    frame_count: int,
    duration_seconds: Optional[float],
) -> str:
    result = "This placement succeeded for the goal." if solved else "This placement did not succeed for the goal."
    remaining = max(0, max_attempts - attempt_number)
    duration_line = (
        f"- Rollout duration: about {duration_seconds:.2f} seconds."
        if duration_seconds is not None
        else "- Rollout duration: unavailable."
    )
    return f"""Latest rollout:
- Previous placement: {list(coords)}
- Goal: {goal_text}
- Result: {result}
- Attached media: {frame_count} sampled frames from the rollout, ordered from early to late.
{duration_line}

You have {remaining} valid attempts remaining. Use the rollout evidence to choose a better next placement.
Return exactly one JSON object:
{{"part1": {{"point": [x, y], "reasoning": "<brief reasoning>"}}}}"""


def build_probe_prompt(probes: Sequence[dict], last_feedback: Optional[dict], frame_count: int) -> str:
    context = ""
    if last_feedback:
        duration_line = (
            f"- Rollout duration: about {last_feedback['duration_seconds']:.2f} seconds."
            if last_feedback.get("duration_seconds") is not None
            else "- Rollout duration: unavailable."
        )
        context = f"""The solving phase is now over.
Latest rollout recap:
- Previous placement: {last_feedback["coords"]}
- Result: {"succeeded" if last_feedback["solved"] else "did not succeed"} for the solving goal.
- Attached media: {frame_count} sampled frames from that latest rollout, ordered from early to late.
{duration_line}

"""
    lines = []
    for probe in probes:
        lines.append(f'- {probe["id"]}: Suppose you drop the orange ball tool at {probe["coords"]}. Will this happen: "{probe["goal_text"]}"?')
    joined = "\n".join(lines)
    return f"""{context}Answer these two yes/no questions. Use your experience from the solving attempts when it is relevant.

Return exactly one JSON object:
{{"answers": [{{"id": "q1", "answer": "yes"}}, {{"id": "q2", "answer": "no"}}]}}

Questions:
{joined}"""


class GoalBank:
    def __init__(self, root: Path, goal_builder_path: Path) -> None:
        self.root = root
        self.goal_builder = load_module(goal_builder_path, "goal_bank_builder")
        self._placement_cache: Dict[str, List[dict]] = {}
        self._aggregate_cache: Dict[str, dict] = {}
        self._predicate_context: Dict[str, dict] = {}

    def discover_goals(self) -> List[dict]:
        goals: List[dict] = []
        for aggregate_path in sorted(self.root.glob("*_*_*/aggregate_goal_bank.json")):
            puzzle_key = aggregate_path.parent.name
            try:
                aggregate = json.loads(aggregate_path.read_text())
            except Exception:
                continue
            parts = puzzle_key.rsplit("_", 2)
            if len(parts) != 3:
                continue
            family, env_id, condition = parts
            for index, item in enumerate(aggregate.get("ranked_goal_bank") or [], start=1):
                probability = item.get("probability_valid_placements")
                if probability is None:
                    continue
                goals.append(
                    {
                        "puzzle_key": puzzle_key,
                        "family": family,
                        "env_id": str(env_id),
                        "condition": condition,
                        "goal_index": index,
                        "signature": item["signature"],
                        "goal_text": model_facing_goal_text(item["display"]),
                        "category": item.get("category"),
                        "broad_category": broad_goal_category(
                            {
                                "signature": item["signature"],
                                "category": item.get("category"),
                                "goal_text": model_facing_goal_text(item["display"]),
                            }
                        ),
                        "probability_valid_placements": float(probability),
                        "count": item.get("count"),
                        "aggregate_path": str(aggregate_path),
                        "placements_path": str(aggregate_path.parent / "placements.jsonl"),
                        "environment_json": str(remap_user_path(aggregate.get("environment_json"))),
                    }
                )
        return goals

    def aggregate_for(self, puzzle_key: str) -> dict:
        if puzzle_key not in self._aggregate_cache:
            path = self.root / puzzle_key / "aggregate_goal_bank.json"
            self._aggregate_cache[puzzle_key] = json.loads(path.read_text())
        return self._aggregate_cache[puzzle_key]

    def placement_rows_for(self, puzzle_key: str) -> List[dict]:
        if puzzle_key not in self._placement_cache:
            path = self.root / puzzle_key / "placements.jsonl"
            rows = []
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
            self._placement_cache[puzzle_key] = rows
        return self._placement_cache[puzzle_key]

    def context_for(self, goal: dict) -> dict:
        key = str(goal["puzzle_key"])
        if key in self._predicate_context:
            return self._predicate_context[key]
        pred = self.goal_builder.get_predicate_module()
        env_path = remap_user_path(goal["environment_json"])
        world_data = pred.load_world_file(env_path)
        world_data["_source_name"] = env_path.stem
        world_data["_env_set"] = str(goal["family"])
        world_data["_env_id"] = str(goal["env_id"])
        label_map, role_map, dynamic_map = self.goal_builder.build_label_maps(pred, world_data)
        ctx = {
            "pred": pred,
            "world_data": world_data,
            "label_map": label_map,
            "role_map": role_map,
            "dynamic_map": dynamic_map,
        }
        self._predicate_context[key] = ctx
        return ctx

    def evaluate_goal_at(self, goal: dict, coords: Sequence[int], trace_dir: Optional[Path] = None) -> dict:
        ctx = self.context_for(goal)
        save_trace_path = None
        if trace_dir is not None:
            ensure_dir(trace_dir)
            save_trace_path = trace_dir / f"trace_{int(coords[0])}_{int(coords[1])}.json"
        row = self.goal_builder.simulate_valid_placement(
            pred=ctx["pred"],
            world_data=ctx["world_data"],
            label_map=ctx["label_map"],
            role_map=ctx["role_map"],
            dynamic_map=ctx["dynamic_map"],
            condition=str(goal["condition"]),
            coords=coords,
            movement_threshold_px=float(self.goal_builder.CENTER_RELATIVE_POSITION_THRESHOLD_PX),
            rotation_threshold_deg=float(self.goal_builder.ROTATION_FIRST_DIRECTION_EPS_DEG * 6.0),
            contact_min_duration_s=float(self.goal_builder.CONTAINER_EVENT_MIN_DURATION_S),
            include_tool_events=False,
            save_trace_path=save_trace_path,
        )
        signatures = {
            self.goal_builder.event_signature(event)
            for event in row.get("event_graph", [])
        } if row.get("valid") else set()
        return {
            "valid": bool(row.get("valid")),
            "answer": "yes" if str(goal["signature"]) in signatures else "no",
            "matched_signature": str(goal["signature"]) in signatures,
            "event_graph": row.get("event_graph", []),
            "placement_row": row,
        }

    def row_has_goal(self, row: dict, signature: str) -> bool:
        if not row.get("valid"):
            return False
        return str(signature) in {self.goal_builder.event_signature(event) for event in row.get("event_graph", [])}

    def choose_probe_pair(
        self,
        *,
        solved_goal: dict,
        tried_coords: Sequence[Sequence[int]],
        run_dir: Path,
        seed: int,
        nonoverlap_distance: float,
    ) -> List[dict]:
        aggregate = self.aggregate_for(str(solved_goal["puzzle_key"]))
        candidates = []
        for idx, item in enumerate(aggregate.get("ranked_goal_bank") or [], start=1):
            if item.get("signature") == solved_goal.get("signature"):
                continue
            prob = item.get("probability_valid_placements")
            if prob is None:
                continue
            candidates.append((abs(float(prob) - 0.5), idx, item))
        candidates.sort(key=lambda item: (item[0], item[1]))
        rows = [row for row in self.placement_rows_for(str(solved_goal["puzzle_key"])) if row.get("valid")]
        rng = random.Random(seed)
        tried = [[int(x), int(y)] for x, y in tried_coords]
        index_match = re.match(r"(\d+)_", run_dir.name)
        if index_match:
            balance_index = int(index_match.group(1))
        else:
            balance_key = f"{seed}|{solved_goal.get('puzzle_key')}|{solved_goal.get('goal_index')}|{solved_goal.get('signature')}"
            balance_index = int(hash_text(balance_key), 16)
        pair_targets = [
            ("yes", "yes"),
            ("yes", "no"),
            ("no", "yes"),
            ("no", "no"),
        ]
        desired_seen_answer, desired_unseen_answer = pair_targets[balance_index % len(pair_targets)]

        def far_from_tried(coords: Sequence[int]) -> bool:
            return all(math.dist([float(coords[0]), float(coords[1])], [float(t[0]), float(t[1])]) >= nonoverlap_distance for t in tried)

        def make_probe_goal(item: dict) -> dict:
            probe_goal = dict(solved_goal)
            probe_goal.update(
                {
                    "signature": item["signature"],
                    "goal_text": model_facing_goal_text(item["display"]),
                    "category": item.get("category"),
                    "broad_category": broad_goal_category(
                        {
                            "signature": item["signature"],
                            "category": item.get("category"),
                            "goal_text": model_facing_goal_text(item["display"]),
                        }
                    ),
                    "probability_valid_placements": float(item.get("probability_valid_placements")),
                    "count": item.get("count"),
                }
            )
            return probe_goal

        def return_pair(probe_goal: dict, seen_coords: Sequence[int], seen_truth: dict, coords: Sequence[int], unseen_truth: dict, kind: str) -> List[dict]:
            return [
                {
                    "id": "q1",
                    "kind": "seen_attempt",
                    "coords": [int(seen_coords[0]), int(seen_coords[1])],
                    "goal_text": probe_goal["goal_text"],
                    "goal_signature": probe_goal["signature"],
                    "goal_probability_valid_placements": probe_goal["probability_valid_placements"],
                    "ground_truth_answer": seen_truth["answer"],
                    "preferred_balanced_answer": desired_seen_answer,
                },
                {
                    "id": "q2",
                    "kind": kind,
                    "coords": [int(coords[0]), int(coords[1])],
                    "goal_text": probe_goal["goal_text"],
                    "goal_signature": probe_goal["signature"],
                    "goal_probability_valid_placements": probe_goal["probability_valid_placements"],
                    "ground_truth_answer": unseen_truth["answer"],
                    "preferred_balanced_answer": desired_unseen_answer,
                },
            ]

        for prefer_balanced_seen, prefer_balanced_unseen, require_nonoverlap in (
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (False, False, True),
            (True, True, False),
            (False, True, False),
            (True, False, False),
            (False, False, False),
        ):
            for _, _, item in candidates:
                probe_goal = make_probe_goal(item)
                seen_options = []
                for coords in tried:
                    truth = self.evaluate_goal_at(probe_goal, coords, trace_dir=run_dir / "probe_truth_traces")
                    if truth.get("valid"):
                        seen_options.append((coords, truth))
                if not seen_options:
                    continue
                rng.shuffle(seen_options)
                for seen_coords, seen_truth in seen_options:
                    if prefer_balanced_seen and seen_truth["answer"] != desired_seen_answer:
                        continue
                    unseen_pool = []
                    for row in rows:
                        coords = row.get("placement_xy") or []
                        if len(coords) != 2:
                            continue
                        if [int(coords[0]), int(coords[1])] in tried:
                            continue
                        if require_nonoverlap and not far_from_tried(coords):
                            continue
                        bank_answer = "yes" if self.row_has_goal(row, str(probe_goal["signature"])) else "no"
                        if prefer_balanced_unseen and bank_answer != desired_unseen_answer:
                            continue
                        unseen_pool.append((rng.random(), coords, bank_answer))
                    unseen_pool.sort()
                    for _, coords, _bank_answer in unseen_pool[:80]:
                        unseen_truth = self.evaluate_goal_at(probe_goal, coords, trace_dir=run_dir / "probe_truth_traces")
                        if not unseen_truth.get("valid"):
                            continue
                        if prefer_balanced_unseen and unseen_truth["answer"] != desired_unseen_answer:
                            continue
                        kind = "unseen_nonoverlap" if require_nonoverlap else "unseen_fallback"
                        return return_pair(probe_goal, seen_coords, seen_truth, coords, unseen_truth, kind)
        raise RuntimeError("Could not select a seen/unseen probe pair.")


def select_goals(goals: Sequence[dict], *, seed: int, limit: Optional[int], start_index: int, prefer_near_half: bool) -> List[dict]:
    items = [dict(goal) for goal in goals]
    if prefer_near_half:
        items.sort(
            key=lambda goal: (
                abs(float(goal["probability_valid_placements"]) - 0.5),
                goal["puzzle_key"],
                goal["goal_index"],
            )
        )
    else:
        rng = random.Random(seed)
        rng.shuffle(items)
    start = max(0, start_index)
    selected = items[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def goals_from_manifest(goals: Sequence[dict], manifest_path: Path) -> List[dict]:
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("selected_goals", manifest if isinstance(manifest, list) else [])
    by_key = {
        (str(goal.get("puzzle_key")), int(goal.get("goal_index")), str(goal.get("signature"))): dict(goal)
        for goal in goals
    }
    selected = []
    missing = []
    for entry in entries:
        key = (str(entry.get("puzzle_key")), int(entry.get("goal_index")), str(entry.get("signature")))
        goal = by_key.get(key)
        if goal is None:
            missing.append({"puzzle_key": key[0], "goal_index": key[1], "signature": key[2]})
            continue
        selected.append(goal)
    if missing:
        raise RuntimeError(f"Goal manifest had {len(missing)} missing goals. First missing: {missing[0]}")
    return selected


def screenshot_for_goal(goal: dict, vtools_root: Path) -> Path:
    condition_dir = "showcase_heatmap_assets_upward" if goal["condition"] == "upward" else "showcase_heatmap_assets"
    manifest_path = vtools_root / condition_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    for entry in data.get("entries") or []:
        if str(entry.get("family")) == str(goal["family"]) and str(entry.get("env_num")) == str(goal["env_id"]):
            return remap_user_path(entry.get("screenshot_path"))
    fallback = vtools_root / condition_dir / str(goal["family"]) / f"env_{goal['env_id']}" / "environment.png"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"No screenshot found for {goal['puzzle_key']}")


def load_world_json(world_path: Path) -> dict:
    with world_path.open() as handle:
        world_dict = json.load(handle)
    world_dict["_source_name"] = world_path.stem
    return world_dict


def build_tool_def(world_dict: dict, condition: str) -> dict:
    del world_dict
    raw_polys = ORANGE_BALL_POLYS
    polys = []
    placement_radius = TOOL_RADIUS_PX
    for poly in raw_polys:
        cast_poly = []
        for x, y in poly:
            fx = float(x)
            fy = float(y)
            cast_poly.append([fx, fy])
            placement_radius = max(placement_radius, math.hypot(fx, fy))
        polys.append(cast_poly)
    return {
        "polys": polys,
        "density": 1.0,
        "friction": 0.5,
        "elasticity": 0.5,
        "color": "orange",
        "placement_radius": placement_radius,
        "inverse_gravity": condition == "upward",
    }


def run_rollout(
    *,
    runtime_script_dir: Path,
    python_bin: str,
    world_path: Path,
    tool_def: dict,
    coords: Sequence[int],
    run_dir: Path,
    attempt_number: int,
    video_basename: str,
) -> Tuple[dict, Optional[Path]]:
    inline = """
import json
import os
import sys
payload = json.loads(sys.stdin.read())
runtime_dir = payload["runtime_script_dir"]
if runtime_dir not in sys.path:
    sys.path.insert(0, runtime_dir)
import make_trial_onetool_3 as mto3
with open(payload["world_path"], "r") as handle:
    world_dict = json.load(handle)
world_dict["_source_name"] = os.path.splitext(os.path.basename(payload["world_path"]))[0]
result_payload, saved_video_path = mto3.run_headless_episode(
    world_dict,
    tool_name="obj1",
    tools_dict={"obj1": payload["tool_def"]},
    drop_xy=(int(payload["coords"][0]), int(payload["coords"][1])),
    no_tool=False,
    record_video=True,
    video_dir=payload["video_dir"],
    video_basename=payload["video_basename"],
)
print(json.dumps({"result_payload": result_payload, "video_path": saved_video_path}))
"""
    req = {
        "runtime_script_dir": str(runtime_script_dir),
        "world_path": str(world_path),
        "tool_def": tool_def,
        "coords": [int(coords[0]), int(coords[1])],
        "video_dir": str(run_dir),
        "video_basename": f"{video_basename}_attempt_{attempt_number:02d}",
    }
    env = os.environ.copy()
    if (BUNDLED_NODE_BIN / "node").exists():
        node_bin = str(BUNDLED_NODE_BIN)
        path_parts = env.get("PATH", "").split(os.pathsep)
        if node_bin not in path_parts:
            env["PATH"] = node_bin + os.pathsep + env.get("PATH", "")
    completed = subprocess.run(
        [python_bin, "-c", inline],
        input=json.dumps(req),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "Simulation subprocess failed.")
    parsed = json.loads(completed.stdout.strip())
    return parsed["result_payload"], (Path(parsed["video_path"]) if parsed.get("video_path") else None)


class GoalProbeRun:
    def __init__(
        self,
        *,
        goal: dict,
        goal_bank: GoalBank,
        provider: Provider,
        vtools_root: Path,
        run_dir: Path,
        python_bin: str,
        max_attempts: int,
        frame_count: int,
        seed: int,
        nonoverlap_distance: float,
    ) -> None:
        self.goal = goal
        self.goal_bank = goal_bank
        self.provider = provider
        self.vtools_root = vtools_root
        self.run_dir = run_dir
        self.python_bin = python_bin
        self.max_attempts = max_attempts
        self.frame_count = frame_count
        self.seed = seed
        self.nonoverlap_distance = nonoverlap_distance
        self.history: List[dict] = []
        self.turn_cache: List[dict] = []
        self.attempts: List[dict] = []
        self.payload: dict = {}
        self.initial_prompt: Optional[str] = None

    def save_progress(self) -> None:
        ensure_dir(self.run_dir)
        model_facing_turn_cache = []
        for turn in self.turn_cache:
            model_facing_turn_cache.append(
                {
                    "turn_index": turn.get("turn_index"),
                    "prompt_kind": turn.get("prompt_kind"),
                    "prompt_hash": turn.get("prompt_hash"),
                    "response_hash": turn.get("response_hash"),
                    "response_text": turn.get("response_text"),
                    "image_count": len(turn.get("image_paths") or []),
                    "image_hashes": turn.get("image_hashes") or [],
                }
            )
        model_facing_payload = {
            "system_prompt": self.provider.system_prompt,
            "conversation_history": self.history,
            "turn_cache": model_facing_turn_cache,
        }
        (self.run_dir / "conversation_history.json").write_text(json.dumps(model_facing_payload, indent=2))
        backend_payload = dict(self.payload)
        backend_payload.update(
            {
                "protocol": "goal_solve_then_seen_unseen_probe",
                "private_goal_metadata": self.goal,
                "model_id": self.provider.model_id,
                "temperature": self.provider.temperature,
                "max_attempts": self.max_attempts,
                "frame_count": self.frame_count,
                "attempts": self.attempts,
                "turn_cache": self.turn_cache,
            }
        )
        (self.run_dir / "backend_run_metadata.json").write_text(json.dumps(backend_payload, indent=2))

    def compact_history_for_final_probe(self) -> None:
        compact: List[dict] = []
        if self.initial_prompt:
            compact.append({"role": "user", "content": self.initial_prompt})
        for attempt in self.attempts:
            coords = attempt.get("coords")
            compact.append({"role": "model", "content": json.dumps({"part1": {"point": coords}})})
            result = "succeeded" if attempt.get("goal_succeeded") else "did not succeed"
            compact.append(
                {
                    "role": "user",
                    "content": f"Attempt {attempt.get('attempt')} result: placement {coords} {result} for the goal.",
                }
            )
        self.history = compact
        self.save_progress()

    def generate(self, prompt: str, *, prompt_kind: str, image_paths: Sequence[Path], history_prompt: Optional[str] = None) -> str:
        text = self.provider.generate(history=self.history, prompt=prompt, image_paths=image_paths)
        self.history.append({"role": "user", "content": history_prompt or prompt})
        self.history.append({"role": "model", "content": text})
        self.turn_cache.append(
            {
                "turn_index": len(self.turn_cache) + 1,
                "prompt_kind": prompt_kind,
                "prompt_hash": hash_text(prompt),
                "response_hash": hash_text(text),
                "response_text": text,
                "image_paths": [str(path) for path in image_paths],
                "image_hashes": [hash_file(path) for path in image_paths],
            }
        )
        self.save_progress()
        return text

    def request_action(self, prompt: str, *, prompt_kind: str, image_paths: Sequence[Path]) -> Tuple[int, int]:
        response = self.generate(prompt, prompt_kind=prompt_kind, image_paths=image_paths)
        coords = parse_action(response)
        if coords is not None:
            self.history[-1]["content"] = json.dumps({"part1": {"point": [coords[0], coords[1]]}})
            self.save_progress()
            return coords
        retry = self.generate(
            """Your previous response did not contain a valid placement.
Return exactly one JSON object:
{"part1": {"point": [x, y], "reasoning": "<brief reasoning>"}}""",
            prompt_kind=f"{prompt_kind}_retry",
            image_paths=[],
        )
        coords = parse_action(retry)
        if coords is None:
            raise RuntimeError("Model failed to produce a valid placement after retry.")
        self.history[-1]["content"] = json.dumps({"part1": {"point": [coords[0], coords[1]]}})
        self.save_progress()
        return coords

    def run(self) -> dict:
        ensure_dir(self.run_dir)
        world_path = remap_user_path(self.goal["environment_json"])
        world_dict = load_world_json(world_path)
        tool_def = build_tool_def(world_dict, str(self.goal["condition"]))
        runtime_script_dir = self.vtools_root / "script"
        screenshot_path = screenshot_for_goal(self.goal, self.vtools_root)
        local_screenshot = self.run_dir / "initial_observation.png"
        shutil.copy2(screenshot_path, local_screenshot)
        shutil.copy2(world_path, self.run_dir / "simulation_world.json")
        self.payload["assets"] = {
            "initial_observation": str(local_screenshot),
            "simulation_world": str(self.run_dir / "simulation_world.json"),
        }
        self.save_progress()

        prompt = build_intro_prompt(self.goal["goal_text"], self.max_attempts)
        self.initial_prompt = prompt
        image_paths: List[Path] = [local_screenshot]
        last_frames: List[Path] = []
        last_feedback: Optional[dict] = None
        solved = False
        valid_attempt_number = 1
        blocked_attempts = 0
        max_blocked_attempts = max(8, self.max_attempts * 3)

        while valid_attempt_number <= self.max_attempts:
            coords = self.request_action(prompt, prompt_kind="solve_attempt", image_paths=image_paths)
            result_payload, source_video = run_rollout(
                runtime_script_dir=runtime_script_dir,
                python_bin=self.python_bin,
                world_path=world_path,
                tool_def=tool_def,
                coords=coords,
                run_dir=self.run_dir,
                attempt_number=valid_attempt_number,
                video_basename=f"{self.goal['puzzle_key']}__goal_{self.goal['goal_index']:04d}",
            )
            placement = (result_payload.get("placements") or [{}])[0]
            if placement.get("obstruction_detected"):
                blocked_attempts += 1
                self.payload.setdefault("blocked_attempts", []).append(
                    {
                        "coords": [int(coords[0]), int(coords[1])],
                        "reason": "overlap_or_boundary",
                    }
                )
                self.save_progress()
                if blocked_attempts >= max_blocked_attempts:
                    break
                prompt = """That placement was blocked because the orange ball would overlap an object or boundary.
Choose a different valid placement and return exactly one JSON object:
{"part1": {"point": [x, y], "reasoning": "<brief reasoning>"}}"""
                image_paths = []
                continue

            attempt_number = valid_attempt_number
            video_path = self.run_dir / f"attempt_{attempt_number:02d}.mp4"
            if source_video and source_video.exists():
                if source_video.resolve() != video_path.resolve():
                    shutil.copy2(source_video, video_path)
                else:
                    video_path = source_video
                frames = extract_sampled_frames(
                    video_path,
                    self.run_dir / f"attempt_{attempt_number:02d}_frames",
                    f"attempt_{attempt_number:02d}",
                    self.frame_count,
                )
            else:
                video_path = None
                frames = []
            duration_seconds = video_duration_seconds(video_path)

            truth = self.goal_bank.evaluate_goal_at(
                self.goal,
                coords,
                trace_dir=self.run_dir / "attempt_truth_traces",
            )
            solved = truth["answer"] == "yes"
            attempt_record = {
                "attempt": attempt_number,
                "coords": [int(coords[0]), int(coords[1])],
                "obstruction_detected": False,
                "goal_succeeded": solved,
                "goal_signature": self.goal["signature"],
                "rollout_video": video_path.name if video_path else None,
                "sampled_frames": [str(path.relative_to(self.run_dir)) for path in frames],
                "rollout_duration_seconds": duration_seconds,
                "goal_probability_valid_placements": self.goal["probability_valid_placements"],
                "truth_valid": truth["valid"],
            }
            self.attempts.append(attempt_record)
            self.save_progress()
            last_frames = frames
            last_feedback = {
                "coords": attempt_record["coords"],
                "solved": solved,
                "duration_seconds": duration_seconds,
            }
            if solved:
                break
            valid_attempt_number += 1
            prompt = build_feedback_prompt(
                self.goal["goal_text"],
                coords,
                solved=False,
                attempt_number=attempt_number,
                max_attempts=self.max_attempts,
                frame_count=self.frame_count,
                duration_seconds=duration_seconds,
            )
            image_paths = frames

        if not self.attempts:
            summary = {
                "run_dir": str(self.run_dir),
                "puzzle_key": self.goal["puzzle_key"],
                "goal_index": self.goal["goal_index"],
                "goal_text": self.goal["goal_text"],
                "goal_probability_valid_placements": self.goal["probability_valid_placements"],
                "solved": False,
                "total_valid_attempts": 0,
                "final_probes": [],
                "status": "no_valid_attempts",
            }
            self.payload["run_status"] = "no_valid_attempts"
            self.save_progress()
            (self.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
            return summary

        try:
            probes = self.goal_bank.choose_probe_pair(
                solved_goal=self.goal,
                tried_coords=[attempt["coords"] for attempt in self.attempts],
                run_dir=self.run_dir,
                seed=self.seed,
                nonoverlap_distance=self.nonoverlap_distance,
            )
        except RuntimeError as exc:
            summary = {
                "run_dir": str(self.run_dir),
                "puzzle_key": self.goal["puzzle_key"],
                "goal_index": self.goal["goal_index"],
                "goal_text": self.goal["goal_text"],
                "goal_probability_valid_placements": self.goal["probability_valid_placements"],
                "solved": solved,
                "total_valid_attempts": len(self.attempts),
                "final_probes": [],
                "status": "probe_selection_failed",
                "error": str(exc),
            }
            self.payload["run_status"] = "probe_selection_failed"
            self.payload["run_error"] = str(exc)
            self.save_progress()
            (self.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
            return summary
        self.compact_history_for_final_probe()
        probe_prompt = build_probe_prompt(probes, last_feedback, self.frame_count)
        probe_response = self.generate(
            probe_prompt,
            prompt_kind="final_seen_unseen_probe",
            image_paths=last_frames,
        )
        parsed = parse_probe_answers(probe_response, [probe["id"] for probe in probes])
        if parsed is None:
            retry_prompt = """Return exactly one JSON object with answers for q1 and q2:
{"answers": [{"id": "q1", "answer": "yes"}, {"id": "q2", "answer": "no"}]}"""
            probe_response = self.generate(retry_prompt, prompt_kind="final_seen_unseen_probe_retry", image_paths=[])
            parsed = parse_probe_answers(probe_response, [probe["id"] for probe in probes])
            if parsed is None:
                raise RuntimeError("Model failed to answer final probes.")
        answer_by_id = {item["id"]: item["answer"] for item in parsed["answers"]}
        for probe in probes:
            probe["model_answer"] = answer_by_id.get(probe["id"])
            probe["correct"] = probe["model_answer"] == probe["ground_truth_answer"]
        self.payload["final_probes"] = probes
        self.save_progress()
        summary = {
            "run_dir": str(self.run_dir),
            "puzzle_key": self.goal["puzzle_key"],
            "goal_index": self.goal["goal_index"],
            "goal_text": self.goal["goal_text"],
            "goal_probability_valid_placements": self.goal["probability_valid_placements"],
            "solved": solved,
            "total_valid_attempts": len(self.attempts),
            "final_probes": probes,
        }
        (self.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve goal-bank goals, then ask one seen and one unseen event probe.")
    parser.add_argument("--backend", choices=("gemini", "openai", "qwen-openai", "mock"), default="gemini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"), default="medium")
    parser.add_argument("--qwen-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", "--temp", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--goal-count", type=int, default=1000)
    parser.add_argument("--goal-manifest", default=None, help="JSON manifest with selected_goals to run in manifest order.")
    parser.add_argument("--prefer-target-goals-near-half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nonoverlap-distance", type=float, default=TOOL_DIAMETER_PX)
    parser.add_argument("--vtools-root", default=str(DEFAULT_VTOOLS_ROOT))
    parser.add_argument("--goal-bank-root", default=str(DEFAULT_GOAL_BANK_ROOT))
    parser.add_argument("--goal-builder", default=str(DEFAULT_GOAL_BUILDER))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.model_id is None:
        if args.backend == "openai":
            args.model_id = "gpt-5"
        elif args.backend == "qwen-openai":
            args.model_id = "qwen3.6-plus"
        else:
            args.model_id = DEFAULT_MODEL_ID
    return args


def make_provider(args: argparse.Namespace, system_prompt: str) -> Provider:
    if args.backend == "mock":
        return MockProvider(args.model_id, args.temperature, system_prompt, args.seed)
    if args.backend == "gemini":
        return GeminiProvider(
            api_key=args.api_key,
            model_id=args.model_id,
            temperature=args.temperature,
            system_prompt=system_prompt,
            seed=args.seed,
        )
    if args.backend == "openai":
        return OpenAIResponsesProvider(
            api_key=args.api_key,
            model_id=args.model_id,
            temperature=args.temperature,
            system_prompt=system_prompt,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
        )
    if args.backend == "qwen-openai":
        return QwenChatProvider(
            api_key=args.api_key,
            base_url=args.base_url,
            model_id=args.model_id,
            temperature=args.temperature,
            system_prompt=system_prompt,
            seed=args.seed,
            enable_thinking=args.qwen_enable_thinking,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def main() -> None:
    args = parse_args()
    ensure_dir(CACHE_ROOT)
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())
    vtools_root = Path(args.vtools_root).expanduser().resolve()
    goal_bank = GoalBank(Path(args.goal_bank_root).expanduser().resolve(), Path(args.goal_builder).expanduser().resolve())
    all_goals = goal_bank.discover_goals()
    if args.goal_manifest:
        selected_1000 = goals_from_manifest(all_goals, Path(args.goal_manifest).expanduser().resolve())
        if args.goal_count is not None:
            selected_1000 = selected_1000[: args.goal_count]
    else:
        selected_1000 = select_goals(
            all_goals,
            seed=args.seed,
            limit=args.goal_count,
            start_index=0,
            prefer_near_half=args.prefer_target_goals_near_half,
        )
    selected = selected_1000[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    provider = make_provider(args, build_system_prompt())
    summaries = []
    for offset, goal in enumerate(selected, start=args.start_index):
        label = f"{offset:04d}_{goal['puzzle_key']}__goal_{goal['goal_index']:04d}"
        run_dir = output_root / label
        if args.skip_existing and (run_dir / "run_summary.json").exists():
            summaries.append(json.loads((run_dir / "run_summary.json").read_text()))
            continue
        runner = GoalProbeRun(
            goal=goal,
            goal_bank=goal_bank,
            provider=provider,
            vtools_root=vtools_root,
            run_dir=run_dir,
            python_bin=args.python_bin,
            max_attempts=args.max_attempts,
            frame_count=args.frame_count,
            seed=args.seed + offset,
            nonoverlap_distance=args.nonoverlap_distance,
        )
        summaries.append(runner.run())
    batch_summary = {
        "output_root": str(output_root),
        "backend": args.backend,
        "selected_goal_count": len(selected),
        "available_goal_count": len(all_goals),
        "nominal_goal_bank_count": len(selected_1000),
        "goal_manifest": str(Path(args.goal_manifest).expanduser().resolve()) if args.goal_manifest else None,
        "summaries": summaries,
    }
    (output_root / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2))
    print(json.dumps(batch_summary, indent=2))


if __name__ == "__main__":
    main()
