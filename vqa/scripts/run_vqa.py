#!/usr/bin/env python3
"""Resumable production runner for the 66-layout static VQA control."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OPENROUTER_MODELS = {
    "gpt": "openai/gpt-5",
    "gemini": "google/gemini-3.1-pro-preview",
    "qwen": "qwen/qwen3.6-plus",
}
DEFAULT_OPENAI_MODELS = {"gpt": "gpt-5"}
ENDPOINTS = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}
KEY_ENVS = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}
WRITE_LOCK = threading.Lock()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_artifact_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    parts = path.parts
    if "perception_vqa" in parts:
        suffix = parts[parts.index("perception_vqa") + 1 :]
        candidate = ROOT.joinpath(*suffix)
        if candidate.exists():
            return candidate
    if "temporal_rollout_vqa" in parts:
        suffix = parts[parts.index("temporal_rollout_vqa") + 1 :]
        candidate = ROOT.joinpath(*suffix)
        if candidate.exists():
            return candidate
    candidate = ROOT / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def answer_schema(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["response_schema"].get("type") == "coordinate_localization":
        required_targets = payload["response_schema"]["required_targets"]
        return {
            "type": "object",
            "properties": {
                target: {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
                for target in required_targets
            },
            "required": required_targets,
            "additionalProperties": False,
        }
    required_ids = payload["response_schema"]["required_question_ids"]
    return {
        "type": "object",
        "properties": {
            question_id: {"type": "string", "enum": list("ABCDEFGHI")}
            for question_id in required_ids
        },
        "required": required_ids,
        "additionalProperties": False,
    }


def build_request(
    payload: dict[str, Any],
    model_id: str,
    provider: str,
    seed: int,
    reasoning_effort: str = "minimal",
    max_tokens: int = 1200,
) -> dict[str, Any]:
    user_text = payload["user_text"]
    if payload.get("scene_json") is not None:
        scene_json_label = str(
            payload.get("scene_json_label")
            or "VISIBLE INITIAL SCENE JSON"
        )
        user_text += f"\n\n{scene_json_label}:\n" + json.dumps(
            payload["scene_json"],
            separators=(",", ":"),
            sort_keys=True,
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    image_paths = list(payload.get("image_paths") or [])
    if payload.get("image_path"):
        image_paths.insert(0, payload["image_path"])
    label_sequence = bool(payload.get("image_sequence_labels"))
    for image_index, image_path in enumerate(image_paths, start=1):
        if label_sequence:
            content.append(
                {
                    "type": "text",
                    "text": f"FRAME {image_index:02d} OF {len(image_paths):02d}",
                }
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url(portable_artifact_path(image_path)),
                    "detail": "high",
                },
            }
        )
    body: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": payload["system"]},
            {"role": "user", "content": content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vqa_answers",
                "strict": True,
                "schema": answer_schema(payload),
            },
        },
        "seed": seed,
    }
    if provider == "openai":
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
        body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if not model_id.startswith("openai/gpt-5"):
            body["temperature"] = 0.2
    return body


def extract_text(response: dict[str, Any]) -> str:
    content = response["choices"][0]["message"].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)


def parse_answer(text: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        return None, [f"invalid_json:{error.msg}"]
    if not isinstance(value, dict):
        return None, ["response_not_object"]
    coordinate_mode = payload["response_schema"].get("type") == "coordinate_localization"
    required = (
        payload["response_schema"]["required_targets"]
        if coordinate_mode
        else payload["response_schema"]["required_question_ids"]
    )
    required_set = set(required)
    observed_set = set(value)
    errors: list[str] = []
    missing = sorted(required_set - observed_set)
    extra = sorted(observed_set - required_set)
    if missing:
        errors.append("missing:" + ",".join(missing))
    if extra:
        errors.append("extra:" + ",".join(extra))
    normalized: dict[str, Any] = {}
    for question_id in required:
        if coordinate_mode:
            point = value.get(question_id)
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(
                    isinstance(coordinate, (int, float))
                    and not isinstance(coordinate, bool)
                    for coordinate in point
                )
            ):
                errors.append(f"invalid_point:{question_id}:{point}")
                normalized[question_id] = point
                continue
            normalized[question_id] = [float(point[0]), float(point[1])]
            if not all(0 <= coordinate <= 599 for coordinate in normalized[question_id]):
                errors.append(f"out_of_bounds:{question_id}:{point}")
            continue
        answer = str(value.get(question_id, "")).strip().upper()
        if not re.fullmatch(r"[A-I]", answer):
            errors.append(f"invalid_option:{question_id}:{answer}")
        normalized[question_id] = answer
    return normalized, errors


def sanitized_error_detail(detail: str, api_key: str) -> str:
    redacted = detail.replace(api_key, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_.*-]{8,}", "[REDACTED]", redacted)


def request_json(
    endpoint: str,
    api_key: str,
    body: dict[str, Any],
    provider: str,
    max_retries: int,
    timeout: float,
    retry_seed: int,
) -> tuple[dict[str, Any], int]:
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["X-Title"] = "VTools static perception VQA"
    rng = random.Random(retry_seed)
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(endpoint, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Provider returned malformed JSON: {error}"
                    ) from error
                time.sleep(min(30.0, 2**attempt + rng.random()))
                continue
            if not isinstance(parsed, dict) or not parsed.get("choices"):
                detail = sanitized_error_detail(
                    json.dumps(parsed, sort_keys=True)[:1000],
                    api_key,
                )
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Provider response omitted choices: {detail}"
                    )
                time.sleep(min(30.0, 2**attempt + rng.random()))
                continue
            return parsed, attempt
        except urllib.error.HTTPError as error:
            detail = sanitized_error_detail(
                error.read().decode("utf-8", errors="replace"),
                api_key,
            )
            retryable = (
                error.code == 429
                or 500 <= error.code < 600
                or (
                    error.code == 400
                    and (
                        "data_inspection_failed" in detail
                        or "inappropriate content" in detail
                    )
                )
            )
            if not retryable or attempt >= max_retries:
                raise RuntimeError(f"HTTP {error.code}: {detail[:1000]}") from error
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(30.0, 2**attempt + rng.random())
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt >= max_retries:
                raise RuntimeError(f"Network failure: {error}") from error
            time.sleep(min(30.0, 2**attempt + rng.random()))
    raise AssertionError("unreachable")


def run_call(
    call: dict[str, Any],
    provider: str,
    model_id: str,
    api_key: str,
    seed: int,
    max_retries: int,
    timeout: float,
    reasoning_effort: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = read_json(portable_artifact_path(call["payload_path"]))
    body = build_request(
        payload,
        model_id,
        provider,
        seed,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
    started = time.time()
    response, retry_count = request_json(
        ENDPOINTS[provider],
        api_key,
        body,
        provider,
        max_retries,
        timeout,
        seed ^ int(sha256_text(call["call_id"])[:8], 16),
    )
    raw_text = extract_text(response)
    parsed, parse_errors = parse_answer(raw_text, payload)
    return {
        "call_id": call["call_id"],
        "layout_id": call["layout_id"],
        "cell_id": call.get("cell_id"),
        "model_key": call["model_key"],
        "model_id_requested": model_id,
        "model_id_returned": response.get("model"),
        "provider": provider,
        "input_condition": call["input_condition"],
        "benchmark_cell_aliases": call["benchmark_cell_aliases"],
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "response_id": response.get("id"),
        "response_text": raw_text,
        "parsed_response": parsed,
        "parse_errors": parse_errors,
        "format_valid": not parse_errors,
        "usage": response.get("usage"),
        "retry_count": retry_count,
        "latency_seconds": round(time.time() - started, 3),
        "payload_sha256": sha256_text(json.dumps(payload, sort_keys=True)),
        "completed_at_unix": time.time(),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(ENDPOINTS), default="openrouter")
    parser.add_argument("--model-key", action="append", choices=("gpt", "gemini", "qwen"))
    parser.add_argument("--model", action="append", default=[], help="Override as key=model_id")
    parser.add_argument(
        "--condition",
        action="append",
        choices=(
            "image_only",
            "json_only",
            "image_coordinates",
            "temporal_rollout_images",
            "temporal_rollout_json",
        ),
    )
    parser.add_argument(
        "--call-manifest",
        type=Path,
        default=ROOT / "call_manifest.jsonl",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="minimal",
    )
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "vqa_seed2026")
    parser.add_argument("--execute", action="store_true", help="Required to make paid API calls")
    args = parser.parse_args()

    defaults = dict(
        DEFAULT_OPENROUTER_MODELS if args.provider == "openrouter" else DEFAULT_OPENAI_MODELS
    )
    for item in args.model:
        key, separator, model_id = item.partition("=")
        if not separator or key not in {"gpt", "gemini", "qwen"} or not model_id:
            raise ValueError(f"Invalid --model override: {item}")
        defaults[key] = model_id

    selected_keys = args.model_key or list(defaults)
    unavailable = [key for key in selected_keys if key not in defaults]
    if unavailable:
        raise ValueError(f"Provider {args.provider} has no default for: {unavailable}")
    conditions = set(args.condition or ("image_only", "json_only"))
    calls = [
        call
        for call in read_jsonl(args.call_manifest)
        if call["model_key"] in selected_keys and call["input_condition"] in conditions
    ]
    calls.sort(key=lambda row: (row["model_key"], row["layout_id"], row["input_condition"]))

    output_path = args.output_dir / args.provider / "responses.jsonl"
    completed: set[str] = set()
    if output_path.exists():
        completed = {
            row["call_id"]
            for row in read_jsonl(output_path)
            if "error" not in row and bool(row.get("format_valid"))
        }
    calls = [call for call in calls if call["call_id"] not in completed]
    if args.limit is not None:
        calls = calls[: args.limit]

    plan = {
        "provider": args.provider,
        "model_ids": {key: defaults[key] for key in selected_keys},
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "conditions": sorted(conditions),
        "remaining_calls": len(calls),
        "already_completed": len(completed),
        "output_path": str(output_path),
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute or not calls:
        return

    key_env = KEY_ENVS[args.provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"Set {key_env} in the environment before using --execute.")

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_call = {
            executor.submit(
                run_call,
                call,
                args.provider,
                defaults[call["model_key"]],
                api_key,
                args.seed,
                args.max_retries,
                args.timeout,
                args.reasoning_effort,
                args.max_tokens,
            ): call
            for call in calls
        }
        for future in concurrent.futures.as_completed(future_to_call):
            call = future_to_call[future]
            try:
                row = future.result()
            except Exception as error:
                failures += 1
                row = {
                    "call_id": call["call_id"],
                    "layout_id": call["layout_id"],
                    "cell_id": call.get("cell_id"),
                    "model_key": call["model_key"],
                    "model_id_requested": defaults[call["model_key"]],
                    "provider": args.provider,
                    "input_condition": call["input_condition"],
                    "seed": args.seed,
                    "error": f"{type(error).__name__}: {error}",
                    "completed_at_unix": time.time(),
                }
            append_jsonl(output_path, row)
            print(
                json.dumps(
                    {
                        "call_id": row["call_id"],
                        "valid": row.get("format_valid"),
                        "error": row.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        raise SystemExit(f"{failures} calls failed; rerun the same command to resume.")


if __name__ == "__main__":
    main()
