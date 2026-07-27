#!/usr/bin/env python3
"""Export a compact, de-identified snapshot of completed model units.

The experiment directories contain regenerable rollout media, simulator traces,
absolute worker paths, and recovery state.  This exporter keeps the material
needed to audit model behavior and recompute the paper measures:

* unit manifests and completion summaries;
* exact rendered prompts and raw responses;
* parsed actions and prospective coordinate predictions;
* simulator success labels and terminal prediction truth;
* transfer-sidecar results when a sidecar is complete; and
* provider model, token, latency, and cost metadata.

Only directories containing ``unit_summary.json`` are exported.  Partial
recovery directories are counted in the coverage report but never serialized
as completed observations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ABSOLUTE_PATH_RE = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\\\)")
SECRET_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{16,}|"
    r"Bearer\s+[A-Za-z0-9._-]{16,})"
)

DROP_KEYS = {
    "environment_json",
    "initial_screenshot",
    "unit_dir",
    "sampled_frames",
    "image_paths",
    "trace_path",
    "structured_trace",
    "rollout_video",
    "response_id",
    "response_extra",
    "event_graph",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if key not in DROP_KEYS
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if SECRET_RE.search(value):
            return SECRET_RE.sub("[REDACTED]", value)
        if ABSOLUTE_PATH_RE.search(value):
            # Prompts and responses should never contain worker paths.  Path
            # metadata is removed above; this is a final defensive guard.
            return "[REDACTED_ABSOLUTE_PATH]"
    return value


def relative_label(path: Path, unit_dir: Path) -> str:
    return path.relative_to(unit_dir).as_posix()


def collect_provider_calls(unit_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(unit_dir.rglob("provider_calls.jsonl")):
        if "transfer_sidecars" in path.parts:
            continue
        for row in read_jsonl(path):
            clean = sanitize(row)
            clean["source"] = relative_label(path, unit_dir)
            records.append(clean)
    return records


def compact_attempt_record(record: dict[str, Any]) -> dict[str, Any]:
    clean = sanitize(record)
    truth = clean.get("truth")
    if isinstance(truth, dict):
        keep = {
            "valid",
            "answer",
            "matched_signature",
            "endpoint_goal_succeeded_original_definition",
            "canonical_dwell_s",
            "canonical_goal_intervals",
            "terminal_persistence_s",
            "terminal_persistence_sample_count",
            "endpoint_final_state_signatures",
            "persistent_final_state_signatures",
            "prediction_endpoints",
        }
        clean["truth"] = {key: truth[key] for key in keep if key in truth}
    render = clean.get("rollout_render")
    if isinstance(render, dict):
        keep = {
            "method",
            "trace_sha256",
            "source_pose_sample_count",
            "sample_indices",
            "sample_times_s",
            "frame_count",
            "frame_dimensions_px",
        }
        clean["rollout_render"] = {
            key: render[key] for key in keep if key in render
        }
    return clean


def collect_attempt_records(
    unit_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    excluded: list[str] = []
    for path in sorted(unit_dir.rglob("attempt_record.json")):
        if "transfer_sidecars" in path.parts:
            continue
        try:
            row = compact_attempt_record(read_json(path))
        except (json.JSONDecodeError, OSError):
            excluded.append(relative_label(path, unit_dir))
            continue
        row["source"] = relative_label(path, unit_dir)
        records.append(row)
    return records, excluded


def collect_sidecars(unit_dir: Path) -> list[dict[str, Any]]:
    root = unit_dir / "transfer_sidecars"
    if not (root / ".incremental_complete").exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for name in ("probe_selection.json", "summary.json"):
            path = root / name
            if path.exists():
                records.append(
                    {
                        "source": relative_label(path, unit_dir),
                        "record": sanitize(read_json(path)),
                    }
                )
        for path in sorted(root.rglob("result.json")):
            records.append(
                {
                    "source": relative_label(path, unit_dir),
                    "record": sanitize(read_json(path)),
                }
            )
        for path in sorted(root.rglob("provider_calls.jsonl")):
            for row in read_jsonl(path):
                records.append(
                    {
                        "source": relative_label(path, unit_dir),
                        "record": sanitize(row),
                    }
                )
    except (json.JSONDecodeError, OSError):
        # A sidecar completion marker can briefly precede a fully flushed file
        # on an active worker. Exclude the entire sidecar bundle rather than
        # publishing a partial branch.
        return []
    return records


def unit_record(summary_path: Path, result_root: Path) -> dict[str, Any]:
    unit_dir = summary_path.parent
    summary = sanitize(read_json(summary_path))
    manifest_path = unit_dir / "unit_manifest.json"
    manifest = sanitize(read_json(manifest_path)) if manifest_path.exists() else None
    attempt_records, excluded_attempt_records = collect_attempt_records(unit_dir)
    return {
        "schema_version": 1,
        "source_unit": unit_dir.relative_to(result_root).as_posix(),
        "manifest": manifest,
        "summary": summary,
        "provider_calls": collect_provider_calls(unit_dir),
        "attempt_records": attempt_records,
        "excluded_unreadable_attempt_records": excluded_attempt_records,
        "sidecar_records": collect_sidecars(unit_dir),
    }


def write_gzip_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            for record in records:
                gz.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_unit_dirs(result_root: Path) -> tuple[list[Path], int]:
    complete = sorted(result_root.glob("*/seed_*/*/unit_summary.json"))
    candidate_dirs = {
        path.parent
        for path in result_root.glob("*/seed_*/*/unit_manifest.json")
    }
    partial_count = len(candidate_dirs - {path.parent for path in complete})
    return complete, partial_count


def export(args: argparse.Namespace) -> dict[str, Any]:
    result_root = args.result_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    summaries, partial_count = discover_unit_dirs(result_root)
    by_model: dict[str, list[Path]] = defaultdict(list)
    for path in summaries:
        data = read_json(path)
        by_model[str(data["model_key"])].append(path)

    shard_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for model_key, paths in sorted(by_model.items()):
        canonical_count = 0
        sidecar_count = 0
        condition_counts: Counter[str] = Counter()
        for path in paths:
            summary = read_json(path)
            if summary.get("goal", {}).get("category") == "canonical":
                canonical_count += 1
            condition_counts.update(summary.get("conditions", {}).keys())
            if (path.parent / "transfer_sidecars" / "summary.json").exists():
                sidecar_count += 1

        for shard_index, start in enumerate(range(0, len(paths), args.shard_size)):
            shard_paths = paths[start : start + args.shard_size]
            shard_path = (
                output_root
                / "data"
                / model_key
                / f"units-{shard_index:04d}.jsonl.gz"
            )
            write_gzip_jsonl(
                shard_path,
                (unit_record(path, result_root) for path in shard_paths),
            )
            shard_rows.append(
                {
                    "model_key": model_key,
                    "path": shard_path.relative_to(output_root).as_posix(),
                    "unit_count": len(shard_paths),
                    "bytes": shard_path.stat().st_size,
                    "sha256": sha256_file(shard_path),
                }
            )

        coverage_rows.append(
            {
                "model_key": model_key,
                "completed_units": len(paths),
                "expected_units": args.expected_units,
                "completion_percent": round(100 * len(paths) / args.expected_units, 3),
                "canonical_units": canonical_count,
                "units_with_transfer_sidecars": sidecar_count,
                "full_condition_units": condition_counts["full"],
                "neither_condition_units": condition_counts["neither"],
                "trace_status_condition_units": condition_counts["trace_status"],
                "status_only_condition_units": condition_counts["status_only"],
            }
        )

    with (output_root / "coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    report = {
        "schema_version": 1,
        "snapshot_id": args.snapshot_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "completed unit_summary.json records only",
        "expected_units_per_main_model": args.expected_units,
        "completed_unit_count": len(summaries),
        "partial_candidate_directory_count": partial_count,
        "coverage": coverage_rows,
        "shards": shard_rows,
        "exclusions": [
            "partial recovery directories without unit_summary.json",
            "rollout PNGs and videos",
            "regenerable full simulator traces",
            "absolute worker filesystem paths",
            "provider response identifiers and credentials",
        ],
    }
    (output_root / "snapshot_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def check(output_root: Path) -> None:
    manifest = read_json(output_root / "snapshot_manifest.json")
    observed = 0
    for shard in manifest["shards"]:
        path = output_root / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        if path.stat().st_size != shard["bytes"]:
            raise SystemExit(f"Size mismatch: {path}")
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                serialized = json.dumps(record)
                if ABSOLUTE_PATH_RE.search(serialized):
                    raise SystemExit(f"Absolute path found in {path}")
                if SECRET_RE.search(serialized):
                    raise SystemExit(f"Possible secret found in {path}")
                observed += 1
    if observed != manifest["completed_unit_count"]:
        raise SystemExit(
            f"Expected {manifest['completed_unit_count']} units, observed {observed}"
        )
    print(
        f"validated {observed} completed units across "
        f"{len(manifest['shards'])} shards"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", default="seed2026_partial_2026-07-27")
    parser.add_argument("--expected-units", type=int, default=1692)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.result_root is None:
        parser.error("--result-root is required unless --check is used")
    return args


def main() -> None:
    args = parse_args()
    if args.check:
        check(args.output_root.resolve())
    else:
        report = export(args)
        print(json.dumps(report["coverage"], indent=2))


if __name__ == "__main__":
    main()
