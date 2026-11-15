#!/usr/bin/env python3
"""Build the 132-cell asset index from the self-contained runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--screenshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        args.manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    environment_root = args.environment_root.expanduser().resolve()
    screenshot_root = args.screenshot_root.expanduser().resolve()
    run_names = sorted({str(goal["run_name"]) for goal in manifest["goals"]})
    if len(run_names) != 132:
        raise ValueError(f"Expected 132 cells, found {len(run_names)}")
    puzzles = {}
    for run_name in run_names:
        family, env_id, gravity = run_name.rsplit("_", 2)
        if gravity not in {"upward", "downward"}:
            raise ValueError(f"Invalid cell name: {run_name}")
        world = environment_root / family / f"{env_id}.json"
        screenshot = screenshot_root / f"{family}_{env_id}.png"
        if not world.exists() or not screenshot.exists():
            raise FileNotFoundError(
                f"Missing assets for {run_name}: {world}, {screenshot}"
            )
        puzzles[run_name] = {
            "world_path": str(world),
            "screenshot_path": str(screenshot),
            "world_sha256": sha256_file(world),
            "screenshot_sha256": sha256_file(screenshot),
            "candidate_copies": "self_contained_runtime_and_static_vqa_render",
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_unix": time.time(),
                "source_manifest": str(args.manifest.expanduser().resolve()),
                "asset_roots": [
                    str(environment_root),
                    str(screenshot_root),
                ],
                "puzzle_count": len(puzzles),
                "inconsistent_assets": [],
                "puzzles": puzzles,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "puzzle_count": len(puzzles),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
