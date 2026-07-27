#!/usr/bin/env python3
"""Attach the exact model-run initial screenshot to each numbered cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def check_release(environment_root: Path) -> None:
    rows = [
        json.loads(line)
        for line in (environment_root / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 132:
        raise ValueError(f"Expected 132 index rows, found {len(rows)}")
    for row in rows:
        cell_dir = environment_root / "cells" / str(row["environment_id"])
        for name in ("environment.json", "tool.json", "initial_scene.png"):
            if not (cell_dir / name).is_file():
                raise FileNotFoundError(cell_dir / name)
        if sha256(cell_dir / "environment.json") != row["environment_sha256"]:
            raise ValueError(f"Environment hash mismatch: {cell_dir}")
        if (
            sha256(cell_dir / "initial_scene.png")
            != row["initial_scene_sha256"]
        ):
            raise ValueError(f"Initial-scene hash mismatch: {cell_dir}")
    print("initial-scene check passed: 132 SHA-verified cells")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--screenshots-root", type=Path)
    parser.add_argument("--run-asset-index", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    environment_root = (
        args.environment_root.resolve()
        if args.environment_root
        else repo_root / "132_base_environments"
    )
    if args.check:
        check_release(environment_root)
        return
    if args.screenshots_root is None or args.run_asset_index is None:
        parser.error(
            "--screenshots-root and --run-asset-index are required unless "
            "--check is used"
        )
    screenshots_root = args.screenshots_root.resolve()
    run_assets = json.loads(args.run_asset_index.read_text())["puzzles"]
    rows: list[dict[str, Any]] = []

    for cell_dir in sorted((environment_root / "cells").iterdir()):
        metadata_path = cell_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        cell_id = str(metadata["legacy_cell_id"])
        layout_id = str(metadata["legacy_layout_id"])
        source = screenshots_root / f"{layout_id}.png"
        expected = str(run_assets[cell_id]["screenshot_sha256"])
        actual = sha256(source)
        if actual != expected:
            raise ValueError(
                f"Screenshot hash mismatch for {cell_id}: {actual} != {expected}"
            )
        destination = cell_dir / "initial_scene.png"
        shutil.copy2(source, destination)
        metadata["initial_scene_path"] = "initial_scene.png"
        metadata["initial_scene_sha256"] = actual
        write_json(metadata_path, metadata)
        rows.append(metadata)

    if len(rows) != 132:
        raise ValueError(f"Expected 132 cells, found {len(rows)}")

    with (environment_root / "index.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with (environment_root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
