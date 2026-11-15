#!/usr/bin/env python3
"""Analyze canonical attempt-1 prompt and seed robustness panels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


OUTCOMES = (
    "success",
    "action_quality_sigma72",
    "prediction_quality_sigma150",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def layout_id(puzzle_key: str) -> str:
    return puzzle_key.rsplit("_", 1)[0]


def load_panel(
    *,
    label: str,
    prompt_variant: str,
    coordinate_rows: Path,
    prediction_rows: Path,
) -> list[dict[str, Any]]:
    prediction = {
        (
            str(row["model_key"]),
            int(row.get("seed") or 2026),
            int(row["attempt"]),
            str(row["balanced_goal_id"]),
            str(row["condition"]),
        ): row
        for row in read_csv(prediction_rows)
    }
    output = []
    seen = set()
    for row in read_jsonl(coordinate_rows):
        if str(row["condition"]) != "neither" or int(row["attempt"]) != 1:
            continue
        key = (
            str(row["model_key"]),
            int(row["seed"]),
            int(row["attempt"]),
            str(row["goal_id"]),
            str(row["condition"]),
        )
        if key in seen:
            continue
        seen.add(key)
        pq = prediction.get(key)
        if pq is None:
            raise KeyError(f"Missing prediction row for {key}")
        output.append(
            {
                "panel": label,
                "prompt_variant": prompt_variant,
                "model_key": str(row["model_key"]),
                "seed": int(row["seed"]),
                "goal_id": str(row["goal_id"]),
                "puzzle_key": str(row["puzzle_key"]),
                "layout_id": layout_id(str(row["puzzle_key"])),
                "gravity": str(row["gravity"]),
                "success": int(bool(row["original_success"])),
                "action_quality_sigma72": float(
                    row["original_aq_sigma72"]
                ),
                "prediction_quality_sigma150": float(
                    pq["prediction_quality_sigma150"]
                ),
            }
        )
    return output


def paired_cluster_interval(
    pairs: list[dict[str, Any]],
    outcome: str,
    *,
    left_label: str,
    right_label: str,
    draws: int,
    seed: int,
) -> dict[str, float]:
    by_layout: dict[str, list[float]] = defaultdict(list)
    differences = []
    for pair in pairs:
        difference = float(pair[left_label][outcome]) - float(
            pair[right_label][outcome]
        )
        differences.append(difference)
        by_layout[str(pair[left_label]["layout_id"])].append(difference)
    layouts = sorted(by_layout)
    rng = random.Random(seed)
    replicates = []
    for _ in range(draws):
        sampled = [rng.choice(layouts) for _ in layouts]
        values = [
            value for sampled_layout in sampled
            for value in by_layout[sampled_layout]
        ]
        replicates.append(sum(values) / len(values))
    return {
        "difference": sum(differences) / len(differences),
        "ci95_lower": percentile(replicates, 0.025),
        "ci95_upper": percentile(replicates, 0.975),
    }


def paired_rows(
    left: Iterable[dict[str, Any]],
    right: Iterable[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
) -> list[dict[str, Any]]:
    def index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (str(row["model_key"]), str(row["goal_id"])): row
            for row in rows
        }

    left_index = index(left)
    right_index = index(right)
    if set(left_index) != set(right_index):
        raise RuntimeError(
            f"Unmatched panels: left_only={len(set(left_index)-set(right_index))}, "
            f"right_only={len(set(right_index)-set(left_index))}"
        )
    return [
        {
            left_label: left_index[key],
            right_label: right_index[key],
        }
        for key in sorted(left_index)
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    main = load_panel(
        label="full_prompt_seed2026",
        prompt_variant="full",
        coordinate_rows=args.main_coordinate_rows,
        prediction_rows=args.main_prediction_rows,
    )
    compact = load_panel(
        label="compact_prompt_seed2026",
        prompt_variant="compact",
        coordinate_rows=args.compact_coordinate_rows,
        prediction_rows=args.compact_prediction_rows,
    )
    seeds = load_panel(
        label="full_prompt_seed42_123",
        prompt_variant="full",
        coordinate_rows=args.seed_coordinate_rows,
        prediction_rows=args.seed_prediction_rows,
    )
    canonical_ids = {str(row["goal_id"]) for row in compact}
    main = [row for row in main if str(row["goal_id"]) in canonical_ids]
    all_rows = main + compact + seeds

    prompt_rows = []
    for model_index, model in enumerate(
        sorted({row["model_key"] for row in compact})
    ):
        full_model = [row for row in main if row["model_key"] == model]
        compact_model = [
            row for row in compact if row["model_key"] == model
        ]
        pairs = paired_rows(
            full_model,
            compact_model,
            left_label="full",
            right_label="compact",
        )
        for outcome_index, outcome in enumerate(OUTCOMES):
            interval = paired_cluster_interval(
                pairs,
                outcome,
                left_label="full",
                right_label="compact",
                draws=args.bootstrap_draws,
                seed=args.seed + 100 * model_index + outcome_index,
            )
            prompt_rows.append(
                {
                    "model_key": model,
                    "outcome": outcome,
                    "pairs": len(pairs),
                    "full_mean": sum(
                        float(pair["full"][outcome]) for pair in pairs
                    )
                    / len(pairs),
                    "compact_mean": sum(
                        float(pair["compact"][outcome]) for pair in pairs
                    )
                    / len(pairs),
                    "full_minus_compact": interval["difference"],
                    "ci95_lower": interval["ci95_lower"],
                    "ci95_upper": interval["ci95_upper"],
                }
            )

    seed_summary = []
    seed_contrasts = []
    rich = main + seeds
    models = sorted({row["model_key"] for row in rich})
    for model_index, model in enumerate(models):
        for seed_value in (42, 123, 2026):
            rows = [
                row
                for row in rich
                if row["model_key"] == model and row["seed"] == seed_value
            ]
            if len(rows) != 132:
                raise RuntimeError(
                    f"Expected 132 rows for {model} seed {seed_value}, "
                    f"found {len(rows)}"
                )
            for outcome in OUTCOMES:
                seed_summary.append(
                    {
                        "model_key": model,
                        "seed": seed_value,
                        "outcome": outcome,
                        "units": len(rows),
                        "mean": sum(float(row[outcome]) for row in rows)
                        / len(rows),
                    }
                )
        reference = [
            row
            for row in rich
            if row["model_key"] == model and row["seed"] == 2026
        ]
        for seed_index, seed_value in enumerate((42, 123)):
            comparison = [
                row
                for row in rich
                if row["model_key"] == model and row["seed"] == seed_value
            ]
            pairs = paired_rows(
                comparison,
                reference,
                left_label="comparison",
                right_label="reference",
            )
            for outcome_index, outcome in enumerate(OUTCOMES):
                interval = paired_cluster_interval(
                    pairs,
                    outcome,
                    left_label="comparison",
                    right_label="reference",
                    draws=args.bootstrap_draws,
                    seed=(
                        args.seed
                        + model_index * 100
                        + seed_index * 10
                        + outcome_index
                    ),
                )
                seed_contrasts.append(
                    {
                        "model_key": model,
                        "comparison_seed": seed_value,
                        "reference_seed": 2026,
                        "outcome": outcome,
                        "pairs": len(pairs),
                        "comparison_minus_reference": interval["difference"],
                        "ci95_lower": interval["ci95_lower"],
                        "ci95_upper": interval["ci95_upper"],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "attempt1_panel_rows.csv", all_rows)
    write_csv(args.output_dir / "prompt_variant_contrasts.csv", prompt_rows)
    write_csv(args.output_dir / "seed_summary.csv", seed_summary)
    write_csv(args.output_dir / "seed_contrasts.csv", seed_contrasts)
    report = {
        "canonical_units_per_model_seed_prompt": 132,
        "prompt_variant_contrasts": prompt_rows,
        "seed_summary": seed_summary,
        "seed_contrasts": seed_contrasts,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "interpretation": (
            "Attempt-1 robustness only; these panels do not alter or re-enter "
            "the confirmatory solver conversations."
        ),
    }
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("main", "compact", "seed"):
        parser.add_argument(
            f"--{prefix}-coordinate-rows",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{prefix}-prediction-rows",
            type=Path,
            required=True,
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
