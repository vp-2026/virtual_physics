#!/usr/bin/env python3
"""Cluster-bootstrap the successful-minus-failed prospective-PQ contrast.

The input is the action-eligible joined table produced by
``analyze_action_prediction_coupling.py``. Each bootstrap replicate resamples
base layouts with replacement and retains every attempted action belonging to
the sampled layouts. This preserves the repeated attempts, goals, and gravity
variants nested within a base layout.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_group(
    rows: list[dict[str, str]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, int, int]:
    by_layout: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_layout[row["base_layout"]].append(row)

    layouts = sorted(by_layout)
    success_sum = np.zeros(len(layouts), dtype=float)
    success_n = np.zeros(len(layouts), dtype=float)
    failure_sum = np.zeros(len(layouts), dtype=float)
    failure_n = np.zeros(len(layouts), dtype=float)
    for index, layout in enumerate(layouts):
        for row in by_layout[layout]:
            pq = float(row["pq_sigma150"])
            if int(row["success"]):
                success_sum[index] += pq
                success_n[index] += 1
            else:
                failure_sum[index] += pq
                failure_n[index] += 1

    estimate = (
        success_sum.sum() / success_n.sum()
        - failure_sum.sum() / failure_n.sum()
    )
    draws = np.empty(replicates, dtype=float)
    chunk_size = 2_000
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        sampled = rng.integers(
            0,
            len(layouts),
            size=(stop - start, len(layouts)),
        )
        sampled_success_n = success_n[sampled].sum(axis=1)
        sampled_failure_n = failure_n[sampled].sum(axis=1)
        valid = (sampled_success_n > 0) & (sampled_failure_n > 0)
        chunk = np.full(stop - start, np.nan, dtype=float)
        chunk[valid] = (
            success_sum[sampled][valid].sum(axis=1)
            / sampled_success_n[valid]
            - failure_sum[sampled][valid].sum(axis=1)
            / sampled_failure_n[valid]
        )
        draws[start:stop] = chunk

    valid_draws = draws[np.isfinite(draws)]
    if len(valid_draws) != replicates:
        raise RuntimeError(
            f"Only {len(valid_draws)} of {replicates} replicates contained "
            "both successful and failed actions."
        )
    low, high = np.quantile(valid_draws, [0.025, 0.975])
    return (
        float(estimate),
        float(low),
        float(high),
        len(layouts),
        int(success_n.sum()),
        int(failure_n.sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = read_rows(args.attempt_rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["condition"])].append(row)

    rng = np.random.default_rng(args.seed)
    output = []
    for (model_key, condition), group in sorted(grouped.items()):
        estimate, low, high, layouts, success_n, failure_n = bootstrap_group(
            group,
            replicates=args.replicates,
            rng=rng,
        )
        output.append(
            {
                "model_key": model_key,
                "condition": condition,
                "contrast": "mean_pq150_success_minus_failure",
                "estimate": estimate,
                "estimate_points": 100.0 * estimate,
                "ci95_low": low,
                "ci95_high": high,
                "ci95_low_points": 100.0 * low,
                "ci95_high_points": 100.0 * high,
                "base_layout_clusters": layouts,
                "successful_action_rows": success_n,
                "failed_action_rows": failure_n,
                "bootstrap_replicates": args.replicates,
                "bootstrap_seed": args.seed,
            }
        )
    write_rows(args.output, output)


if __name__ == "__main__":
    main()
