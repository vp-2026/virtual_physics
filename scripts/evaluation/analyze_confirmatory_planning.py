#!/usr/bin/env python3
"""Frozen confirmatory planning analysis for the seed-2026 VTools run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PRIMARY_ARMS = ("full", "frames_only", "status_only", "neither")
ALL_ARMS = (*PRIMARY_ARMS, "trace_status")
CONTRASTS = {
    "full_minus_neither": {"full": 1.0, "neither": -1.0},
    "rollout_main_effect": {
        "full": 0.5,
        "frames_only": 0.5,
        "status_only": -0.5,
        "neither": -0.5,
    },
    "status_main_effect": {
        "full": 0.5,
        "status_only": 0.5,
        "frames_only": -0.5,
        "neither": -0.5,
    },
    "rollout_x_status": {
        "full": 1.0,
        "frames_only": -1.0,
        "status_only": -1.0,
        "neither": 1.0,
    },
    "trace_minus_full": {"trace_status": 1.0, "full": -1.0},
    "trace_minus_status_only": {
        "trace_status": 1.0,
        "status_only": -1.0,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else math.nan


def layout_macro_values(
    rows: Sequence[dict[str, Any]],
) -> list[float]:
    by_layout_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_layout_model[
            (str(row["layout_id"]), str(row["model_key"]))
        ].append(float(row["value"]))
    by_layout: dict[str, list[float]] = defaultdict(list)
    for (layout, _model), values in by_layout_model.items():
        by_layout[layout].append(mean(values))
    return [
        mean(model_values)
        for _layout, model_values in sorted(by_layout.items())
    ]


def bootstrap_interval(
    layout_values: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    if not layout_values:
        return math.nan, math.nan
    rng = random.Random(seed)
    values = list(layout_values)
    replicates = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    ]
    return quantile(replicates, 0.025), quantile(replicates, 0.975)


def grouping_keys(row: dict[str, Any]) -> Iterable[tuple[str, str]]:
    yield "pooled", "all"
    yield "model", str(row["model_key"])
    yield "gravity", str(row["gravity"])
    yield "model_x_gravity", (
        f"{row['model_key']}|{row['gravity']}"
    )
    yield "category", str(row["category"])
    yield "canonical", (
        "canonical" if row["is_canonical"] else "noncanonical"
    )


def collect_unit_rows(
    result_root: Path,
    allowed_goal_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_root.rglob("unit_summary.json")):
        summary = read_json(path)
        goal = summary["goal"]
        goal_id = str(goal["balanced_goal_id"])
        if allowed_goal_ids is not None and goal_id not in allowed_goal_ids:
            continue
        layout_id = f"{goal['family']}_{goal['env_id']}"
        shared_success = bool(
            summary["shared_attempt_1"]["goal_succeeded"]
        )
        paper_budget = int(goal.get("attempt_limit_manifest") or 8)
        for arm, result in summary["conditions"].items():
            first_success = result.get("first_success_attempt")
            row = {
                "model_key": str(summary["model_key"]),
                "model_id": str(summary["model_id"]),
                "seed": int(summary["seed"]),
                "goal_id": goal_id,
                "layout_id": layout_id,
                "puzzle_key": str(goal["puzzle_key"]),
                "gravity": str(goal["condition"]),
                "category": str(
                    goal.get("category_5")
                    or goal.get("category")
                    or ""
                ),
                "is_canonical": (
                    str(goal.get("source") or "")
                    == "canonical_world_gcond"
                ),
                "arm": str(arm),
                "shared_attempt_1_success": shared_success,
                "attempt_count": int(result["attempt_count"]),
                "first_success_attempt": (
                    int(first_success)
                    if first_success is not None
                    else None
                ),
                "solve_by_8": int(first_success is not None),
                "paper_comparable_budget": paper_budget,
                "solve_by_paper_budget": int(
                    first_success is not None
                    and int(first_success) <= paper_budget
                ),
                "attempts_used": (
                    int(first_success)
                    if first_success is not None
                    else 9
                ),
                "attempts_used_paper_budget": (
                    int(first_success)
                    if first_success is not None
                    and int(first_success) <= paper_budget
                    else paper_budget + 1
                ),
                "blocked_action_count": int(
                    result.get("blocked_action_count") or 0
                ),
                "duplicate_action_count": int(
                    result.get("duplicate_action_count") or 0
                ),
                "completion_reason": str(
                    result.get("completion_reason") or ""
                ),
            }
            for attempt in range(1, 9):
                row[f"solve_by_{attempt}"] = int(
                    first_success is not None
                    and int(first_success) <= attempt
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No unit summaries under {result_root}")
    return rows


def contrast_observations(
    unit_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_unit: dict[
        tuple[str, int, str], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in unit_rows:
        by_unit[
            (
                str(row["model_key"]),
                int(row["seed"]),
                str(row["goal_id"]),
            )
        ][str(row["arm"])] = row
    outcomes = (
        "solve_by_8",
        "solve_by_paper_budget",
        "attempts_used",
        "attempts_used_paper_budget",
        *(f"solve_by_{attempt}" for attempt in range(2, 8)),
    )
    rows = []
    for _unit, arms in sorted(by_unit.items()):
        reference = next(iter(arms.values()))
        if reference["shared_attempt_1_success"]:
            continue
        for contrast, weights in CONTRASTS.items():
            if not set(weights).issubset(arms):
                continue
            for outcome in outcomes:
                rows.append(
                    {
                        "model_key": reference["model_key"],
                        "seed": reference["seed"],
                        "goal_id": reference["goal_id"],
                        "layout_id": reference["layout_id"],
                        "puzzle_key": reference["puzzle_key"],
                        "gravity": reference["gravity"],
                        "category": reference["category"],
                        "is_canonical": reference["is_canonical"],
                        "contrast": contrast,
                        "outcome": outcome,
                        "value": sum(
                            weight * float(arms[arm][outcome])
                            for arm, weight in weights.items()
                        ),
                    }
                )
    return rows


def summarize_contrasts(
    observations: Sequence[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in observations:
        for scope, level in grouping_keys(row):
            groups[
                (
                    scope,
                    level,
                    str(row["contrast"]),
                    str(row["outcome"]),
                )
            ].append(row)
    summaries = []
    for (scope, level, contrast, outcome), rows in sorted(
        groups.items()
    ):
        layout_values = layout_macro_values(rows)
        low, high = bootstrap_interval(
            layout_values,
            draws=draws,
            seed=(
                seed
                ^ int(
                    hashlib_seed(
                        f"{scope}|{level}|{contrast}|{outcome}"
                    )
                )
            ),
        )
        summaries.append(
            {
                "scope": scope,
                "level": level,
                "contrast": contrast,
                "outcome": outcome,
                "paired_units": len(rows),
                "base_layouts": len(
                    {str(row["layout_id"]) for row in rows}
                ),
                "goal_micro_estimate": mean(
                    float(row["value"]) for row in rows
                ),
                "equal_model_layout_macro_estimate": mean(layout_values),
                "layout_cluster_bootstrap_95_low": low,
                "layout_cluster_bootstrap_95_high": high,
            }
        )
    return summaries


def hashlib_seed(text: str) -> int:
    import hashlib

    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def arm_curve_rows(
    unit_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [
        row for row in unit_rows if not row["shared_attempt_1_success"]
    ]
    groups: dict[
        tuple[str, str, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in eligible:
        for scope, level in grouping_keys(row):
            for attempt in range(2, 9):
                groups[
                    (scope, level, str(row["arm"]), attempt)
                ].append(row)
    output = []
    for (scope, level, arm, attempt), rows in sorted(groups.items()):
        output.append(
            {
                "scope": scope,
                "level": level,
                "arm": arm,
                "attempt": attempt,
                "eligible_units": len(rows),
                "solve_rate": mean(
                    float(row[f"solve_by_{attempt}"]) for row in rows
                ),
                "mean_attempts_used": mean(
                    float(row["attempts_used"]) for row in rows
                ),
            }
        )
    return output


def exact_paired_primary_test(
    unit_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[
        tuple[str, int, str], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in unit_rows:
        if row["shared_attempt_1_success"]:
            continue
        by_unit[
            (
                str(row["model_key"]),
                int(row["seed"]),
                str(row["goal_id"]),
            )
        ][str(row["arm"])] = row
    full_only = 0
    neither_only = 0
    paired = 0
    for arms in by_unit.values():
        if "full" not in arms or "neither" not in arms:
            continue
        paired += 1
        full = int(arms["full"]["solve_by_8"])
        neither = int(arms["neither"]["solve_by_8"])
        full_only += int(full == 1 and neither == 0)
        neither_only += int(full == 0 and neither == 1)
    discordant = full_only + neither_only
    if discordant:
        tail = min(full_only, neither_only)
        probability = min(
            1.0,
            2.0
            * sum(
                math.comb(discordant, count)
                for count in range(tail + 1)
            )
            / (2.0**discordant),
        )
    else:
        probability = 1.0
    return {
        "test": "exact two-sided McNemar sign test on discordant model-goal pairs",
        "outcome": "solve_by_8",
        "contrast": "full_minus_neither",
        "eligibility": "shared_attempt_1_failed",
        "paired_model_goal_units": paired,
        "full_only_successes": full_only,
        "neither_only_successes": neither_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": probability,
        "uncertainty_note": (
            "This unit-level test is retained as a sensitivity analysis. It "
            "does not account for dependence among goals in one layout; the "
            "confirmatory p-value is the base-layout sign-flip test."
        ),
    }


def cluster_sign_flip_primary_test(
    observations: Sequence[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Test the primary paired contrast at the 66-layout cluster level."""
    primary_rows = [
        row
        for row in observations
        if row["contrast"] == "full_minus_neither"
        and row["outcome"] == "solve_by_8"
    ]
    layout_values = layout_macro_values(primary_rows)
    if not layout_values:
        raise RuntimeError("No primary observations for cluster sign-flip test")
    observed = mean(layout_values)
    threshold = abs(observed) - 1e-15
    cluster_count = len(layout_values)
    if cluster_count <= 20:
        total = 1 << cluster_count
        extreme = 0
        for bits in range(total):
            permuted = mean(
                value if (bits >> index) & 1 else -value
                for index, value in enumerate(layout_values)
            )
            extreme += int(abs(permuted) >= threshold)
        p_value = extreme / total
        method = "exact_enumeration"
        evaluated = total
        mc_se = 0.0
    else:
        rng = random.Random(seed)
        extreme = 0
        for _ in range(draws):
            bits = rng.getrandbits(cluster_count)
            permuted = mean(
                value if (bits >> index) & 1 else -value
                for index, value in enumerate(layout_values)
            )
            extreme += int(abs(permuted) >= threshold)
        # The +1 correction includes the observed assignment and prevents a
        # zero Monte Carlo p-value.
        p_value = (extreme + 1) / (draws + 1)
        method = "monte_carlo"
        evaluated = draws
        mc_se = math.sqrt(
            p_value * (1.0 - p_value) / (draws + 1)
        )
    return {
        "test": "two-sided paired sign-flip test of layout-level mean effects",
        "outcome": "solve_by_8",
        "contrast": "full_minus_neither",
        "eligibility": "shared_attempt_1_failed",
        "cluster_unit": "base_layout",
        "base_layouts": cluster_count,
        "equal_model_layout_macro_estimate": observed,
        "method": method,
        "sign_assignments_evaluated": evaluated,
        "two_sided_p": p_value,
        "monte_carlo_standard_error": mc_se,
        "seed": seed,
    }


def _gravity_interactions_for_rows(
    observations: Sequence[dict[str, Any]],
    *,
    draws: int,
    seed: int,
    analysis_stratum: str,
    interpretation: str,
) -> list[dict[str, Any]]:
    by_key: dict[
        tuple[str, str, str, str], list[float]
    ] = defaultdict(list)
    for row in observations:
        by_key[
            (
                str(row["model_key"]),
                str(row["layout_id"]),
                str(row["contrast"]),
                str(row["outcome"]),
                str(row["gravity"]),
            )
        ].append(float(row["value"]))
    interaction_rows = []
    prefixes = {
        key[:4] for key in by_key
    }
    for model, layout, contrast, outcome in sorted(prefixes):
        upward = by_key.get(
            (model, layout, contrast, outcome, "upward")
        )
        downward = by_key.get(
            (model, layout, contrast, outcome, "downward")
        )
        if not upward or not downward:
            continue
        interaction_rows.append(
            {
                "model_key": model,
                "layout_id": layout,
                "contrast": contrast,
                "outcome": outcome,
                "value": mean(upward) - mean(downward),
            }
        )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in interaction_rows:
        groups[
            ("all", row["contrast"], row["outcome"])
        ].append(row)
        groups[
            (row["model_key"], row["contrast"], row["outcome"])
        ].append(row)
    output = []
    for (model, contrast, outcome), rows in sorted(groups.items()):
        values = [float(row["value"]) for row in rows]
        low, high = bootstrap_interval(
            values,
            draws=draws,
            seed=seed ^ hashlib_seed(
                f"gravity|{model}|{contrast}|{outcome}"
            ),
        )
        output.append(
            {
                "analysis_stratum": analysis_stratum,
                "interpretation": interpretation,
                "model_key": model,
                "contrast": contrast,
                "outcome": outcome,
                "base_layouts": len(rows),
                "upward_minus_downward": mean(values),
                "layout_cluster_bootstrap_95_low": low,
                "layout_cluster_bootstrap_95_high": high,
            }
        )
    return output


def gravity_interactions(
    observations: Sequence[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep canonical paired and diverse unpaired gravity summaries distinct."""
    canonical = [
        row for row in observations if bool(row.get("is_canonical"))
    ]
    diverse = [
        row for row in observations if not bool(row.get("is_canonical"))
    ]
    return [
        *_gravity_interactions_for_rows(
            canonical,
            draws=draws,
            seed=seed ^ hashlib_seed("canonical_paired"),
            analysis_stratum="canonical_exact_layout_pair",
            interpretation=(
                "paired upward-minus-downward contrast within each of the 66 "
                "canonical base layouts"
            ),
        ),
        *_gravity_interactions_for_rows(
            diverse,
            draws=draws,
            seed=seed ^ hashlib_seed("diverse_unpaired"),
            analysis_stratum="diverse_layout_adjusted_unpaired",
            interpretation=(
                "descriptive layout-adjusted contrast; noncanonical goals are "
                "not asserted to be exact upward/downward pairs"
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-goal-ids-from-manifest",
        type=Path,
        default=None,
        help=(
            "Optional manifest defining a frozen sensitivity subset. Results "
            "for other goal IDs are ignored without making additional calls."
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument(
        "--permutation-draws",
        type=int,
        default=100000,
        help=(
            "Monte Carlo sign assignments for the base-layout primary test; "
            "small synthetic analyses with at most 20 layouts are enumerated."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.result_root = args.result_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.include_goal_ids_from_manifest is not None:
        args.include_goal_ids_from_manifest = (
            args.include_goal_ids_from_manifest.expanduser().resolve()
        )
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    if args.permutation_draws < 1:
        parser.error("--permutation-draws must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_goal_ids = None
    if args.include_goal_ids_from_manifest is not None:
        subset_payload = read_json(args.include_goal_ids_from_manifest)
        allowed_goal_ids = {
            str(goal["balanced_goal_id"])
            for goal in subset_payload["goals"]
        }
    unit_rows = collect_unit_rows(
        args.result_root,
        allowed_goal_ids=allowed_goal_ids,
    )
    observations = contrast_observations(unit_rows)
    summaries = summarize_contrasts(
        observations,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    curves = arm_curve_rows(unit_rows)
    gravity = gravity_interactions(
        observations,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    primary_cluster_test = cluster_sign_flip_primary_test(
        observations,
        draws=args.permutation_draws,
        seed=args.seed,
    )
    unit_sensitivity_test = exact_paired_primary_test(unit_rows)
    write_csv(args.output_dir / "unit_condition_rows.csv", unit_rows)
    write_csv(args.output_dir / "paired_observations.csv", observations)
    write_csv(args.output_dir / "contrast_summary.csv", summaries)
    write_csv(args.output_dir / "solve_curves.csv", curves)
    write_csv(
        args.output_dir / "gravity_interactions.csv",
        gravity,
    )
    (
        args.output_dir
        / "primary_layout_cluster_sign_flip_test.json"
    ).write_text(
        json.dumps(primary_cluster_test, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (
        args.output_dir
        / "unit_level_mcnemar_sensitivity.json"
    ).write_text(
        json.dumps(unit_sensitivity_test, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "completed_model_goal_units": len(
            {
                (
                    row["model_key"],
                    row["seed"],
                    row["goal_id"],
                )
                for row in unit_rows
            }
        ),
        "unit_condition_rows": len(unit_rows),
        "postfork_eligible_model_goal_units": len(
            {
                (
                    row["model_key"],
                    row["seed"],
                    row["goal_id"],
                )
                for row in unit_rows
                if not row["shared_attempt_1_success"]
            }
        ),
        "models": sorted({row["model_key"] for row in unit_rows}),
        "base_layouts": len({row["layout_id"] for row in unit_rows}),
        "bootstrap_draws": args.bootstrap_draws,
        "permutation_draws": args.permutation_draws,
        "seed": args.seed,
        "goal_filter_manifest": (
            str(args.include_goal_ids_from_manifest)
            if args.include_goal_ids_from_manifest is not None
            else None
        ),
        "included_goal_id_count": (
            len(allowed_goal_ids)
            if allowed_goal_ids is not None
            else len({str(row["goal_id"]) for row in unit_rows})
        ),
        "primary_estimand": (
            "equal-model, equal-base-layout full-minus-neither solve_by_8 "
            "among shared-attempt-1 failures"
        ),
        "paper_comparable_sensitivity": (
            "solve_by_paper_budget and attempts_used_paper_budget use each "
            "goal's frozen submitted K_g; fixed-eight outcomes are retained "
            "for the reviewer-response feedback baseline"
        ),
        "gravity_policy": {
            "canonical": (
                "exact upward/downward pairing within 66 base layouts"
            ),
            "diverse": (
                "within-gravity arm effects plus descriptive layout-adjusted "
                "unpaired gravity contrast; no exact goal pairing claim"
            ),
        },
        "primary_layout_cluster_sign_flip_test": primary_cluster_test,
        "unit_level_mcnemar_sensitivity": unit_sensitivity_test,
        "artifacts": {
            "unit_condition_rows": str(
                (args.output_dir / "unit_condition_rows.csv").resolve()
            ),
            "paired_observations": str(
                (args.output_dir / "paired_observations.csv").resolve()
            ),
            "contrast_summary": str(
                (args.output_dir / "contrast_summary.csv").resolve()
            ),
            "solve_curves": str(
                (args.output_dir / "solve_curves.csv").resolve()
            ),
            "gravity_interactions": str(
                (args.output_dir / "gravity_interactions.csv").resolve()
            ),
            "primary_layout_cluster_sign_flip_test": str(
                (
                    args.output_dir
                    / "primary_layout_cluster_sign_flip_test.json"
                ).resolve()
            ),
            "unit_level_mcnemar_sensitivity": str(
                (
                    args.output_dir
                    / "unit_level_mcnemar_sensitivity.json"
                ).resolve()
            ),
        },
    }
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
