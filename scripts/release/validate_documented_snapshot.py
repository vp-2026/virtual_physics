#!/usr/bin/env python3
"""Validate the closed collection, its missingness accounting, and preserved tables.

This reads local files only. It never invokes a model or writes to GitHub.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import export_model_outputs as exporter

SNAPSHOT = "snapshot_seed2026_documented_gaps_2026-11-05"
EXPECTED = {"gpt5": 1692, "gemini31pro": 1691, "qwen36plus": 1652}
SIDECARS = {"gpt5": 132, "gemini31pro": 132, "qwen36plus": 131}
CONDITIONS = ("full", "neither", "trace_status")
REASONS = {"procedural_repair_ceiling", "provider_content_inspection",
           "format_repair_ceiling", "provider_missing_choices", "interrupted_checkpoint"}
EXTRA_SECRET = re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
EXCLUDED_GOAL = re.compile(r"\bgoal[-_ ]?b\b", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def terminal(summary):
    require(summary["seed"] == 2026, "Unexpected seed")
    for condition in CONDITIONS:
        branch = summary["conditions"][condition]
        count = branch["attempt_count"]
        absorbed = (branch.get("completion_reason") == "shared_attempt_success_absorbed"
                    and count == 1 and branch["solved_by_budget"])
        early = (condition != "neither" and branch["solved_by_budget"]
                 and branch["first_success_attempt"] == count)
        require(1 <= count <= 8 and (count == 8 or absorbed or early),
                "Nonterminal branch cannot be included as a completed episode")


def sidecar_complete(summary, expected_conditions):
    result = summary["results"]
    pre = result["pre_feedback"]
    require(set(pre) == {"visual_early_probe", "visual_terminal_probe", "trace_early_probe",
                         "trace_terminal_probe", "visual_clear_path_probe"}, "Incomplete pre-feedback bundle")
    for name, item in pre.items():
        require((name == "visual_clear_path_probe" and item.get("available") is False)
                or item.get("protocol_finalized") or item.get("schema_valid"), "Nonterminal probe")
    require(set(result["conditions"]) == set(expected_conditions), "Incomplete sidecar conditions")
    for branch in result["conditions"].values():
        for moment in ("early_prediction", "terminal_prediction"):
            require(branch[moment].get("protocol_finalized") or branch[moment].get("schema_valid"),
                    "Incomplete held-out prediction")


def scan(value, label):
    exporter.validate_release_value(value, Path(label))
    text = value if isinstance(value, str) else json.dumps(value)
    require(not EXTRA_SECRET.search(text), "Possible credential in " + label)
    require(not EXCLUDED_GOAL.search(text), "Excluded goal material in " + label)


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def recovery_row(record):
    calls = record["provider_calls"]
    token_logs = record.get("token_budget_recovery_calls", [])
    expanded = [r for r in token_logs if r["request_max_tokens"] > r["base_request_max_tokens"]]
    return {"model_key": record["summary"]["model_key"],
            "benchmark_index": record["summary"]["goal"]["benchmark_index"],
            "source_unit": record["source_unit"], "fresh_recovery_replica": True,
            "recorded_logical_calls": len(calls),
            "recorded_format_repair_calls": sum((r.get("format_repair_index") or 0) > 0 for r in calls),
            "token_recovery_log_records": len(token_logs),
            "expanded_allowance_log_records": len(expanded),
            "expanded_allowance_observed": bool(expanded),
            "max_request_tokens_in_recovery_log": max((r["request_max_tokens"] for r in token_logs), default=0),
            "recorded_logical_call_wall_seconds": round(sum(r.get("wall_seconds", 0) or 0 for r in calls), 6)}


def validate(repo):
    root = repo / "model_outputs" / SNAPSHOT
    manifest = json.loads((root / "snapshot_manifest.json").read_text())
    require(manifest["collection_status"] == "closed_with_documented_incomplete_units", "Wrong release boundary")
    require(manifest["paper_tables_recomputed"] is False, "Frozen tables must remain separate")
    require(manifest["partial_episodes_are_experimental_failures"] is False, "Incorrect missing-data treatment")
    indices = {m: set() for m in EXPECTED}
    canonical = {m: set() for m in EXPECTED}
    sides = {m: set() for m in EXPECTED}
    names = set()
    recovered = set()
    recovery_rows = []
    fingerprints = {}
    successes = collections.Counter()
    condition_counts = collections.Counter()
    goals = {}
    shard_paths = set()
    for shard in manifest["shards"]:
        path = root / shard["path"]
        require(path.resolve().is_relative_to(root.resolve()) and path not in shard_paths,
                "Unsafe or duplicated shard path")
        shard_paths.add(path)
        require(path.stat().st_size == shard["bytes"] and exporter.sha256_file(path) == shard["sha256"],
                "Shard size or hash mismatch")
        count = 0
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                scan(record, shard["path"])
                summary = record["summary"]
                terminal(summary)
                model = summary["model_key"]
                index = summary["goal"]["benchmark_index"]
                name = record["source_unit"]
                require(model == shard["model_key"] and model in EXPECTED, "Model/shard mismatch")
                require(isinstance(index, int) and 0 <= index < 1692, "Invalid benchmark index")
                require(name not in names and index not in indices[model], "Duplicate episode")
                require(name.startswith(model + "/seed_2026/") and int(name.split("/")[2].split("_")[0]) == index,
                        "Episode label/index mismatch")
                names.add(name)
                indices[model].add(index)
                fingerprints[name] = fingerprint(record)
                goal_fields = {k: summary["goal"][k] for k in
                               ("balanced_goal_id", "signature", "asset_world_sha256", "asset_screenshot_sha256", "attempt_budget")}
                require(goals.setdefault(index, goal_fields) == goal_fields, "Cross-model task/asset mismatch")
                if summary["goal"]["source"] == "canonical_world_gcond":
                    canonical[model].add(index)
                bundles = [x["record"] for x in record["sidecar_records"]
                           if x["source"] == "transfer_sidecars/summary.json"]
                require(len(bundles) <= 1 and (bundles or not record["sidecar_records"]), "Unfinalized sidecar bundle")
                if bundles:
                    sidecar_complete(bundles[0], summary["conditions"])
                    require(index in canonical[model], "Noncanonical sidecar in canonical count")
                    sides[model].add(index)
                if record.get("recovery_provenance"):
                    require(model == "qwen36plus" and not record["excluded_unreadable_attempt_records"], "Invalid recovery export")
                    provenance = record["recovery_provenance"]
                    require(provenance["outcome_based_selection"] is False
                            and provenance["in_original_frozen_paper_tables"] is False, "Missing recovery disclosure")
                    recovered.add(name)
                    recovery_rows.append(recovery_row(record))
                for condition in summary["conditions"]:
                    condition_counts[(model, condition)] += 1
                for condition in CONDITIONS:
                    successes[(model, condition)] += bool(summary["conditions"][condition]["solved_by_budget"])
                count += 1
        require(count == shard["unit_count"], "Shard count mismatch")
    require(shard_paths == set((root / "data").rglob("*.gz")), "Unmanifested or absent shard")
    require(len(names) == manifest["completed_unit_count"] == 5035, "Wrong completed total")
    require(len(recovered) == 113 and recovered == set(manifest["recovered_main_units"]), "Recovery manifest mismatch")
    recovery_csv = rows(root / "recovery_exceptions.csv")
    recovery_rows.sort(key=lambda r: r["source_unit"])
    require(len(recovery_csv) == len(recovery_rows), "Recovery exception row count mismatch")
    for left, right in zip(recovery_rows, recovery_csv):
        require(all(str(value) == right[key] for key, value in left.items()), "Recovery exception report mismatch")
    # Every previously released record must be preserved exactly as a JSON value.
    base_count = 0
    base = repo / "model_outputs" / manifest["source_snapshot"]
    for path in sorted((base / "data").rglob("*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                name = record["source_unit"]
                require(name not in recovered and fingerprints.get(name) == fingerprint(record),
                        "Previously published record changed or omitted")
                base_count += 1
    require(base_count == 4922, "Wrong prior-snapshot population")
    missing_doc = json.loads((root / "missing_units.json").read_text())
    missing = missing_doc["units"]
    csv_missing = rows(root / "missing_units.csv")
    require(len(missing) == len(csv_missing) == 42, "Missingness manifest count mismatch")
    require(missing_doc["missingness_is_not_assumed_random"] is True, "Missingness assumption absent")
    for left, right in zip(missing, csv_missing):
        require(all(str(value) == right[key] for key, value in left.items()), "CSV/JSON gap disagreement")
        require(left["status"] == "incomplete" and left["scored_as_experimental_failure"] is False,
                "Missing unit mislabeled as a failure")
        require(left["observed_stop_reason"] in REASONS, "Undocumented stopping reason")
        require(left["scope"] in ("main", "canonical_sidecar"), "Unknown missingness scope")
        require(left["model_key"] in EXPECTED, "Unknown missing model")
    require(len({(r["model_key"], r["scope"], r["benchmark_index"]) for r in missing}) == 42,
            "Duplicate gap")
    for model in EXPECTED:
        gaps = {r["benchmark_index"] for r in missing if r["model_key"] == model and r["scope"] == "main"}
        side_gaps = {r["benchmark_index"] for r in missing if r["model_key"] == model and r["scope"] == "canonical_sidecar"}
        require(len(indices[model]) == EXPECTED[model] and not indices[model] & gaps
                and indices[model] | gaps == set(range(1692)), "Incomplete main index accounting")
        require(len(canonical[model]) == 132 and len(sides[model]) == SIDECARS[model]
                and not sides[model] & side_gaps and sides[model] | side_gaps == canonical[model],
                "Incomplete canonical sidecar accounting")
    require(sum(r["scope"] == "main" for r in missing) == manifest["incomplete_main_unit_count"] == 41,
            "Main missingness total mismatch")
    require(sum(map(len, sides.values())) == manifest["completed_canonical_sidecar_count"] == 395,
            "Sidecar completion total mismatch")
    require(manifest["incomplete_canonical_sidecar_count"] == 1, "Sidecar gap count mismatch")
    cov = rows(root / "coverage.csv")
    require(len(cov) == 3 and {r["model_key"] for r in cov} == set(EXPECTED), "Coverage rows mismatch")
    for row, embedded in zip(cov, manifest["coverage"]):
        require(all(str(value) == row[key] for key, value in embedded.items()), "Manifest/CSV coverage disagreement")
        model = row["model_key"]
        for key, value in {"completed_units": EXPECTED[model], "expected_units": 1692,
                           "incomplete_units": 1692 - EXPECTED[model], "canonical_units": 132,
                           "units_with_transfer_sidecars": SIDECARS[model], "expected_canonical_sidecars": 132,
                           "incomplete_canonical_sidecars": 132 - SIDECARS[model],
                           "fresh_recovery_units": 113 if model == "qwen36plus" else 0}.items():
            require(int(row[key]) == value, "Coverage value mismatch: " + key)
        for condition in (*CONDITIONS, "status_only"):
            require(int(row[condition + "_condition_units"]) == condition_counts[(model, condition)],
                    "Condition population mismatch")
    planning = rows(root / "planning_available.csv")
    require(len(planning) == 9 and len({(r["model_key"], r["condition"]) for r in planning}) == 9,
            "Planning rows mismatch")
    for row in planning:
        model, condition = row["model_key"], row["condition"]
        n = EXPECTED[model]
        k = successes[(model, condition)]
        require(int(row["completed_units"]) == n and int(row["expected_units"]) == 1692
                and int(row["incomplete_units"]) == 1692 - n and int(row["solved_by_eight"]) == k,
                "Planning denominator/numerator mismatch")
        for key, value in {"solve_rate_among_completed": k / n, "all_expected_lower_bound": k / 1692,
                           "all_expected_upper_bound": (k + 1692 - n) / 1692}.items():
            require(math.isclose(float(row[key]), value, abs_tol=1e-12), "Planning sensitivity calculation mismatch")
    populations = json.loads((root / "analysis_populations.json").read_text())
    require(populations["completed_main_by_model"] == {m: sorted(v) for m, v in indices.items()}, "Analysis population mismatch")
    for key, values in (("common_completed_main_indices", indices), ("common_canonical_main_indices", canonical),
                        ("common_available_canonical_sidecar_indices", sides)):
        require(populations[key] == sorted(set.intersection(*values.values())), "Matched population mismatch")
    frozen = json.loads((root / "frozen_analysis_manifest.json").read_text())
    require(frozen["modified"] is False and frozen["files"], "Missing frozen-table manifest")
    for item in frozen["files"]:
        path = repo / item["path"]
        require(path.resolve().is_relative_to((repo / "model_outputs/analysis_seed2026_2026-07-27").resolve()),
                "Unsafe frozen-table path")
        require(exporter.sha256_file(path) == item["sha256"] and len(rows(path)) == item["rows"],
                "Frozen paper table changed")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix != ".gz":
            scan(path.read_text(), path.relative_to(root).as_posix())
    return {"completed_main": len(names), "completed_sidecars": sum(map(len, sides.values())),
            "incomplete_main": 41, "incomplete_sidecars": 1, "preserved_prior_records": base_count,
            "fresh_complete_recovery_records": len(recovered), "shards": len(shard_paths),
            "matched_main": len(set.intersection(*indices.values())),
            "matched_canonical_main": len(set.intersection(*canonical.values())),
            "available_matched_sidecars": len(set.intersection(*sides.values())),
            "frozen_tables_verified": len(frozen["files"]), "validation": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(validate(args.repo.resolve()), indent=2, sort_keys=True))
