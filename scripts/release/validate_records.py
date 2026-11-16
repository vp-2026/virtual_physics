"""Validate published model records and completed episode boundaries."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any

ABSOLUTE_PATH_RE = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\\\)")

SECRET_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{16,}|"
    r"Bearer\s+[A-Za-z0-9._-]{16,})"
)

EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"
)

UUID_RE = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])",
    re.IGNORECASE,
)

IPV4_RE = re.compile(
    r"(?<![0-9.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![0-9.])"
)

GOAL_B_RE = re.compile(
    r"Goal-A\s*(?:->|→|to)\s*Goal-B",
    re.IGNORECASE,
)

FORBIDDEN_RELEASE_KEYS = {
    "account_id",
    "goal_transfer",
    "goal_transfer_eligible",
    "provider_response_id",
    "request_id",
    "response_id",
    "target_goal_id",
    "transfer_pair_selection",
    "transfer_source",
    "transfer_target_goal_id",
    "user_id",
}

def validate_release_value(value: Any, path: Path) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            forbidden = FORBIDDEN_RELEASE_KEYS.intersection(item)
            if forbidden:
                raise SystemExit(
                    f"Forbidden release field found in {path}: "
                    f"{sorted(forbidden)}"
                )
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            if ABSOLUTE_PATH_RE.search(item):
                raise SystemExit(f"Absolute path found in {path}")
            if SECRET_RE.search(item):
                raise SystemExit(f"Possible secret found in {path}")
            if EMAIL_RE.search(item):
                raise SystemExit(f"Email address found in {path}")
            if UUID_RE.search(item):
                raise SystemExit(f"UUID found in {path}")
            if IPV4_RE.search(item):
                raise SystemExit(f"IP address found in {path}")
            if GOAL_B_RE.search(item):
                raise SystemExit(f"Goal-B material found in {path}")

CONDITIONS = ("full", "neither", "trace_status")

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

def scan(value, label):
    validate_release_value(value, Path(label))
    text = value if isinstance(value, str) else json.dumps(value)
    require(not EXTRA_SECRET.search(text), "Possible credential in " + label)
    require(not EXCLUDED_GOAL.search(text), "Excluded goal material in " + label)
