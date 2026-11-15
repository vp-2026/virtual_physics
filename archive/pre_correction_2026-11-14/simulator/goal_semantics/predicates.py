#!/usr/bin/env python3
"""Predicate metadata used for rollout trace analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    event_type: str
    primary_role: str | None = None
    secondary_role: str | None = None
    direct_tool_contact_invalidates: bool = False
    requires_no_tool_baseline: bool = False
    terminal_role: str | None = None


PREDICATE_REGISTRY: dict[str, PredicateSpec] = {
    "CONTACT_TOOL_TARGET": PredicateSpec("CONTACT_TOOL_TARGET", "collision", "tool", "target"),
    "CONTACT_TOOL_ANY_BLUE_MOVABLE": PredicateSpec("CONTACT_TOOL_ANY_BLUE_MOVABLE", "collision", "tool", "blue"),
    "CONTACT_TOOL_GOAL_CONTAINER": PredicateSpec("CONTACT_TOOL_GOAL_CONTAINER", "collision", "tool", "goal"),
    "CONTACT_TARGET_GOAL_CONTAINER": PredicateSpec("CONTACT_TARGET_GOAL_CONTAINER", "collision", "target", "goal"),
    "CONTACT_TARGET_ANY_BLUE": PredicateSpec("CONTACT_TARGET_ANY_BLUE", "collision", "target", "blue"),
    "CONTACT_BLUE_GOAL_CONTAINER": PredicateSpec("CONTACT_BLUE_GOAL_CONTAINER", "collision", "blue", "goal"),
    "TARGET_MOVES_LEFT": PredicateSpec("TARGET_MOVES_LEFT", "move_onset", terminal_role="target"),
    "TARGET_MOVES_RIGHT": PredicateSpec("TARGET_MOVES_RIGHT", "move_onset", terminal_role="target"),
    "BLUE_OBJECT_MOVES_LEFT": PredicateSpec("BLUE_OBJECT_MOVES_LEFT", "move_onset", terminal_role="blue"),
    "BLUE_OBJECT_MOVES_RIGHT": PredicateSpec("BLUE_OBJECT_MOVES_RIGHT", "move_onset", terminal_role="blue"),
    "BLUE_RECTANGLE_MOVES_LEFT": PredicateSpec("BLUE_RECTANGLE_MOVES_LEFT", "move_onset", terminal_role="blue_rectangle"),
    "BLUE_RECTANGLE_MOVES_RIGHT": PredicateSpec("BLUE_RECTANGLE_MOVES_RIGHT", "move_onset", terminal_role="blue_rectangle"),
    "BLUE_SQUARE_MOVES_LEFT": PredicateSpec("BLUE_SQUARE_MOVES_LEFT", "move_onset", terminal_role="blue_square"),
    "BLUE_SQUARE_MOVES_RIGHT": PredicateSpec("BLUE_SQUARE_MOVES_RIGHT", "move_onset", terminal_role="blue_square"),
    "BLUE_RECTANGLE_ROTATES_CLOCKWISE": PredicateSpec("BLUE_RECTANGLE_ROTATES_CLOCKWISE", "move_onset", terminal_role="blue_rectangle"),
    "BLUE_RECTANGLE_ROTATES_COUNTERCLOCKWISE": PredicateSpec("BLUE_RECTANGLE_ROTATES_COUNTERCLOCKWISE", "move_onset", terminal_role="blue_rectangle"),
    "GOAL_CONTAINER_MOVES_LEFT": PredicateSpec("GOAL_CONTAINER_MOVES_LEFT", "move_onset", terminal_role="goal"),
    "GOAL_CONTAINER_MOVES_RIGHT": PredicateSpec("GOAL_CONTAINER_MOVES_RIGHT", "move_onset", terminal_role="goal"),
    "ALL_MOVABLE_OBJECTS_MOVE": PredicateSpec("ALL_MOVABLE_OBJECTS_MOVE", "move_onset", terminal_role="movable"),
    "TARGET_MOVES_CLOSER_TO_GOAL": PredicateSpec("TARGET_MOVES_CLOSER_TO_GOAL", "move_onset", terminal_role="target"),
    "TARGET_MOVES_FARTHER_FROM_GOAL": PredicateSpec("TARGET_MOVES_FARTHER_FROM_GOAL", "move_onset", terminal_role="target"),
    "TARGET_IN_GOAL_AFTER_SETTLE": PredicateSpec("TARGET_IN_GOAL_AFTER_SETTLE", "goal_entry", terminal_role="target"),
}

for _name, _role in {
    "INDIRECT_TRANSFER_TO_TARGET": "target",
    "INDIRECT_TRANSFER_TO_BLUE": "blue",
    "INDIRECT_TRANSFER_TO_CONTAINER": "goal",
}.items():
    PREDICATE_REGISTRY[_name] = PredicateSpec(
        _name,
        "move_onset",
        terminal_role=_role,
        direct_tool_contact_invalidates=True,
        requires_no_tool_baseline=True,
    )

for _name, _role in {
    "COUNTERFACTUAL_RED_TARGET_DIFFERENT_FINAL_POSITION": "target",
    "COUNTERFACTUAL_BLUE_OBJECT_DIFFERENT_FINAL_POSITION": "blue",
    "COUNTERFACTUAL_BLUE_RECTANGLE_DIFFERENT_FINAL_POSITION": "blue_rectangle",
    "COUNTERFACTUAL_BLUE_SQUARE_DIFFERENT_FINAL_POSITION": "blue_square",
    "COUNTERFACTUAL_GREEN_GOAL_CONTAINER_DIFFERENT_FINAL_POSITION": "goal",
}.items():
    PREDICATE_REGISTRY[_name] = PredicateSpec(_name, "settled", terminal_role=_role, requires_no_tool_baseline=True)


def get_predicate_spec(predicate_name: str) -> PredicateSpec | None:
    return PREDICATE_REGISTRY.get(predicate_name)
