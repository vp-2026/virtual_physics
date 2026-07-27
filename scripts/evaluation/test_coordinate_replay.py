#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replay = load(
    ROOT / "run_coordinate_replay.py",
    "coordinate_replay_test_module",
)


class CoordinateReplayTest(unittest.TestCase):
    def test_primary_signatures_restore_submitted_endpoint_truth(self):
        class Builder:
            @staticmethod
            def event_signature(event):
                return event["signature"]

        signatures = replay.primary_row_signatures(
            Builder(),
            {
                "event_graph": [{"signature": "movement|during"}],
                "endpoint_final_state_signatures": [
                    "final_state|submitted_endpoint"
                ],
                "persistent_final_state_signatures": [],
            },
        )
        self.assertEqual(
            signatures,
            {
                "movement|during",
                "final_state|submitted_endpoint",
            },
        )

    def test_nearest_point_uses_lexicographic_tie_break(self):
        self.assertEqual(
            replay.nearest_point(
                (105, 105),
                [(100, 110), (110, 100), (100, 100), (110, 110)],
            ),
            (100, 100),
        )

    def test_goal_success_uses_signature_or_canonical_dwell(self):
        diverse = {
            "source": "diverse",
            "signature": "movement|red|x|left",
        }
        canonical = {
            "source": "canonical_world_gcond",
            "signature": "canonical",
        }
        simulation = {
            "valid": True,
            "signatures": ["movement|red|x|left"],
            "canonical_in_goal_dwell_2s": True,
        }
        self.assertTrue(
            replay.goal_success_from_simulation(diverse, simulation)
        )
        self.assertTrue(
            replay.goal_success_from_simulation(canonical, simulation)
        )
        self.assertFalse(
            replay.goal_success_from_simulation(
                diverse,
                {**simulation, "valid": False},
            )
        )

    def test_collect_actions_reads_fixed_branch_states(self):
        with tempfile.TemporaryDirectory() as directory:
            unit = (
                Path(directory)
                / "gpt5"
                / "seed_2026"
                / "0000_test"
            )
            branch = unit / "branches" / "full"
            branch.mkdir(parents=True)
            world = unit / "assets" / "simulation_world.json"
            world.parent.mkdir(parents=True)
            world.write_text("{}\n", encoding="utf-8")
            summary = {
                "model_key": "gpt5",
                "seed": 2026,
                "goal": {
                    "balanced_goal_id": "goal",
                    "puzzle_key": "Basic_1_upward",
                    "condition": "upward",
                    "category_5": "movement",
                    "source": "diverse",
                    "signature": "sig",
                    "environment_json": "/missing/world.json",
                },
            }
            (unit / "unit_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            (branch / "state.json").write_text(
                json.dumps(
                    {
                        "attempts": [
                            {
                                "attempt": 1,
                                "coords": [101, 202],
                                "goal_succeeded": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            actions, goals = replay.collect_actions(Path(directory))
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["original_coords"], [101, 202])
            self.assertTrue(actions[0]["jitter_scope"])
            self.assertEqual(goals["goal"]["signature"], "sig")


if __name__ == "__main__":
    unittest.main()
