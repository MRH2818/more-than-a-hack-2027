"""Synthetic tests for the fair BAR-ATH checks (possession/gate.py). No match data.

    python -m unittest tests.test_gate
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

from possession import config, evidence, gate, masktest  # noqa: E402
from test_pipeline import kickoff_events, synthetic_match  # noqa: E402


class GateHelperTests(unittest.TestCase):
    def test_live_lengths_look_like_live_gaps(self):
        lengths = gate.live_lengths(seed=0)
        self.assertTrue(np.all(lengths > config.MAX_INTERP_GAP))
        self.assertTrue(np.all(lengths <= gate.LIVE_MAX_S * config.FPS))
        median_s = np.median(lengths) / config.FPS
        self.assertLess(abs(median_s - config.EXT_LIVE_GAP_MEDIAN_S), 0.5)
        np.testing.assert_array_equal(lengths, gate.live_lengths(seed=0))

    def test_kickoff_only_drops_every_other_tag(self):
        events = kickoff_events() + [
            {"code": "BUILD UP", "start_frame": 100, "end_frame": 300, "tags": {"Team": "Barcelona"}},
            {"code": "SET PIECES", "start_frame": 500, "end_frame": 520,
             "tags": {"Type": "Corner", "Team": "Barcelona"}}]
        kept = gate.kickoff_only(events)
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(e["tags"]["Type"] == "Kick Off (Start)" for e in kept))

    def test_video_agreement_counts(self):
        labels = np.array(["H", "H", "A", "D", "A", "H"])
        state = np.array([evidence.HOME, evidence.AWAY, evidence.AWAY, evidence.HOME,
                          evidence.NONE, evidence.HOME], dtype=np.int8)
        frames = np.array([True, True, True, True, True, False])
        out = gate.video_agreement(labels, state, frames)
        self.assertEqual(out["video_frames"], 4)             # NONE and the excluded frame drop out
        self.assertAlmostEqual(out["video_team_agree_pct"], 66.7)   # H ok, H wrong, A ok
        self.assertAlmostEqual(out["video_dead_pct"], 25.0)


class GateRunTests(unittest.TestCase):
    def test_run_variant_on_synthetic_match(self):
        arrays = synthetic_match()
        events = kickoff_events()
        n = arrays["slots"].shape[0]
        graded = np.zeros(n, dtype=bool)
        graded[11185:90652] = True
        graded[110808:190198] = True
        blocks = [(1, 11185, 90651), (2, 110808, 190197)]
        masked_ball, hidden = masktest.mask_ball(arrays["ball"], graded, 0.15, 0, blocks=blocks,
                                                 lengths=gate.live_lengths(seed=0))
        self.assertGreater(hidden.sum(), 0)
        row = gate.run_variant({}, arrays, dict(arrays, ball=masked_ball), hidden, events, graded,
                               np.zeros(n, dtype=np.int8), np.zeros(n, dtype=bool))
        self.assertGreaterEqual(row["live_hidden_team_agree_pct"], 90.0)
        self.assertEqual(row["video_frames"], 0)
        self.assertIn("heldout_phase_agree_pct", row)


class GoalDeadTests(unittest.TestCase):
    @staticmethod
    def tag(code, start, end, kind=None):
        return {"code": code, "start_frame": start, "end_frame": end, "tags": {"Type": kind}}

    def test_goal_cores(self):
        from possession import tagging
        events = [self.tag("GOALS", 1000, 4025), self.tag("BUILD UP", 900, 1500),
                  self.tag("SET PIECES", 4000, 4125, "Kick Off (After Goal)"),
                  # second goal: another restart overlaps the kickoff -> no stretch claimed
                  self.tag("GOALS", 6000, 7000), self.tag("SET PIECES", 6500, 7375, "Throw-in"),
                  self.tag("SET PIECES", 7200, 7300, "Kick Off (After Goal)"),
                  # third goal: a restart between the goal clip and the kickoff -> skipped
                  self.tag("GOALS", 8000, 8200), self.tag("SET PIECES", 8300, 8400, "Corner"),
                  self.tag("SET PIECES", 8600, 8700, "Kick Off (After Goal)")]
        self.assertEqual(tagging.goal_cores(events, [(1, 0, 9999)]), [(1525, 3975)])
        self.assertEqual(tagging.goal_cores(events, [(1, 0, 3000)]), [])     # kickoff in another half
        self.assertEqual(tagging.goal_cores(kickoff_events(), [(1, 0, 199318)]), [])

    def test_pipeline_forces_dead_between_goal_and_kickoff(self):
        from unittest import mock
        from possession import pipeline
        arrays = synthetic_match()
        base, _ = pipeline.label(arrays, events=kickoff_events())
        runs = np.flatnonzero(np.isin(base, ("H", "A")))
        start = int(runs[len(runs) // 2])                        # somewhere in live play
        events = kickoff_events() + [
            self.tag("GOALS", start - 100, start + 525),
            self.tag("SET PIECES", start + 500, start + 625, "Kick Off (After Goal)")]
        with mock.patch.object(config, "USE_GOAL_DEAD", False):
            off, _ = pipeline.label(arrays, events=events)
        with mock.patch.object(config, "USE_GOAL_DEAD", True):
            on, _ = pipeline.label(arrays, events=events)
        core = slice(start - 100 + config.FPS, start + 500 - config.FPS)
        self.assertTrue(np.all(on[core] == "D"))
        self.assertFalse(np.all(off[core] == "D"))


if __name__ == "__main__":
    unittest.main()
