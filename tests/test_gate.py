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


if __name__ == "__main__":
    unittest.main()
