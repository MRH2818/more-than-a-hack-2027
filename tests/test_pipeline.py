"""Synthetic tests for ball repair, evidence, dead-ball cues, the decoder and the full pipeline.

    python -m unittest discover -s tests -v
"""

import itertools
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

import possession_metrics as pm  # noqa: E402
from possession import ball, deadball, decoder, evidence, pipeline  # noqa: E402

PITCH = (105.3, 68.0)


def path_score(labels, emissions, cost):
    score = sum(emissions[t, k] for t, k in enumerate(labels))
    score -= sum(cost[a][b] for a, b in zip(labels[:-1], labels[1:]) if a != b)
    return score


def feasible(labels, m):
    runs = [len(list(group)) for _, group in itertools.groupby(labels)]
    keys = [key for key, _ in itertools.groupby(labels)]
    return all(length >= m[key] for key, length in zip(keys, runs))


class DecoderTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.RandomState(0)
        m = [2, 3, 2]
        cost = [[0, 1.5, 0.7], [1.5, 0, 0.7], [0.4, 0.4, 0]]
        for _ in range(25):
            emissions = rng.randn(8, 3)
            best = max((path_score(seq, emissions, cost), seq)
                       for seq in itertools.product(range(3), repeat=8) if feasible(seq, m))
            got = decoder.decode(emissions, m, cost)
            self.assertTrue(feasible(tuple(got), m))
            self.assertAlmostEqual(path_score(tuple(got), emissions, cost), best[0])

    def test_bonuses_match_brute_force(self):
        rng = np.random.RandomState(1)
        m = [2, 3, 2]
        cost = [[0, 1.5, 0.7], [1.5, 0, 0.7], [0.4, 0.4, 0]]
        for _ in range(25):
            emissions = rng.randn(8, 3)
            entry = np.abs(rng.randn(8, 3)) * (rng.rand(8, 3) < 0.3)
            leave = np.abs(rng.randn(8, 3)) * (rng.rand(8, 3) < 0.3)

            def score(seq):
                total = path_score(seq, emissions, cost)
                for t in range(1, len(seq)):
                    if seq[t] != seq[t - 1]:
                        total += entry[t][seq[t]] + leave[t][seq[t - 1]]
                return total

            best = max(score(seq) for seq in itertools.product(range(3), repeat=8)
                       if feasible(seq, m))
            got = decoder.decode(emissions, m, cost, entry_bonus=entry, exit_bonus=leave)
            self.assertTrue(feasible(tuple(got), m))
            self.assertAlmostEqual(score(tuple(got)), best)

    def test_short_touch_does_not_switch(self):
        e = np.zeros((300, 3))
        e[:, 2] = -5
        e[0:100, 0] = 1          # home controls
        e[150:154, 1] = 1        # away touches for 4 frames: a deflection
        e[200:240, 0] = 1        # home again
        labels = decoder.decode(e, [25, 25, 25], [[0, 10, 4], [10, 0, 4], [4, 4, 0]])
        self.assertTrue(np.all(labels == decoder.H))

    def test_sustained_control_switches_and_snaps(self):
        e = np.zeros((300, 3))
        e[:, 2] = -5
        e[0:100, 0] = 1
        e[160:300, 1] = 1        # away wins it at 160 and keeps it
        labels = decoder.decode(e, [25, 25, 25], [[0, 10, 4], [10, 0, 4], [4, 4, 0]])
        control = np.zeros(300, dtype=np.int8)
        control[0:100] = evidence.HOME
        control[160:300] = evidence.AWAY
        snapped = decoder.snap_turnovers(labels, control, evidence.HOME, evidence.AWAY, 100)
        self.assertEqual(int(np.flatnonzero(snapped == decoder.A)[0]), 160)
        self.assertEqual(len(np.flatnonzero(np.diff(snapped) != 0)), 1)


class BallAndEvidenceTests(unittest.TestCase):
    def test_interpolation_and_oob(self):
        xy = np.full((20, 2), np.nan)
        xy[0:5] = [[10, 10]] * 5
        xy[10:20] = [[20, 10]] * 10
        filled, interpolated = ball.fill_short_gaps(xy, max_gap=12)
        self.assertTrue(interpolated[5:10].all())
        self.assertAlmostEqual(filled[7, 0], 10 + 10 * 3 / 6.0)
        oob = ball.out_of_bounds(np.array([[-2.0, 5.0], [50.0, 70.0], [50.0, 30.0]]), PITCH)
        self.assertEqual(list(oob), [True, True, False])

    def test_control_states(self):
        d_home = np.array([0.5, 0.5, 3.0, np.nan, 0.8])
        d_away = np.array([3.0, 0.9, 0.4, np.nan, 0.6])
        self.assertEqual(list(evidence.control(d_home, d_away)),
                         [evidence.HOME, evidence.DUEL, evidence.AWAY, evidence.NONE,
                          evidence.DUEL])

    def test_gmm_separates_slow_from_fast(self):
        x = np.concatenate([np.full(500, 0.8), np.full(1500, 2.5)]) + \
            np.random.RandomState(1).randn(2000) * 0.1
        odds = deadball.gmm_logodds(x, np.ones(2000, dtype=bool))
        self.assertGreater(odds[:500].mean(), 2)
        self.assertLess(odds[500:].mean(), -2)


def synthetic_match():
    """Full-length arrays: in each half home holds the ball, then away, then it goes out."""
    n = 199319
    slots = np.full((n, 4, 3), np.nan)
    team = np.array([0, 1, 0, 1], dtype=np.int8)
    ball_xy = np.full((n, 3), np.nan)
    slots[:, 2] = [0.2, 0.5, 1.0]           # spare home player, far away
    slots[:, 3] = [0.8, 0.5, 1.0]           # spare away player

    script = []
    for first in (11185, 113742):
        script += [(first, first + 2000, 0, 0.5),       # home carries along x
                   (first + 2000, first + 4000, 1, 0.5),  # away wins it
                   (first + 4000, first + 4500, None, -0.05)]  # ball out over the touchline
    for start, stop, holder, y in script:
        steps = np.linspace(0.3, 0.6, stop - start)
        if holder is None:
            ball_xy[start:stop] = np.column_stack([np.full(stop - start, 0.5),
                                                   np.full(stop - start, y),
                                                   np.zeros(stop - start)])
            continue
        slots[start:stop, holder] = np.column_stack([steps, np.full(stop - start, y),
                                                     np.full(stop - start, 3.0)])
        slots[start:stop, 1 - holder] = np.column_stack([steps, np.full(stop - start, y + 0.1),
                                                         np.full(stop - start, 3.0)])
        ball_xy[start:stop] = np.column_stack([steps + 0.002, np.full(stop - start, y),
                                               np.full(stop - start, 5.0)])
    return {"slots": slots, "ball": ball_xy, "team": team, "present": np.ones(n, dtype=bool),
            "pitch": np.array(PITCH)}


def kickoff_events(first=11185, second=113742):
    """Smart-Tagging-style kickoff tags for both halves of the synthetic match."""
    return [{"code": "SET PIECES", "start_frame": frame, "end_frame": frame + 25,
             "tags": {"Type": "Kick Off (Start)", "Team": team}}
            for frame, team in ((first, "Barcelona"), (second, "Atletic Club"))]


class PipelineTests(unittest.TestCase):
    def test_end_to_end_valid_and_sensible(self):
        labels, parts = pipeline.label(synthetic_match(), events=kickoff_events())
        table = {"frame": np.arange(len(labels)), "label": labels}
        import pandas as pd
        self.assertIsNone(pm.validate_submission(pd.DataFrame(table), 199319))

        self.assertEqual(labels[5000], "X")                  # before kickoff: ungraded
        self.assertEqual(labels[11185 + 1000], "H")
        self.assertEqual(labels[11185 + 3000], "A")
        self.assertEqual(labels[11185 + 4200], "D")          # ball beyond the touchline
        self.assertTrue(np.all(labels[110808:113742] == "D"))  # published pre-kickoff stretch
        self.assertEqual(labels[113742 + 1000], "H")
        # the turnover lands within half a second of the moment away takes the ball
        switch = 11185 + int(np.flatnonzero(labels[11185:11185 + 4000] == "A")[0])
        self.assertLessEqual(abs(switch - (11185 + 2000)), 12)


if __name__ == "__main__":
    unittest.main()
