"""Synthetic tests for the simulated stoppage ball (possession/external.py). No match data.

    python -m unittest tests.test_stoppage
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

from possession import config, external, learned  # noqa: E402

PITCH = np.array([100.0, 60.0])


def frozen_match(n_live=200, n_dead=300):
    """Live, a stoppage with the ball frozen where it died, live again from another spot."""
    n = 2 * n_live + n_dead
    xy = np.zeros((n, 2))
    xy[:n_live, 0] = np.linspace(20.0, 40.0, n_live)          # rolling at 2.5 m/s
    xy[:n_live, 1] = 30.0
    xy[n_live:n_live + n_dead] = xy[n_live - 1]               # frozen
    xy[n_live + n_dead:, 0] = np.linspace(60.0, 70.0, n_live)  # restart 20 m away
    xy[n_live + n_dead:, 1] = 10.0
    answers = np.array(["H"] * n_live + ["D"] * n_dead + ["A"] * n_live)
    slots = np.full((n, 4, 3), np.nan)
    walk = np.linspace(0.0, 1.0, n)
    # slot 0 walks to the restart spot; the others stand far away
    slots[:, 0, 0] = (50.0 + 10.0 * np.minimum(1.0, walk * n / (n_live + n_dead))) / PITCH[0]
    slots[:, 0, 1] = (20.0 - 10.0 * np.minimum(1.0, walk * n / (n_live + n_dead))) / PITCH[1]
    slots[:, 1:, 0] = 0.1
    slots[:, 1:, 1] = 0.9
    slots[:, :, 2] = 1.0
    ball = np.column_stack([xy / PITCH, np.full(n, np.nan)])
    arrays = {"slots": slots, "team": np.array([0, 0, 1, 1], dtype=np.int8), "ball": ball,
              "pitch": PITCH, "present": np.ones(n, dtype=bool)}
    return arrays, answers, [(1, 0, n - 1)], (n_live, n_live + n_dead)


class StoppageBallTests(unittest.TestCase):
    def test_live_frames_untouched_and_stoppage_moves(self):
        arrays, answers, _, (start, stop) = frozen_match()
        for mode in ("place", "carry", "mix"):
            out = external.simulate_stoppage_ball(arrays, answers, mode, seed=3)
            live = answers != "D"
            np.testing.assert_array_equal(out[live], arrays["ball"][live])
            xy = out[start:stop, :2] * PITCH
            self.assertGreater(len(np.unique(xy.round(3), axis=0)), 10, mode)
            # no teleport: every step is physically possible, including into the restart
            path = np.vstack([arrays["ball"][start - 1:start, :2] * PITCH, xy,
                              arrays["ball"][stop:stop + 1, :2] * PITCH])
            steps = np.hypot(*np.diff(path, axis=0).T) * config.FPS
            self.assertLess(steps.max(), 36.0, mode)
            # it ends at the restart spot
            restart = arrays["ball"][stop, :2] * PITCH
            self.assertLess(np.hypot(*(xy[-1] - restart)), 1.0, mode)

    def test_place_lies_still_and_carry_follows_the_taker(self):
        arrays, answers, _, (start, stop) = frozen_match()
        restart = arrays["ball"][stop, :2] * PITCH
        placed = external.simulate_stoppage_ball(arrays, answers, "place", seed=0)[:, :2] * PITCH
        np.testing.assert_allclose(placed[stop - 50:stop], np.tile(restart, (50, 1)), atol=1e-6)
        carried = external.simulate_stoppage_ball(arrays, answers, "carry", seed=0)[:, :2] * PITCH
        taker = arrays["slots"][:, 0, :2] * PITCH
        middle = slice(start + 100, stop - 20)
        gap = np.hypot(*(carried[middle] - taker[middle]).T)
        self.assertLess(gap.max(), 1.0)
        self.assertGreater(gap.min(), 0.2)

    def test_deterministic_and_seeded(self):
        arrays, answers, _, _ = frozen_match()
        a = external.simulate_stoppage_ball(arrays, answers, "mix", seed=5)
        b = external.simulate_stoppage_ball(arrays, answers, "mix", seed=5)
        np.testing.assert_array_equal(a, b)

    def test_stoppage_without_restart_is_left_alone(self):
        arrays, answers, _, (start, _) = frozen_match()
        answers = answers.copy()
        answers[start:] = "D"                                   # dead until the end of the half
        out = external.simulate_stoppage_ball(arrays, answers, "mix", seed=0)
        np.testing.assert_array_equal(out, arrays["ball"])

    def test_frozen_mode_keeps_the_old_bench_and_its_cache_key(self):
        arrays, answers, blocks, _ = frozen_match()
        saved = config.STOPPAGE_BALL
        try:
            config.STOPPAGE_BALL = "frozen"
            key_frozen = learned.bench_key()
            first = external.degrade(arrays, answers, blocks, 0.25, seed=2)
            self.assertTrue(all(name.startswith("EXT_") for name, _ in key_frozen))
            config.STOPPAGE_BALL = "mix"
            second = external.degrade(arrays, answers, blocks, 0.25, seed=2)
            self.assertNotEqual(learned.bench_key(), key_frozen)
        finally:
            config.STOPPAGE_BALL = saved
        # here (ball never near a line): same hiding, same noise on live frames
        np.testing.assert_array_equal(np.isnan(first["ball"][:, 0]), np.isnan(second["ball"][:, 0]))
        live = (answers != "D") & ~np.isnan(first["ball"][:, 0])
        np.testing.assert_allclose(first["ball"][live], second["ball"][live], equal_nan=True)


class DeadWhenSeenProbeTests(unittest.TestCase):
    def test_probe_runs_on_a_synthetic_match_and_prints_aggregates(self):
        import contextlib
        import io
        from unittest import mock

        from possession import io_atd, probe, tagging
        from test_pipeline import kickoff_events, synthetic_match

        # the synthetic ball goes out over the touchline 4000 frames into each half
        events = kickoff_events() + [
            {"code": "SET PIECES", "start_frame": first + 4300, "end_frame": first + 4500,
             "tags": {"Type": "Throw-in", "Team": "N/A"}} for first in (11185, 113742)]
        out = io.StringIO()
        with mock.patch.object(io_atd, "load", return_value=synthetic_match()), \
                mock.patch.object(tagging, "load_events", return_value=events), \
                contextlib.redirect_stdout(out):
            probe.dead_when_seen(("v6",))
        text = out.getvalue()
        self.assertIn("set-piece tags used: 2", text)
        self.assertIn("share of the wait labelled dead", text)
        self.assertIn("tags withheld", text)
        self.assertIn("ball seen", text)

    def test_dead_checks_run_and_islands_are_counted(self):
        import contextlib
        import io
        from unittest import mock

        from possession import io_atd, probe, tagging
        from test_pipeline import kickoff_events, synthetic_match

        events = kickoff_events() + [
            {"code": "SET PIECES", "start_frame": 11185 + 4400, "end_frame": 11185 + 4500,
             "tags": {"Type": "Goal Kick", "Team": "Barcelona"}},
            {"code": "GOALS", "start_frame": 11185 + 1000, "end_frame": 11185 + 1100,
             "tags": {"Team": "Barcelona"}},
            {"code": "SET PIECES", "start_frame": 11185 + 1900, "end_frame": 11185 + 2000,
             "tags": {"Type": "Kick Off (After Goal)", "Team": "Atletic Club"}}]
        out = io.StringIO()
        with mock.patch.object(io_atd, "load", return_value=synthetic_match()), \
                mock.patch.object(tagging, "load_events", return_value=events), \
                contextlib.redirect_stdout(out):
            probe.dead_checks(("v6",))
        text = out.getvalue()
        for expected in ("goal-to-kickoff frames", "8 s before goal kick", "islands",
                         "open play"):
            self.assertIn(expected, text)
        labels = np.array(list("DD" + "H" * 50 + "DD" + "A" * 100 + "H" * 20 + "D" + "H" * 300))
        self.assertEqual(probe.live_islands(labels), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
