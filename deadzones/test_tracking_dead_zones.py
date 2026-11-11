import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from deadzones.dead_zones import Event
from deadzones.tracking_dead_zones import detect, features, load_tracking, public_metrics, runs


class TrackingDeadZonesTests(unittest.TestCase):
    def setUp(self):
        self.n = 500
        self.meta = {"fps": 25, "n_frames": self.n, "pitch": [105, 68], "teams": [0] * 11 + [1] * 11}
        self.f = pd.DataFrame({"frame": np.arange(self.n), "players_seen": 22, "home_seen": 11, "away_seen": 11,
                               "player_median_mps": .2, "player_p90_mps": .6,
                               "ball_x_m": np.nan, "ball_y_m": np.nan, "ball_speed_mps": np.nan,
                               "ball_valid": False, "nearest_player_m": 1., "outside_pitch_m": np.nan,
                               "kickoff_formation_fraction": 1., "homography_valid": True, "homography_error": .2})

    def event(self, kind="throw-in"):
        return Event("r", 4, 16, "SET PIECES", "1st half", "N/A", kind)

    def outside(self, lo=100, hi=200):
        self.f.loc[lo:hi-1, ["ball_x_m", "ball_y_m", "ball_speed_mps", "outside_pitch_m"]] = [50, -2, .2, 2]
        self.f.loc[lo:hi-1, "ball_valid"] = True

    def launch(self, lo=200, hi=250):
        self.f.loc[lo:hi-1, ["ball_x_m", "ball_y_m", "ball_speed_mps", "outside_pitch_m", "player_median_mps", "player_p90_mps"]] = [55, 34, 4, -34, 2, 3]
        self.f.loc[lo:hi-1, "ball_valid"] = True

    def test_idle_players_without_ball_do_not_mark_ordinary_restart_dead(self):
        r = detect(self.f, [self.event()], self.meta)
        self.assertFalse(r["supported"].any())
        self.assertTrue(r["idle_review"].any())

    def test_kickoff_waiting_requires_following_centre_departure(self):
        e = self.event("kick off (start)")
        self.assertFalse(detect(self.f, [e], self.meta)["supported"].any())
        self.launch()
        r = detect(self.f, [e], self.meta)
        self.assertTrue(r["supported"][100])
        self.assertFalse(r["supported"][195:].any())  # 0.2-second end guard
        self.assertFalse(r["supported"][:5].any())
        self.assertTrue(r["resume"][210])

    def test_kickoff_visible_conflicting_ball_interrupts_waiting(self):
        self.launch(100, 250)
        self.f.loc[100:249, "ball_x_m"] = 90.  # far from centre, not corroborating kickoff
        r = detect(self.f, [self.event("kick off (start)")], self.meta)
        self.assertFalse(r["supported"].any())

    def test_kickoff_wrong_formation_abstains(self):
        self.launch()
        self.f["kickoff_formation_fraction"] = .4
        self.assertFalse(detect(self.f, [self.event("kick off (start)")], self.meta)["supported"].any())

    def test_outside_persistence_and_exclusive_boundaries(self):
        self.outside()
        r = detect(self.f, [self.event()], self.meta)
        self.assertEqual(runs(r["supported"]), [(105, 195)])
        self.assertEqual(int(r["supported"].sum()), 90)

    def test_single_frame_outlier_abstains(self):
        self.outside(100, 101)
        self.assertFalse(detect(self.f, [self.event()], self.meta)["supported"].any())

    def test_missing_ball_breaks_outside_run(self):
        self.outside()
        self.f.loc[140:159, "ball_valid"] = False
        r = detect(self.f, [self.event()], self.meta)
        self.assertFalse(r["supported"][135:165].any())

    def test_quality_gates_block_false_idle_and_outside(self):
        self.outside()
        for column, value in (("players_seen", 3), ("players_seen", 35), ("home_seen", 2),
                              ("homography_valid", False), ("homography_error", 5)):
            f = self.f.copy()
            f[column] = value
            self.assertFalse(detect(f, [self.event()], self.meta)["supported"].any(), column)

    def test_stationary_setup_needs_launch_and_appropriate_location(self):
        self.f.loc[100:199, ["ball_x_m", "ball_y_m", "ball_speed_mps", "outside_pitch_m"]] = [52.5, 34, .1, -34]
        self.f.loc[100:199, "ball_valid"] = True
        e = self.event("free kick")
        r = detect(self.f, [e], self.meta)
        self.assertTrue(r["setup_review"].any())
        self.assertFalse(r["supported"].any())
        self.launch()
        self.assertTrue(detect(self.f, [e], self.meta)["supported"].any())
        # The same central stationary ball is not throw-in preparation.
        self.assertFalse(detect(self.f, [self.event()], self.meta)["supported"].any())

    def test_player_movement_alone_does_not_prove_restart(self):
        self.f.loc[200:, ["player_median_mps", "player_p90_mps"]] = [3, 5]
        r = detect(self.f, [self.event("kick off (start)")], self.meta)
        self.assertFalse(r["supported"].any())
        self.assertFalse(r["resume"].any())

    def test_overlapping_tags_do_not_double_count_frames(self):
        self.outside()
        e2 = Event("r2", 5, 17, "SET PIECES", "1st half", "N/A", "goal kick")
        r = detect(self.f, [self.event(), e2], self.meta)
        self.assertEqual(r["supported"].sum(), 90)

    def test_positive_movement_vetoes_old_goal_core(self):
        events = [Event("g", 0, 15, "GOALS", "1st half", "Barcelona", ""),
                  Event("p", 0, 2, "PROGRESSION", "1st half", "Barcelona", ""),
                  Event("k", 14, 19, "SET PIECES", "1st half", "N/A", "kick off (after goal)")]
        self.launch()
        r = detect(self.f, events, self.meta)
        self.assertTrue(r["original"][210])
        self.assertFalse(r["combined"][210])
        self.assertTrue(r["combined"][100])

    def test_motion_uses_positions_when_reported_speed_is_zero(self):
        p = np.zeros((self.n, 22, 3), dtype=float)
        p[:, :, 0] = np.arange(self.n)[:, None] * .08  # 2m/s
        b = np.zeros((self.n, 3), dtype=float)
        geometry = (np.ones(self.n, bool), np.zeros(self.n), np.ones(self.n, bool))
        f = features(p, b, self.meta, geometry)
        self.assertAlmostEqual(f.player_median_mps[100], 2, places=4)
        b[100, :2] = 1000
        f = features(p, b, self.meta, geometry)
        self.assertFalse(f.ball_valid[100])
        self.assertFalse(f.ball_valid[101])

    def test_loader_converts_units_and_rejects_frame_gaps(self):
        meta = {"n_frames": 2, "teams": [0, 1], "pitch": [105, 68]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.txt"
            path.write_text("1:0.5,0.5,3.6;0.1,0.2,7.2:0.5,0.5,36\n2:0.5,0.5,3.6;0.1,0.2,7.2:0.5,0.5,36\n")
            players, ball = load_tracking(path, meta)
            np.testing.assert_allclose(players[0, 0], [52.5, 34, 1])
            np.testing.assert_allclose(ball[0], [52.5, 34, 10])
            path.write_text(path.read_text().replace("2:", "3:"))
            with self.assertRaises(ValueError):
                load_tracking(path, meta)

    def test_public_metrics_ignore_unknown_reference(self):
        s = public_metrics(np.array([True, True, True]), np.array(["D", "H", "X"]))
        self.assertEqual(s["precision"], .5)
        self.assertEqual(s["graded_predictions"], 2)


if __name__ == "__main__":
    unittest.main()
