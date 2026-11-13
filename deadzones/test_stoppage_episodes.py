import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from deadzones.dead_zones import Event
from deadzones.stoppage_episodes import (EpisodeParams, bridge_observation_gaps,
                                         reconstruct_stoppages)
from deadzones.tracking_dead_zones import Params, detect, export, features


class StoppageEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.n = 800
        self.meta = {"fps": 25, "n_frames": self.n, "pitch": [105, 68],
                     "teams": [0] * 11 + [1] * 11}
        self.f = pd.DataFrame({"frame": np.arange(self.n), "players_seen": 22,
            "home_seen": 11, "away_seen": 11, "player_median_mps": .2,
            "player_p90_mps": .6, "ball_x_m": np.nan, "ball_y_m": np.nan,
            "ball_speed_mps": np.nan, "ball_valid": False, "nearest_player_m": 1.,
            "outside_pitch_m": np.nan, "kickoff_formation_fraction": 1.,
            "homography_valid": True, "homography_error": .2})
        self.event = Event("r", 4, 24, "SET PIECES", "1st half", "N/A", "throw-in")

    def outside(self, lo=100, hi=200):
        self.f.loc[lo:hi-1, ["ball_x_m", "ball_y_m", "ball_speed_mps", "outside_pitch_m"]] = [50, -2, .2, 2]
        self.f.loc[lo:hi-1, "ball_valid"] = True

    def departure(self, lo=200, hi=260):
        self.f.loc[lo:hi-1, "ball_x_m"] = 50.
        self.f.loc[lo:hi-1, "ball_y_m"] = -.8 + .12 * np.arange(hi - lo)
        self.f.loc[lo:hi-1, "outside_pitch_m"] = -self.f.loc[lo:hi-1, "ball_y_m"]
        self.f.loc[lo:hi-1, "ball_speed_mps"] = 3.
        self.f.loc[lo:hi-1, "ball_valid"] = True

    def reconstruct(self, event=None):
        return reconstruct_stoppages(self.f, [event or self.event], self.meta,
            self.f.homography_valid.to_numpy(), Params())

    def test_observed_stop_without_restart_stays_review_only(self):
        self.outside()
        result = detect(self.f, [self.event], self.meta)
        self.assertTrue(result["supported"].any())
        self.assertFalse(result["episode_combined"].any())
        np.testing.assert_array_equal(result["combined"], result["fragment_combined"])
        self.assertEqual(len(result["episode_review"]), 1)

    def test_throw_in_episode_ends_at_observed_reentry_with_guard(self):
        self.outside()
        self.departure()
        result = self.reconstruct()
        self.assertEqual(len(result["episodes"]), 1)
        row = result["episodes"][0]
        self.assertEqual((row["start_frame"], row["end_frame_exclusive"]), (105, 200))
        self.assertEqual(row["restart_motion_frame"], 207)
        self.assertTrue(result["mask"][105:200].all())
        self.assertFalse(result["mask"][200:].any())

    def test_stop_boundary_uses_observed_crossing_before_strong_out_seed(self):
        self.outside()
        self.departure()
        self.f.loc[85:99, "ball_y_m"] = -.1 * (np.arange(85, 100) - 90)
        self.f.loc[85:99, "ball_x_m"] = 50.
        self.f.loc[85:99, "ball_valid"] = True
        self.f.loc[85:99, "ball_speed_mps"] = 2.5
        self.f.loc[85:99, "outside_pitch_m"] = -self.f.loc[85:99, "ball_y_m"]
        row = self.reconstruct()["episodes"][0]
        self.assertEqual(row["stop_observed_frame"], 91)
        self.assertEqual(row["start_frame"], 96)
        self.f.loc[95, "ball_valid"] = False
        self.assertEqual(self.reconstruct()["episodes"][0]["stop_observed_frame"], 100)

    def test_short_bracketed_occlusion_does_not_fragment_stoppage(self):
        self.outside()
        self.departure()
        self.f.loc[130:144, "ball_valid"] = False
        result = self.reconstruct()
        self.assertEqual(len(result["episodes"]), 1)
        self.assertTrue(result["mask"][130:145].all())

    def test_long_occlusion_is_not_filled(self):
        self.outside()
        self.departure()
        self.f.loc[120:184, "ball_valid"] = False
        result = self.reconstruct()
        self.assertFalse(result["mask"][120:185].any())
        self.assertTrue(result["review"])

    def test_geometry_failure_cannot_be_bridged(self):
        self.outside()
        self.departure()
        self.f.loc[130:134, "homography_valid"] = False
        self.assertFalse(self.reconstruct()["mask"][130:135].any())

    def test_visible_infield_motion_interrupts_waiting(self):
        self.outside()
        self.departure()
        self.f.loc[130:149, ["ball_y_m", "outside_pitch_m", "ball_speed_mps"]] = [10, -10, 4]
        self.assertFalse(self.reconstruct()["mask"][130:150].any())

    def test_speed_channel_without_coordinate_departure_is_rejected(self):
        self.outside()
        self.departure()
        self.f.loc[200:259, ["ball_y_m", "outside_pitch_m"]] = [1, -1]
        self.assertFalse(self.reconstruct()["mask"].any())

    def test_reentry_on_wrong_touchline_or_teleport_is_rejected(self):
        self.outside()
        self.departure()
        self.f.loc[200:259, "ball_y_m"] = 60 - .12 * np.arange(60)
        self.f.loc[200:259, "outside_pitch_m"] = -5.
        self.assertFalse(self.reconstruct()["mask"].any())

    def test_touchline_exit_cannot_seed_a_goal_kick(self):
        self.outside()
        self.departure()
        self.assertFalse(self.reconstruct(replace(self.event, kind="goal kick"))["mask"].any())

    def test_goal_kick_launch_does_not_require_whole_team_to_accelerate(self):
        self.f.loc[100:199, ["ball_x_m", "ball_y_m", "ball_speed_mps", "outside_pitch_m"]] = [5, 34, .1, -5]
        self.f.loc[100:259, "ball_valid"] = True
        self.f.loc[200:259, "ball_x_m"] = 5 + .12 * np.arange(60)
        self.f.loc[200:259, ["ball_y_m", "ball_speed_mps", "outside_pitch_m"]] = [34, 3, -5]
        event = replace(self.event, kind="goal kick")
        result = detect(self.f, [event], self.meta)
        self.assertFalse(result["supported"].any())
        self.assertTrue(result["episode_mask"][105:195].all())
        self.assertFalse(result["episode_mask"][195:].any())

    def test_restart_cannot_borrow_a_departure_from_the_next_half(self):
        self.outside()
        self.departure()
        next_half = Event("h2", 8, 12, "SET PIECES", "2nd half", "N/A", "kick off (start)")
        result = reconstruct_stoppages(self.f, [self.event, next_half], self.meta,
            np.ones(self.n, bool), replace(Params(), lookback_s=0))
        self.assertFalse(result["mask"].any())

    def test_missing_half_and_departure_before_tag_abstain(self):
        self.outside()
        self.departure()
        self.assertFalse(self.reconstruct(replace(self.event, half=""))["mask"].any())
        self.assertFalse(self.reconstruct(replace(self.event, start=12))["mask"].any())

    def test_bridge_does_not_extend_edges_or_cross_known_conflicts(self):
        observed = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(bridge_observation_gaps(observed, ~observed, 1),
                                      [False, True, True, True, False])
        np.testing.assert_array_equal(bridge_observation_gaps(observed, observed, 1), observed)

    def test_duplicate_tags_and_input_order_do_not_change_union(self):
        self.outside()
        self.departure()
        other = replace(self.event, id="r2", start=5)
        args = (self.meta, np.ones(self.n, bool), Params())
        a = reconstruct_stoppages(self.f, [self.event, other], *args)
        b = reconstruct_stoppages(self.f, [other, self.event], *args)
        np.testing.assert_array_equal(a["mask"], self.reconstruct()["mask"])
        self.assertEqual(a["episodes"], b["episodes"])

    def test_goal_override_does_not_return_after_restart_occlusion(self):
        events = [Event("g", 0, 15, "GOALS", "1st half", "Barcelona", ""),
                  Event("p", 0, 2, "PROGRESSION", "1st half", "Barcelona", ""),
                  Event("k", 14, 19, "SET PIECES", "1st half", "N/A", "kick off (after goal)")]
        self.f.loc[200:249, ["ball_x_m", "ball_y_m", "ball_speed_mps", "outside_pitch_m",
                           "player_median_mps", "player_p90_mps"]] = [55, 34, 4, -34, 2, 3]
        self.f.loc[200:249, "ball_valid"] = True
        result = detect(self.f, events, self.meta)
        self.assertTrue(result["original"][300])
        self.assertTrue(result["fragment_combined"][300])
        self.assertFalse(result["episode_combined"][200:325].any())

    def test_unreliable_nearby_player_cannot_anchor_setup(self):
        players = np.zeros((self.n, 22, 3))
        players[:, :, :2] = [10, 10]
        players[:, 0, 0] = (np.arange(self.n) - 100) * 50
        players[:, 0, 1] = 0
        ball = np.zeros((self.n, 3))
        geometry = (np.ones(self.n, bool), np.zeros(self.n), np.ones(self.n, bool))
        measured = features(players, ball, self.meta, geometry)
        self.assertAlmostEqual(measured.nearest_player_m[100], np.sqrt(200))

    def test_episode_params_reject_nonfinite_and_negative_values(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                EpisodeParams(max_bridge_s=value)

    def test_export_policies_preserve_ownership_and_report_selected_mask(self):
        self.outside()
        result = detect(self.f, [self.event], self.meta)
        base = np.resize(np.array(["H", "A", "D", "X"]), self.n)
        with tempfile.TemporaryDirectory() as folder:
            for policy, key in (("episodes", "episode_combined"), ("fragments", "fragment_combined")):
                out = Path(folder) / policy
                out.mkdir()
                args = SimpleNamespace(out=out, overlay_policy=policy,
                    **{k: Path(__file__) for k in ("raw", "metadata", "homography", "tags")})
                summary = export(result, self.f, self.meta, args, base=base)
                labels = pd.read_csv(out / "overlaid.csv").label.to_numpy()
                mask = result[key]
                np.testing.assert_array_equal(labels[~mask], base[~mask])
                self.assertTrue((labels[mask] == "D").all())
                evidence = pd.read_csv(out / "frame_evidence.csv.gz")
                np.testing.assert_array_equal(evidence.selected_dead, mask)
                self.assertEqual(summary["overlay_policy"], policy)
                self.assertEqual(list(pd.read_csv(out / "episodes.csv").columns)[0], "event_id")


if __name__ == "__main__":
    unittest.main()
