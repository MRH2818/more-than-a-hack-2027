import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from deadzones.dead_zones import Event
from deadzones.restart_detection import (RestartParams, apply_restart_overlay,
                                         assign_restart_teams, tag_team_label)
from deadzones.tracking_dead_zones import (DEFAULT_OVERLAY_POLICY, detect, export, features)


class RestartTeamTests(unittest.TestCase):
    def setUp(self):
        self.n = 400
        self.meta = {"fps": 25, "n_frames": self.n, "pitch": [105, 68],
                     "team_names": ["Home Club", "Away Club"], "teams": [0] * 11 + [1] * 11}
        self.f = pd.DataFrame({"frame": np.arange(self.n), "ball_valid": True,
                               "outside_pitch_m": -10., "nearest_home_m": 10.,
                               "nearest_away_m": 10.})
        self.quality = np.ones(self.n, bool)
        self.detection = {"event_id": "r", "half": "1st half", "restart_type": "free kick",
                          "restart_earliest_frame": 100, "restart_latest_frame": 110,
                          "stop_observed_frame": 50, "window_end_frame_exclusive": 300,
                          "tag_team": "Home Club"}

    def evidence(self, team="H", start=80, end=136):
        column = "nearest_home_m" if team == "H" else "nearest_away_m"
        other = "nearest_away_m" if team == "H" else "nearest_home_m"
        self.f.loc[start:end-1, column] = 1.
        self.f.loc[start:end-1, other] = 6.

    def infer(self, detections=None, params=RestartParams()):
        return assign_restart_teams(self.f, detections or [self.detection], self.meta,
                                    self.quality, params)

    def test_tag_and_tracking_assign_only_after_upper_bound_for_one_second(self):
        self.evidence()
        result = self.infer()
        self.assertTrue((result["labels"][111:136] == "H").all())
        self.assertTrue((result["labels"][:111] == "X").all())
        self.assertTrue((result["labels"][136:] == "X").all())
        self.assertTrue(result["uncertain"][100:111].all())
        self.assertEqual(result["assignments"][0]["source"], "tag_and_pre_tracking")

    def test_away_mapping_is_from_metadata_not_home_bias(self):
        self.evidence("A")
        self.detection["tag_team"] = " away   CLUB "
        self.assertTrue((self.infer()["labels"][111:136] == "A").all())
        self.assertEqual(tag_team_label("Other Club", self.meta), "X")

    def test_tag_alone_never_assigns(self):
        result = self.infer()
        self.assertTrue((result["labels"] == "X").all())
        self.assertEqual(result["assignments"][0]["reason"], "Tag has no corroborating tracking")

    def test_tag_conflict_with_tracking_abstains(self):
        self.evidence("A")
        self.assertTrue((self.infer()["labels"] == "X").all())

    def test_unknown_tag_uses_sustained_post_tracking_without_backfilling_flight(self):
        self.detection["tag_team"] = "N/A"
        self.evidence("A", start=119)
        result = self.infer()
        self.assertTrue((result["labels"][:119] == "X").all())
        self.assertTrue((result["labels"][119:136] == "A").all())
        self.assertEqual(result["assignments"][0]["source"], "post_tracking")

    def test_missing_metadata_names_does_not_guess_a_tag_team(self):
        self.meta.pop("team_names")
        self.evidence("A", start=111)
        result = self.infer()
        self.assertEqual(result["assignments"][0]["tag_team_label"], "X")
        self.assertTrue((result["labels"][111:136] == "A").all())

    def test_unknown_tag_with_only_pre_tracking_abstains(self):
        self.detection["tag_team"] = "N/A"
        self.evidence(end=100)
        self.assertTrue((self.infer()["labels"] == "X").all())

    def test_pre_post_conflict_abstains_even_without_tag(self):
        self.detection["tag_team"] = "N/A"
        self.evidence(end=100)
        self.evidence("A", start=111)
        self.assertTrue((self.infer()["labels"] == "X").all())

    def test_one_frame_nearest_player_is_not_sustained_evidence(self):
        self.evidence(start=111, end=112)
        self.assertTrue((self.infer()["labels"] == "X").all())

    def test_equal_distance_contest_and_missing_team_tracks_abstain(self):
        self.evidence()
        self.f["nearest_away_m"] = 1.2
        self.assertTrue((self.infer()["labels"] == "X").all())
        self.f["nearest_away_m"] = np.nan
        self.assertTrue((self.infer()["labels"] == "X").all())

    def test_stale_pre_tracking_cannot_authorize_flight_assignment(self):
        self.evidence(start=75, end=85)
        self.assertTrue((self.infer()["labels"] == "X").all())

    def test_wide_bounds_and_missing_features_abstain(self):
        self.evidence()
        self.detection["restart_earliest_frame"] = 75
        self.assertTrue((self.infer()["labels"] == "X").all())
        self.detection["restart_earliest_frame"] = 100
        self.f = self.f.drop(columns=["nearest_home_m", "nearest_away_m"])
        self.assertTrue((self.infer()["labels"] == "X").all())

    def test_missing_ball_quality_loss_and_ball_out_stop_assignment(self):
        for kind in ("ball", "quality", "outside"):
            self.setUp()
            self.evidence(end=100)  # known tag and pre-contact tracking agree
            if kind == "ball":
                self.f.loc[120, "ball_valid"] = False
            elif kind == "quality":
                self.quality[120] = False
            else:
                self.f.loc[120, "outside_pitch_m"] = 1.
            result = self.infer()
            self.assertTrue((result["labels"][111:120] == "H").all(), kind)
            self.assertTrue((result["labels"][120:] == "X").all(), kind)

    def test_opponent_touch_interrupts_assignment_instead_of_being_smoothed_away(self):
        self.evidence(end=100)
        self.evidence("A", start=120, end=121)
        result = self.infer()
        self.assertTrue((result["labels"][111:120] == "H").all())
        self.assertTrue((result["labels"][120:] == "X").all())

    def test_event_limit_and_next_restart_limit_ownership_window(self):
        self.evidence()
        self.detection["window_end_frame_exclusive"] = 120
        self.assertTrue((self.infer()["labels"][120:] == "X").all())
        self.detection["window_end_frame_exclusive"] = 300
        other = {**self.detection, "event_id": "r2", "restart_earliest_frame": 120,
                 "restart_latest_frame": 130, "tag_team": "N/A"}
        result = self.infer([self.detection, other])
        first = next(r for r in result["assignments"] if r["event_id"] == "r")
        self.assertEqual(first["assignment_end_frame_exclusive"], 120)
        self.assertTrue((result["labels"][120:131] == "X").all())

    def test_overlapping_opposing_tags_abstain_and_duplicates_are_order_independent(self):
        self.evidence()
        other = {**self.detection, "event_id": "r2", "tag_team": "Away Club"}
        self.assertTrue((self.infer([self.detection, other])["labels"] == "X").all())
        unknown = {**self.detection, "event_id": "r3", "tag_team": "N/A"}
        self.assertTrue((self.infer([self.detection, other, unknown])["labels"] == "X").all())
        other["tag_team"] = "Home Club"
        a = self.infer([self.detection, other])
        b = self.infer([other, self.detection])
        np.testing.assert_array_equal(a["labels"], self.infer()["labels"])
        self.assertEqual(a["assignments"], b["assignments"])

    def test_overlay_preserves_base_outside_masks_and_stops_at_new_dead_interval(self):
        self.evidence()
        result = self.infer()
        base = np.resize(np.array(["H", "A", "D", "X"]), self.n)
        dead = np.zeros(self.n, bool)
        dead[120:123] = True
        output, applied = apply_restart_overlay(base, dead, result)
        self.assertTrue((output[111:120] == "H").all())
        self.assertTrue((output[120:123] == "D").all())
        self.assertFalse(applied[120:].any())
        np.testing.assert_array_equal(output[~(dead | applied)], base[~(dead | applied)])
        off, applied_off = apply_restart_overlay(base, dead, result, enabled=False)
        np.testing.assert_array_equal(off, np.where(dead, "D", base))
        self.assertFalse(applied_off.any())

    def test_invalid_bounds_parameters_and_overlay_alignment_are_rejected(self):
        for key, value in (("local_window_s", 0), ("max_uncertainty_s", float("nan")),
                           ("min_vote_share", 1.1), ("min_team_evidence_s", 2)):
            with self.assertRaises(ValueError):
                RestartParams(**{key: value})
        self.detection["restart_latest_frame"] = self.n
        with self.assertRaises(ValueError):
            self.infer()
        self.detection["restart_latest_frame"] = 110
        with self.assertRaises(ValueError):
            apply_restart_overlay(["H"], np.zeros(1, bool), self.infer())


class RestartIntegrationTests(unittest.TestCase):
    def fixture(self):
        n = 400
        meta = {"fps": 25, "n_frames": n, "pitch": [105, 68],
                "teams": [0] * 11 + [1] * 11, "team_names": ["Home Club", "Away Club"]}
        f = pd.DataFrame({"frame": np.arange(n), "players_seen": 22, "home_seen": 11,
            "away_seen": 11, "player_median_mps": .2, "player_p90_mps": .6,
            "ball_x_m": np.nan, "ball_y_m": np.nan, "ball_speed_mps": np.nan,
            "ball_valid": False, "nearest_player_m": 1., "nearest_home_m": 1.,
            "nearest_away_m": 6., "outside_pitch_m": np.nan,
            "kickoff_formation_fraction": 1., "homography_valid": True, "homography_error": .2})
        f.loc[100:199, ["ball_x_m", "ball_y_m", "ball_speed_mps", "outside_pitch_m"]] = [5, 34, .1, -5]
        f.loc[100:259, "ball_valid"] = True
        f.loc[200:259, "ball_x_m"] = 5 + .12 * np.arange(60)
        f.loc[200:259, ["ball_y_m", "ball_speed_mps", "outside_pitch_m"]] = [34, 3, -5]
        event = Event("r", 4, 14, "SET PIECES", "1st half", "Home Club", "goal kick")
        return f, meta, event

    def test_detector_exports_nonzero_bounds_and_actual_confirmation_latency(self):
        f, meta, event = self.fixture()
        result = detect(f, [event], meta)
        detection = result["restart_detections"][0]
        self.assertEqual((detection["restart_earliest_frame"], detection["restart_latest_frame"]), (195, 205))
        # 1.5m setup displacement takes longer than the initial 0.4s motion test.
        self.assertGreater(detection["confirmation_frame"], 209)
        self.assertEqual(detection["confirmation_available_frame"], detection["confirmation_frame"] + 5)
        self.assertEqual(detection["bounds_kind"], "heuristic_observation_bracket")
        self.assertFalse(detection["frame_verified"])
        self.assertTrue((result["restart_teams"]["labels"][206:231] == "H").all())

    def test_default_export_keeps_episodes_experimental_but_applies_local_teams(self):
        f, meta, event = self.fixture()
        result = detect(f, [event], meta)
        self.assertEqual(DEFAULT_OVERLAY_POLICY, "fragments")
        self.assertFalse(result["combined"].any())
        self.assertTrue(result["episode_combined"].any())
        base = np.full(len(f), "A")
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(out=Path(folder), **{k: Path(__file__)
                for k in ("raw", "metadata", "homography", "tags")})
            summary = export(result, f, meta, args, base=base)
            self.assertEqual(summary["overlay_policy"], "fragments")
            self.assertTrue(summary["episode_policy_experimental"])
            self.assertEqual(summary["restart_team_changed_frames"], 25)
            labels = pd.read_csv(Path(folder) / "overlaid.csv").label.to_numpy()
            self.assertTrue((labels[:206] == "A").all())
            self.assertTrue((labels[206:231] == "H").all())
            self.assertTrue((labels[231:] == "A").all())
            bounds = pd.read_csv(Path(folder) / "restart_detections.csv")
            self.assertEqual(bounds.restart_earliest_frame[0], 195)
            self.assertEqual(bounds.restart_latest_frame[0], 205)
            self.assertEqual(pd.read_csv(Path(folder) / "restart_assignments.csv").status[0], "assigned")
            self.assertEqual(pd.read_csv(Path(folder) / "restart_assignments.csv").applied_frames[0], 25)
            args.overlay_policy = "episodes"
            args.restart_team_policy = "off"
            export(result, f, meta, args, base=base)
            labels = pd.read_csv(Path(folder) / "overlaid.csv").label.to_numpy()
            np.testing.assert_array_equal(labels, np.where(result["episode_combined"], "D", base))
            self.assertEqual(pd.read_csv(Path(folder) / "restart_assignments.csv").applied_frames[0], 0)

    def test_per_team_nearest_distances_exclude_unreliable_tracks(self):
        n = 100
        meta = {"fps": 25, "teams": [0, 0, 1], "pitch": [105, 68]}
        players = np.zeros((n, 3, 3))
        players[:, 0, 0] = (np.arange(n) - 50) * 50
        players[:, 1, 0] = 5
        players[:, 2, 0] = 2
        ball = np.zeros((n, 3))
        geometry = (np.ones(n, bool), np.zeros(n), np.ones(n, bool))
        f = features(players, ball, meta, geometry)
        self.assertEqual(f.nearest_home_m[50], 5)
        self.assertEqual(f.nearest_away_m[50], 2)


if __name__ == "__main__":
    unittest.main()
