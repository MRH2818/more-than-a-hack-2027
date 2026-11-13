"""Synthetic tests for restarts, the ball-free model, zone calibration and the masked-ball gate."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

import possession_metrics as pm  # noqa: E402
from possession import ballfree, diagnostics, evidence, masktest, pipeline, restarts  # noqa: E402
from test_pipeline import kickoff_events, synthetic_match  # noqa: E402

PITCH = (105.3, 68.0)


class RestartTests(unittest.TestCase):
    def test_opponent_of_last_toucher_restarts(self):
        codes = np.array([0] * 100 + [2] * 50 + [0] * 100, dtype=np.int8)
        state = np.zeros(250, dtype=np.int8)
        state[60:90] = evidence.HOME              # home had it last
        went_out = np.zeros(250, dtype=bool)
        went_out[95:110] = True                   # ball seen over the line
        prior, info = restarts.restart_prior(codes, state, went_out, window=50, weight=0.2)
        self.assertEqual(info, {"stoppages": 1, "out_of_play": 1, "assigned": 1})
        self.assertTrue(np.allclose(prior[150:200, 1], 0.2))   # away restarts
        self.assertTrue(np.all(prior[:, 0] == 0))
        self.assertTrue(np.all(prior[200:] == 0))

    def test_no_out_of_play_cue_no_prior(self):
        codes = np.array([0] * 100 + [2] * 50 + [0] * 100, dtype=np.int8)
        state = np.full(250, evidence.HOME, dtype=np.int8)
        prior, info = restarts.restart_prior(codes, state, np.zeros(250, dtype=bool))
        self.assertEqual(info["assigned"], 0)
        self.assertFalse(prior.any())

    def test_forced_dead_is_skipped(self):
        codes = np.array([0] * 100 + [2] * 50 + [0] * 100, dtype=np.int8)
        state = np.full(250, evidence.HOME, dtype=np.int8)
        forced = np.zeros(250, dtype=bool)
        forced[100:150] = True
        prior, info = restarts.restart_prior(codes, state, np.ones(250, dtype=bool), forced)
        self.assertEqual(info["stoppages"], 0)
        self.assertFalse(prior.any())


def shape_match(n=4000, seed=0):
    """Home attacks +x. Whoever has the ball sits 15 m further forward and pushes on."""
    rng = np.random.RandomState(seed)
    slots = np.full((n, 6, 3), np.nan)
    team = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    holder = (np.arange(n) // 200) % 2           # alternating spells of 8 s
    for slot in range(6):
        side = team[slot]
        ahead = (holder == side) * 15.0
        own_goal_x = 20.0 if side == 0 else 85.0
        direction = 1.0 if side == 0 else -1.0
        x = own_goal_x + direction * (10.0 + ahead) + rng.randn(n)
        slots[:, slot, 0] = x / PITCH[0]
        slots[:, slot, 1] = (20.0 + 12.0 * (slot % 3) + rng.randn(n)) / PITCH[1]
        slots[:, slot, 2] = 2.0
    codes = np.where(holder == 0, 0, 1).astype(np.int8)
    return {"slots": slots, "team": team, "pitch": np.array(PITCH)}, codes


class BallFreeTests(unittest.TestCase):
    def test_attack_sign(self):
        arrays, _ = shape_match()
        sign = ballfree.attack_sign(arrays["slots"], arrays["team"], PITCH, [(1, 0, 3999)])
        self.assertTrue(np.all(sign == 1.0))

    def test_learns_shape_and_fills_missing_ball(self):
        arrays, codes = shape_match()
        known = np.ones(len(codes), dtype=bool)
        known[2000:] = False                     # ball unknown in the second half of the clip
        fit_codes = codes.copy()
        out, info = ballfree.emission(arrays, fit_codes, known, [(1, 0, 3999)])
        self.assertFalse(info["skipped"])
        self.assertGreater(info["pseudo_label_agreement_pct"], 95)
        unknown = np.arange(len(codes)) >= 2000
        margin = out[:, 0] - out[:, 1]
        right = np.sign(margin[unknown]) == np.where(codes[unknown] == 0, 1, -1)
        self.assertGreater(right[50:-50].mean(), 0.9)
        self.assertTrue(np.all(out[~unknown] == 0))   # only applied where the ball is unknown
        self.assertLessEqual(np.abs(margin).max(), ballfree.config.BALLFREE_WEIGHT *
                             ballfree.config.BALLFREE_CLIP + 1e-12)


class ZoneTests(unittest.TestCase):
    def test_radius_between_modes(self):
        rng = np.random.RandomState(0)
        near = np.exp(rng.randn(6000) * 0.3 + np.log(0.6))
        far = np.exp(rng.randn(4000) * 0.5 + np.log(6.0))
        d = np.concatenate([near, far])
        radius, info = evidence.calibrate_zone(d, d + 1.0, np.ones(len(d), dtype=bool))
        self.assertFalse(info["fallback"])
        self.assertGreater(radius, 0.9)
        self.assertLess(radius, 2.0)

    def test_degenerate_falls_back(self):
        d = np.full(100, 1.0)
        radius, info = evidence.calibrate_zone(d, d, np.ones(100, dtype=bool))
        self.assertTrue(info["fallback"])
        self.assertEqual(radius, evidence.config.POSSESSION_ZONE_M)


class GapDeadTests(unittest.TestCase):
    def test_long_gap_only(self):
        from possession import deadball
        observed = np.ones(300, dtype=bool)
        observed[20:30] = False                  # a 0.4 s glitch
        observed[100:200] = False                # a 4 s gap
        gaps = deadball.long_gaps(observed, min_gap=50)
        self.assertFalse(gaps[20:30].any())
        self.assertTrue(gaps[100:200].all())
        self.assertEqual(int(gaps.sum()), 100)

    def test_gap_in_possession_becomes_dead(self):
        match = synthetic_match()
        gap = slice(11185 + 600, 11185 + 1100)   # 20 s with no ball, slow players
        match["ball"][gap] = np.nan
        match["slots"][gap, :, 2] = 0.5
        plain, _ = pipeline.label(match, events=[], options={"gapdead": False})
        cue, parts = pipeline.label(match, events=[], options={"gapdead": True, "cleandead": True,
                                                "touches": True, "softcontrol": True})
        self.assertTrue(np.all(plain[gap] == "H"))
        self.assertGreater(np.mean(cue[gap] == "D"), 0.9)
        self.assertIn("gap_long", parts["dead_parts"])


class CleanDeadTests(unittest.TestCase):
    def test_brief_still_and_slow_players_do_not_stop_play(self):
        match = synthetic_match()
        lull = slice(11185 + 600, 11185 + 700)   # 4 s lull: ball seen, briefly still, slow players
        start_x = match["ball"][11185 + 600, 0]
        # The synthetic ball drifts below the still threshold; make it roll at ~1.3 m/s into
        # and out of the pause so the pause itself is the only still stretch (1.6 s).
        match["ball"][11185 + 560:11185 + 600, 0] = start_x + np.linspace(-0.02, 0, 40)
        match["ball"][lull, 0] = start_x
        match["ball"][11185 + 640:11185 + 700, 0] += np.linspace(0, 0.03, 60)
        match["slots"][lull, :, 2] = 0.3
        match["slots"][lull, 0:2, 1] += 0.06       # nobody within the possession zone
        plain, _ = pipeline.label(match, events=[], options={"cleandead": False})
        clean, _ = pipeline.label(match, events=[], options={"cleandead": True})
        self.assertGreaterEqual(int(np.sum(plain[lull] == "D")), 25)  # old cues invent one
        self.assertTrue(np.all(clean[lull] == "H"))
        self.assertEqual(clean[11185 + 4200], "D")   # real out-of-play stoppage kept

    def test_persistent(self):
        from possession import deadball
        mask = np.zeros(200, dtype=bool)
        mask[10:30] = True
        mask[100:180] = True
        kept = deadball.persistent(mask, 50)
        self.assertFalse(kept[10:30].any())
        self.assertTrue(kept[100:180].all())


class TaggingRuleTests(unittest.TestCase):
    def test_pre_kickoff_from_tag(self):
        from possession import tagging
        blocks = [(1, 100, 9999), (2, 20000, 39999)]
        events = kickoff_events(first=100, second=22934)
        forced, lengths = tagging.pre_kickoff(events, blocks, 40000)
        self.assertEqual(lengths, [0, 2934])
        self.assertTrue(forced[20000:22934].all())
        self.assertEqual(int(forced.sum()), 2934)

    def test_late_or_missing_kickoff_tag_is_ignored(self):
        from possession import tagging
        blocks = [(1, 100, 99999)]
        late = kickoff_events(first=100 + 6 * 60 * 25, second=10 ** 7)
        forced, lengths = tagging.pre_kickoff(late, blocks, 100000)
        self.assertEqual(lengths, [0])
        self.assertFalse(forced.any())
        forced, lengths = tagging.pre_kickoff([], blocks, 100000)
        self.assertFalse(forced.any())

    def test_kickoff_is_first_ball_movement_after_tag(self):
        from possession import tagging
        blocks = [(1, 100, 9999), (2, 20000, 39999)]
        moving = np.zeros(40000, dtype=bool)
        moving[22934:23500] = True                 # ball first played 82 frames after the tag
        moving[30000:30100] = True
        events = kickoff_events(first=100, second=22852)
        _, lengths = tagging.pre_kickoff(events, blocks, 40000, moving)
        self.assertEqual(lengths, [0, 2934])
        moving[:] = False                          # never seen moving: fall back to the tag
        _, lengths = tagging.pre_kickoff(events, blocks, 40000, moving)
        self.assertEqual(lengths, [0, 2852])

    def test_tagged_restart_prior(self):
        codes = np.array([0] * 100 + [2] * 60 + [0] * 140, dtype=np.int8)
        prior, info = restarts.tagged_prior([(120, 150, 1)], codes, window=50, weight=0.2)
        self.assertEqual(info, {"tagged": 1, "assigned": 1})
        self.assertTrue(np.allclose(prior[160:210, 1], 0.2))   # from first alive frame after tag
        self.assertEqual(float(prior.sum()), 0.2 * 50)

    def test_line_aware_keeps_player_cue_at_the_line(self):
        from possession import deadball
        n = 400
        slots = np.full((n, 2, 3), np.nan)
        slots[:, :, 2] = np.r_[np.full(200, 3.0), np.full(200, 0.3)][:, None]
        ball = {"observed": np.ones(n, dtype=bool), "oob": np.zeros(n, dtype=bool),
                "speed": np.full(n, 1.5), "known": np.ones(n, dtype=bool),
                "line_distance": np.full(n, 0.4)}            # held at the touchline
        graded = np.ones(n, dtype=bool)
        _, plain = deadball.dead_loglik(slots, ball, graded, clean=True)
        _, aware = deadball.dead_loglik(slots, ball, graded, clean=True, line_aware=True)
        self.assertEqual(float(plain["player_speed"][250:350].max()), 0.0)
        self.assertGreater(float(aware["player_speed"][250:350].min()), 0.0)


def straight_then_turn(n=200, turn=100, speed=10.0, fps=25):
    """Ball at `speed` m/s along +x, then back along -x from frame `turn`."""
    x = np.empty(n)
    step = speed / fps
    x[:turn + 1] = 30.0 + step * np.arange(turn + 1)
    x[turn:] = x[turn] - step * np.arange(n - turn)
    xy = np.column_stack([x, np.full(n, 34.0)])
    return {"xy": xy}, xy


class TouchTests(unittest.TestCase):
    def test_turn_next_to_home_player_is_a_home_touch(self):
        from possession import touches
        ball, xy = straight_then_turn()
        d_home = np.full(200, 1.6)               # beyond 1 m: plain control sees nothing
        d_away = np.full(200, 6.0)
        state, info = touches.detect(ball, d_home, d_away)
        hits = np.flatnonzero(state == evidence.HOME)
        self.assertEqual(len(hits), 1)
        self.assertLessEqual(abs(int(hits[0]) - 100), 2)
        self.assertEqual(info["home"], 1)

    def test_steady_ball_has_no_touches(self):
        from possession import touches
        xy = np.column_stack([30.0 + 0.4 * np.arange(200), np.full(200, 34.0)])
        state, info = touches.detect({"xy": xy}, np.full(200, 0.5), np.full(200, 6.0))
        self.assertFalse((state != evidence.NONE).any())
        self.assertEqual(info["velocity_peaks"], 0)

    def test_contested_or_far_touch_gives_nothing(self):
        from possession import touches
        ball, _ = straight_then_turn()
        state, info = touches.detect(ball, np.full(200, 1.6), np.full(200, 2.0))
        self.assertFalse((state != evidence.NONE).any())
        self.assertEqual(info["contested"], 1)
        state, info = touches.detect(ball, np.full(200, 4.0), np.full(200, 8.0))
        self.assertEqual(info["nobody"], 1)

    def test_touches_sharpen_a_turnover(self):
        """Away wins the ball at frame 150 but never comes within 1 m: only touches see it."""
        e = np.zeros((400, 3))
        e[:, 2] = -5
        e[0:100, 0] = 1                          # home in clean control
        touch_state = np.full(400, evidence.NONE, dtype=np.int8)
        touch_state[[150, 175, 200, 240]] = evidence.AWAY
        from possession import decoder, touches
        cost = [[0, 10, 4], [10, 0, 4], [4, 4, 0]]
        plain = decoder.decode(e, [25, 25, 25], cost)
        with_touches = decoder.decode(e + np.c_[touches.loglik(touch_state), np.zeros(400)],
                                      [25, 25, 25], cost)
        self.assertTrue(np.all(plain == decoder.H))
        snapped = decoder.snap_turnovers(with_touches, touch_state, evidence.HOME,
                                         evidence.AWAY, 100)
        self.assertEqual(int(np.flatnonzero(snapped == decoder.A)[0]), 150)

    def test_soft_control(self):
        d_home = np.array([0.5, 1.5, 2.5, 1.5, np.nan])
        d_away = np.array([5.0, 5.0, 5.0, 2.0, np.nan])
        out = evidence.soft_control_loglik(d_home, d_away)
        self.assertEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[1, 0], 0.5)
        self.assertEqual(out[2, 0], 0.0)
        self.assertEqual(out[3, 0], 0.0)         # opponent only 0.5 m farther: contested
        self.assertFalse(out[:, 1].any())
        self.assertFalse(out[4].any())


class TransitionCheckTests(unittest.TestCase):
    def test_offsets(self):
        labels = np.array(["H"] * 100 + ["A"] * 100 + ["D"] * 50 + ["H"] * 50)
        graded = np.ones(300, dtype=bool)
        events = [{"code": "ATTACKING TRANSITION", "start_frame": 125,
                   "tags": {"Team": "Atletic Club"}},        # away gained at 100: 1 s off
                  {"code": "DEFENSIVE TRANSITION", "start_frame": 260,
                   "tags": {"Team": "Atletic Club"}},        # home gained at 250: 0.4 s off
                  {"code": "ATTACKING TRANSITION", "start_frame": 10,
                   "tags": {"Team": "N/A"}}]                 # no team: ignored
        out = diagnostics.transition_agreement(labels, events, graded)
        self.assertEqual(out["transition_tags"], 2)
        self.assertEqual(out["trans_within_2s_pct"], 100.0)
        self.assertEqual(out["trans_median_s"], 0.7)


class MaskTests(unittest.TestCase):
    def test_mask_hides_only_seen_frames_with_margins(self):
        ball_xy = np.zeros((5000, 3))
        ball_xy[1000:1100] = np.nan                 # a real gap
        graded = np.zeros(5000, dtype=bool)
        graded[500:4500] = True
        masked, hidden = masktest.mask_ball(ball_xy, graded, 0.2, seed=0,
                                            blocks=[(1, 500, 4499)])
        seen = ~np.isnan(ball_xy[:, 0])
        self.assertTrue(np.all(seen[hidden] & graded[hidden]))
        self.assertTrue(np.all(np.isnan(masked[hidden, 0])))
        self.assertGreater(hidden.sum(), 0.19 * (seen & graded).sum())
        self.assertFalse(hidden[999] or hidden[1100])  # never touching the real gap
        again, hidden_again = masktest.mask_ball(ball_xy, graded, 0.2, seed=0,
                                                 blocks=[(1, 500, 4499)])
        self.assertTrue(np.array_equal(hidden, hidden_again))  # deterministic


class FullPipelineWithOptions(unittest.TestCase):
    def test_all_options_valid_and_sensible(self):
        labels, parts = pipeline.label(synthetic_match(), events=kickoff_events(),
                                       options={"tagrestarts": True, "lineaware": True,
                                                "restarts": True, "ballfree": True, "zone": True,
                                                "gapdead": True, "cleandead": True,
                                                "touches": True, "softcontrol": True})
        frame = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
        self.assertIsNone(pm.validate_submission(frame, 199319))
        self.assertEqual(labels[11185 + 1000], "H")
        self.assertEqual(labels[11185 + 3000], "A")
        self.assertEqual(labels[11185 + 4200], "D")
        self.assertTrue(np.all(labels[110808:113742] == "D"))
        summary = diagnostics.summarise(labels, parts, [])
        self.assertIn("restarts", summary)
        self.assertIn("ballfree", summary)

    def test_unknown_option_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.resolve_options({"magic": True})

    def test_tag_agreement(self):
        labels = np.array(["H"] * 100 + ["D"] * 50 + ["A"] * 100)
        graded = np.ones(250, dtype=bool)
        events = [{"code": "BUILD UP", "start_frame": 0, "end_frame": 100,
                   "tags": {"Team": "Barcelona"}},
                  {"code": "DEFENDING IN MIDDLE THIRD", "start_frame": 150, "end_frame": 250,
                   "tags": {"Team": "Athletic Club"}},
                  {"code": "SET PIECES", "start_frame": 140, "end_frame": 160, "tags": {}}]
        out = diagnostics.tag_agreement(labels, events, graded)
        self.assertEqual(out["phase_team_agree_pct"], 50.0)   # defending away => home has it
        self.assertEqual(out["setpiece_near_dead_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
