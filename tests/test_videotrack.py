"""Synthetic tests for the video ball path filter and its pipeline switch. Never reads cache/."""

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

from possession import pipeline, videotrack  # noqa: E402
from test_pipeline import kickoff_events, synthetic_match  # noqa: E402

PITCH = np.array([105.3, 68.0])


def scene(g=100, k=5, speed=10.0, seed=0):
    """True ball rolling along x at `speed` m/s; slot 0 sees it (conf 0.6, 0.3 m noise)."""
    rng = np.random.RandomState(seed)
    truth = np.column_stack([20.0 + speed / 25.0 * np.arange(g), np.full(g, 30.0)])
    pos = np.full((g, k, 2), np.nan)
    conf = np.full((g, k), np.nan)
    pos[:, 0] = truth + rng.randn(g, 2) * 0.3
    conf[:, 0] = 0.6
    return truth, pos, conf


class ChainTests(unittest.TestCase):
    def test_follows_ball_and_ignores_static_decoy(self):
        truth, pos, conf = scene()
        pos[:, 1] = [80.0, 10.0]                 # a head/line mark detected every frame
        conf[:, 1] = 0.4
        out = videotrack._chain_gap(pos, conf, truth[0] - [0.4, 0], videotrack.default_params())
        err = np.hypot(*(out - truth).T)
        self.assertTrue(np.all(np.isfinite(err)))
        self.assertLess(np.percentile(err, 95), 1.5)

    def test_isolated_low_confidence_hits_are_dropped(self):
        g = 60
        pos = np.full((g, 3, 2), np.nan)
        conf = np.full((g, 3), np.nan)
        for t in (5, 25, 45):                    # single frames, far apart, weak
            pos[t, 0] = [50.0 + t, 20.0]
            conf[t, 0] = 0.3
        out = videotrack._chain_gap(pos, conf, None, videotrack.default_params())
        self.assertTrue(np.isnan(out).all())

    def test_short_misses_are_bridged(self):
        truth, pos, conf = scene()
        pos[40:48] = np.nan                      # detector misses 8 frames
        conf[40:48] = np.nan
        out = videotrack._chain_gap(pos, conf, truth[0], videotrack.default_params())
        self.assertTrue(np.all(np.isfinite(out[40:48])))
        self.assertLess(np.max(np.hypot(*(out[40:48] - truth[40:48]).T)), 1.5)

    def test_speed_limit_breaks_impossible_jumps(self):
        truth, pos, conf = scene(g=40)
        pos[20:25] = np.nan                      # 5 frames unseen...
        conf[20:25] = np.nan
        pos[25:, 0, 0] += 40.0                   # ...then 40 m away: no real ball does that
        out = videotrack._chain_gap(pos, conf, truth[0], videotrack.default_params())
        self.assertTrue(np.isnan(out[20:25]).all())        # not bridged by a fake path
        self.assertTrue(np.isfinite(out[:20]).all())       # both real chains still kept
        self.assertTrue(np.isfinite(out[25:]).all())

    def test_fill_only_touches_atd_gaps(self):
        n = 300
        truth = np.column_stack([10.0 + 0.4 * np.arange(n), np.full(n, 30.0)])
        known = np.ones(n, dtype=bool)
        known[100:200] = False
        pos = np.full((n, 5, 2), np.nan)
        conf = np.full((n, 5), np.nan)
        pos[:, 0] = truth
        conf[:, 0] = 0.7
        out, info = videotrack.fill(np.where(known[:, None], truth, np.nan), known, pos, conf,
                                    np.ones(n, dtype=bool))
        self.assertTrue(np.isnan(out[known]).all())
        self.assertLess(np.nanmax(np.hypot(*(out[100:200] - truth[100:200]).T)), 1e-6)
        self.assertEqual(info["gaps"], 1)
        self.assertEqual(info["filled_pct_of_gaps"], 100.0)

    def test_candidates_on_pitch(self):
        h = np.diag([1920.0, 1080.0, 1.0])
        matrices = np.repeat(h[None], 2, axis=0)
        candidates = np.full((2, 2, 5), np.nan)
        candidates[0, 0] = [960, 540, 0.8, 10, 10]   # pitch centre
        candidates[1, 0] = [9000, 540, 0.8, 10, 10]  # far off the pitch
        pos, conf = videotrack.candidates_on_pitch(candidates, matrices, PITCH)
        np.testing.assert_allclose(pos[0, 0], PITCH / 2)
        self.assertTrue(np.isnan(pos[1, 0]).all() and np.isnan(conf[1, 0]))


class PipelineSwitchTests(unittest.TestCase):
    def test_missing_detections_fall_back_cleanly(self):
        original = videotrack.load_candidates
        videotrack.load_candidates = lambda *a, **k: None
        try:
            labels, parts = pipeline.label(synthetic_match(), events=kickoff_events(),
                                           options={"videoball": True})
        finally:
            videotrack.load_candidates = original
        self.assertEqual(parts["video"], {"available": False})
        self.assertEqual(labels[11185 + 1000], "H")

    def test_video_fills_a_live_gap(self):
        """ATD loses the ball mid-possession while away holds it; the video still sees it."""
        match = synthetic_match()
        n = len(match["ball"])
        gap = slice(11185 + 2500, 11185 + 3500)      # inside the away spell, 40 s without ball
        video_norm = match["ball"][:, :2].copy()
        match["ball"][gap] = np.nan
        candidates = np.full((n, 5, 5), np.nan)
        candidates[:, 0, 0] = video_norm[:, 0] * 1920
        candidates[:, 0, 1] = video_norm[:, 1] * 1080
        candidates[:, 0, 2] = 0.7
        done = np.ones(n, dtype=bool)
        h = np.diag([1920.0, 1080.0, 1.0])

        import load_homography
        original = (videotrack.load_candidates, load_homography.load)
        videotrack.load_candidates = lambda *a, **k: (candidates, done)
        load_homography.load = lambda *a, **k: (np.repeat(h[None], n, axis=0), None, None)
        try:
            _, plain = pipeline.label(match, events=kickoff_events(),
                                      options={"videoball": False})
            _, video = pipeline.label(match, events=kickoff_events(),
                                      options={"videoball": True})
        finally:
            videotrack.load_candidates, load_homography.load = original
        self.assertTrue(video["video"]["available"])
        from possession import evidence
        self.assertFalse((plain["control"][gap] == evidence.AWAY).any())
        self.assertGreater(np.mean(video["control"][gap] == evidence.AWAY), 0.9)


if __name__ == "__main__":
    unittest.main()
