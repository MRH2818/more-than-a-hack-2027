"""Synthetic tests for the learned evidence (possession/learned.py). No match data, no IDSSE.

    python -m unittest tests.test_learned
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import isolation  # noqa: E402,F401  (must come before possession imports)

from possession import learned, pipeline  # noqa: E402

PITCH = (105.0, 68.0)


def toy_match(n=6000, seed=0):
    """Two players per team; home holds the ball, then away, then the ball goes dead and
    disappears; repeated. Returns arrays and per-frame answers."""
    rng = np.random.RandomState(seed)
    slots = np.full((n, 4, 3), np.nan)
    team = np.array([0, 1, 0, 1], dtype=np.int8)
    ball = np.full((n, 3), np.nan)
    answers = np.full(n, "D", dtype="<U1")
    slots[:, 2] = [0.3, 0.3, 0.5]
    slots[:, 3] = [0.7, 0.7, 0.5]
    period = 1500
    for start in range(0, n, period):
        for offset, holder, letter in ((0, 0, "H"), (600, 1, "A")):
            a, b = start + offset, min(n, start + offset + 600)
            if a >= n:
                continue
            x = np.linspace(0.3, 0.7, b - a)
            y = 0.5 + 0.01 * rng.randn(b - a)
            slots[a:b, holder] = np.column_stack([x, y, np.full(b - a, 4.0)])
            slots[a:b, 1 - holder] = np.column_stack([x + 0.05, y + 0.05, np.full(b - a, 4.0)])
            ball[a:b] = np.column_stack([x + 0.003, y, np.full(b - a, 6.0)])
            answers[a:b] = letter
        a, b = start + 1200, min(n, start + period)
        if a < n:
            slots[a:b, 0] = [0.5, 0.45, 0.3]
            slots[a:b, 1] = [0.5, 0.55, 0.3]
    arrays = {"slots": slots, "team": team, "ball": ball, "pitch": np.array(PITCH),
              "present": np.ones(n, dtype=bool)}
    return arrays, answers


def swap_teams(arrays):
    return dict(arrays, team=(1 - arrays["team"]).astype(np.int8))


def fit_toy_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    arrays, answers = toy_match()
    names, x, y = learned.training_rows(arrays, answers, stride=3)
    classifier = HistGradientBoostingClassifier(max_iter=30, random_state=0)
    classifier.fit(x, y)
    prior = np.bincount(y, minlength=3) / float(len(y))
    return learned.Model(classifier, names, np.log(prior))


class HelperTests(unittest.TestCase):
    def test_window_mean_matches_brute_force(self):
        rng = np.random.RandomState(0)
        x = rng.randn(200)
        x[rng.rand(200) < 0.3] = np.nan
        for before, after in ((0, 0), (3, 3), (5, 0), (0, 7)):
            got = learned._window_mean(x, before, after)
            for t in range(200):
                window = x[max(0, t - before):t + after + 1]
                expected = np.nanmean(window) if np.isfinite(window).any() else np.nan
                if np.isnan(expected):
                    self.assertTrue(np.isnan(got[t]))
                else:
                    self.assertAlmostEqual(got[t], expected)

    def test_gap_features(self):
        observed = np.array([1, 1, 0, 0, 0, 1, 0], dtype=bool)
        values = {"v": np.arange(7, dtype=np.float64) * 10}
        out = learned._gap_features(observed, values)
        np.testing.assert_array_equal(out["gap_since"][:6], [0, 0, 1, 2, 3, 0])
        np.testing.assert_array_equal(out["gap_until"][:6], [0, 0, 3, 2, 1, 0])
        np.testing.assert_array_equal(out["gap_length"][2:5], [4, 4, 4])
        np.testing.assert_array_equal(out["v_last_seen"][2:5], [10, 10, 10])
        np.testing.assert_array_equal(out["v_next_seen"][2:5], [50, 50, 50])
        self.assertTrue(np.isnan(out["v_next_seen"][6]))


class ExtraFeatureTests(unittest.TestCase):
    def test_since_until(self):
        mask = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
        since, until = learned._since_until(mask, cap=99)
        np.testing.assert_array_equal(since, [99, 0, 1, 2, 0, 1])
        np.testing.assert_array_equal(until, [1, 0, 2, 1, 0, 99])

    def test_level_2_is_a_superset_and_team_symmetric(self):
        arrays, _ = toy_match(n=1500)
        base = learned.features(arrays, 1)
        rich = learned.features(arrays, 2)
        self.assertTrue(set(base) < set(rich))
        mirrored = learned.features(swap_teams(arrays), 2)
        swapped = learned.swapped(rich)
        self.assertTrue(any(name.endswith(learned.SIGNED) for name in rich))
        for name in rich:                    # includes the signed home-minus-away features
            np.testing.assert_allclose(mirrored[name], swapped[name], equal_nan=True,
                                       err_msg=name)


class BoundaryTests(unittest.TestCase):
    def test_boundary_target(self):
        answers = np.array(list("HHHHAAAAXXDD"))
        target = learned.boundary_target(answers, half=1)
        np.testing.assert_array_equal(target, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    def test_refine_moves_to_the_peak_but_not_past_neighbours(self):
        labels = np.array(list("H" * 10 + "A" * 10 + "D" * 10))
        score = np.zeros(30)
        score[7] = 1.0                      # true turnover 3 frames earlier
        score[29] = 1.0                     # far from the A->D boundary, beyond its midpoint
        out = learned.refine_boundaries(labels, score, window=20, margin=0.05)
        self.assertEqual("".join(out), "H" * 7 + "A" * 13 + "D" * 10)

    def test_refine_keeps_boundary_without_a_clear_peak(self):
        labels = np.array(list("H" * 10 + "A" * 10))
        out = learned.refine_boundaries(labels, np.full(20, 0.3), window=5, margin=0.05)
        np.testing.assert_array_equal(out, labels)

    def test_merge_short_stoppages(self):
        labels = np.array(list("HHHDDHHHDDDDAAADH"))
        out = learned.merge_short_stoppages(labels, max_frames=2)
        self.assertEqual("".join(out), "HHHHHHHHDDDDAAADH")


class SymmetryTests(unittest.TestCase):
    def test_swapping_teams_in_the_data_swaps_paired_features(self):
        arrays, _ = toy_match(n=1500)
        f = learned.features(arrays)
        g = learned.features(swap_teams(arrays))
        swapped = learned.swapped(f)
        self.assertEqual(sorted(f), sorted(g))
        for name in f:
            np.testing.assert_allclose(g[name], swapped[name], equal_nan=True, err_msg=name)

    def test_model_is_team_symmetric(self):
        model = fit_toy_model()
        arrays, _ = toy_match(n=3000, seed=1)
        direct = model.log_odds(learned.features(arrays))
        mirror = model.log_odds(learned.features(swap_teams(arrays)))
        np.testing.assert_allclose(direct, mirror[:, [1, 0, 2]], atol=1e-9)


class HybridTests(unittest.TestCase):
    def test_hybrid_of_a_model_with_itself_is_that_model(self):
        model = fit_toy_model()
        arrays, _ = toy_match(n=1500, seed=4)
        hybrid = learned.Hybrid(model, model)
        np.testing.assert_allclose(hybrid.log_odds(hybrid.featurize(arrays)),
                                   model.log_odds(model.featurize(arrays)), atol=1e-9)

    def test_dead_probability_comes_from_the_dead_model(self):
        model = fit_toy_model()
        other = learned.Model(model.classifiers, model.names, model.log_prior)
        other.log_odds = lambda f: model.log_odds(f)[:, [1, 0, 2]] * 0.5   # a different team view
        arrays, _ = toy_match(n=1500, seed=4)
        hybrid = learned.Hybrid(model, other)
        got = hybrid.log_odds(hybrid.featurize(arrays))
        want = model.log_odds(model.featurize(arrays))
        np.testing.assert_allclose(got[:, 2], want[:, 2], atol=1e-9)
        alive = lambda lo: np.logaddexp(lo[:, 0] + model.log_prior[0], lo[:, 1] + model.log_prior[1])
        np.testing.assert_allclose(alive(got), alive(want), atol=1e-9)
        self.assertFalse(np.allclose(got[:, :2], want[:, :2]))


class PlayerDegradeTests(unittest.TestCase):
    def test_dropouts_follow_ball_state_and_are_seeded(self):
        from possession import config, external
        n = 200000
        slots = np.zeros((n, 4, 3))
        missing = np.zeros(n, dtype=bool)
        missing[n // 2:] = True
        a = external.degrade_players(slots, missing, np.random.RandomState(0))
        b = external.degrade_players(slots, missing, np.random.RandomState(0))
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        gone = np.isnan(a[:, :, 0])
        seen_share, gap_share = gone[:n // 2].mean(), gone[n // 2:].mean()
        self.assertLess(seen_share, gap_share)
        self.assertLess(abs(seen_share - config.EXT_PLAYER_MISSING_SEEN), 0.01)
        self.assertGreater(gap_share, 0.5 * config.EXT_PLAYER_MISSING_GAP)
        self.assertFalse(np.isnan(slots).any())          # input untouched


class LearnedPipelineTests(unittest.TestCase):
    def test_model_separates_classes_and_pipeline_decodes(self):
        model = fit_toy_model()
        arrays, answers = toy_match(n=6000, seed=2)
        odds = model.log_odds(learned.features(arrays))
        best = np.array(learned.CLASSES)[np.argmax(odds, axis=1)]
        self.assertGreater(np.mean(best == answers), 0.9)

        blocks = [(1, 0, len(answers) - 1)]
        labels, _ = pipeline.label(arrays, events=[], blocks=blocks,
                                   options={"learned": True, "videoball": False}, model=model)
        self.assertGreater(np.mean(labels == answers), 0.9)
        self.assertTrue(set(np.unique(labels)) <= {"H", "A", "D"})

    def test_cached_model_round_trips_and_survives_a_bad_cache(self):
        import pickle
        import tempfile
        from unittest import mock
        from possession import config, external
        toy = fit_toy_model()
        arrays, _ = toy_match(n=1500, seed=3)
        expected = toy.log_odds(learned.features(arrays))
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(config, "CACHE_DIR", folder), \
                mock.patch.object(external, "download", lambda: None), \
                mock.patch.object(external, "match_ids", lambda: ["toy"]), \
                mock.patch.object(learned, "train", lambda *a, **k: toy) as _:
            first = learned.default_model()
            files = os.listdir(folder)
            self.assertEqual(len(files), 1)
            with open(os.path.join(folder, files[0]), "rb") as handle:
                _, parts = pickle.load(handle)
            self.assertIsInstance(parts, dict)               # plain parts, not the Model class
            again = learned.default_model()                    # loaded from the file
            np.testing.assert_array_equal(again.log_odds(learned.features(arrays)), expected)
            with open(os.path.join(folder, files[0]), "wb") as handle:
                handle.write(b"not a pickle")
            recovered = learned.default_model()                # bad cache -> trains again
            np.testing.assert_array_equal(recovered.log_odds(learned.features(arrays)), expected)
            self.assertIsNotNone(first)

    def test_default_model_is_a_hybrid_when_team_settings_are_given(self):
        import tempfile
        from unittest import mock
        from possession import config, external
        toy = fit_toy_model()
        seen = []

        def fake_train(*args, **kwargs):
            seen.append(config.LEARNED_SEEDS)
            return toy

        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(config, "CACHE_DIR", folder), \
                mock.patch.object(config, "LEARNED_TEAM_SETTINGS", {"LEARNED_SEEDS": (1, 2, 3)}), \
                mock.patch.object(external, "download", lambda: None), \
                mock.patch.object(external, "match_ids", lambda: ["toy"]), \
                mock.patch.object(learned, "train", fake_train):
            before = config.LEARNED_SEEDS
            model = learned.default_model()
            self.assertIsInstance(model, learned.Hybrid)
            self.assertEqual(seen, [before, (1, 2, 3)])
            self.assertEqual(config.LEARNED_SEEDS, before)       # overrides restored
            self.assertEqual(len(os.listdir(folder)), 2)         # one cache file per model
            self.assertIsInstance(learned.default_model(), learned.Hybrid)
            self.assertEqual(len(seen), 2)                       # second call came from cache

    def test_off_by_default_never_trains(self):
        arrays, answers = toy_match(n=3000)
        labels, parts = pipeline.label(arrays, events=[], blocks=[(1, 0, len(answers) - 1)],
                                       options={"videoball": False})
        self.assertFalse(parts["options"]["learned"])


if __name__ == "__main__":
    unittest.main()
