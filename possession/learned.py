"""Learned per-frame evidence: P(H / A / D | tracking features), trained on OTHER games.

The hand-made evidence (control zones, dead-ball cues) is replaced by a gradient-boosted
classifier trained on the open IDSSE Bundesliga matches (possession/external.py), whose tracking
is first degraded to look like BAR-ATH's and whose answers are converted to this challenge's
definition. No BAR-ATH answers are used anywhere: on BAR-ATH the model only predicts.

    python -m possession.learned crossval      # IDSSE: tune matches (leave-one-out) + held-out

The decoder still does the segmentation. The classifier's log-odds against its training prior,
times LEARNED_WEIGHT, become the per-frame emissions.

Symmetry: every row is also shown with the teams swapped (H <-> A), and predictions average the
two views, so the model cannot learn "home" or "away" as such, only "the team near the ball".
Deterministic: fixed seeds, single-threaded-equivalent results from HistGradientBoosting.
"""

import argparse
import os
import pickle

import numpy as np

from . import ball as ball_mod
from . import config, evidence

CLASSES = ("H", "A", "D")
PAIRED = "__home", "__away"
SIGNED = "__signed"          # home-minus-away quantities: negated when the teams are swapped


# -- small helpers ------------------------------------------------------------------------------

def _window_mean(x, before, after):
    """Mean of x over frames [t-before, t+after], ignoring NaN; NaN if none valid."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    values = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    n = len(x)
    idx = np.arange(n)
    lo = np.clip(idx - before, 0, n)
    hi = np.clip(idx + after + 1, 0, n)
    total = values[hi] - values[lo]
    count = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _gap_features(observed, values):
    """For each frame: frames since the ball was last seen, frames until it is next seen, and
    `values` (dict of (T,) arrays) taken at the last and the next sighting."""
    n = len(observed)
    idx = np.arange(n)
    last = np.where(observed, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(observed, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    cap = 60 * config.FPS
    since = np.where(last >= 0, idx - last, cap).astype(np.float64)
    until = np.where(nxt < n, nxt - idx, cap).astype(np.float64)
    out = {"gap_since": np.minimum(since, cap), "gap_until": np.minimum(until, cap)}
    out["gap_length"] = np.where(observed, 0.0, np.minimum(since + until, 2 * cap))
    for name, v in values.items():
        out[name + "_last_seen"] = np.where(last >= 0, v[np.maximum(last, 0)], np.nan)
        out[name + "_next_seen"] = np.where(nxt < n, v[np.minimum(nxt, n - 1)], np.nan)
    return out


# -- features -----------------------------------------------------------------------------------

def features(arrays, level=1):
    """Dict name -> (T,) float array. Paired features end in __home / __away.

    level 1: the v7 set. level 2 adds who last / next had clear control and how long ago, ball
    direction changes and wider context (_features_extra)."""
    import warnings
    with warnings.catch_warnings():
        # Frames with no tracked player (outside play) give empty means: NaN is intended.
        warnings.simplefilter("ignore", RuntimeWarning)
        f = _features(arrays)
        if level >= 2:
            f.update(_features_extra(arrays, f))
        return f


def _since_until(mask, cap):
    """Frames since `mask` was last True and until it is next True (capped)."""
    n = len(mask)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(mask, idx, -1))
    nxt = np.minimum.accumulate(np.where(mask, idx, n)[::-1])[::-1]
    since = np.where(last >= 0, idx - last, cap).astype(np.float64)
    until = np.where(nxt < n, nxt - idx, cap).astype(np.float64)
    return np.minimum(since, cap), np.minimum(until, cap)


def _features_extra(arrays, f):
    pitch = tuple(float(v) for v in arrays["pitch"])
    ball = ball_mod.track(arrays["ball"], pitch)
    xy = ball["xy"]
    n = len(xy)
    out = {}
    cap = 30 * config.FPS

    # Who was last / is next in clear control (ball in a player's zone, no opponent near).
    state = evidence.control(f["dist" + PAIRED[0]], f["dist" + PAIRED[1]],
                             possession_zone=config.POSSESSION_ZONE_M,
                             duel_zone=config.DUEL_ZONE_M)
    since, until = {}, {}
    for code, suffix in ((evidence.HOME, PAIRED[0]), (evidence.AWAY, PAIRED[1])):
        since[suffix], until[suffix] = _since_until(state == code, cap)
        out["ctrl_since" + suffix] = since[suffix]
        out["ctrl_until" + suffix] = until[suffix]
        out["ctrl_share_500" + suffix] = _window_mean((state == code).astype(np.float64),
                                                      500, 500)
    # Signed: positive when home was the more recent / is the sooner controller.
    out["last_ctrl_lead" + SIGNED] = since[PAIRED[1]] - since[PAIRED[0]]
    out["next_ctrl_lead" + SIGNED] = until[PAIRED[1]] - until[PAIRED[0]]
    duel_since, duel_until = _since_until(state == evidence.DUEL, cap)
    out["duel_since"], out["duel_until"] = duel_since, duel_until

    # Ball direction change and acceleration (touches, passes, shots).
    vel = ball_mod.velocity(xy)
    k = 5
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[k:] = xy[k:] - xy[:-k]
    after[:-k] = xy[k:] - xy[:-k]
    norm = np.hypot(before[:, 0], before[:, 1]) * np.hypot(after[:, 0], after[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(norm > 1e-6, (before * after).sum(axis=1) / norm, np.nan)
    turn = 1.0 - np.clip(cosine, -1.0, 1.0)                      # 0 straight, 2 reversed
    out["ball_turn"] = turn
    out["ball_turn_max_12"] = _window_mean(turn, 12, 12)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    accel = np.full(n, np.nan)
    accel[1:-1] = (speed[2:] - speed[:-2]) * config.FPS / 2.0
    out["ball_accel"] = np.clip(accel, -80.0, 80.0)
    out["ball_speed_change_25"] = _window_mean(speed, 0, 25) - _window_mean(speed, 25, 0)

    # Wider context.
    out["speed_rel_500"] = _window_mean(f["speed_rel_25"], 500, 500)
    out["ball_seen_frac_750"] = _window_mean(f["ball_observed"], 750, 750)
    for suffix in PAIRED:
        out["ctrl_500_500" + suffix] = _window_mean(f["ctrl_now" + suffix], 500, 500)
    return out


def _features(arrays):
    slots, team = arrays["slots"], arrays["team"]
    pitch = tuple(float(v) for v in arrays["pitch"])
    length, width = pitch
    n = slots.shape[0]
    ball = ball_mod.track(arrays["ball"], pitch)
    xy = ball["xy"]
    f = {}

    # Ball state.
    f["ball_observed"] = ball["observed"].astype(np.float64)
    f["ball_known"] = ball["known"].astype(np.float64)
    speed = np.where(ball["known"], ball["speed"], np.nan)
    f["ball_speed"] = np.minimum(speed, 40.0)
    f["line_dist"] = np.clip(ball["line_distance"], -10.0, 40.0)
    f["goal_line_dist"] = np.minimum(xy[:, 0], length - xy[:, 0])
    f["touch_line_dist"] = np.minimum(xy[:, 1], width - xy[:, 1])
    f["oob"] = ball["oob"].astype(np.float64)
    for half_window in (12, 50):
        f["ball_speed_mean_{}".format(half_window)] = _window_mean(f["ball_speed"],
                                                                  half_window, half_window)
    f["ball_seen_frac_75"] = _window_mean(f["ball_observed"], 75, 75)
    f["ball_seen_frac_250"] = _window_mean(f["ball_observed"], 250, 250)
    f.update(_gap_features(ball["observed"], {"line_dist": f["line_dist"],
                                              "ball_speed": f["ball_speed"],
                                              "goal_line_dist": f["goal_line_dist"]}))

    # Players: distances to the ball, counts near it, speeds, shape.
    px = slots[:, :, 0] * length
    py = slots[:, :, 1] * width
    pspeed = slots[:, :, 2] if slots.shape[2] > 2 else np.full(px.shape, np.nan)
    pspeed = np.where(np.isnan(px), np.nan, pspeed)
    dist = np.hypot(px - xy[:, None, 0], py - xy[:, None, 1])
    all_speed = np.nanmean(np.where(np.isfinite(pspeed), pspeed, np.nan), axis=1) \
        if np.isfinite(pspeed).any() else np.full(n, np.nan)
    # Speeds are relative to the match's own median DURING PLAY. arrays["play_mask"] (the graded
    # halves) keeps warm-up, half-time and post-match frames out of that median; without it the
    # median is over every frame (other games' arrays hold the two halves only).
    play = arrays.get("play_mask")
    windowed = _window_mean(all_speed, 25, 25)
    scale = np.nanmedian(windowed[play] if play is not None and play.any() else windowed)
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    rel_speed = all_speed / scale
    for half_window in (25, 75, 150):
        f["speed_rel_{}".format(half_window)] = _window_mean(rel_speed, half_window, half_window)
    f["speed_rel_past_75"] = _window_mean(rel_speed, 75, 0)
    f["speed_rel_future_75"] = _window_mean(rel_speed, 0, 75)
    with np.errstate(invalid="ignore"):
        slow = np.where(np.isnan(pspeed), np.nan, (pspeed < 1.0).astype(np.float64))
    f["slow_frac_25"] = _window_mean(np.nanmean(slow, axis=1) if np.isfinite(slow).any()
                                     else np.full(n, np.nan), 25, 25)

    centroids = {}
    near_home = near_away = None
    for side, suffix in ((0, PAIRED[0]), (1, PAIRED[1])):
        mine = team[None, :] == side
        d = np.where(mine & np.isfinite(dist), dist, np.inf)
        d_sorted = np.sort(d, axis=1)
        nearest = np.where(np.isinf(d_sorted[:, 0]), np.nan, d_sorted[:, 0])
        second = np.where(np.isinf(d_sorted[:, 1]), np.nan, d_sorted[:, 1])
        f["dist" + suffix] = np.minimum(nearest, 60.0)
        f["dist2" + suffix] = np.minimum(second, 60.0)
        known = np.isfinite(xy[:, 0])
        for radius in (3.0, 10.0):
            count = np.sum(d <= radius, axis=1).astype(np.float64)
            f["count_{:g}m".format(radius) + suffix] = np.where(known, count, np.nan)
        with np.errstate(invalid="ignore"):
            cx = np.nanmean(np.where(mine, px, np.nan), axis=1)
            cy = np.nanmean(np.where(mine, py, np.nan), axis=1)
            spread = np.nanmean(np.hypot(px - cx[:, None], py - cy[:, None]) *
                                np.where(mine, 1.0, np.nan), axis=1)
            tspeed = np.nanmean(np.where(mine, pspeed, np.nan), axis=1) / scale
        centroids[side] = (cx, cy)
        f["spread" + suffix] = spread
        f["team_speed_rel_50" + suffix] = _window_mean(tspeed, 50, 50)
        f["centroid_ball_dx" + suffix] = np.abs(cx - xy[:, 0])
        if side == 0:
            near_home = nearest
        else:
            near_away = nearest
    f["centroid_gap"] = np.hypot(centroids[0][0] - centroids[1][0],
                                 centroids[0][1] - centroids[1][1])
    f["dist_min"] = np.fmin(f["dist" + PAIRED[0]], f["dist" + PAIRED[1]])

    # Control evidence over time: soft control (as in the hand-made model) averaged over windows,
    # including past-only and future-only ones so a turnover shows up as a change.
    soft = evidence.soft_control_loglik(near_home, near_away) / config.CONTROL_LL
    for i, suffix in enumerate(PAIRED):
        s = np.where(np.isfinite(xy[:, 0]), soft[:, i], np.nan)
        f["ctrl_now" + suffix] = s
        for before, after in ((12, 12), (50, 50), (125, 125), (75, 0), (0, 75), (250, 0),
                              (0, 250)):
            f["ctrl_{}_{}".format(before, after) + suffix] = _window_mean(s, before, after)
    return f


def swapped(f):
    """Same features with home and away exchanged."""
    out = {}
    for name, values in f.items():
        if name.endswith(PAIRED[0]):
            out[name] = f[name[:-len(PAIRED[0])] + PAIRED[1]]
        elif name.endswith(PAIRED[1]):
            out[name] = f[name[:-len(PAIRED[1])] + PAIRED[0]]
        elif name.endswith(SIGNED):
            out[name] = -np.asarray(values)
        else:
            out[name] = values
    return out


def matrix(f, names):
    return np.column_stack([np.asarray(f[name], dtype=np.float32) for name in names])


# -- model --------------------------------------------------------------------------------------

class Model:
    """Classifier(s) plus what is needed to turn their output into decoder emissions."""

    def __init__(self, classifier, names, log_prior, level=1):
        self.classifiers = list(classifier) if isinstance(classifier, (list, tuple)) \
            else [classifier]
        self.names, self.log_prior, self.level = names, log_prior, level

    @property
    def classifier(self):
        return self.classifiers[0]

    def featurize(self, arrays):
        return features(arrays, getattr(self, "level", 1))

    def log_odds(self, f):
        """(T, 3) log P(class | x) - log P(class), averaged over the two team views (and over
        the classifiers of an ensemble)."""
        direct_x, mirror_x = matrix(f, self.names), matrix(swapped(f), self.names)
        prob = 0.0
        for classifier in self.classifiers:
            prob = prob + classifier.predict_proba(direct_x) \
                + classifier.predict_proba(mirror_x)[:, [1, 0, 2]]
        prob = np.clip(prob / (2.0 * len(self.classifiers)), 1e-6, 1.0)
        return np.log(prob) - self.log_prior


def training_rows(arrays, answers, stride, level=1):
    """Feature rows (original and team-swapped) and class targets, every `stride`-th frame."""
    f = features(arrays, level)
    names = sorted(f)
    target = np.full(len(answers), -1, dtype=np.int8)
    for code, letter in enumerate(CLASSES):
        target[answers == letter] = code
    keep = np.flatnonzero(target >= 0)[::stride]
    x_direct = matrix(f, names)[keep]
    x_mirror = matrix(swapped(f), names)[keep]
    y = target[keep]
    y_mirror = np.where(y == 0, 1, np.where(y == 1, 0, y)).astype(np.int8)
    return names, np.vstack([x_direct, x_mirror]), np.concatenate([y, y_mirror])


def bench_key():
    """Everything that changes what a degraded IDSSE copy looks like."""
    key = tuple((name, getattr(config, name)) for name in sorted(vars(config))
                if name.startswith("EXT_"))
    if config.STOPPAGE_BALL != "frozen":
        key += (("STOPPAGE_BALL", config.STOPPAGE_BALL),)
    return key


def _rows_cached(match, seed, noise, control_min_s, stride, level):
    """training_rows for one degraded match, cached on disk (IDSSE-derived, local only)."""
    import hashlib
    from . import external
    key = repr((match, seed, noise, control_min_s, stride, level, FEATURE_VERSION, bench_key()))
    folder = os.path.join(config.CACHE_DIR, "external", "rows")
    path = os.path.join(folder, hashlib.sha256(key.encode()).hexdigest()[:16] + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as saved:
            return [str(v) for v in saved["names"]], saved["x"], saved["y"]
    arrays, answers, _ = external.prepared((match,), noise, seed=seed,
                                           control_min_s=control_min_s)[match]
    names, x, y = training_rows(arrays, answers, stride, level)
    os.makedirs(folder, exist_ok=True)
    np.savez(path, names=np.array(names), x=x, y=y)
    return names, x, y


FEATURE_VERSION = 2


def train(matches, seeds=None, stride=None, noise=None, control_min_s=None, level=None,
          trees=None, members=None):
    """Fit the classifier(s) on degraded copies of `matches` (IDSSE ids).

    noise may be one value or several (training rows are pooled over them); members > 1 fits
    an ensemble that differs in random_state and in which degradation seed each member skips.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    seeds = config.LEARNED_SEEDS if seeds is None else seeds
    stride = config.LEARNED_STRIDE if stride is None else stride
    noise = config.LEARNED_NOISES if noise is None else noise
    noises = tuple(noise) if isinstance(noise, (tuple, list)) else (noise,)
    level = config.LEARNED_LEVEL if level is None else level
    trees = config.LEARNED_TREES if trees is None else trees
    members = config.LEARNED_MEMBERS if members is None else members
    xs, ys, names = [], [], None
    for level_noise in noises:
        for seed in seeds:
            for match in matches:
                names, x, y = _rows_cached(match, seed, level_noise, control_min_s, stride, level)
                xs.append(x)
                ys.append(y)
    x, y = np.vstack(xs), np.concatenate(ys)
    classifiers = []
    for member in range(members):
        classifier = HistGradientBoostingClassifier(
            max_iter=trees, learning_rate=0.1, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
            random_state=member)
        if members > 1:
            # Each member sees a fixed, different 80% of the rows (deterministic bagging).
            keep = np.random.RandomState(1000 + member).rand(len(y)) < 0.8
            classifier.fit(x[keep], y[keep])
        else:
            classifier.fit(x, y)
        classifiers.append(classifier)
    prior = np.bincount(y, minlength=3) / float(len(y))
    return Model(classifiers, names, np.log(prior), level)


# -- boundary refinement ---------------------------------------------------------------------------
# A second classifier scores "a possession starts or ends right here" per frame; after decoding,
# each boundary moves to the best-scoring frame nearby. spellF1 is an IoU score: a boundary 1 s
# off costs about a third of that possession's credit (EVALUATION.md).

def boundary_target(answers, half):
    """(T,) 1 where a label change (between graded labels) lies within `half` frames."""
    change = np.zeros(len(answers), dtype=bool)
    differs = (answers[1:] != answers[:-1]) & (answers[1:] != "X") & (answers[:-1] != "X")
    change[1:] = differs
    reach = np.convolve(change.astype(np.int32), np.ones(2 * half + 1, dtype=np.int32),
                        mode="same")
    return (reach > 0).astype(np.int8)


def _boundary_rows_cached(match, seed, noise, control_min_s, stride, level, half):
    import hashlib
    from . import external
    key = repr(("boundary", match, seed, noise, control_min_s, stride, level, half,
                FEATURE_VERSION, bench_key()))
    folder = os.path.join(config.CACHE_DIR, "external", "rows")
    path = os.path.join(folder, hashlib.sha256(key.encode()).hexdigest()[:16] + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as saved:
            return [str(v) for v in saved["names"]], saved["x"], saved["y"]
    arrays, answers, _ = external.prepared((match,), noise, seed=seed,
                                           control_min_s=control_min_s)[match]
    f = features(arrays, level)
    names = sorted(f)
    target = boundary_target(answers, half)
    graded = np.flatnonzero(answers != "X")
    # Every boundary-neighbourhood frame, plus every `stride`-th other frame.
    keep = np.union1d(graded[target[graded] == 1], graded[::stride])
    x = np.vstack([matrix(f, names)[keep], matrix(swapped(f), names)[keep]])
    y = np.concatenate([target[keep], target[keep]])
    os.makedirs(folder, exist_ok=True)
    np.savez(path, names=np.array(names), x=x, y=y)
    return names, x, y


class BoundaryModel:
    def __init__(self, classifier, names, level):
        self.classifier, self.names, self.level = classifier, names, level

    def score(self, arrays):
        """(T,) P(boundary here), averaged over the two team views and 5 frames."""
        f = features(arrays, self.level)
        p = 0.5 * (self.classifier.predict_proba(matrix(f, self.names))[:, 1]
                   + self.classifier.predict_proba(matrix(swapped(f), self.names))[:, 1])
        return np.convolve(p, np.ones(5) / 5.0, mode="same")


def train_boundary(matches, seeds=None, stride=None, noise=None, control_min_s=None, level=2,
                   half=None, trees=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    seeds = config.LEARNED_SEEDS if seeds is None else seeds
    stride = config.LEARNED_STRIDE if stride is None else stride
    noise = config.EXT_BALL_NOISE_M if noise is None else noise
    half = config.BOUNDARY_HALF if half is None else half
    xs, ys, names = [], [], None
    for seed in seeds:
        for match in matches:
            names, x, y = _boundary_rows_cached(match, seed, noise, control_min_s, stride, level,
                                                half)
            xs.append(x)
            ys.append(y)
    classifier = HistGradientBoostingClassifier(
        max_iter=config.LEARNED_TREES if trees is None else trees, learning_rate=0.1,
        max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
        random_state=0)
    classifier.fit(np.vstack(xs), np.concatenate(ys))
    return BoundaryModel(classifier, names, level)


def refine_boundaries(labels, score, window=None, margin=None):
    """Move each label change to the best-scoring frame within `window` frames, never past the
    midpoint to the neighbouring change, and only if that frame beats the current one by
    `margin`. labels: (T,) letters; frames labelled X separate independent stretches."""
    window = config.BOUNDARY_WINDOW if window is None else window
    margin = config.BOUNDARY_MARGIN if margin is None else margin
    out = labels.copy()
    changes = np.flatnonzero((labels[1:] != labels[:-1]) & (labels[1:] != "X")
                             & (labels[:-1] != "X")) + 1
    edges = np.flatnonzero((labels[1:] != labels[:-1])) + 1          # includes X edges
    for b in changes.tolist():
        i = int(np.searchsorted(edges, b))
        prev_edge = int(edges[i - 1]) if i > 0 else 0
        next_edge = int(edges[i + 1]) if i + 1 < len(edges) else len(labels)
        lo = max(b - window, (prev_edge + b) // 2 + 1, 1)
        hi = min(b + window, (b + next_edge) // 2)
        if hi <= lo:
            continue
        target = lo + int(np.argmax(score[lo:hi + 1]))
        if target == b or score[target] < score[b] + margin:
            continue
        if target < b:
            out[target:b] = labels[b]
        else:
            out[b:target] = labels[b - 1]
    return out


def merge_short_stoppages(labels, max_frames):
    """Relabel D runs of at most max_frames that sit between two spells of the SAME team."""
    from .ball import runs_of
    out = labels.copy()
    for start, stop in runs_of(labels == "D"):
        if stop - start > max_frames or start == 0 or stop >= len(labels):
            continue
        before, after = labels[start - 1], labels[stop]
        if before == after and before in ("H", "A"):
            out[start:stop] = before
    return out


def model_path(key=None):
    """Cache file for the model trained with settings `key` (one file per setting)."""
    import hashlib
    tag = hashlib.sha256(repr(key).encode()).hexdigest()[:10] if key is not None else "default"
    return os.path.join(config.CACHE_DIR, "learned_model_{}.pkl".format(tag))


class Hybrid:
    """Dead-versus-live from one model, home-versus-away (given live) from another.

    Lets a model with better turnover timing supply the team split without changing which
    frames are called dead. Same interface as Model."""

    def __init__(self, dead_model, team_model):
        self.dead_model, self.team_model = dead_model, team_model
        self.log_prior = dead_model.log_prior
        self.level = max(dead_model.level, team_model.level)

    def featurize(self, arrays):
        return self.dead_model.featurize(arrays), self.team_model.featurize(arrays)

    def log_odds(self, f):
        dead = self.dead_model.log_odds(f[0]) + self.dead_model.log_prior
        team = self.team_model.log_odds(f[1]) + self.team_model.log_prior
        alive = np.logaddexp(dead[:, 0], dead[:, 1])
        split = team[:, :2] - np.logaddexp(team[:, 0], team[:, 1])[:, None]
        log_p = np.column_stack([alive + split[:, 0], alive + split[:, 1], dead[:, 2]])
        return log_p - self.log_prior


def default_model(rebuild=False):
    """The model used on BAR-ATH. With config.LEARNED_TEAM_SETTINGS, a Hybrid whose team split
    comes from a second model trained with those config overrides."""
    model = _cached_model(rebuild)
    overrides = config.LEARNED_TEAM_SETTINGS
    if not overrides:
        return model
    saved = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(config, name, value)
        team_model = _cached_model(rebuild)
    finally:
        for name, value in saved.items():
            setattr(config, name, value)
    return Hybrid(model, team_model)


def _cached_model(rebuild=False):
    """One model for the current config: trained on all IDSSE matches, cached under cache/."""
    from . import external
    external.download()          # pinned, hash-checked; does nothing when already present
    matches = tuple(external.match_ids())
    if config.LEARNED_USE_METRICA:
        from . import metrica
        metrica.download()       # pinned commit, hash-checked
        matches += metrica.MATCHES
    key = (FEATURE_VERSION, config.LEARNED_LEVEL, config.LEARNED_NOISES, config.LEARNED_MEMBERS,
           matches,
           bench_key(),
           config.LEARNED_SEEDS, config.LEARNED_STRIDE, config.LEARNED_TREES,
           config.EXT_BALL_NOISE_M, config.EXT_CONTROL_MIN_S, tuple(external.match_ids()),
           # bench version: 2 = referees excluded, players degraded
           3, config.EXT_LINE_HIDE_M, config.EXT_LINE_HIDE_P, config.EXT_GOAL_GAPS_PER_MIN,
           config.EXT_GOAL_GAP_SCALE_M, config.LEARNED_CONTROL_MIN_S, config.EXT_DEGRADE_PLAYERS, config.EXT_PLAYER_MISSING_SEEN,
           config.EXT_PLAYER_MISSING_GAP, config.EXT_PLAYER_DROP_MEDIAN_S,
           config.EXT_PLAYER_DROP_SIGMA)
    path = model_path(key)
    if not rebuild and os.path.exists(path):
        # The file holds plain parts, never the Model class itself: a pickled class remembers
        # the module it was defined in, which is "__main__" under `python -m possession.learned`.
        try:
            with open(path, "rb") as handle:
                saved_key, parts = pickle.load(handle)
            if saved_key == key and isinstance(parts, dict):
                return Model(parts["classifiers"], parts["names"], parts["log_prior"],
                             parts["level"])
        except Exception:          # unreadable or old-format cache: train again
            pass
    model = train(matches, control_min_s=config.LEARNED_CONTROL_MIN_S)
    with open(path, "wb") as handle:
        pickle.dump((key, {"classifiers": model.classifiers, "names": model.names,
                           "log_prior": model.log_prior, "level": model.level}), handle)
    return model


_MEMO = {}


MODES = ("full", "dead", "team", "blend", "teamgap")


def emissions(model, arrays, mode="full"):
    """(T, 3) emissions for the decoder. The model's output is memoised per (model, arrays)
    object pair, so settings searches over the decoder do not re-predict.

    mode "full": all three columns; "dead": only the dead-vs-alive log-odds (D column, H/A 0);
    "team": only home-vs-away given alive (H/A columns, D 0); "blend": same as full (the caller
    adds it to the hand-made evidence)."""
    lo = _log_odds(model, arrays)
    if mode in ("full", "blend"):
        return config.LEARNED_WEIGHT * lo
    log_p = lo + model.log_prior
    prior = np.exp(model.log_prior)
    alive = np.logaddexp(log_p[:, 0], log_p[:, 1])
    out = np.zeros_like(lo)
    if mode == "dead":
        out[:, 2] = (log_p[:, 2] - model.log_prior[2]) - (alive - np.log(prior[0] + prior[1]))
    elif mode in ("team", "teamgap"):
        for k in (0, 1):
            out[:, k] = (log_p[:, k] - alive) - np.log(prior[k] / (prior[0] + prior[1]))
    else:
        raise ValueError("unknown learned mode {!r}".format(mode))
    return config.LEARNED_WEIGHT * out


def _log_odds(model, arrays):
    key = (id(model), id(arrays["ball"]), id(arrays["slots"]))
    if key not in _MEMO:
        if len(_MEMO) > 32:
            _MEMO.clear()
        _MEMO[key] = (model, arrays["ball"], arrays["slots"], model.log_odds(model.featurize(arrays)))
        # (the key ignores play_mask: one arrays object is always used with the same mask)
    return _MEMO[key][3]


# -- evaluation on other games --------------------------------------------------------------------

def crossval(settings_grid=None, seed=1, control_min_s=None):
    """Leave-one-match-out on the tune matches (choose settings), then train on all tune
    matches and report the held-out test matches once."""
    from . import external
    out = {}
    tune = external.prepared(external.TUNE_MATCHES, config.EXT_BALL_NOISE_M, seed=seed,
                             control_min_s=control_min_s)
    test = external.prepared(external.TEST_MATCHES, config.EXT_BALL_NOISE_M, seed=seed,
                             control_min_s=control_min_s)
    models = {}
    for match in external.TUNE_MATCHES:
        others = tuple(m for m in external.TUNE_MATCHES if m != match)
        models[match] = train(others, control_min_s=control_min_s)
    models["test"] = train(external.TUNE_MATCHES, control_min_s=control_min_s)
    out["models"] = models
    out["tune"] = tune
    out["test"] = test
    return out


def evaluate(models, data, settings=None, options=None, mode="learned"):
    """Mean scores with the learned evidence on, each match scored by the model given for it
    (models[match], else models['test'])."""
    from . import external, pipeline
    settings = settings or {}
    saved = {key: getattr(config, key) for key in settings}
    rows = []
    try:
        for key, value in settings.items():
            setattr(config, key, value)
        for match, (arrays, answers, blocks) in data.items():
            model = models.get(match, models.get("test"))
            opts = dict(options or {}, **{mode: True}, **external.NO_VIDEO)
            labels, _ = pipeline.label(arrays, events=[], blocks=blocks, options=opts,
                                       model=model)
            rows.append(external.score(labels, answers))
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
    return {"spellF1": float(np.mean([r["spellF1"] for r in rows])),
            "macroF1": float(np.mean([r["macroF1"] for r in rows])),
            "poss_ratio": float(np.mean([r["n_pred"] / float(r["n_gt"]) for r in rows]))}


SEARCH = {
    "LEARNED_WEIGHT": (0.25, 0.5, 1.0, 2.0),
    "SWITCH_TURNOVER": (1.0, 2.0, 4.0, 8.0, 16.0),
    "SWITCH_TO_DEAD": (2.0, 4.0, 8.0, 16.0),
    "SWITCH_FROM_DEAD": (4.0, 8.0, 16.0, 32.0),
    "MIN_POSSESSION": (50, 75, 100, 150),
    "MIN_DEAD": (12, 25, 50),
}


def tune(run, rounds=2, search=None):
    """Coordinate search of the weight and decoder settings on the tune matches, each scored
    by the model that did not see it; then the held-out test matches, once."""
    from . import external
    search = search or SEARCH
    current = {key: getattr(config, key) for key in search}
    best = evaluate(run["models"], run["tune"], current)
    print("start  ", current, external._fmt(best), flush=True)
    for _ in range(rounds):
        for key, values in search.items():
            for value in values:
                if value == current[key]:
                    continue
                trial = dict(current, **{key: value})
                result = evaluate(run["models"], run["tune"], trial)
                if result["spellF1"] > best["spellF1"] + 1e-6:
                    current, best = trial, result
                    print("better ", current, external._fmt(best), flush=True)
    print("chosen:", current)
    print("held-out test, chosen:", external._fmt(evaluate(run["models"], run["test"], current)))
    return current


def _feature_summary(f, mask):
    out = {}
    for name, values in f.items():
        v = np.asarray(values, dtype=np.float64)[mask]
        finite = v[np.isfinite(v)]
        q = np.percentile(finite, [10, 25, 50, 75, 90]) if len(finite) else np.full(5, np.nan)
        out[name] = (float(np.mean(~np.isfinite(v))), q)
    return out


def shift():
    """How each feature's distribution on BAR-ATH (graded frames) differs from the degraded
    IDSSE copies the model is trained on. Aggregates only (quantiles and missing rates)."""
    import sys
    from . import external, io_atd
    sys.path.insert(0, config.ROOT)
    from games import GAME, half_video_blocks
    arrays = io_atd.load()
    mask = np.zeros(arrays["slots"].shape[0], dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    ours = _feature_summary(features(dict(arrays, play_mask=mask)), mask)
    theirs = []
    for match, (degraded, answers, _) in external.prepared(
            external.TUNE_MATCHES, config.EXT_BALL_NOISE_M, seed=config.LEARNED_SEEDS[0]).items():
        theirs.append(_feature_summary(features(degraded), np.ones(len(answers), dtype=bool)))
    rows = []
    for name, (nan_ours, q_ours) in ours.items():
        nan_theirs = float(np.mean([t[name][0] for t in theirs]))
        q_theirs = np.nanmean([t[name][1] for t in theirs], axis=0)
        spread = max(q_theirs[3] - q_theirs[1], 1e-6)            # IDSSE interquartile range
        score = float(np.nanmax(np.abs(q_ours - q_theirs)) / spread) if np.isfinite(
            q_ours).any() else float("nan")
        rows.append((score + 5.0 * abs(nan_ours - nan_theirs), name, nan_ours, nan_theirs,
                     q_ours, q_theirs))
    rows.sort(key=lambda r: -r[0] if np.isfinite(r[0]) else 0.0)
    print("{:<30} {:>6} {:>13}  {:<30} {:<30}".format(
        "feature", "shift", "missing o/t", "BAR-ATH p10/p50/p90", "IDSSE p10/p50/p90"))
    for score, name, nan_o, nan_t, q_o, q_t in rows:
        print("{:<30} {:>6.2f} {:>6.2f}/{:<6.2f}  {:<30} {:<30}".format(
            name, score, nan_o, nan_t,
            str([round(float(v), 2) for v in q_o[[0, 2, 4]]]),
            str([round(float(v), 2) for v in q_t[[0, 2, 4]]])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("crossval")
    sub.add_parser("tune")
    sub.add_parser("shift")
    sub.add_parser("train")
    args = parser.parse_args()
    if args.command == "shift":
        shift()
        return
    if args.command == "train":
        default_model(rebuild=True)
        print("trained and cached under", config.CACHE_DIR)
        return
    from . import external
    run = crossval()
    if args.command == "tune":
        tune(run)
        return
    print("hand-made (current defaults):")
    print("  tune  ", external._fmt(external.evaluate({}, run["tune"])))
    print("  test  ", external._fmt(external.evaluate({}, run["test"])))
    print("learned (config defaults):")
    print("  tune  ", external._fmt(evaluate(run["models"], run["tune"])))
    print("  test  ", external._fmt(evaluate(run["models"], run["test"])))


if __name__ == "__main__":
    main()
