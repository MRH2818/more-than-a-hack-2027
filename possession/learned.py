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

def features(arrays):
    """Dict name -> (T,) float array. Paired features end in __home / __away."""
    import warnings
    with warnings.catch_warnings():
        # Frames with no tracked player (outside play) give empty means: NaN is intended.
        warnings.simplefilter("ignore", RuntimeWarning)
        return _features(arrays)


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
    scale = np.nanmedian(_window_mean(all_speed, 25, 25))
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
        else:
            out[name] = values
    return out


def matrix(f, names):
    return np.column_stack([np.asarray(f[name], dtype=np.float32) for name in names])


# -- model --------------------------------------------------------------------------------------

class Model:
    """Classifier plus what is needed to turn its output into decoder emissions."""

    def __init__(self, classifier, names, log_prior):
        self.classifier, self.names, self.log_prior = classifier, names, log_prior

    def log_odds(self, f):
        """(T, 3) log P(class | x) - log P(class), averaged over the two team views."""
        direct = self.classifier.predict_proba(matrix(f, self.names))
        mirror = self.classifier.predict_proba(matrix(swapped(f), self.names))[:, [1, 0, 2]]
        prob = np.clip(0.5 * (direct + mirror), 1e-6, 1.0)
        return np.log(prob) - self.log_prior


def training_rows(arrays, answers, stride):
    """Feature rows (original and team-swapped) and class targets, every `stride`-th frame."""
    f = features(arrays)
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


def train(matches, seeds=None, stride=None, noise=None, control_min_s=None):
    """Fit the classifier on degraded copies of `matches` (IDSSE ids)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from . import external

    seeds = config.LEARNED_SEEDS if seeds is None else seeds
    stride = config.LEARNED_STRIDE if stride is None else stride
    noise = config.EXT_BALL_NOISE_M if noise is None else noise
    xs, ys, names = [], [], None
    for seed in seeds:
        for match, (arrays, answers, _) in external.prepared(matches, noise, seed=seed,
                                                             control_min_s=control_min_s).items():
            names, x, y = training_rows(arrays, answers, stride)
            xs.append(x)
            ys.append(y)
    x, y = np.vstack(xs), np.concatenate(ys)
    classifier = HistGradientBoostingClassifier(
        max_iter=config.LEARNED_TREES, learning_rate=0.1, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
    classifier.fit(x, y)
    prior = np.bincount(y, minlength=3) / float(len(y))
    return Model(classifier, names, np.log(prior))


def model_path():
    return os.path.join(config.CACHE_DIR, "learned_model.pkl")


def default_model(rebuild=False):
    """The model used on BAR-ATH: trained on all IDSSE matches, cached under cache/."""
    from . import external
    external.download()          # pinned, hash-checked; does nothing when already present
    path = model_path()
    key = (config.LEARNED_SEEDS, config.LEARNED_STRIDE, config.LEARNED_TREES,
           config.EXT_BALL_NOISE_M, config.EXT_CONTROL_MIN_S, tuple(external.match_ids()),
           # bench version: 2 = referees excluded, players degraded
           2, config.EXT_DEGRADE_PLAYERS, config.EXT_PLAYER_MISSING_SEEN,
           config.EXT_PLAYER_MISSING_GAP, config.EXT_PLAYER_DROP_MEDIAN_S,
           config.EXT_PLAYER_DROP_SIGMA)
    if not rebuild and os.path.exists(path):
        with open(path, "rb") as handle:
            saved_key, model = pickle.load(handle)
        if saved_key == key:
            return model
    model = train(external.match_ids())
    with open(path, "wb") as handle:
        pickle.dump((key, model), handle)
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
        _MEMO[key] = (model, arrays["ball"], arrays["slots"], model.log_odds(features(arrays)))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("crossval")
    sub.add_parser("tune")
    sub.add_parser("train")
    args = parser.parse_args()
    if args.command == "train":
        default_model(rebuild=True)
        print("wrote", model_path())
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
