"""Per-frame dead-ball evidence: log-odds that the ball is dead rather than alive."""

import numpy as np

from . import config
from .ball import runs_of


def mean_player_speed(slots, window=config.PLAYER_SPEED_WINDOW):
    """Mean speed (m/s) over tracked slots, then a centred moving average over `window` frames."""
    speed = slots[:, :, 2] if slots.shape[2] > 2 else np.full(slots.shape[:2], np.nan)
    counts = np.sum(~np.isnan(speed), axis=1)
    totals = np.nansum(speed, axis=1)
    per_frame = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    valid = ~np.isnan(per_frame)
    values = np.where(valid, per_frame, 0.0)
    kernel = np.ones(window)
    summed = np.convolve(values, kernel, mode="same")
    n = np.convolve(valid.astype(np.float64), kernel, mode="same")
    return np.where(n > 0, summed / np.maximum(n, 1), np.nan)


def gmm_fit(data, iterations=200):
    """Unsupervised 2-component 1-D Gaussian mixture fitted by EM. Returns (mean, var, weight)
    sorted so component 0 has the lower mean, or None when the data cannot support a fit.

    Initialised from the 20th/80th percentiles, so it is deterministic. No labels involved.
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    if len(data) < 10 or np.std(data) == 0:
        return None
    mean = np.percentile(data, [20, 80]).astype(np.float64)
    var = np.full(2, np.var(data))
    weight = np.array([0.5, 0.5])
    for _ in range(iterations):
        log_p = (-0.5 * (data[:, None] - mean) ** 2 / var - 0.5 * np.log(2 * np.pi * var)
                 + np.log(weight))
        log_p -= log_p.max(axis=1, keepdims=True)
        resp = np.exp(log_p)
        resp /= resp.sum(axis=1, keepdims=True)
        total = resp.sum(axis=0)
        weight = np.maximum(total / len(data), 1e-12)
        mean = (resp * data[:, None]).sum(axis=0) / np.maximum(total, 1e-12)
        var = np.maximum((resp * (data[:, None] - mean) ** 2).sum(axis=0)
                         / np.maximum(total, 1e-12), 1e-6)
    order = np.argsort(mean)
    return mean[order], var[order], weight[order]


def gmm_logodds_at(model, x):
    """log p(low component | x) - log p(high component | x); 0 where x is NaN."""
    mean, var, weight = model
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(len(x))
    seen = ~np.isnan(x)
    values = x[seen]

    def log_component(k):
        return (-0.5 * (values - mean[k]) ** 2 / var[k] - 0.5 * np.log(2 * np.pi * var[k])
                + np.log(weight[k]))

    out[seen] = log_component(0) - log_component(1)
    return out


def gmm_logodds(x, fit_mask, iterations=200):
    """Fit the mixture on x[fit_mask] and return its low-vs-high log-odds for every frame.

    The low-speed component is read as 'stoppage'. This is clustering, not supervision.
    """
    model = gmm_fit(x[fit_mask & ~np.isnan(x)], iterations)
    if model is None:
        return np.zeros(len(x))
    return gmm_logodds_at(model, x)


def gap_out_of_play(ball, near_line=config.NEAR_LINE_M, min_gap=config.GAP_OOB_MIN):
    """True across tracking gaps whose last sighting was within near_line of a boundary."""
    out = np.zeros(len(ball["observed"]), dtype=bool)
    for start, stop in runs_of(~ball["known"]):
        if start == 0 or stop - start < min_gap:
            continue
        if ball["line_distance"][start - 1] <= near_line:
            out[start:stop] = True
    return out


def persistent(mask, min_run):
    """mask, keeping only runs of at least min_run frames."""
    out = np.zeros(len(mask), dtype=bool)
    for start, stop in runs_of(mask):
        if stop - start >= min_run:
            out[start:stop] = True
    return out


def long_gaps(observed, min_gap=config.GAP_DEAD_MIN):
    """True across every ball-missing run of at least min_gap frames."""
    out = np.zeros(len(observed), dtype=bool)
    for start, stop in runs_of(~observed):
        if stop - start >= min_gap:
            out[start:stop] = True
    return out


def setpiece_mask(set_pieces, ball, n_frames):
    """True from each tagged set piece's start until the ball is first seen moving inside it."""
    out = np.zeros(n_frames, dtype=bool)
    moving = ball["observed"] & (ball["speed"] >= config.BALL_MOVING_MS)
    for start, stop in set_pieces:
        start, stop = max(0, start), min(n_frames, stop)
        hits = np.flatnonzero(moving[start:stop])
        if len(hits):
            out[start:start + hits[0]] = True
    return out


def dead_loglik(slots, ball, graded_mask, set_pieces=(), gap_dead=False,
                clean=False, line_aware=False):
    """(T,) log-odds of dead vs alive, summed from independent cues. Also returns the parts."""
    n = len(ball["observed"])
    with np.errstate(invalid="ignore"):
        inside = ball["observed"] & ~ball["oob"]
        moving = inside & (ball["speed"] >= config.BALL_MOVING_MS)
        still = inside & (ball["speed"] < config.BALL_STATIONARY_MS)

    speed_odds = np.clip(gmm_logodds(mean_player_speed(slots), graded_mask),
                         -config.PLAYER_SPEED_CLIP, config.PLAYER_SPEED_CLIP)
    if clean:
        still = persistent(still, config.STILL_MIN)
        with np.errstate(invalid="ignore"):
            clear = inside & (ball["line_distance"] > config.OOB_MARGIN_M) if line_aware \
                else inside
        speed_odds = np.where(clear & (speed_odds > 0), 0.0, speed_odds)
    parts = {
        "player_speed": config.PLAYER_SPEED_WEIGHT * speed_odds,
        "oob": config.OOB_LL * ball["oob"],
        "moving": -config.MOVING_ALIVE_LL * moving,
        "still": config.STATIONARY_DEAD_LL * still,
        "gap_oob": config.GAP_OOB_LL * gap_out_of_play(ball),
        "setpiece": config.SETPIECE_LL * setpiece_mask(set_pieces, ball, n),
    }
    if gap_dead:
        parts["gap_long"] = config.GAP_DEAD_LL * long_gaps(ball["observed"])
    return sum(parts.values()), parts
