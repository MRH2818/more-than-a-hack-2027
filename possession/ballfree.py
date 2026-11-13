"""Ball-free team evidence: which team is in possession, judged from player positions only.

A team with the ball pushes up, stretches the pitch and moves forward; the team without it drops
and compacts. A logistic model learns that mapping from this match's own frames where the ball
is known, with the first-pass labels as targets (pseudo-labels -- no ground truth involved),
and is applied only where the ball is unknown. Deterministic: closed-form Newton updates.
"""

import warnings

import numpy as np

from . import config

FEATURES = ("centroid_x", "spread_x", "front_x", "spread_y", "mean_speed", "push")


def _nan_reduce(function, values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return function(values, axis=1)


def attack_sign(slots, team, pitch, blocks):
    """(T,) +1 where home attacks towards +x, -1 where towards -x, 0 outside the blocks.

    Read from where each team stands on average over the half: a team defends the goal on its
    own side, so the team with the lower mean x attacks towards +x.
    """
    length = pitch[0]
    x = slots[:, :, 0] * length
    sign = np.zeros(len(slots))
    for _, first, last in blocks:
        window = x[first:last + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            home = np.nanmean(window[:, team == 0])
            away = np.nanmean(window[:, team == 1])
        sign[first:last + 1] = 1.0 if home < away else -1.0
    return sign


def features(slots, team, pitch, blocks, push_half=config.BALLFREE_PUSH_HALF, fps=config.FPS):
    """(T, 6) home-minus-away team-shape features in attacking coordinates (m, m/s)."""
    length, width = pitch
    x = slots[:, :, 0] * length - length / 2.0
    y = slots[:, :, 1] * width
    speed = slots[:, :, 2] if slots.shape[2] > 2 else np.full(x.shape, np.nan)
    sign = attack_sign(slots, team, pitch, blocks)

    per_team = []
    for side in (0, 1):
        direction = sign if side == 0 else -sign
        members = team == side
        forward = x[:, members] * direction[:, None]
        centroid = _nan_reduce(np.nanmean, forward)
        push = np.full(len(x), np.nan)
        push[push_half:-push_half] = ((centroid[2 * push_half:] - centroid[:-2 * push_half])
                                      * fps / (2.0 * push_half))
        per_team.append([centroid,
                         _nan_reduce(np.nanstd, forward),
                         _nan_reduce(np.nanmax, forward),
                         _nan_reduce(np.nanstd, y[:, members]),
                         _nan_reduce(np.nanmean, speed[:, members]),
                         push])
    return np.column_stack([home - away for home, away in zip(*per_team)])


def fit_logistic(X, y, l2=config.BALLFREE_L2, iterations=50):
    """L2-regularised logistic regression by Newton's method. Returns (mean, scale, weights)."""
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    penalty = np.full(Z.shape[1], l2)
    penalty[0] = 0.0
    w = np.zeros(Z.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        gradient = Z.T @ (y - p) - penalty * w
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return mean, scale, w


def predict_logit(model, X):
    mean, scale, w = model
    return np.column_stack([np.ones(len(X)), (X - mean) / scale]) @ w


def emission(arrays, codes, known, blocks, weight=config.BALLFREE_WEIGHT,
             clip=config.BALLFREE_CLIP):
    """(T, 2) H/A log-likelihood on ball-unknown frames, plus aggregate info about the fit.

    codes: first-pass labels as 0 H, 1 A, 2 D, -1 ungraded. known: ball observed/interpolated.
    """
    n = len(codes)
    out = np.zeros((n, 2))
    X = features(arrays["slots"], arrays["team"], tuple(arrays["pitch"]), blocks)
    usable = ~np.isnan(X).any(axis=1)
    fit = usable & known & ((codes == 0) | (codes == 1))
    target = (codes[fit] == 0).astype(np.float64)
    info = {"fit_frames": int(fit.sum())}
    if fit.sum() < 100 or target.min() == target.max():
        return out, dict(info, skipped=True)

    model = fit_logistic(X[fit], target)
    prior = np.log(target.mean() / (1.0 - target.mean()))
    apply = usable & ~known & (codes >= 0)
    ratio = np.clip(predict_logit(model, X[apply]) - prior, -clip, clip)
    out[apply, 0] = 0.5 * weight * ratio
    out[apply, 1] = -0.5 * weight * ratio

    fitted = predict_logit(model, X[fit]) > prior
    info.update({"skipped": False,
                 "pseudo_label_agreement_pct": round(100.0 * float(np.mean(fitted == (target == 1))), 1),
                 "applied_frames": int(apply.sum()),
                 "coefficients": dict(zip(FEATURES, np.round(model[2][1:], 3).tolist()))})
    return out, info
