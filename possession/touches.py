"""Touches: moments the ball's velocity changes sharply next to one team's player.

The ball does not change speed or direction on its own (beyond slow friction), so a sharp change
means a player played it: a reception, pass, shot, clearance or interception. This gives team
evidence at exactly the moments that matter for turnovers, at distances where a plain "within
1 m" rule sees nothing.
"""

import numpy as np

from . import ball as ball_mod
from . import config, evidence


def velocity_change(xy, half=config.TOUCH_HALF, fps=config.FPS):
    """(T,) |v(t + half) - v(t - half)| in m/s, NaN where either side is unknown."""
    vel = ball_mod.velocity(xy, fps)
    out = np.full(len(xy), np.nan)
    if len(xy) > 2 * half:
        out[half:-half] = np.hypot(vel[2 * half:, 0] - vel[:-2 * half, 0],
                                   vel[2 * half:, 1] - vel[:-2 * half, 1])
    return out


def peaks(values, threshold, half):
    """Frames where values >= threshold and is the maximum within +-half (first one on ties)."""
    v = np.where(np.isnan(values), -np.inf, values)
    out = []
    for t in np.flatnonzero(v >= threshold):
        lo, hi = max(0, t - half), min(len(v), t + half + 1)
        window = v[lo:hi]
        if v[t] == window.max() and lo + int(np.argmax(window)) == t:
            out.append(int(t))
    return np.array(out, dtype=np.int64)


def detect(ball, d_home, d_away, half=config.TOUCH_HALF, threshold=config.TOUCH_DV_MS,
           radius=config.TOUCH_RADIUS_M, margin=config.TOUCH_MARGIN_M):
    """(T,) int8 of evidence.HOME / AWAY at touch frames, NONE elsewhere; plus aggregate info."""
    n = len(d_home)
    out = np.full(n, evidence.NONE, dtype=np.int8)
    candidates = peaks(velocity_change(ball["xy"], half), threshold, half)
    info = {"velocity_peaks": len(candidates), "home": 0, "away": 0, "contested": 0, "nobody": 0}
    for t in candidates:
        lo, hi = max(0, t - 2), min(n, t + 3)
        with np.errstate(invalid="ignore"):
            home = np.nanmin(d_home[lo:hi]) if np.isfinite(d_home[lo:hi]).any() else np.inf
            away = np.nanmin(d_away[lo:hi]) if np.isfinite(d_away[lo:hi]).any() else np.inf
        nearest = min(home, away)
        if nearest > radius:
            info["nobody"] += 1
        elif abs(home - away) < margin:
            info["contested"] += 1
        elif home < away:
            out[t] = evidence.HOME
            info["home"] += 1
        else:
            out[t] = evidence.AWAY
            info["away"] += 1
    return out, info


def loglik(touch_state, weight=config.TOUCH_LL):
    out = np.zeros((len(touch_state), 2))
    out[touch_state == evidence.HOME, 0] = weight
    out[touch_state == evidence.AWAY, 1] = weight
    return out
