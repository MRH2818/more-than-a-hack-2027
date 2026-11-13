"""Per-frame control evidence from players and ball (Vidal-Codina et al. 2022 zones)."""

import numpy as np

from . import config

NONE, HOME, AWAY, DUEL = 0, 1, 2, 3


def team_distances(slots, team, ball_xy_m, pitch):
    """(d_home, d_away): distance (m) from the ball to the nearest tracked player of each team.

    NaN where the ball is unknown or that team has no tracked player.
    """
    length, width = pitch
    dx = slots[:, :, 0] * length - ball_xy_m[:, None, 0]
    dy = slots[:, :, 1] * width - ball_xy_m[:, None, 1]
    distance = np.hypot(dx, dy)
    distance = np.where(np.isnan(distance), np.inf, distance)

    out = []
    for side in (0, 1):
        nearest = np.min(np.where(team[None, :] == side, distance, np.inf), axis=1)
        out.append(np.where(np.isinf(nearest), np.nan, nearest))
    return out[0], out[1]


def control(d_home, d_away, possession_zone=config.POSSESSION_ZONE_M,
            duel_zone=config.DUEL_ZONE_M):
    """Per-frame control state: NONE (loose / unknown), HOME, AWAY or DUEL."""
    with np.errstate(invalid="ignore"):
        home_near = d_home <= duel_zone
        away_near = d_away <= duel_zone
        duel = home_near & away_near
        home = (d_home <= possession_zone) & ~duel & ~(d_away < d_home)
        away = (d_away <= possession_zone) & ~duel & ~(d_home < d_away)
    state = np.full(len(d_home), NONE, dtype=np.int8)
    state[home] = HOME
    state[away] = AWAY
    state[duel] = DUEL
    return state


def control_loglik(state, weight=config.CONTROL_LL):
    """(T, 2) per-frame log-likelihood added to the H and A states."""
    out = np.zeros((len(state), 2))
    out[state == HOME, 0] = weight
    out[state == AWAY, 1] = weight
    return out


def calibrate_zone(d_home, d_away, fit_mask, value_range=config.ZONE_RANGE_M):
    """Possession-zone radius (m) learned from this match without labels, plus the fit.

    Fits a 2-component mixture to log(distance from the ball to the nearest player) on fit_mask
    frames. The low component is 'ball at a player's feet'; the radius is the first distance
    above its mean where the high ('ball travelling') component becomes more likely.
    Returns (radius, info). Falls back to config.POSSESSION_ZONE_M if the fit is degenerate.
    """
    from .deadball import gmm_fit, gmm_logodds_at

    nearest = np.fmin(d_home, d_away)
    with np.errstate(invalid="ignore", divide="ignore"):
        keep = fit_mask & np.isfinite(nearest) & (nearest > 0.01) & (nearest < 10.0)
        model = gmm_fit(np.log(nearest[keep]))
    info = {"fit_frames": int(keep.sum())}
    if model is None:
        return config.POSSESSION_ZONE_M, dict(info, fallback=True)

    grid = np.exp(np.linspace(np.log(0.1), np.log(10.0), 2000))
    odds = gmm_logodds_at(model, np.log(grid))
    above = (grid > np.exp(model[0][0])) & (odds < 0)
    if not above.any():
        return config.POSSESSION_ZONE_M, dict(info, fallback=True)
    radius = float(np.clip(grid[np.argmax(above)], *value_range))
    info.update({"near_mode_m": round(float(np.exp(model[0][0])), 2),
                 "far_mode_m": round(float(np.exp(model[0][1])), 2),
                 "near_share": round(float(model[2][0]), 3),
                 "raw_radius_m": round(float(grid[np.argmax(above)]), 2),
                 "fallback": False})
    return radius, info


def soft_control_loglik(d_home, d_away, zone=config.POSSESSION_ZONE_M,
                        margin=config.TOUCH_MARGIN_M, weight=config.CONTROL_LL):
    """(T, 2) control evidence that fades from `weight` at `zone` to 0 at 2 * zone, given only
    when the nearest opponent is at least `margin` farther than the nearest team-mate."""
    out = np.zeros((len(d_home), 2))
    home = np.where(np.isnan(d_home), np.inf, d_home)
    away = np.where(np.isnan(d_away), np.inf, d_away)
    nearest = np.minimum(home, away)
    strength = np.clip((2.0 * zone - nearest) / zone, 0.0, 1.0) * weight
    with np.errstate(invalid="ignore"):
        clear = np.abs(home - away) >= margin
    out[:, 0] = np.where(clear & (home < away), strength, 0.0)
    out[:, 1] = np.where(clear & (away < home), strength, 0.0)
    return out
