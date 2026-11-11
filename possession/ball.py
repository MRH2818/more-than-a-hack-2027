"""Ball track repair: metres, smoothing, velocity, short-gap interpolation, out-of-bounds."""

import numpy as np

from . import config

# Savitzky-Golay window 7, polynomial order 2 -- the standard closed-form coefficients.
_SAVGOL7 = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0


def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def to_metres(xy_normalised, pitch):
    return xy_normalised[..., :2] * np.asarray(pitch, dtype=np.float64)


def smooth(xy):
    """Savitzky-Golay inside every observed run of at least 7 frames; shorter runs left raw."""
    out = xy.copy()
    half = len(_SAVGOL7) // 2
    for start, stop in runs_of(~np.isnan(xy[:, 0])):
        if stop - start < len(_SAVGOL7):
            continue
        for axis in range(xy.shape[1]):
            out[start + half:stop - half, axis] = np.convolve(
                xy[start:stop, axis], _SAVGOL7[::-1], mode="valid")
    return out


def fill_short_gaps(xy, max_gap=config.MAX_INTERP_GAP):
    """Linear interpolation across interior gaps of at most max_gap frames.

    Returns (filled, interpolated_mask). Gaps at the array edges or longer than max_gap stay NaN.
    """
    out = xy.copy()
    interpolated = np.zeros(len(xy), dtype=bool)
    missing = np.isnan(xy[:, 0])
    for start, stop in runs_of(missing):
        if start == 0 or stop == len(xy) or stop - start > max_gap:
            continue
        before, after = xy[start - 1], xy[stop]
        steps = np.arange(1, stop - start + 1)[:, None] / float(stop - start + 1)
        out[start:stop] = before + (after - before) * steps
        interpolated[start:stop] = True
    return out, interpolated


def velocity(xy, fps=config.FPS):
    """(T, 2) m/s by central differences inside each observed run; NaN elsewhere."""
    out = np.full_like(xy, np.nan)
    for start, stop in runs_of(~np.isnan(xy[:, 0])):
        if stop - start < 2:
            continue
        out[start:stop] = np.gradient(xy[start:stop], axis=0) * fps
    return out


def out_of_bounds(xy_m, pitch, margin=config.OOB_MARGIN_M):
    """True where the ball is observed beyond a touchline or goal line by more than margin."""
    length, width = pitch
    x, y = xy_m[:, 0], xy_m[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def distance_to_line(xy_m, pitch):
    """Distance (m) from the ball to the nearest boundary line; negative outside the pitch."""
    length, width = pitch
    x, y = xy_m[:, 0], xy_m[:, 1]
    return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))


def track(ball_normalised, pitch):
    """Everything downstream needs from the ball, as a dict of (T,) / (T, 2) arrays."""
    raw = to_metres(ball_normalised, pitch)
    observed = ~np.isnan(raw[:, 0])
    smoothed = smooth(raw)
    filled, interpolated = fill_short_gaps(smoothed)
    vel = velocity(filled)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    return {
        "xy": filled,
        "observed": observed,
        "interpolated": interpolated,
        "known": observed | interpolated,
        "speed": speed,
        "oob": out_of_bounds(filled, pitch) & observed,
        "line_distance": distance_to_line(filled, pitch),
    }
