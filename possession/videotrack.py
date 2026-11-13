"""Turn per-frame ball candidates from the video into one physically consistent ball path.

Used only to fill frames where the provided ATD has no ball. Inside each ATD gap a dynamic
programme picks chains of candidates that a real ball could have followed (speed limit, short
skips allowed), rewarding confident detections and charging for starting a chain that is not
connected to the last ATD sighting. Everything else stays missing.

    python -m possession.videotrack check              # accuracy on hidden ATD stretches
    python -m possession.videotrack check --sweep      # same, for a grid of strictness settings
"""

import argparse
import os
import sys

import numpy as np

from . import config
from .ball import runs_of

sys.path.insert(0, os.path.join(config.ROOT, "starter"))

NEG = -1e18


def load_candidates(path=None):
    """(candidates (T, K, 5) pixel x, pixel y, conf, w, h; done (T,)) or None if not run yet."""
    path = config.VIDEO_DETECTIONS if path is None else path
    if not os.path.exists(path):
        return None
    saved = np.load(path)
    return saved["candidates"].astype(np.float64), saved["done"].astype(bool)


def candidates_on_pitch(candidates, matrices, pitch, off_pitch=config.VIDEO_OFF_PITCH_M):
    """(positions (T, K, 2) metres, conf (T, K)); NaN for missing or clearly off-pitch."""
    n, k, _ = candidates.shape
    inverse = np.full((n, 3, 3), np.nan)
    ok = ~np.isnan(matrices).any(axis=(1, 2))
    inverse[ok] = np.linalg.inv(matrices[ok])
    px, py = candidates[:, :, 0], candidates[:, :, 1]
    den = inverse[:, 2, 0, None] * px + inverse[:, 2, 1, None] * py + inverse[:, 2, 2, None]
    x = (inverse[:, 0, 0, None] * px + inverse[:, 0, 1, None] * py + inverse[:, 0, 2, None]) / den
    y = (inverse[:, 1, 0, None] * px + inverse[:, 1, 1, None] * py + inverse[:, 1, 2, None]) / den
    pos = np.stack([x * pitch[0], y * pitch[1]], axis=-1)
    conf = candidates[:, :, 2].copy()
    with np.errstate(invalid="ignore"):
        off = ((pos[..., 0] < -off_pitch) | (pos[..., 0] > pitch[0] + off_pitch)
               | (pos[..., 1] < -off_pitch) | (pos[..., 1] > pitch[1] + off_pitch))
    bad = off | np.isnan(pos).any(axis=-1) | np.isnan(conf)
    pos[bad] = np.nan
    conf[bad] = np.nan
    return pos, conf


def _chain_gap(pos, conf, anchor, p):
    """Best set of non-overlapping chains through one gap. Returns (G, 2) metres, NaN unfilled.

    pos (G, K, 2), conf (G, K); anchor (2,) last ATD position before the gap, or None.
    """
    g, k = conf.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        emit = np.where(np.isnan(conf), NEG, np.log(conf / p["conf_ref"]))
    score = np.full((g, k), NEG)
    back_t = np.full((g, k), -1, dtype=np.int64)    # previous frame in chain, -1 = chain start
    back_j = np.full((g, k), -1, dtype=np.int64)
    idle = np.zeros(g + 1)                           # idle[t + 1]: best with no chain open after t
    idle_end = [None] * (g + 1)                      # (t, j) of the chain ending there, or None
    step = p["vmax"] / config.FPS

    for t in range(g):
        valid = emit[t] > NEG / 2
        best = np.full(k, idle[t] - p["start"])      # open a new chain
        bt = np.full(k, -1, dtype=np.int64)
        bj = np.full(k, -1, dtype=np.int64)
        if anchor is not None and t < p["max_skip"] + 1:
            reach = step * (t + 1) + p["pos_tol"]
            d = np.hypot(*(pos[t] - anchor).T)
            from_anchor = np.where(d <= reach, -p["miss"] * t, NEG)
            better = from_anchor > best
            best = np.where(better, from_anchor, best)
            bt = np.where(better, -2, bt)            # -2: chained to the ATD anchor
        for gap in range(1, p["max_skip"] + 2):
            s = t - gap
            if s < 0:
                break
            prev = score[s]
            live = prev > NEG / 2
            if not live.any():
                continue
            d = np.hypot(pos[t][:, None, 0] - pos[s][None, :, 0],
                         pos[t][:, None, 1] - pos[s][None, :, 1])
            feasible = (d <= step * gap + p["pos_tol"]) & live[None, :]
            cand = np.where(feasible, prev[None, :] - p["miss"] * (gap - 1), NEG)
            arg = np.argmax(cand, axis=1)
            val = cand[np.arange(k), arg]
            better = val > best
            best = np.where(better, val, best)
            bt = np.where(better, s, bt)
            bj = np.where(better, arg, bj)
        score[t] = np.where(valid, best + emit[t], NEG)
        back_t[t], back_j[t] = bt, bj
        top = int(np.argmax(score[t]))
        if score[t, top] > idle[t]:
            idle[t + 1], idle_end[t + 1] = score[t, top], (t, top)
        else:
            idle[t + 1], idle_end[t + 1] = idle[t], idle_end[t]

    out = np.full((g, 2), np.nan)
    end = idle_end[g]
    while end is not None:
        t, j = end
        chain = []
        while True:
            chain.append((t, j))
            pt, pj = back_t[t, j], back_j[t, j]
            if pt < 0:
                break
            t, j = int(pt), int(pj)
        from_anchor = back_t[chain[-1][0], chain[-1][1]] == -2
        chain.reverse()
        for (t0, j0), (t1, j1) in zip(chain[:-1], chain[1:]):
            span = t1 - t0
            for i in range(span):
                out[t0 + i] = pos[t0, j0] + (pos[t1, j1] - pos[t0, j0]) * i / span
        last_t, last_j = chain[-1]
        out[last_t] = pos[last_t, last_j]
        first_t = chain[0][0]
        # A chain opened from idle continues the best set that ended before it; one chained to
        # the ATD anchor is the first thing in the gap.
        end = None if from_anchor else idle_end[first_t]
    return out


def default_params():
    return {"vmax": config.VIDEO_VMAX, "pos_tol": config.VIDEO_POS_TOL,
            "max_skip": config.VIDEO_MAX_SKIP, "miss": config.VIDEO_MISS,
            "conf_ref": config.VIDEO_CONF_REF, "start": config.VIDEO_START}


def fill(atd_xy_m, atd_known, pos, conf, done, params=None):
    """(T, 2) metres: video ball on frames where the ATD has none, NaN elsewhere; plus info."""
    p = default_params() if params is None else params
    n = len(atd_known)
    out = np.full((n, 2), np.nan)
    gaps = 0
    for start, stop in runs_of(~atd_known & done):
        gaps += 1
        anchor = atd_xy_m[start - 1] if start > 0 and atd_known[start - 1] else None
        out[start:stop] = _chain_gap(pos[start:stop], conf[start:stop], anchor, p)
    filled = ~np.isnan(out[:, 0])
    missing = ~atd_known & done
    return out, {"gaps": gaps, "gap_frames": int(missing.sum()),
                 "filled_pct_of_gaps": round(100.0 * filled.sum() / max(1, missing.sum()), 1)}


SWEEP_START = (2.0, 4.0, 8.0, 1e9)     # 1e9: only chains linked to the last ATD sighting
SWEEP_CONF_REF = (0.25, 0.35, 0.5)


def check(fraction, seed, sweep=False):
    """Hide ATD-seen stretches, fill them from video, compare with the hidden ATD ball.

    The ATD is provided input, not ground truth: this calibrates the filter against the
    provider's own ball (self-supervised). With sweep, a grid of strictness settings is scored
    on the same hidden set -- choose on one seed, confirm on another.
    """
    from load_homography import load as load_homography
    from . import io_atd, masktest, probe
    sys.path.insert(0, config.ROOT)
    from games import GAME, half_video_blocks

    loaded = load_candidates()
    if loaded is None:
        print("no detections yet: run python -m possession.video_ball detect")
        return
    candidates, done = loaded
    arrays = io_atd.load()
    pitch = np.asarray(arrays["pitch"], dtype=np.float64)
    blocks = half_video_blocks(GAME)
    n = len(done)
    graded = probe.graded_mask(n, blocks) & done
    matrices, _, _ = load_homography(source="tactical")
    pos, conf = candidates_on_pitch(candidates, matrices, pitch)
    atd = arrays["ball"][:, :2] * pitch
    known = ~np.isnan(atd[:, 0])

    masked_ball, hidden = masktest.mask_ball(arrays["ball"], graded, fraction, seed, blocks=blocks)
    masked_known = ~np.isnan(masked_ball[:, 0])
    print("seed {}: hidden ATD frames {} ({:.1f}% of graded)".format(
        seed, int(hidden.sum()), 100.0 * hidden.sum() / max(1, graded.sum())))

    settings = [default_params()]
    if sweep:
        settings = [dict(default_params(), start=st, conf_ref=cr)
                    for st in SWEEP_START for cr in SWEEP_CONF_REF]
    print("{:>7} {:>8} {:>8} {:>6} {:>6} {:>8} {:>8} {:>10}".format(
        "start", "conf_ref", "filled", "p50_m", "p90_m", "<=2m", "<=5m", "real_fill"))
    for params in settings:
        filled, _ = fill(np.where(masked_known[:, None], atd, np.nan), masked_known, pos, conf,
                         done, params)
        got = hidden & ~np.isnan(filled[:, 0])
        err = np.hypot(*(filled[got] - atd[got]).T)
        _, real_info = fill(np.where(known[:, None], atd, np.nan), known, pos, conf, done, params)
        start = "anchor" if params["start"] >= 1e8 else "{:g}".format(params["start"])
        if len(err):
            print("{:>7} {:>8g} {:>7.1%} {:>6.2f} {:>6.2f} {:>7.1%} {:>7.1%} {:>9}%".format(
                start, params["conf_ref"], got.sum() / max(1, hidden.sum()), np.median(err),
                np.percentile(err, 90), np.mean(err <= 2), np.mean(err <= 5),
                real_info["filled_pct_of_gaps"]), flush=True)
        else:
            print("{:>7} {:>8g}  nothing filled".format(start, params["conf_ref"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check")
    p_check.add_argument("--fraction", type=float, default=0.15)
    p_check.add_argument("--seed", type=int, default=0)
    p_check.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    check(args.fraction, args.seed, args.sweep)


if __name__ == "__main__":
    main()
