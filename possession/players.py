"""Aggregate facts about the PLAYER tracking, to make other games' tracking look like ours.

    python -m possession.players            # BAR-ATH (the human runs this; aggregates only)
    python -m possession.players --idsse    # the IDSSE matches, clean and degraded

Prints only counts, shares and quantiles: no frame numbers, no coordinates. The same function
runs on both datasets, so the numbers are directly comparable.
"""

import argparse
import os
import sys

import numpy as np

from . import config
from .ball import runs_of

_SAVGOL7 = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0


def _q(values, qs=(10, 50, 90), digits=2):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return [round(float(v), digits) for v in np.percentile(values, qs)]


def stats(arrays, mask):
    """Dict of aggregates over the frames in `mask`."""
    slots, team = arrays["slots"], arrays["team"]
    length, width = (float(v) for v in arrays["pitch"])
    seen = ~np.isnan(slots[:, :, 0])
    minutes = mask.sum() / float(config.FPS) / 60.0
    ball_seen = ~np.isnan(arrays["ball"][:, 0])
    out = {"minutes": round(minutes, 1)}

    for side, name in ((0, "home"), (1, "away")):
        count = seen[:, team == side].sum(axis=1)
        out["tracked_{}_p10_p50_p90".format(name)] = _q(count[mask], digits=1)
        out["tracked_{}_ball_seen_vs_missing_mean".format(name)] = [
            round(float(count[mask & ball_seen].mean()), 2) if (mask & ball_seen).any() else None,
            round(float(count[mask & ~ball_seen].mean()), 2) if (mask & ~ball_seen).any() else None]

    run_lengths, starts, jumps, residuals, fd_ratio, speed_steps = [], 0, 0, [], [], []
    for s in range(slots.shape[1]):
        for a, b in runs_of(seen[:, s] & mask):
            run_lengths.append(b - a)
            starts += 1
            if b - a < 9:
                continue
            x = slots[a:b, s, 0] * length
            y = slots[a:b, s, 1] * width
            step = np.hypot(np.diff(x), np.diff(y))
            jumps += int(np.sum(step > 2.0))            # > 2 m in 40 ms = > 50 m/s: an id jump
            for v in (x, y):
                smooth = np.convolve(v, _SAVGOL7[::-1], mode="valid")
                residuals.append(np.abs(v[3:-3] - smooth))
            if slots.shape[2] > 2:
                given = slots[a:b, s, 2]
                derived = np.hypot(np.gradient(x), np.gradient(y)) * config.FPS
                moving = derived > 1.0
                if moving.any():
                    fd_ratio.append(given[moving] / derived[moving])
                speed_steps.append(np.abs(np.diff(given)))
    run_lengths = np.array(run_lengths, dtype=np.float64) / config.FPS
    out["tracklets_per_min"] = round(starts / max(minutes, 1e-9), 2)
    out["tracklet_s_p10_p50_p90"] = _q(run_lengths, digits=1)
    out["share_of_player_frames_in_tracklets_under_60s"] = round(
        float(run_lengths[run_lengths < 60].sum() / max(run_lengths.sum(), 1e-9)), 3)
    out["id_jumps_per_min"] = round(jumps / max(minutes, 1e-9), 2)
    if residuals:
        out["position_jitter_cm_p50_p90"] = _q(np.concatenate(residuals) * 100.0, (50, 90), 1)
    if fd_ratio:
        out["speed_given_over_derived_p10_p50_p90"] = _q(np.concatenate(fd_ratio))
        out["speed_step_ms_p50_p90_p99"] = _q(np.concatenate(speed_steps), (50, 90, 99), 3)
        speed = slots[:, :, 2][mask]
        out["player_speed_ms_p50_p90_p99"] = _q(speed[np.isfinite(speed)], (50, 90, 99))
    return out


def show(title, values):
    print("== {} ==".format(title))
    for key, value in values.items():
        print("  {:<48} {}".format(key, value))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idsse", action="store_true")
    args = parser.parse_args()
    if args.idsse:
        from . import external
        for match in external.match_ids():
            arrays, answers, blocks = external.load(match)
            mask = np.ones(len(answers), dtype=bool)
            show(match + " clean", stats(arrays, mask))
            degraded = external.degrade(arrays, answers, blocks, config.EXT_BALL_NOISE_M, seed=1)
            show(match + " degraded", stats(degraded, mask))
        return
    from . import io_atd
    sys.path.insert(0, config.ROOT)
    from games import GAME, half_video_blocks
    arrays = io_atd.load()
    mask = np.zeros(arrays["slots"].shape[0], dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    show("BAR-ATH graded frames", stats(arrays, mask))


if __name__ == "__main__":
    main()
