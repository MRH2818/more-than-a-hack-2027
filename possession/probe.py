"""Aggregate facts about the ball track that decide which improvement matters most.

    python -m possession.probe

Prints only counts, shares and quantiles: no frame numbers, no coordinates.
"""

import os
import sys

import numpy as np

from . import ball as ball_mod
from . import config, deadball, evidence, io_atd

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks  # noqa: E402

GAP_BINS = (1, 13, 26, 51, 126, 251, 751, 10 ** 9)  # frames: <0.5 s, 1 s, 2 s, 5 s, 10 s, 30 s


def graded_mask(n, blocks):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def gap_lengths(observed, blocks):
    """Lengths of ball-missing runs inside each graded half (edge runs included)."""
    out = []
    for _, first, last in blocks:
        out += [stop - start for start, stop in ball_mod.runs_of(~observed[first:last + 1])]
    return np.array(out, dtype=np.int64)


def gap_table(lengths, total_frames):
    rows = []
    for low, high in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        chosen = lengths[(lengths >= low) & (lengths < high)]
        label = "{:.1f}-{:.1f}s".format((low - 1) / config.FPS, (high - 1) / config.FPS) \
            if high < 10 ** 9 else ">{:.0f}s".format((low - 1) / config.FPS)
        rows.append((label, len(chosen), round(100.0 * chosen.sum() / total_frames, 2)))
    return rows


def main():
    arrays = io_atd.load()
    n = arrays["slots"].shape[0]
    pitch = tuple(arrays["pitch"])
    blocks = half_video_blocks(GAME)
    graded = graded_mask(n, blocks)
    ball = ball_mod.track(arrays["ball"], pitch)
    observed = ball["observed"]

    print("== ball coverage on graded frames ==")
    print("observed_pct            ", round(100.0 * observed[graded].mean(), 1))
    print("interpolated_pct        ", round(100.0 * ball["interpolated"][graded].mean(), 2))
    lengths = gap_lengths(observed, blocks)
    print("gaps                    ", len(lengths))
    if len(lengths):
        print("gap_s_p50_p90_p99_max   ",
              [round(float(q) / config.FPS, 1) for q in np.percentile(lengths, [50, 90, 99, 100])])
    print("{:<12} {:>6} {:>18}".format("gap length", "gaps", "% of graded frames"))
    for label, count, share in gap_table(lengths, int(graded.sum())):
        print("{:<12} {:>6} {:>18}".format(label, count, share))

    print("\n== where the ball goes missing ==")
    speed = deadball.mean_player_speed(arrays["slots"])
    for name, mask in (("ball observed", graded & observed), ("ball missing", graded & ~observed)):
        values = speed[mask]
        values = values[~np.isnan(values)]
        if len(values):
            print("{:<14} mean player speed m/s p25/p50/p75: {}".format(
                name, [round(float(q), 2) for q in np.percentile(values, [25, 50, 75])]))
    starts_near_line = 0
    gap_starts = 0
    for _, first, last in blocks:
        for start, stop in ball_mod.runs_of(~observed[first:last + 1]):
            if start == 0 or stop - start <= config.MAX_INTERP_GAP:
                continue
            gap_starts += 1
            starts_near_line += int(ball["line_distance"][first + start - 1] <= config.NEAR_LINE_M)
    if gap_starts:
        print("long gaps whose last sighting is within {} m of a line: {:.1f}%".format(
            config.NEAR_LINE_M, 100.0 * starts_near_line / gap_starts))

    print("\n== ball to nearest player (observed, inside the pitch) ==")
    d_home, d_away = evidence.team_distances(arrays["slots"], arrays["team"], ball["xy"], pitch)
    nearest = np.fmin(d_home, d_away)
    mask = graded & observed & ~ball["oob"] & np.isfinite(nearest)
    values = nearest[mask]
    if len(values):
        print("quantiles m p5/p10/p25/p50/p75/p90:",
              [round(float(q), 2) for q in np.percentile(values, [5, 10, 25, 50, 75, 90])])
        for radius in (0.5, 1.0, 1.5, 2.0, 3.0):
            print("share within {:.1f} m: {:.1f}%".format(radius, 100.0 * np.mean(values <= radius)))
    radius, info = evidence.calibrate_zone(d_home, d_away, graded & observed & ~ball["oob"])
    print("calibrated possession zone:", round(radius, 2), "m", info)


def gap_model():
    """How the ATD ball gaps split between (our predicted) stoppages and live play, and how long
    each kind is. Aggregates only; used to make other games' tracking look like ours."""
    from . import pipeline, tagging
    arrays = io_atd.load()
    n = arrays["slots"].shape[0]
    blocks = half_video_blocks(GAME)
    graded = graded_mask(n, blocks)
    labels, parts = pipeline.label(arrays, events=tagging.load_events())
    observed = parts["ball"]["observed"]
    dead = (labels == "D") & graded
    alive = np.isin(labels, ("H", "A")) & graded
    print("missing share | predicted D      {:.1%}".format(np.mean(~observed[dead])))
    print("missing share | predicted H/A    {:.1%}".format(np.mean(~observed[alive])))
    rows = {"mostly D": [], "mostly live": []}
    for _, first, last in blocks:
        for start, stop in ball_mod.runs_of(~observed[first:last + 1]):
            a, b = first + start, first + stop
            key = "mostly D" if np.mean(dead[a:b]) >= 0.5 else "mostly live"
            rows[key].append(b - a)
    for key, lengths in rows.items():
        lengths = np.array(lengths) / float(config.FPS)
        if len(lengths):
            print("{:<12} gaps {:>4}  length s p10/p50/p90 {}".format(
                key, len(lengths), np.round(np.percentile(lengths, [10, 50, 90]), 1).tolist()))
    # Where predicted stoppages are, how much of each is covered by a gap.
    covered = []
    for _, first, last in blocks:
        for start, stop in ball_mod.runs_of(dead[first:last + 1]):
            covered.append(np.mean(~observed[first + start:first + stop]))
    covered = np.array(covered)
    print("stoppages {}: share mostly (>50%) without ball {:.1%}, fully without {:.1%}".format(
        len(covered), np.mean(covered > 0.5), np.mean(covered >= 0.99)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gaps":
        gap_model()
    else:
        main()
