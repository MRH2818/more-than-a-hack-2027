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


OFFSETS_S = (-20, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15)


def visibility_profile(observed, anchors, valid):
    """Share of frames with a ball in bins of time relative to the anchors (aggregate over all
    anchors). Bins are [OFFSETS_S[i], OFFSETS_S[i+1]) seconds; frames outside `valid` skipped."""
    out = []
    for low, high in zip(OFFSETS_S[:-1], OFFSETS_S[1:]):
        seen = total = 0
        for anchor in anchors:
            a, b = anchor + low * config.FPS, anchor + high * config.FPS
            a, b = max(0, a), min(len(observed), b)
            if b > a:
                seen += int((observed[a:b] & valid[a:b]).sum())
                total += int(valid[a:b].sum())
        out.append(seen / float(total) if total else float("nan"))
    return out


def print_profile(name, count, profile):
    print("{:<34} n={:<4} ".format(name, count) + "  ".join(
        "[{:+d},{:+d})s {:.0%}".format(low, high, value)
        for low, high, value in zip(OFFSETS_S[:-1], OFFSETS_S[1:], profile)))


def restart_visibility():
    """Label-free: how often the tracking has a ball around each automatic set-piece tag. Used
    to check that other games' stoppages are degraded realistically (the first calibration
    used our own predicted stoppages, which is circular). Aggregates only."""
    from . import tagging
    arrays = io_atd.load()
    n = arrays["slots"].shape[0]
    blocks = half_video_blocks(GAME)
    graded = graded_mask(n, blocks)
    observed = ball_mod.track(arrays["ball"], tuple(arrays["pitch"]))["observed"]
    events = [e for e in tagging.load_events()
              if (e["code"] or "").strip().upper() == "SET PIECES"
              and 0 <= e["start_frame"] < n and graded[e["start_frame"]]]
    print("ball seen, all graded frames: {:.1%}".format(observed[graded].mean()))
    durations = np.array([e["end_frame"] - e["start_frame"] for e in events]) / float(config.FPS)
    if len(durations):
        print("set-piece tags {}: tag length s p10/p50/p90 {}".format(
            len(events), np.round(np.percentile(durations, [10, 50, 90]), 1).tolist()))
    by_type = {}
    for e in events:
        kind = (e["tags"].get("Type") or "?").strip().lower()
        by_type.setdefault(kind, []).append(e)
    print("ball-seen share by time relative to the tag START:")
    print_profile("all set pieces", len(events),
                  visibility_profile(observed, [e["start_frame"] for e in events], graded))
    for kind, group in sorted(by_type.items(), key=lambda item: -len(item[1])):
        if len(group) >= 5:
            print_profile(kind, len(group),
                          visibility_profile(observed, [e["start_frame"] for e in group], graded))
    print("ball-seen share by time relative to the tag END:")
    print_profile("all set pieces", len(events),
                  visibility_profile(observed, [e["end_frame"] for e in events], graded))
    inside = np.zeros(n, dtype=bool)
    for e in events:
        inside[max(0, e["start_frame"]):min(n, e["end_frame"])] = True
    inside &= graded
    if inside.any():
        print("ball seen inside set-piece tags: {:.1%} of {} frames".format(
            observed[inside].mean(), int(inside.sum())))


def goal_check():
    """Does the current pipeline already call the goal-to-kickoff stretches dead? Aggregates."""
    from . import pipeline, tagging
    arrays = io_atd.load()
    blocks = half_video_blocks(GAME)
    events = tagging.load_events()
    labels, _ = pipeline.label(arrays, events=events)
    cores = tagging.goal_cores(events, blocks)
    total = sum(b - a for a, b in cores)
    dead = sum(int((labels[a:b] == "D").sum()) for a, b in cores)
    starts = 0
    for a, b in cores:
        seg = labels[a:b]
        starts += int((np.isin(seg[1:], ("H", "A")) & (seg[1:] != seg[:-1])).sum())
    edge = 3 * config.FPS
    first = sum(int((labels[a:min(b, a + edge)] != "D").sum()) for a, b in cores)
    last = sum(int((labels[max(a + edge, b - edge):b] != "D").sum()) for a, b in cores)
    inside_runs = 0
    for a, b in cores:
        for start, stop in ball_mod.runs_of(np.isin(labels[a:b], ("H", "A"))):
            inside_runs += int(start > 0 and stop < b - a)
    print("not-dead frames: first 3 s of a stretch {}, last 3 s {}, in between {}; live runs "
          "that both start and end inside a stretch: {}".format(
              first, last, sum(int((labels[a:b] != "D").sum()) for a, b in cores) - first - last,
              inside_runs))
    lengths = [round((b - a) / float(config.FPS), 1) for a, b in cores]
    print("goal-to-kickoff stretches: {} (lengths s: {}), {} frames".format(
        len(cores), sorted(lengths), total))
    if total:
        print("labelled dead inside them: {:.1%}; not dead: {} frames; possessions starting "
              "inside: {}".format(dead / float(total), total - dead, starts))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gaps":
        gap_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "goals":
        goal_check()
    elif len(sys.argv) > 1 and sys.argv[1] == "restarts":
        restart_visibility()
    else:
        main()
