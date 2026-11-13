"""Aggregate-only summaries of a labelling. Safe to share: no frame numbers, no coordinates."""

import os
import sys

import numpy as np

from . import config, evidence, tagging

sys.path.insert(0, os.path.join(config.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def _quantiles(values):
    if not len(values):
        return None
    return [round(float(q), 1) for q in np.percentile(values, [10, 50, 90])]


def tag_agreement(labels, events, graded, near=2 * config.FPS):
    """No-ground-truth consistency checks against Smart Tagging (aggregates only).

    phase_team_agree_pct: on frames inside a phase tag with a team (attacking phases name the
    team in possession, defending phases the team without it) that we label H or A, the share
    where our team matches. setpiece_near_dead_pct: share of graded SET PIECES tags with one of
    our D frames within 2 s of the tag start. Neither is a target to tune to -- phases are not
    possessions -- but a change that lowers them is suspicious.
    """
    n = len(labels)
    agree = total = 0
    for event in events:
        code = (event["code"] or "").strip().upper()
        side = tagging.side_of(event["tags"].get("Team"))
        if side is None or code not in config.TAG_ATTACKING + config.TAG_DEFENDING:
            continue
        holder = side if code in config.TAG_ATTACKING else 1 - side
        window = labels[max(0, event["start_frame"]):min(n, event["end_frame"])]
        window = window[graded[max(0, event["start_frame"]):min(n, event["end_frame"])]]
        team_frames = (window == "H") | (window == "A")
        total += int(team_frames.sum())
        agree += int(np.sum(window == ("H" if holder == 0 else "A")))
    pieces = [start for start, _ in tagging.set_piece_intervals(events)
              if 0 <= start < n and graded[start]]
    near_dead = sum(bool(np.any(labels[max(0, s - near):min(n, s + near)] == "D"))
                    for s in pieces)
    return {
        "phase_team_agree_pct": round(100.0 * agree / total, 1) if total else None,
        "setpiece_tags": len(pieces),
        "setpiece_near_dead_pct": round(100.0 * near_dead / len(pieces), 1) if pieces else None,
    }


DEAD_BINS_S = (0, 2, 5, 10, 30, 10 ** 6)


def dead_breakdown(segments, parts, fps=config.FPS):
    """Decoded stoppages by length, and which cues back them (aggregates only).

    Per length bin: count, then the share of those stoppages where the ball is missing for
    most of it, where it is seen beyond a line, where it is seen still, and where NONE of the
    ball cues fire (only the player-speed or tagging cues).
    """
    ball = parts["ball"]
    cues = parts["dead_parts"]
    rows = {}
    for low, high in zip(DEAD_BINS_S[:-1], DEAD_BINS_S[1:]):
        chosen = [(a, b) for a, b in segments if low * fps <= b - a < high * fps]
        name = "{}-{}s".format(low, high) if high < 10 ** 6 else ">{}s".format(low)
        if not chosen:
            rows[name] = {"n": 0}
            continue
        missing = [np.mean(~ball["observed"][a:b]) > 0.5 for a, b in chosen]
        oob = [bool(np.any(cues["oob"][a:b] > 0) or np.any(cues["gap_oob"][a:b] > 0))
               for a, b in chosen]
        still = [bool(np.any(cues["still"][a:b] > 0)) for a, b in chosen]
        only_players = [not (m or o or st) for m, o, st in zip(missing, oob, still)]
        rows[name] = {"n": len(chosen),
                      "ball_missing_pct": round(100.0 * np.mean(missing)),
                      "out_pct": round(100.0 * np.mean(oob)),
                      "still_pct": round(100.0 * np.mean(still)),
                      "players_only_pct": round(100.0 * np.mean(only_players))}
    return rows


def gain_frames(labels, graded):
    """{0: frames where a home spell starts, 1: same for away}, inside graded frames."""
    out = {}
    for side, letter in ((0, "H"), (1, "A")):
        mine = (labels == letter) & graded
        out[side] = np.flatnonzero(mine[1:] & ~mine[:-1]) + 1
    return out


def transition_agreement(labels, events, graded, fps=config.FPS):
    """How well our possession changes line up with Smart Tagging's transition tags.

    For each graded transition tag, the gaining team is the tagged team (attacking transition)
    or its opponent (defensive transition); we look for the nearest start of a spell of the
    gaining team in our labels. Reported: share within 2 s and 5 s of the tag start, and the
    median absolute offset of those within 10 s. Tags are automatic (no human review) and are
    used only as a yardstick here, never as an input.
    """
    starts = gain_frames(labels, graded)
    offsets = []
    for event in events:
        code = (event["code"] or "").strip().upper()
        side = tagging.side_of(event["tags"].get("Team"))
        if side is None or code not in config.TAG_GAIN + config.TAG_LOSS:
            continue
        frame = event["start_frame"]
        if not (0 <= frame < len(labels)) or not graded[frame]:
            continue
        gainer = side if code in config.TAG_GAIN else 1 - side
        candidates = starts[gainer]
        offsets.append(np.min(np.abs(candidates - frame)) / float(fps) if len(candidates)
                       else np.inf)
    offsets = np.array(offsets)
    if not len(offsets):
        return {}
    close = offsets[offsets <= 10]
    return {
        "transition_tags": len(offsets),
        "trans_within_2s_pct": round(100.0 * float(np.mean(offsets <= 2)), 1),
        "trans_within_5s_pct": round(100.0 * float(np.mean(offsets <= 5)), 1),
        "trans_median_s": round(float(np.median(close)), 2) if len(close) else None,
    }


def summarise(labels, parts, events=None):
    graded = parts["graded"]
    codes = pm.label_codes(np.where(graded, labels, "X"))
    runs = pm.valid_runs(codes)
    segments = pm.segments(codes, runs)
    spells = [(a, b) for a, b, k in segments if k in (pm.H, pm.A)]
    dead = [(a, b) for a, b, k in segments if k == pm.D]
    minutes = graded.sum() / config.FPS / 60.0
    state = parts["control"][graded]
    ball = parts["ball"]

    out = {
        "graded_frames": int(graded.sum()),
        "label_share_pct": {name: round(100.0 * float(np.mean(labels[graded] == name)), 1)
                            for name in ("H", "A", "D", "X")},
        "possessions": len(spells),
        "possessions_per_min": round(len(spells) / minutes, 2),
        "possession_s_p10_p50_p90": _quantiles([(b - a) / config.FPS for a, b in spells]),
        "dead_segments": len(dead),
        "dead_s_p10_p50_p90": _quantiles([(b - a) / config.FPS for a, b in dead]),
        "ball_observed_pct": round(100.0 * float(ball["observed"][graded].mean()), 1),
        "ball_interpolated_pct": round(100.0 * float(ball["interpolated"][graded].mean()), 1),
        "control_pct": {name: round(100.0 * float(np.mean(state == code)), 1)
                        for name, code in (("none", evidence.NONE), ("home", evidence.HOME),
                                           ("away", evidence.AWAY), ("duel", evidence.DUEL))},
        "dead_cue_active_pct": {name: round(100.0 * float(np.mean(part[graded] != 0)), 1)
                                for name, part in parts["dead_parts"].items()},
    }
    summary = {"options": parts.get("options")}
    summary.update(out)
    summary["possession_zone_m"] = round(float(parts.get("zone_m", config.POSSESSION_ZONE_M)), 2)
    summary["pre_kickoff_frames"] = parts.get("pre_kickoff_frames")
    for key in ("video", "whistle", "zone_info", "touches", "restarts", "tag_restarts", "ballfree"):
        if parts.get(key) is not None:
            summary[key] = parts[key]
    summary["dead_by_length"] = dead_breakdown(dead, parts)
    if events:
        summary.update(tag_agreement(labels, events, graded))
        summary.update(transition_agreement(labels, events, graded))
    return summary


def print_summary(summary):
    for key, value in summary.items():
        if key == "dead_by_length":
            print("dead_by_length")
            for name, row in value.items():
                print("    {:<10} {}".format(name, row))
            continue
        print("{:<28} {}".format(key, value))
