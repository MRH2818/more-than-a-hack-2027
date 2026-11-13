"""Smart Tagging as soft evidence: set-piece intervals and a weak team prior from phases."""

import os
import sys

import numpy as np

from . import config

sys.path.insert(0, config.ROOT)
sys.path.insert(0, os.path.join(config.ROOT, "starter"))


def load_events():
    """Smart Tagging instances as dicts (starter/load_smart_tagging.load), or [] if absent."""
    import load_smart_tagging
    try:
        return load_smart_tagging.load()
    except IOError:
        return []


def side_of(team_name):
    """0 home, 1 away, None for N/A or anything unrecognised."""
    name = (team_name or "").strip().lower()
    if name in config.TAG_HOME_NAMES:
        return 0
    if name in config.TAG_AWAY_NAMES:
        return 1
    return None


def set_piece_intervals(events):
    return [(event["start_frame"], event["end_frame"]) for event in events
            if (event["code"] or "").strip().upper() == "SET PIECES"]


def team_prior(events, n_frames, weight=config.TAG_TEAM_LL):
    """(T, 2) log-likelihood added to H and A from phase tags. Overlapping tags add up."""
    out = np.zeros((n_frames, 2))
    for event in events:
        code = (event["code"] or "").strip().upper()
        side = side_of(event["tags"].get("Team"))
        if side is None:
            continue
        if code in config.TAG_ATTACKING:
            in_possession = side
        elif code in config.TAG_DEFENDING:
            in_possession = 1 - side
        else:
            continue
        start, stop = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        out[start:stop, in_possession] += weight
    return out


def set_pieces_with_team(events):
    """[(start, end, side)] for SET PIECES tags naming a team (side 0 home, 1 away)."""
    out = []
    for event in events:
        if (event["code"] or "").strip().upper() != "SET PIECES":
            continue
        side = side_of(event["tags"].get("Team"))
        if side is not None:
            out.append((event["start_frame"], event["end_frame"], side))
    return out


def pre_kickoff(events, blocks, n_frames, moving=None, max_delay=config.KICKOFF_MAX_DELAY,
                search=config.KICKOFF_SEARCH):
    """(T,) True on graded frames before their half's kickoff, plus the count per half.

    The kickoff is the first frame at or after the tagged one where `moving` (ball seen moving)
    is True, within `search` frames; the tag start itself if there is none or no mask is given.
    """
    starts = [event["start_frame"] for event in events
              if (event["code"] or "").strip().upper() == "SET PIECES"
              and (event["tags"].get("Type") or "").strip().lower() == config.KICKOFF_TYPE]
    out = np.zeros(n_frames, dtype=bool)
    lengths = []
    for _, first, last in blocks:
        inside = [s for s in starts if first <= s <= min(last, first + max_delay)]
        kick = min(inside) if inside else first
        if inside and moving is not None:
            hits = np.flatnonzero(moving[kick:min(n_frames, kick + search)])
            if len(hits):
                kick += int(hits[0])
        out[first:kick] = True
        lengths.append(kick - first)
    return out, lengths
