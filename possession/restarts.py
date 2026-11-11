"""Who restarts after the ball goes out (Laws 15-17): the opponents of the last team in control.

Works on a first-pass decoding of one graded half. Free kicks and other stoppages with no
out-of-play cue get no prior: the fouled team is usually the one in possession, so the last-touch
rule does not apply, and the post-restart control evidence is left to decide.
"""

import numpy as np

from . import config, evidence
from .ball import runs_of

DEAD = 2


def restart_prior(codes, state, went_out, forced=None, window=config.RESTART_WINDOW,
                  weight=config.RESTART_LL, lookback=config.RESTART_LOOKBACK, fps=config.FPS):
    """(n, 2) log-likelihood for H and A after each decoded out-of-play stoppage.

    codes: first-pass decoding (0 H, 1 A, 2 D) of one half. state: control states for the same
    frames. went_out: True where the ball is seen beyond a line or vanished near one. forced:
    True on frames whose D label is a published fact (no restart prior there).
    Returns (prior, info) where info counts stoppages by outcome (aggregates only).
    """
    n = len(codes)
    out = np.zeros((n, 2))
    forced = np.zeros(n, dtype=bool) if forced is None else forced
    info = {"stoppages": 0, "out_of_play": 0, "assigned": 0}
    for start, stop in runs_of(codes == DEAD):
        if start == 0 or stop == n or forced[start:stop].any():
            continue
        info["stoppages"] += 1
        # The ball may be seen out a little before the decoder starts the stoppage.
        search = max(0, start - fps)
        hits = np.flatnonzero(went_out[search:stop])
        if not len(hits):
            continue
        info["out_of_play"] += 1
        left = search + int(hits[0])
        history = state[max(0, left - lookback):left]
        touched = np.flatnonzero((history == evidence.HOME) | (history == evidence.AWAY))
        if not len(touched):
            continue
        last = history[touched[-1]]
        restarter = 1 if last == evidence.HOME else 0  # columns: 0 = H, 1 = A
        out[stop:min(n, stop + window), restarter] += weight
        info["assigned"] += 1
    return out, info


def tagged_prior(set_pieces, codes, window=config.RESTART_WINDOW, weight=config.RESTART_LL,
                 search=config.TAG_RESTART_SEARCH):
    """(T, 2) prior for the team Smart Tagging names on each set piece, from the first frame at
    or after the tag start that the first pass labels alive. codes: 0 H, 1 A, 2 D, -1 ungraded."""
    n = len(codes)
    out = np.zeros((n, 2))
    info = {"tagged": len(set_pieces), "assigned": 0}
    alive = (codes == 0) | (codes == 1)
    for start, end, side in set_pieces:
        start, stop = max(0, start), min(n, max(start, end) + search)
        hits = np.flatnonzero(alive[start:stop])
        if not len(hits):
            continue
        first = start + int(hits[0])
        out[first:min(n, first + window), side] += weight
        info["assigned"] += 1
    return out, info
