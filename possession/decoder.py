"""Segmental Viterbi over H, A, D with minimum durations and switch costs.

Model: a sequence of segments, each a single label held for at least min_duration[label] frames.
Score = sum of per-frame emissions (log-likelihoods) - sum of switch costs. Exact maximisation by
dynamic programming in O(T * K^2) with K = 3, using cumulative sums for the duration constraint:

    L[k, t]  best score of a path whose current segment has label k, has lasted >= m_k frames,
             and ends at t
    E[k, t]  best score of a path that ENTERS label k at frame t
             = max_j L[j, t-1] - cost[j, k] + e[t, k]

    L[k, t] = max( L[k, t-1] + e[t, k],                     # stay
                   E[k, t-m_k+1] + C_k[t] - C_k[t-m_k+1] )  # segment just reached m_k frames

C_k is the cumulative emission of label k. Equivalent to an HMM with a chain of m_k substates per
label, without materialising them.
"""

import numpy as np

H, A, D = 0, 1, 2
NEG = -1e18


def decode(emissions, min_duration, switch_cost, initial=None, entry_bonus=None,
           exit_bonus=None):
    """Best label sequence (T,) for emissions (T, 3).

    min_duration: (3,) ints >= 1. switch_cost: (3, 3), diagonal ignored. initial: (3,) log prior
    on the first label (default flat). The whole sequence is assumed to be one continuous run;
    call it once per half.
    """
    emissions = np.asarray(emissions, dtype=np.float64)
    n, k_labels = emissions.shape
    m = np.asarray(min_duration, dtype=np.int64)
    cost = np.asarray(switch_cost, dtype=np.float64)
    initial = np.zeros(k_labels) if initial is None else np.asarray(initial, dtype=np.float64)
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    cumulative = np.vstack([np.zeros((1, k_labels)), np.cumsum(emissions, axis=0)])

    def span(k, first, last):  # sum of emissions of label k over frames first..last inclusive
        return cumulative[last + 1, k] - cumulative[first, k]

    entry_from = np.full((n, k_labels), -1, dtype=np.int8)   # -1: entered at frame 0
    by_chain = np.zeros((n, k_labels), dtype=bool)

    # Plain Python floats in the loop: K is 3, and numpy call overhead would dominate.
    e = emissions.tolist()
    cost_list = cost.tolist()
    # Optional (n, K) per-frame bonuses: entry_bonus[t][k] is added when a segment of label k
    # starts at frame t, exit_bonus[t][j] when a segment of label j ends just before frame t.
    entry = np.asarray(entry_bonus, dtype=np.float64).tolist() if entry_bonus is not None \
        else None
    leave = np.asarray(exit_bonus, dtype=np.float64).tolist() if exit_bonus is not None \
        else None
    m_list = m.tolist()
    L_rows = [[NEG] * k_labels for _ in range(n)]
    E_rows = [[NEG] * k_labels for _ in range(n)]

    for t in range(n):
        prev = L_rows[t - 1] if t else None
        for k in range(k_labels):
            if t == 0:
                E_rows[0][k] = initial[k] + e[0][k]
                entry_from[0, k] = -1
            else:
                best, arg = NEG, -1
                for j in range(k_labels):
                    if j == k:
                        continue
                    score = prev[j] - cost_list[j][k]
                    if entry is not None:
                        score += entry[t][k]
                    if leave is not None:
                        score += leave[t][j]
                    if score > best:
                        best, arg = score, j
                if arg >= 0 and best > NEG / 2:
                    E_rows[t][k] = best + e[t][k]
                    entry_from[t, k] = arg

            stay = prev[k] + e[t][k] if t else NEG
            start = t - m_list[k] + 1
            chain = NEG
            if start >= 0 and E_rows[start][k] > NEG / 2:
                chain = E_rows[start][k] + span(k, start + 1, t) if t > start else E_rows[start][k]
            if chain > stay:
                L_rows[t][k] = chain
                by_chain[t, k] = True
            else:
                L_rows[t][k] = stay

    labels = np.empty(n, dtype=np.int8)
    final = L_rows[n - 1]
    k = int(np.argmax(final))
    if final[k] <= NEG / 2:
        raise ValueError("no feasible path: sequence shorter than every minimum duration")
    t = n - 1
    while t >= 0:
        if by_chain[t, k]:
            start = t - m_list[k] + 1
            labels[start:t + 1] = k
            source = entry_from[start, k]
            t = start - 1
            if t >= 0:
                k = int(source)
        else:
            labels[t] = k
            t -= 1
    return labels


def snap_turnovers(labels, control_state, home_code, away_code, window):
    """Move each H<->A boundary to the first clean control frame of the new team after the old
    team's last one, searching `window` frames either side. Leaves D boundaries alone."""
    out = labels.copy()
    team_code = {H: home_code, A: away_code}
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for b in boundaries.tolist():
        old, new = int(labels[b - 1]), int(labels[b])
        if D in (old, new):
            continue
        low, high = max(0, b - window), min(len(labels), b + window)
        segment = control_state[low:high]
        old_hits = np.flatnonzero(segment[:b - low] == team_code[old])
        last_old = low + int(old_hits[-1]) if len(old_hits) else low - 1
        new_hits = np.flatnonzero(segment[last_old - low + 1:] == team_code[new])
        if not len(new_hits):
            continue
        target = last_old + 1 + int(new_hits[0])
        if target < b:
            if np.any(out[target:b] == D):
                continue
            out[target:b] = new
        elif target > b:
            if np.any(out[b:target] == D):
                continue
            out[b:target] = old
    return out
