"""
Possession / duel / loss / gain detection from tracking data.
Implements Sec. 2.3.1-2.3.3 of Vidal-Codina et al., "Automatic event detection
in football using tracking data" (set piece triggers, 2.3.4, not included).

Inputs (per match/period, one row per frame, units in metres):
    ball    : (T, 2) array of ball x, y
    players : (T, N, 2) array of player x, y (NaN where a player is not on the pitch)
    teams   : (N,) array of 0/1 team labels
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class Params:
    fps: float = 25.0
    r_pz: float = 1.5       # possession zone radius (m)      -- placeholder, tune
    r_dz: float = 3.0       # duel zone radius, >= r_pz (m)   -- placeholder, tune
    eps_s: float = 0.3      # min ball displacement for a loss (m/frame)
    eps_theta: float = 0.95  # direction change if cos(angle) < eps_theta
    eps_v: float = 2.0      # min speed change for a gain (m/s)
    smooth: bool = True     # Savitzky-Golay on noisy ball data
    sg_window: int = 7      # 7 frames = ~0.25 s at 25 Hz
    sg_order: int = 2


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def control_frames(ball, players, teams, p: Params):
    """2.3.1: returns (in_pz, poss, duel, members) where members[f] is the set of
    player indices involved in control frame f (None if f is not a control frame)."""
    teams = np.asarray(teams)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)  # (T, N)
    with np.errstate(invalid="ignore"):
        in_pz = dist <= p.r_pz
        in_dz = dist <= p.r_dz  # NaN compares False
    # duel: at least two opponents (one from each team) inside the DZ around the ball
    duel = in_dz[:, teams == 0].any(1) & in_dz[:, teams == 1].any(1)
    poss = in_pz.any(1) & ~duel  # duel prevails when both trigger
    members: list[set[int] | None] = [None] * len(ball)
    for f in np.flatnonzero(duel):
        members[f] = {int(i) for i in np.flatnonzero(in_dz[f])}
    for f in np.flatnonzero(poss):
        members[f] = {int(i) for i in np.flatnonzero(in_pz[f])}
    return in_pz, poss, duel, members


def ball_kinematics(ball, p: Params):
    """Incoming (frame f-1 -> f) and outgoing (f -> f+1) direction/speed per frame."""
    T = len(ball)
    d0 = np.full((T, 2), np.nan)
    d1 = np.full((T, 2), np.nan)
    d0[1:] = ball[1:] - ball[:-1]
    d1[:-1] = ball[1:] - ball[:-1]
    ds = np.linalg.norm(d1, axis=1)            # displacement f -> f+1
    v0 = np.linalg.norm(d0, axis=1) * p.fps    # incoming speed
    v1 = ds * p.fps                            # outgoing speed
    return d0, d1, ds, v0, v1


def detect_losses(in_pz, members, ds, teams, p: Params):
    """2.3.2: player A loses possession at control frame f if
    (1) ball outside A's PZ at f+1 and displacement ds[f] > eps_s, and
    (2) A is absent from the next control frame fc > f."""
    ctrl = [f for f, m in enumerate(members) if m is not None]
    rows = []
    for k in range(len(ctrl) - 1):
        f, fc = ctrl[k], ctrl[k + 1]
        mf, mc = members[f], members[fc]
        if mf is None or mc is None or f + 1 >= len(members):
            continue
        for a in mf:
            if in_pz[f + 1, a] or not ds[f] > p.eps_s:
                continue
            if a in mc:
                continue  # regained later with no one else in control (Fig 3b2)
            rows.append(dict(frame=f, player=a, next_control_frame=fc,
                             to_teammate=bool(all(teams[b] == teams[a] for b in mc))))
    return pd.DataFrame(rows, columns=["frame", "player", "next_control_frame", "to_teammate"])


def detect_gains(in_pz, poss, d0, d1, v0, v1, p: Params):
    """2.3.3: for each run of consecutive possession frames f0..fn of the same
    player, a gain occurs at f0 if the ball changed direction
    (u0[f0] . u1[fn] < eps_theta) or speed (|v0 - v1| > eps_v on >=1 frame)."""
    T, N = in_pz.shape
    u0, u1 = _unit(d0), _unit(d1)
    rows = []
    for a in range(N):
        m = poss & in_pz[:, a]
        if not m.any():
            continue
        edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        for f0, fn in zip(starts, ends):
            if f0 < 1 or fn + 1 >= T:
                continue
            with np.errstate(invalid="ignore"):
                cos = float(np.dot(u0[f0], u1[fn]))
                dir_chg = cos < p.eps_theta
                spd_chg = bool((np.abs(v0[f0:fn + 1] - v1[f0:fn + 1]) > p.eps_v).any())
            if dir_chg or spd_chg:
                rows.append(dict(frame=f0, player=a, last_frame=fn,
                                 dir_change=bool(dir_chg), speed_change=spd_chg))
    return pd.DataFrame(rows, columns=["frame", "player", "last_frame", "dir_change", "speed_change"])


def run(ball, players, teams, p: Params = Params()):
    ball = np.asarray(ball, float)
    players = np.asarray(players, float)
    if p.smooth:
        ball = np.asarray(savgol_filter(ball, p.sg_window, p.sg_order, axis=0), dtype=float)
    in_pz, poss, duel, members = control_frames(ball, players, teams, p)
    d0, d1, ds, v0, v1 = ball_kinematics(ball, p)
    losses = detect_losses(in_pz, members, ds, np.asarray(teams), p)
    gains = detect_gains(in_pz, poss, d0, d1, v0, v1, p)
    ctrl = pd.DataFrame(dict(frame=np.arange(len(ball)), possession=poss, duel=duel))
    return dict(control=ctrl[ctrl.possession | ctrl.duel], losses=losses, gains=gains)


if __name__ == "__main__":
    # Synthetic demo: ball rolls in, player 0 (team 0) sits at origin and redirects it,
    # then it rolls to player 1 (team 0) at (10, 0).
    T = 120
    ball = np.zeros((T, 2))
    ball[:50, 0] = np.linspace(-8, 0, 50)            # approach along x
    ball[50:, 0] = np.linspace(0, 10, T - 50)        # leaves toward player 1
    ball[50:, 1] = np.linspace(0, 6, T - 50) * 0     # keep straight, change speed instead
    ball[:50, 0] = np.linspace(-8, 0, 50) ** 3 / 64  # slow near player => speed change
    players = np.full((T, 2, 2), np.nan)
    players[:, 0] = [0, 0]
    players[:, 1] = [10, 0]
    out = run(ball, players, teams=[0, 0], p=Params(smooth=False))
    for k, v in out.items():
        print(f"--- {k}\n{v.head()}\n")