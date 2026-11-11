"""Load BAR-ATH tactical tracking data into labelled pandas tables.

Usage:
    from load_tracking import load_tracking
    players, ball = load_tracking("data/tactical/BAR-ATH_tactical_atd.txt")

Line format (from the .xml):  frame : p1 ; p2 ; ... ; p46 : ball
Each player / the ball is  x,y,speed   (x and y normalized 0-1, speed in km/h).
Tracks 1-24 are Barcelona, tracks 25-46 are Athletic Club.
"""
import numpy as np
import pandas as pd

FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0      # metres
N_BAR, N_ATH = 24, 22
N_TRACKS = N_BAR + N_ATH            # 46


def load_tracking(path):
    frames, player_rows, ball_rows = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame, plist, blist = line.split(":")
            frames.append(int(frame))
            vals = np.array(plist.replace(";", ",").split(","), dtype="float32")
            player_rows.append(vals.reshape(N_TRACKS, 3))
            ball_rows.append(np.array(blist.split(","), dtype="float32"))

    frames = np.array(frames)
    P = np.stack(player_rows)       # (n_frames, 46, 3)
    B = np.stack(ball_rows)         # (n_frames, 3)

    # --- players: one row per track per frame, empty (NaN) tracks dropped ---
    f_idx, t_idx = np.nonzero(~np.isnan(P[:, :, 0]))
    is_bar = t_idx < N_BAR
    local_no = np.where(is_bar, t_idx + 1, t_idx - N_BAR + 1)   # 1..24 / 1..22
    team_code = np.where(is_bar, "BAR", "ATH")
    players = pd.DataFrame({
        "frame": frames[f_idx],
        "time_s": (frames[f_idx] - 1) / FPS,
        "team": np.where(is_bar, "Barcelona", "Athletic Club"),
        "track": [f"{c}_{n:02d}" for c, n in zip(team_code, local_no)],
        "is_goalkeeper": (t_idx == N_BAR - 1) | (t_idx == N_TRACKS - 1),
        "x_m": P[f_idx, t_idx, 0] * PITCH_L,
        "y_m": P[f_idx, t_idx, 1] * PITCH_W,
        "speed_kph": P[f_idx, t_idx, 2],
    })

    # --- ball: one row per frame ---
    ball = pd.DataFrame({
        "frame": frames,
        "time_s": (frames - 1) / FPS,
        "x_m": B[:, 0] * PITCH_L,
        "y_m": B[:, 1] * PITCH_W,
        "speed_kph": B[:, 2],
    })
    return players, ball


if __name__ == "__main__":
    import sys
    p, b = load_tracking(sys.argv[1] if len(sys.argv) > 1
                         else "data/tactical/BAR-ATH_tactical_atd.txt")
    print(p.head(10)); print(b.head(5))
    print("tracks seen per team:\n", p.groupby("team")["track"].nunique())