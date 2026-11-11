"""Parse the raw ATD once into numpy arrays indexed by VIDEO frame, and cache them.

    python -m possession.io_atd          # parse + cache, prints a shape summary (no values)

Arrays (T = video frames, S = track slots, P = fields per slot, B = fields in the ball group):
    slots   float64 (T, S, P)   normalised 0-1 x, y, then speed (m/s) and anything else present
    ball    float64 (T, B)      normalised x, y, then whatever else the ball group carries
    team    int8    (S,)        0 home, 1 away -- from starter/baseline.team_of_each_track
    present bool    (T,)        True where the raw file had a record for that frame

Row t is video frame t = frame_id - 1 (DATA.md). Frames with no record stay NaN.
float64 on purpose: the nearest-player check against the published baseline must reproduce its
tie-breaks exactly, and the baseline computes in Python floats.
"""

import os
import sys

import numpy as np

from . import config

sys.path.insert(0, config.ROOT)
sys.path.insert(0, os.path.join(config.ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

CACHE_NAME = "atd_{source}.npz"


def _floats(group):
    return [float(value) for value in group.split(config.VALUE_SEP)]


def parse_raw(raw_path, n_frames):
    """Stream the raw file into (slots, ball, present). Shapes are learned from the first record."""
    slots = ball = None
    present = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(config.FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                raise ValueError("frame_id {} outside 1..{}".format(frame + 1, n_frames))

            groups = parts[1].split(config.GROUP_SEP)
            ball_values = _floats(parts[2])
            if slots is None:
                n_fields = len(groups[0].split(config.VALUE_SEP))
                slots = np.full((n_frames, len(groups), n_fields), np.nan)
                ball = np.full((n_frames, len(ball_values)), np.nan)

            if len(groups) != slots.shape[1] or len(ball_values) != ball.shape[1]:
                raise ValueError("frame_id {}: record shape differs from the first record"
                                 .format(frame + 1))
            slots[frame] = [_floats(group) for group in groups]
            ball[frame] = ball_values
            present[frame] = True

    if slots is None:
        raise ValueError("{}: no records parsed".format(raw_path))
    return slots, ball, present


def build(game=GAME, data_dir=None, source="tactical"):
    """Parse the ATD of one source into a dict of arrays."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    slots, ball, present = parse_raw(raw_path, n_frames)
    team = team_of_each_track(meta_path)
    if len(team) != slots.shape[1]:
        raise ValueError("metadata lists {} slots, raw file has {}".format(
            len(team), slots.shape[1]))
    return {"slots": slots, "ball": ball, "team": team, "present": present,
            "pitch": np.array(GAMES[game]["pitch"], dtype=np.float64)}


def cache_path(source="tactical"):
    return os.path.join(config.CACHE_DIR, CACHE_NAME.format(source=source))


def load(source="tactical", game=GAME, data_dir=None, rebuild=False):
    """Cached arrays, parsing the raw file on first use."""
    path = cache_path(source)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as stored:
            return {key: stored[key] for key in stored.files}
    arrays = build(game, data_dir, source)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def summary(arrays):
    """Shapes and coverage only -- safe to share."""
    slots, ball = arrays["slots"], arrays["ball"]
    ball_seen = ~np.isnan(ball[:, 0])
    tracked = (~np.isnan(slots[:, :, 0])).sum(axis=1)
    return {
        "frames": slots.shape[0],
        "records_present": int(arrays["present"].sum()),
        "slots": slots.shape[1],
        "fields_per_slot": slots.shape[2],
        "fields_in_ball_group": ball.shape[1],
        "slots_home": int(np.sum(arrays["team"] == 0)),
        "slots_away": int(np.sum(arrays["team"] == 1)),
        "ball_seen_pct": round(100.0 * ball_seen.mean(), 2),
        "mean_tracked_objects": round(float(tracked.mean()), 2),
        # Columns of the ball group beyond x,y that are ever non-NaN: tells us whether extra
        # channels (z, state, owner) actually carry anything.
        "ball_extra_columns_non_nan_pct": [
            round(100.0 * float(np.mean(~np.isnan(ball[:, k]))), 2)
            for k in range(2, ball.shape[1])],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    for key, value in summary(load(args.source, rebuild=args.rebuild)).items():
        print("{:<32} {}".format(key, value))
