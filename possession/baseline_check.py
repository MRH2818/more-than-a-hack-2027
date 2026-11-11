"""Gate for the loader: rebuild baseline_smooth50 from the cached arrays and compare.

    python -m possession.baseline_check

If the new loader is right, the nearest-player codes and the smoothed labels match the committed
evaluation/results/baseline_smooth50.csv on every frame. Prints counts only -- safe to share.
"""

import os
import sys

import numpy as np
import pandas as pd

from . import config, io_atd

sys.path.insert(0, os.path.join(config.ROOT, "starter"))
from baseline import AWAY, HOME, NOBODY, smooth  # noqa: E402

REFERENCE = os.path.join(config.ROOT, "evaluation", "results", "baseline_smooth50.csv")


def nearest_player_codes(arrays):
    """Vectorised equivalent of baseline.nearest_player_labels: team of the nearest slot, in
    metres, NOBODY where the ball is missing. np.nanargmin returns the first minimum, which is the
    same tie-break as the baseline's strict '<' over slots in order."""
    length, width = arrays["pitch"]
    slots, ball = arrays["slots"], arrays["ball"]
    codes = np.full(len(ball), NOBODY, dtype=np.int8)

    have_ball = ~np.isnan(ball[:, 0])
    dx = (slots[have_ball, :, 0] * length) - (ball[have_ball, None, 0] * length)
    dy = (slots[have_ball, :, 1] * width) - (ball[have_ball, None, 1] * width)
    distance = dx * dx + dy * dy
    any_player = ~np.all(np.isnan(distance), axis=1)

    rows = np.flatnonzero(have_ball)[any_player]
    nearest = np.nanargmin(distance[any_player], axis=1)
    codes[rows] = np.where(arrays["team"][nearest] == 0, HOME, AWAY)
    return codes


def compare(window=50):
    arrays = io_atd.load()
    labels = smooth(nearest_player_codes(arrays), window)
    reference = pd.read_csv(REFERENCE)["label"].astype(str).str.strip().values
    if len(reference) != len(labels):
        return {"identical": False, "reason": "length {} vs {}".format(len(labels),
                                                                      len(reference))}
    differ = labels != reference
    return {"identical": not differ.any(), "frames_differing": int(differ.sum()),
            "frames": len(labels)}


if __name__ == "__main__":
    for key, value in compare().items():
        print("{:<18} {}".format(key, value))
