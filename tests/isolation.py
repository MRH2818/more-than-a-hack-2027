"""Imported first by every test module: keeps synthetic tests away from the real match files.

CLAUDE.md: Claude runs the tests, and must never run code against the match data. Config
defaults may switch on parts that read cache/ or data/ (e.g. the video ball), so the tests turn
them off and point every such path at a directory that does not exist.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from possession import config  # noqa: E402

NOWHERE = os.path.join(ROOT, "tests", "_no_such_dir")
config.USE_VIDEO_BALL = False
config.USE_WHISTLE = False
config.VIDEO_DETECTIONS = os.path.join(NOWHERE, "detections.npz")
config.CACHE_DIR = NOWHERE

import load_homography  # noqa: E402


def _refuse(*args, **kwargs):
    raise RuntimeError("a test tried to read the real homography file")


load_homography.load = _refuse
os.environ["MTAH_DATA_DIR"] = NOWHERE
