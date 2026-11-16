"""Picture of the labels over time: the public ground truth on top, submissions below.

    python -m possession.timeline
    python -m possession.timeline --rows "v9=evaluation/results/possession_v9.csv" \
        "Baseline=evaluation/results/baseline_smooth50.csv" --out cache/timeline.png

Each strip is one label per frame over the graded window of the ground-truth file (home / away
/ dead ball). Scores in the strip titles come from the official scorer. The picture shows
frame-level labels, so it is written under cache/ by default and must never be committed; it
is a sanity look, like run_evaluation.py, not something to tune on.
"""

import argparse
import os
import sys

import numpy as np

from . import config

sys.path.insert(0, os.path.join(config.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

DEFAULT_GT = os.path.join(config.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_ROWS = ("v9 (ours)=" + os.path.join("evaluation", "results", "possession_v9.csv"),
                "Official baseline (smooth50)=" + os.path.join("evaluation", "results",
                                                             "baseline_smooth50.csv"))
DEFAULT_OUT = os.path.join(config.CACHE_DIR, "timeline.png")

SURFACE, TEXT, SECONDARY, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
COLOURS = {pm.H: "#2a78d6", pm.A: "#eb6834", pm.D: "#c3c2b7"}      # home, away, dead ball
NAMES = {pm.H: "Home in possession", pm.A: "Away in possession", pm.D: "Dead ball"}


def runs_of_codes(codes):
    """(start, stop, code) for each run of equal codes."""
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return [(int(a), int(b), int(codes[a])) for a, b in zip(starts, stops)]


def window(gt_codes):
    """[first, stop) covering every graded frame of the ground truth."""
    runs = pm.valid_runs(gt_codes)
    if not runs:
        raise ValueError("the ground-truth file has no graded frames")
    return runs[0][0], runs[-1][1]


def describe(gt_codes, pred_codes):
    scores = pm.evaluate(gt_codes, pred_codes)
    return "spellF1 {:.3f}   macroF1 {:.3f}   {} possessions, mean {:.1f} s".format(
        scores["spell_f1_mean"], scores["macro_f1"], scores["n_pred_spells"],
        scores["mean_spell_duration_s"])


def draw(rows, first, stop, out, home="Home", away="Away", note=""):
    """rows: [(name, detail, codes)] where codes covers the whole video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    minutes = (stop - first) / float(config.FPS) / 60.0
    figure, axes = plt.subplots(len(rows), 1, figsize=(12, 1.15 * len(rows) + 1.3),
                                sharex=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for axis, (name, detail, codes) in zip(axes, rows):
        axis.set_facecolor(SURFACE)
        for a, b, code in runs_of_codes(codes[first:stop]):
            if code in COLOURS:
                axis.broken_barh([(a / float(config.FPS) / 60.0, (b - a) / float(config.FPS) / 60.0)],
                                 (0, 1), facecolors=COLOURS[code], edgecolor=SURFACE,
                                 linewidth=0.6)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        for side in ("top", "left", "right"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(AXIS)
        axis.tick_params(axis="x", colors=MUTED, length=3)
        axis.text(0.0, 1.12, name, transform=axis.transAxes, color=TEXT, fontsize=10.5,
                  fontweight="bold", ha="left", va="bottom")
        if detail:
            axis.text(1.0, 1.12, detail, transform=axis.transAxes, color=SECONDARY,
                      fontsize=9.5, ha="right", va="bottom")
    axes[-1].set_xlim(0, minutes)
    axes[-1].set_xlabel("Minutes into the window" + (" ({})".format(note) if note else ""),
                        color=SECONDARY, fontsize=9.5)
    labels = {pm.H: NAMES[pm.H].replace("Home", home), pm.A: NAMES[pm.A].replace("Away", away),
              pm.D: NAMES[pm.D]}
    figure.legend(handles=[Patch(facecolor=COLOURS[c], label=labels[c]) for c in (pm.H, pm.A, pm.D)],
                  loc="upper right", ncol=3, frameon=False, fontsize=9.5, labelcolor=SECONDARY,
                  bbox_to_anchor=(0.985, 1.0), handlelength=1.2, columnspacing=1.6)
    figure.subplots_adjust(left=0.02, right=0.985, top=1.0 - 0.75 / figure.get_figheight(),
                           bottom=0.62 / figure.get_figheight(), hspace=0.62)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--rows", nargs="*", default=list(DEFAULT_ROWS), metavar="NAME=CSV")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--home", default="Barcelona")
    parser.add_argument("--away", default="Athletic Club")
    args = parser.parse_args()

    gt_codes = pm.label_codes(pm.read_label_csv(args.gt))
    first, stop = window(gt_codes)
    gt_stats = pm.possession_stats(gt_codes, pm.valid_runs(gt_codes))
    rows = [("Ground truth", "{} possessions, mean {:.1f} s".format(
        gt_stats["n_spells"], gt_stats["mean_duration_s"]), gt_codes)]
    for item in args.rows:
        name, _, path = item.partition("=")
        if not path:
            parser.error("--rows takes NAME=CSV, got {!r}".format(item))
        pred = pm.label_codes(pm.read_label_csv(path))
        shown = pm.fill_abstentions(pred, pm.valid_runs(gt_codes))    # as the scorer sees it
        rows.append((name, describe(gt_codes, pred), shown))
    draw(rows, first, stop, args.out, args.home, args.away,
         "public window, sanity check only")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
