"""The no-ground-truth gate: hide real ball stretches and see whether the labels survive.

    python -m possession.masktest                      # all variants
    python -m possession.masktest --variants base ballfree

For each variant, label the match twice: with the real ball, and with extra ball gaps cut into
stretches where the ball WAS seen (gap lengths drawn from the match's own long-gap
distribution, fixed seed). On the hidden frames the real-ball answer is the reference, so the
scores measure how well each variant copes when the ball disappears -- the situation behind 40%
of graded frames. Also prints the plain consistency checks for each unmasked run.
Aggregates only.
"""

import argparse
import os
import sys
import time

import numpy as np

from . import config, diagnostics, io_atd, pipeline, probe, tagging

sys.path.insert(0, os.path.join(config.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks  # noqa: E402

VARIANTS = {
    "base": {},
    "restarts": {"restarts": True},
    "ballfree": {"ballfree": True},
    "zone": {"zone": True},
    "gapdead": {"gapdead": True},
    "cleandead": {"cleandead": True},
    "clean+gap": {"cleandead": True, "gapdead": True},
    "lineaware": {"lineaware": True},
    "tagrestarts": {"tagrestarts": True},
    "line+tag": {"lineaware": True, "tagrestarts": True},
    "touches": {"touches": True},
    "soft": {"softcontrol": True},
    "touch+soft": {"touches": True, "softcontrol": True},
    "touch+rest": {"touches": True, "restarts": True, "tagrestarts": True},
    "touch+free": {"touches": True, "ballfree": True},
    "touch+line": {"touches": True, "lineaware": True},
    "whistle": {"whistle": True},
    "video+whistle": {"videoball": True, "whistle": True},
    "video": {"videoball": True},
    "videofull": {"videofull": True},
    "video+touch": {"videoball": True, "touches": True},
    "touch+all": {"touches": True, "softcontrol": True, "restarts": True, "tagrestarts": True,
                  "ballfree": True, "lineaware": True},
    "gap+rest": {"gapdead": True, "restarts": True},
    "all": {"restarts": True, "ballfree": True, "zone": True, "gapdead": True,
            "cleandead": True},
}


def mask_ball(ball_array, graded, fraction, seed=0, max_len=30 * config.FPS, blocks=None):
    """Copy of the ball array with extra NaN spans covering ~fraction of the observed graded
    frames, plus the boolean mask of the frames hidden."""
    observed = ~np.isnan(ball_array[:, 0])
    blocks = blocks if blocks is not None else []
    lengths = probe.gap_lengths(observed, blocks) if blocks else np.array([], dtype=np.int64)
    lengths = lengths[(lengths > config.MAX_INTERP_GAP) & (lengths <= max_len)]
    if not len(lengths):
        lengths = np.array([2 * config.FPS])

    rng = np.random.RandomState(seed)
    eligible = observed & graded
    cumulative = np.concatenate(([0], np.cumsum(eligible)))
    hidden = np.zeros(len(ball_array), dtype=bool)
    target = fraction * eligible.sum()
    candidates = rng.permutation(np.flatnonzero(eligible))
    draws = rng.choice(lengths, size=len(candidates))
    for start, length in zip(candidates, draws):
        if hidden.sum() >= target:
            break
        stop = start + int(length)
        # Keep one seen frame either side so a hidden span never merges with a real gap.
        if start < 1 or stop + 1 > len(ball_array):
            continue
        if cumulative[stop + 1] - cumulative[start - 1] != stop - start + 2:
            continue
        if hidden[start - 1:stop + 1].any():
            continue
        hidden[start:stop] = True
    masked = ball_array.copy()
    masked[hidden] = np.nan
    return masked, hidden


def compare(reference, candidate, hidden, runs):
    ref_codes, cand_codes = pm.label_codes(reference), pm.label_codes(candidate)
    team = hidden & np.isin(reference, ("H", "A"))
    return {
        "hidden_frame_agree_pct": round(100.0 * float(np.mean(candidate[hidden] == reference[hidden])), 1),
        "hidden_team_agree_pct": round(100.0 * float(np.mean(candidate[team] == reference[team])), 1)
        if team.any() else None,
        "spellF1_vs_real_ball": round(pm.spell_scores(ref_codes, cand_codes, runs)["mean_f1"], 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="*", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--fraction", type=float, default=0.15,
                        help="share of ball-seen graded frames to hide")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arrays = io_atd.load()
    events = tagging.load_events()
    n = arrays["slots"].shape[0]
    blocks = half_video_blocks(GAME)
    graded = probe.graded_mask(n, blocks)
    runs = pm.valid_runs(pm.label_codes(np.where(graded, "H", "X")))

    masked_ball, hidden = mask_ball(arrays["ball"], graded, args.fraction, args.seed,
                                    blocks=blocks)
    masked_arrays = dict(arrays, ball=masked_ball)
    print("hidden frames: {:.1f}% of graded".format(100.0 * hidden.sum() / graded.sum()))

    base_reference = None
    rows = []
    for name in args.variants:
        began = time.time()
        real, parts = pipeline.label(arrays, events=events, options=VARIANTS[name])
        cut, _ = pipeline.label(masked_arrays, events=events, options=VARIANTS[name])
        if name == "base":
            base_reference = real
        own = compare(real, cut, hidden, runs)
        summary = diagnostics.summarise(real, parts, events)
        row = {"variant": name, **own,
               "possessions": summary["possessions"],
               "poss_median_s": (summary["possession_s_p10_p50_p90"] or [None] * 3)[1],
               "D_share_pct": summary["label_share_pct"]["D"],
               "dead_segments": summary["dead_segments"],
               "phase_team_agree_pct": summary.get("phase_team_agree_pct"),
               "setpiece_near_dead_pct": summary.get("setpiece_near_dead_pct"),
               "trans_within_2s_pct": summary.get("trans_within_2s_pct"),
               "trans_within_5s_pct": summary.get("trans_within_5s_pct"),
               "trans_median_s": summary.get("trans_median_s"),
               "zone_m": summary["possession_zone_m"],
               "pre_kickoff": "/".join(str(v) for v in summary.get("pre_kickoff_frames") or []),
               "video_fill_pct": (summary.get("video") or {}).get("filled_pct_of_gaps", "-"),
               "whistles": (summary.get("whistle") or {}).get("graded_events", "-"),
               "touches_h/a": "{home}/{away}".format(**summary["touches"])
               if summary.get("touches") else "-"}
        if base_reference is not None and name != "base":
            row["spellF1_vs_base_real"] = round(pm.spell_scores(
                pm.label_codes(base_reference), pm.label_codes(real), runs)["mean_f1"], 3)
        rows.append(row)
        print("{:<9} done in {:.0f}s".format(name, time.time() - began), flush=True)

    print()
    keys = []
    for row in rows:
        keys += [key for key in row if key not in keys]
    for key in keys:
        print("{:<26}".format(key) + "".join("{:>12}".format(str(row.get(key, "-")))
                                           for row in rows))


if __name__ == "__main__":
    main()
