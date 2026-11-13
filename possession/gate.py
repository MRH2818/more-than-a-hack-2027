"""Fair no-ground-truth checks on BAR-ATH: every check uses information the model was NOT given.

    python -m possession.gate --variants base learned l-team

Why this exists (2026-11-12 audit): masktest and the tag checks favoured the hand-made model.
masktest hid the ball in live play for stoppage-length spans (up to 30 s; real live-play gaps
are about 5 s), and phase agreement was measured against tags that every model also reads.

Checks, per variant (aggregates only):
  live_hidden_team_agree_pct   hide the ball for live-play-length spans (lognormal, median
                               EXT_LIVE_GAP_MEDIAN_S, capped at 15 s); on hidden frames, share
                               where the team equals the same model's real-ball answer
  heldout_phase_agree_pct      label WITHOUT phase tags and set-piece tags (kickoff tag kept),
                               then agreement with the phase tags' team
  heldout_setpiece_dead_pct    same run: share of set-piece tags with a D frame within 2 s
  heldout_trans_*              same run: distance from tagged transitions to our turnovers
  video_team_agree_pct         label WITHOUT the video ball; on ATD-gap frames where the video
                               ball is clearly at one team's feet and we say H or A, share where
                               we name that team (video_dead_pct: share of those frames we call D)
Plus the plain counts of the full model (all inputs on).
"""

import argparse
import os
import sys
import time

import numpy as np

from . import config, diagnostics, evidence, io_atd, masktest, pipeline, probe, tagging

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks  # noqa: E402

LIVE_MAX_S = 15.0


def live_lengths(n=4000, seed=0):
    """Span lengths (frames) like BAR-ATH's live-play ball gaps (config EXT_LIVE_GAP_*)."""
    rng = np.random.RandomState(seed)
    seconds = config.EXT_LIVE_GAP_MEDIAN_S * np.exp(config.EXT_LIVE_GAP_SIGMA * rng.randn(n))
    seconds = np.clip(seconds, (config.MAX_INTERP_GAP + 1) / float(config.FPS), LIVE_MAX_S)
    return np.round(seconds * config.FPS).astype(np.int64)


def kickoff_only(events):
    """Events reduced to the kickoff tags (needed for the forced pre-kickoff stretch)."""
    return [event for event in events
            if (event["code"] or "").strip().upper() == "SET PIECES"
            and (event["tags"].get("Type") or "").strip().lower() == config.KICKOFF_TYPE]


def video_agreement(labels, state, frames):
    """Team agreement with a clear video-ball control state on `frames` (bool mask)."""
    clear = frames & ((state == evidence.HOME) | (state == evidence.AWAY))
    said_team = clear & np.isin(labels, ("H", "A"))
    want = np.where(state == evidence.HOME, "H", "A")
    return {
        "video_frames": int(clear.sum()),
        "video_team_agree_pct": round(100.0 * float(np.mean(labels[said_team] == want[said_team])), 1)
        if said_team.any() else None,
        "video_dead_pct": round(100.0 * float(np.mean(labels[clear] == "D")), 1)
        if clear.any() else None,
    }


def run_variant(spec, arrays, masked_arrays, hidden, events, graded, video_state, video_frames):
    options = {k: v for k, v in spec.items() if k != "_settings"}
    settings = spec.get("_settings", {})
    saved = {key: getattr(config, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(config, key, value)
        full, parts = pipeline.label(arrays, events=events, options=options)
        cut, _ = pipeline.label(masked_arrays, events=events, options=options)
        no_tags, _ = pipeline.label(arrays, events=kickoff_only(events), options=options)
        no_video, _ = pipeline.label(arrays, events=events,
                                     options=dict(options, videoball=False))
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
    team = hidden & np.isin(full, ("H", "A"))
    summary = diagnostics.summarise(full, parts, events)
    held = diagnostics.tag_agreement(no_tags, events, graded)
    held_trans = diagnostics.transition_agreement(no_tags, events, graded)
    row = {
        "live_hidden_team_agree_pct": round(100.0 * float(np.mean(cut[team] == full[team])), 1),
        "live_hidden_frame_agree_pct": round(100.0 * float(np.mean(cut[hidden] == full[hidden])), 1),
        "heldout_phase_agree_pct": held.get("phase_team_agree_pct"),
        "heldout_setpiece_dead_pct": held.get("setpiece_near_dead_pct"),
        "heldout_trans_within_2s_pct": held_trans.get("trans_within_2s_pct"),
        "heldout_trans_within_5s_pct": held_trans.get("trans_within_5s_pct"),
        "heldout_trans_median_s": held_trans.get("trans_median_s"),
    }
    row.update(video_agreement(no_video, video_state, video_frames))
    row.update({
        "possessions": summary["possessions"],
        "dead_segments": summary["dead_segments"],
        "D_share_pct": summary["label_share_pct"]["D"],
        "phase_agree_with_tags_pct": summary.get("phase_team_agree_pct"),
        "trans_median_s_with_tags": summary.get("trans_median_s"),
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="*", choices=list(masktest.VARIANTS),
                        default=["base", "learned"])
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arrays = io_atd.load()
    events = tagging.load_events()
    n = arrays["slots"].shape[0]
    blocks = half_video_blocks(GAME)
    graded = probe.graded_mask(n, blocks)
    masked_ball, hidden = masktest.mask_ball(arrays["ball"], graded, args.fraction, args.seed,
                                             blocks=blocks, lengths=live_lengths(seed=args.seed))
    masked_arrays = dict(arrays, ball=masked_ball)
    print("hidden frames: {:.1f}% of graded, in live-play-length spans".format(
        100.0 * hidden.sum() / graded.sum()))

    video_xy, info = pipeline.video_ball(arrays, graded)
    if video_xy is None:
        video_state = np.zeros(n, dtype=np.int8)
        video_frames = np.zeros(n, dtype=bool)
    else:
        pitch = tuple(arrays["pitch"])
        d_home, d_away = evidence.team_distances(arrays["slots"], arrays["team"], video_xy, pitch)
        video_state = evidence.control(d_home, d_away, possession_zone=config.POSSESSION_ZONE_M,
                                       duel_zone=config.DUEL_ZONE_M)
        video_frames = graded & np.isnan(arrays["ball"][:, 0]) & ~np.isnan(video_xy[:, 0])

    rows = {}
    for name in args.variants:
        began = time.time()
        rows[name] = run_variant(masktest.VARIANTS[name], arrays, masked_arrays, hidden, events,
                                 graded, video_state, video_frames)
        print("{:<12} done in {:.0f}s".format(name, time.time() - began), flush=True)

    print()
    print("{:<30}".format("check") + "".join("{:>13}".format(name[:12]) for name in rows))
    for key in next(iter(rows.values())):
        print("{:<30}".format(key) + "".join("{:>13}".format(str(row.get(key, "-")))
                                             for row in rows.values()))


if __name__ == "__main__":
    main()
