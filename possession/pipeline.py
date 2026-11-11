"""Arrays in, labels out. The whole method in one function, so run.py stays a thin wrapper.

Pass 1 decodes from ball/player/tagging evidence. If restarts or the ball-free model are on,
they read the pass-1 labels, add their evidence, and pass 2 decodes again.
"""

import os
import sys

import numpy as np

from . import ball as ball_mod
from . import (audio, ballfree, config, deadball, decoder, evidence, restarts, tagging, touches,
               videotrack)

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks  # noqa: E402

LETTERS = np.array(["H", "A", "D"])
OPTIONS = ("restarts", "ballfree", "zone", "gapdead", "cleandead", "lineaware",
           "tagrestarts", "touches", "softcontrol", "videoball", "videofull",
           "whistle")


def resolve_options(options=None):
    """Which optional components are on: config defaults, overridden by `options`."""
    out = {"restarts": config.USE_RESTARTS, "ballfree": config.USE_BALLFREE,
           "zone": config.CALIBRATE_ZONE, "gapdead": config.USE_GAP_DEAD,
           "cleandead": config.USE_CLEAN_DEAD, "lineaware": config.USE_LINE_AWARE,
           "tagrestarts": config.USE_TAG_RESTARTS, "touches": config.USE_TOUCHES,
           "softcontrol": config.USE_SOFT_CONTROL, "videoball": config.USE_VIDEO_BALL,
           "videofull": False, "whistle": config.USE_WHISTLE}
    for key, value in (options or {}).items():
        if key not in out:
            raise ValueError("unknown option {!r}; expected one of {}".format(key, OPTIONS))
        out[key] = bool(value)
    return out


def emissions(arrays, events, graded_mask, blocks, calibrate_zone=False, gap_dead=False,
              clean_dead=False, line_aware=False, use_touches=False, soft_control=False,
              video_xy=None):
    """(T, 3) per-frame log-likelihoods for H, A, D plus the intermediate pieces.

    video_xy: optional (T, 2) metres of video ball where the ATD has none; used for the team
    (control and touch) evidence only, never for the dead-ball cues.
    """
    n = arrays["slots"].shape[0]
    pitch = tuple(arrays["pitch"])
    ball = ball_mod.track(arrays["ball"], pitch)

    team_xy = ball["xy"]
    if video_xy is not None:
        team_xy = np.where(np.isnan(ball["xy"][:, :1]), video_xy, ball["xy"])
    d_home, d_away = evidence.team_distances(arrays["slots"], arrays["team"], team_xy, pitch)
    zone, zone_info = config.POSSESSION_ZONE_M, None
    if calibrate_zone:
        zone, zone_info = evidence.calibrate_zone(d_home, d_away,
                                                  graded_mask & ball["observed"] & ~ball["oob"])
    state = evidence.control(d_home, d_away, possession_zone=zone)

    out = np.zeros((n, 3))
    if soft_control:
        out[:, :2] += evidence.soft_control_loglik(d_home, d_away, zone)
    else:
        out[:, :2] += evidence.control_loglik(state)
    # Turnovers snap to the first frame the new team is seen in control (or touching the ball).
    snap_state = state.copy()
    touch_info = None
    if use_touches:
        touch_state, touch_info = touches.detect(dict(ball, xy=team_xy), d_home, d_away)
        out[:, :2] += touches.loglik(touch_state)
        hit = touch_state != evidence.NONE
        snap_state[hit] = touch_state[hit]
    out[:, :2] += tagging.team_prior(events, n)
    dead, dead_parts = deadball.dead_loglik(arrays["slots"], ball, graded_mask,
                                            tagging.set_piece_intervals(events), gap_dead,
                                            clean_dead, line_aware)
    out[:, 2] += dead

    with np.errstate(invalid="ignore"):
        moving = ball["observed"] & (ball["speed"] >= config.BALL_MOVING_MS)
    forced, pre_kickoff_frames = tagging.pre_kickoff(events, blocks, n, moving)
    out[forced, :2] = -1e6
    out[forced, 2] = 0.0
    return out, {"ball": ball, "control": state, "dead_parts": dead_parts, "forced": forced,
                 "pre_kickoff_frames": pre_kickoff_frames, "snap_state": snap_state,
                 "touches": touch_info,
                 "zone_m": zone, "zone_info": zone_info}


def video_ball(arrays, graded):
    """(T, 2) metres of video ball filling ATD gaps on graded frames, plus info; (None, info)
    when the detections have not been produced yet."""
    loaded = videotrack.load_candidates()
    if loaded is None:
        print("WARNING: video ball requested but no detections in cache/ -- labelling without it")
        return None, {"available": False}
    sys.path.insert(0, os.path.join(config.ROOT, "starter"))
    from load_homography import load as load_homography
    candidates, done = loaded
    if not done[graded].all():
        raise RuntimeError("video detections cover only {:.1%} of graded frames; finish them with "
                           "python -m possession.video_ball detect".format(done[graded].mean()))
    pitch = np.asarray(arrays["pitch"], dtype=np.float64)
    matrices, _, _ = load_homography(source="tactical")
    pos, conf = videotrack.candidates_on_pitch(candidates, matrices, pitch)
    atd = arrays["ball"][:, :2] * pitch
    known = ~np.isnan(atd[:, 0])
    filled, info = videotrack.fill(atd, known, pos, conf, done & graded)
    return filled, dict(info, available=True)


def _decode_blocks(scores, parts, blocks, n, bonus=None):
    m = [config.MIN_POSSESSION, config.MIN_POSSESSION, config.MIN_DEAD]
    cost = np.array([
        [0.0, config.SWITCH_TURNOVER, config.SWITCH_TO_DEAD],
        [config.SWITCH_TURNOVER, 0.0, config.SWITCH_TO_DEAD],
        [config.SWITCH_FROM_DEAD, config.SWITCH_FROM_DEAD, 0.0],
    ])
    codes = np.full(n, -1, dtype=np.int8)
    for _, first, last in blocks:
        window = slice(first, last + 1)
        entry, leave = (None, None) if bonus is None else (bonus[0][window], bonus[1][window])
        decoded = decoder.decode(scores[window], m, cost, entry_bonus=entry, exit_bonus=leave)
        codes[window] = decoder.snap_turnovers(decoded, parts["snap_state"][window],
                                               evidence.HOME, evidence.AWAY, config.SNAP_WINDOW)
    return codes


def label(arrays, events=None, game=GAME, options=None, blocks=None):
    """(T,) array of 'H'/'A'/'D'/'X' for the whole video, plus diagnostics pieces."""
    opts = resolve_options(options)
    n = arrays["slots"].shape[0]
    events = tagging.load_events() if events is None else events
    blocks = half_video_blocks(game) if blocks is None else blocks
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    video_xy, video_info = None, None
    if opts["videoball"] or opts["videofull"]:
        video_xy, video_info = video_ball(arrays, graded)
        if video_xy is not None and opts["videofull"]:
            # Everything (dead-ball cues included) sees the video ball as if the ATD had it.
            pitch = np.asarray(arrays["pitch"], dtype=np.float64)
            fill = np.isnan(arrays["ball"][:, 0]) & ~np.isnan(video_xy[:, 0])
            ball_array = arrays["ball"].copy()
            ball_array[fill, :2] = video_xy[fill] / pitch
            if ball_array.shape[1] > 2:
                ball_array[fill, 2:] = np.nan
            arrays = dict(arrays, ball=ball_array)
            video_xy = None
    scores, parts = emissions(arrays, events, graded, blocks, calibrate_zone=opts["zone"],
                              gap_dead=opts["gapdead"],
                              clean_dead=opts["cleandead"],
                              line_aware=opts["lineaware"], use_touches=opts["touches"],
                              soft_control=opts["softcontrol"], video_xy=video_xy)
    parts["video"] = video_info
    bonus = None
    if opts["whistle"]:
        features = audio.load_features()
        if features is None:
            parts["whistle"] = {"available": False}
        else:
            events = audio.whistles(features)
            bonus = audio.boundary_bonus(events, n)
            scores = scores.copy()
            scores[:, 2] += np.where(parts["forced"], 0.0, audio.dead_evidence(events, n))
            parts["whistle"] = {"available": True, "events": len(events),
                                "graded_events": int(sum(graded[a] for a, _ in events))}
    codes = _decode_blocks(scores, parts, blocks, n, bonus)

    if opts["restarts"] or opts["ballfree"] or opts["tagrestarts"]:
        extra = np.zeros((n, 3))
        if opts["restarts"]:
            went_out = (parts["dead_parts"]["oob"] > 0) | (parts["dead_parts"]["gap_oob"] > 0)
            totals = {}
            for _, first, last in blocks:
                window = slice(first, last + 1)
                prior, info = restarts.restart_prior(codes[window], parts["control"][window],
                                                     went_out[window], parts["forced"][window])
                extra[window, :2] += prior
                for key, value in info.items():
                    totals[key] = totals.get(key, 0) + value
            parts["restarts"] = totals
        if opts["tagrestarts"]:
            prior, info = restarts.tagged_prior(tagging.set_pieces_with_team(events), codes)
            extra[:, :2] += prior
            parts["tag_restarts"] = info
        if opts["ballfree"]:
            prior, info = ballfree.emission(arrays, codes, parts["ball"]["known"], blocks)
            extra[:, :2] += prior
            parts["ballfree"] = info
        codes = _decode_blocks(scores + extra, parts, blocks, n, bonus)

    labels = np.full(n, "X", dtype="<U1")
    labels[codes >= 0] = LETTERS[codes[codes >= 0]]
    parts["graded"] = graded
    parts["options"] = opts
    return labels, parts


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, value) for i, value in enumerate(labels))
    return path
