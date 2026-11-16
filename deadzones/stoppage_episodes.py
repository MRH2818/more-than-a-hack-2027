"""Reconstruct bounded stoppages from observations, without possession labels.

Short D fragments are useful evidence but can split one predicted possession into
two. An overlay therefore needs an observed waiting period AND a plausible restart.
No tag boundary, missing track, or moving player is itself a restart timestamp.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EpisodeParams:
    max_bridge_s: float = 2.0
    departure_confirmation_s: float = 0.4
    min_departure_displacement_m: float = 0.5
    max_throw_travel_m: float = 15.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bridge_observation_gaps(observed, allowed, maximum):
    """Close only short, bracketed gaps; never extend either end of a run."""
    out = np.asarray(observed, dtype=bool).copy()
    for a, b in runs(~out):
        if a > 0 and b < len(out) and b - a <= maximum and allowed[a:b].all():
            out[a:b] = True
    return out


def restart_location(x, y, kind, pitch):
    length, width = pitch
    if kind.startswith("kick off"):
        return np.hypot(x - length / 2, y - width / 2) <= 2
    if kind == "throw-in":
        return np.minimum(np.abs(y), np.abs(y - width)) <= 2
    if kind == "goal kick":
        return ((np.minimum(np.abs(x), np.abs(x - length)) <= 8)
                & (np.abs(y - width / 2) <= 12))
    if kind == "corner kick":
        return ((np.minimum(np.abs(x), np.abs(x - length)) <= 3)
                & (np.minimum(np.abs(y), np.abs(y - width)) <= 3))
    if kind == "free kick":
        return (x >= 0) & (x <= length) & (y >= 0) & (y <= width)
    return np.zeros(len(x), dtype=bool)


def observed_exit(start, lower_bound, signed_distance, reliable, maximum):
    """Refine a strong out seed to a contiguous, observed boundary crossing.

    Do not extrapolate across missing samples or assume that the first tracked
    outside point was when the ball crossed the line.
    """
    for frame in range(start - 1, max(lower_bound, start - maximum) - 1, -1):
        if not reliable[frame]:
            break
        if signed_distance[frame] <= 0:
            return frame + 1
    return start


def reconstruct_stoppages(f, events, meta, quality, detector_params,
                          params=EpisodeParams()):
    """Return continuous D episodes and rejected candidates for review.

Outside observations must agree with the restart type. Ordinary occlusions may
only join two observations, for at most max_bridge_s, with valid player/geometry
quality throughout. A visible moving ball inside the pitch breaks waiting.
Kickoff formation retains the detector's explicit missing-ball exception.

Departure requires actual coordinate displacement over several consecutive
observations, not just the ATD speed channel. Goal kicks, corners and free kicks
also require a stationary setup at the correct restart location. Throw-ins can
use a short outside-to-inside trajectory from the same touchline.
"""
    p = detector_params
    n, fps = len(f), meta["fps"]
    length, width = meta["pitch"]
    x, y = f.ball_x_m.to_numpy(), f.ball_y_m.to_numpy()
    valid = f.ball_valid.to_numpy()
    quality = np.asarray(quality, dtype=bool)
    speed = f.ball_speed_mps.to_numpy()
    outside = f.outside_pitch_m.to_numpy()
    near = (f.nearest_player_m <= 4).to_numpy()
    calm = ((f.player_median_mps <= p.max_player_median_mps)
            & (f.player_p90_mps <= p.max_player_p90_mps)).to_numpy()
    still = valid & (speed <= p.max_ball_still_mps)
    moving = quality & valid & (speed >= p.min_ball_launch_mps) & (outside <= 0)
    confirmation = max(2, round(params.departure_confirmation_s * fps))
    gap = max(1, round(params.max_bridge_s * fps))
    setup_length = max(1, round(p.min_setup_s * fps))
    out_length = max(1, round(p.min_out_s * fps))
    guard = round(p.boundary_guard_s * fps)
    selected = np.zeros(n, dtype=bool)
    episodes, review, detections = [], [], []
    # Features use centered velocities. Do not report their observed transition
    # as an exact contact frame, even when waiting and motion are adjacent.
    timing_margin = max(1, round(.2 * fps))
    restarts = sorted((e for e in events if e.code == "SET PIECES"),
                      key=lambda e: (e.start, e.end, e.id))

    for event in restarts:
        if not event.half:
            continue
        tag_start = max(0, round(event.start * fps))
        lo = max(0, round((event.start - p.lookback_s) * fps))
        hi = min(n, round(event.end * fps))
        # A second-half clip must not borrow a first-half observation or restart.
        starts = [e.start for e in restarts
                  if e.half == event.half and e.kind == "kick off (start)"]
        if starts:
            lo = max(lo, round((min(starts) - p.lookback_s) * fps))
        other_halves = [round((e.start - p.lookback_s) * fps) for e in restarts
                        if e.half and e.half != event.half
                        and e.kind == "kick off (start)" and e.start > event.start]
        if other_halves:
            hi = min(hi, min(other_halves))
        if hi <= lo:
            continue

        location = restart_location(x, y, event.kind, meta["pitch"])
        setup = quality & still & near & location
        # A stationary free-kick location anywhere in the pitch is too weak by
        # itself: retain the original collective-calm requirement for the seed.
        setup_seed = setup & calm
        if event.kind == "throw-in":
            boundary_distance = np.maximum(-y, y - width)
            out = boundary_distance >= p.outside_margin_m
        elif event.kind in {"goal kick", "corner kick"}:
            boundary_distance = np.maximum(-x, x - length)
            out = boundary_distance >= p.outside_margin_m
        else:
            boundary_distance = np.full(n, np.nan)
            out = np.zeros(n, dtype=bool)
        out &= quality & valid
        waiting = out | setup
        formation = np.zeros(n, dtype=bool)
        if event.kind.startswith("kick off"):
            formation = (quality & (f.kickoff_formation_fraction >= .9).to_numpy()
                         & (f.player_median_mps <= p.kickoff_idle_median_mps).to_numpy()
                         & (f.player_p90_mps <= p.kickoff_idle_p90_mps).to_numpy()
                         & (~valid | (location & still)))
            waiting |= formation
        # Observed in-pitch motion, poor geometry and sparse players are barriers.
        allowed = quality & ~valid
        waiting = bridge_observation_gaps(waiting[lo:hi], allowed[lo:hi], gap)

        for a, b in runs(waiting):
            a, b = int(lo + a), int(lo + b)
            anchors = []
            for c, d in runs(out[a:b]):
                if d - c >= out_length:
                    anchors.append((int(a + c), "observed_ball_out"))
            for c, d in runs(setup_seed[a:b]):
                c, d = int(a + c), int(a + d)
                if d - c >= setup_length and np.hypot(np.ptp(x[c:d]), np.ptp(y[c:d])) <= 1:
                    anchors.append((c, "stationary_setup"))
            for c, d in runs(formation[a:b]):
                if d - c >= setup_length:
                    anchors.append((int(a + c), "kickoff_formation"))
            if not anchors:
                continue
            start, stop_evidence = min(anchors)
            if stop_evidence == "observed_ball_out":
                start = observed_exit(start, lo, boundary_distance, quality & valid, gap)
            search_start = max(b, tag_start)
            search_end = min(hi, b + gap + confirmation)
            restart = None
            confirmed = None
            for c, d in runs(moving[search_start:search_end]):
                c, d = int(search_start + c), int(search_start + d)
                if c - b > gap or d - c < confirmation:
                    continue
                if not quality[b:c].all() or moving[b:c].any():
                    continue
                last = c + confirmation - 1
                travel = np.hypot(x[last] - x[c], y[last] - y[c])
                if travel < params.min_departure_displacement_m:
                    continue
                # Reject teleporting reappearances, including the first valid
                # frame after the adjacent-jump filter has recovered.
                seen = np.flatnonzero(valid[start:b]) + start
                if len(seen):
                    previous = int(seen[-1])
                    distance = np.hypot(x[c] - x[previous], y[c] - y[previous])
                    if distance > 45 * (c - previous) / fps:
                        continue
                if event.kind.startswith("kick off"):
                    plausible = np.hypot(x[c] - length / 2, y[c] - width / 2) <= 15
                    plausible_confirmation = last
                else:
                    recent = np.flatnonzero(setup[max(start, b - gap):b]) + max(start, b - gap)
                    plausible = False
                    plausible_confirmation = last
                    if len(recent) >= setup_length:
                        origin = int(recent[-1])
                        displaced_end = min(d, c + max(confirmation, round(p.max_launch_delay_s * fps)))
                        displaced = np.flatnonzero(np.hypot(x[c:displaced_end] - x[origin],
                                                            y[c:displaced_end] - y[origin]) >= 1.5)
                        plausible = len(displaced) > 0
                        if plausible:
                            plausible_confirmation = max(last, c + int(displaced[0]))
                    if event.kind == "throw-in" and len(seen):
                        origin = int(seen[-1])
                        same_side = y[origin] <= 2 or y[origin] >= width - 2
                        inward = ((y[origin] <= 2 and y[last] > y[c])
                                  or (y[origin] >= width - 2 and y[last] < y[c]))
                        throw_departure = (same_side and inward
                                           and np.hypot(x[c] - x[origin], y[c] - y[origin])
                                           <= params.max_throw_travel_m)
                        if throw_departure:
                            plausible = True
                            plausible_confirmation = last
                if plausible:
                    restart = c
                    confirmed = plausible_confirmation
                    break
            row = {"event_id": event.id, "restart_type": event.kind, "half": event.half,
                   "stop_evidence": stop_evidence, "stop_observed_frame": start,
                   "last_waiting_frame_exclusive": b, "restart_motion_frame": restart,
                   "frame_verified": False}
            if restart is None:
                review.append({**row, "reason": "No corroborated restart within bounded search"})
                continue
            bounds = {"restart_earliest_frame": max(lo, b - timing_margin),
                      "restart_latest_frame": min(hi - 1, restart + timing_margin),
                      "confirmation_frame": confirmed,
                      "confirmation_available_frame": min(n - 1, confirmed + timing_margin),
                      "timing_margin_frames": timing_margin,
                      "bounds_kind": "heuristic_observation_bracket"}
            bounds["restart_uncertainty_s"] = (bounds["restart_latest_frame"]
                                                - bounds["restart_earliest_frame"]) / fps
            detections.append({**row, **bounds, "tag_team": event.team,
                               "window_end_frame_exclusive": hi})
            # [b, restart] is an uncertain restart bracket. Do not extend D to
            # its late edge just because the first sustained motion is delayed.
            sf, ef = start + guard, min(b, restart - guard)
            if ef <= sf:
                continue
            selected[sf:ef] = True
            episodes.append({**row, **bounds, "start_frame": sf, "end_frame_exclusive": ef,
                             "duration_s": (ef - sf) / fps,
                             "restart_uncertainty_frames": restart - b})
    return {"mask": selected, "episodes": episodes, "review": review, "restarts": detections,
            "params": asdict(params)}
