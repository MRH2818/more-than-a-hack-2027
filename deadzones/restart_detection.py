"""Bounded, auditable team handoffs after observed restarts (offline).

Restart brackets come from stoppage_episodes' waiting/departure observations.
They are heuristic bounds, not statistical confidence intervals. Tag names alone
never assign possession, and the existing ownership model handles everything
outside the short post-restart window.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from deadzones.stoppage_episodes import runs


@dataclass(frozen=True)
class RestartParams:
    local_window_s: float = 1.0
    pre_evidence_s: float = 1.0
    min_team_evidence_s: float = 0.2
    max_pre_evidence_age_s: float = 0.4
    max_uncertainty_s: float = 1.0
    max_player_distance_m: float = 2.5
    min_team_distance_margin_m: float = 1.0
    min_vote_share: float = 0.8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.min_vote_share > 1:
            raise ValueError("min_vote_share must be at most one")
        if self.min_team_evidence_s > self.local_window_s:
            raise ValueError("team evidence must fit in the local window")


def tag_team_label(name, meta):
    """Exact normalized metadata names; do not guess aliases or similar clubs."""
    def normalize(value):
        return " ".join(str(value or "").casefold().split())
    names = [normalize(value) for value in meta.get("team_names", [])]
    value = normalize(name)
    if len(names) != 2 or not all(names) or names[0] == names[1]:
        return "X"
    return "HA"[names.index(value)] if value in names else "X"


def team_evidence(votes, start, end, minimum, share):
    """Require a sustained run as well as a dominant vote; ties abstain."""
    segment = votes[start:end]
    counts = [int((segment == team).sum()) for team in "HA"]
    total = sum(counts)
    if not total or counts[0] == counts[1]:
        return "X", None, None, 0
    team = "HA"[int(counts[1] > counts[0])]
    count = max(counts)
    supported = [(int(a + start), int(b + start)) for a, b in runs(segment == team)
                 if b - a >= minimum]
    if count / total < share or not supported:
        return "X", None, None, count
    return team, supported[0][0], supported[-1][1] - 1, count


def assign_restart_teams(f, detections, meta, quality, params=RestartParams()):
    """Generate local H/A proposals independently of any baseline or reference.

    Assignments start strictly after the inclusive restart uncertainty range. A
    known tag needs agreeing tracking; an unknown tag needs post-restart tracking.
    Conflicting tag/track evidence abstains. Proposals end on loss of evidence,
    contested/opposing proximity, another uncertainty range, or the one-second cap.
    """
    n, fps = len(f), meta["fps"]
    quality = np.asarray(quality, dtype=bool)
    if quality.shape != (n,):
        raise ValueError("quality must align with the feature frames")
    proposals = np.zeros((n, 2), dtype=bool)
    uncertain = np.zeros(n, dtype=bool)
    rows = []
    ordered = sorted(detections, key=lambda r: (r["restart_earliest_frame"],
                                               r["restart_latest_frame"], r["event_id"]))
    for detection in ordered:
        lower, upper = detection["restart_earliest_frame"], detection["restart_latest_frame"]
        limit = detection["window_end_frame_exclusive"]
        if not 0 <= lower <= upper < limit <= n:
            raise ValueError("restart bounds must be ordered and inside their event window")
        uncertain[lower:upper + 1] = True
    required = {"nearest_home_m", "nearest_away_m"}
    have_features = required.issubset(f.columns)
    votes = np.full(n, "X", dtype="<U1")
    contested = np.zeros(n, dtype=bool)
    live_observed = (quality & f.ball_valid.to_numpy()
                     & np.isfinite(f.outside_pitch_m.to_numpy())
                     & (f.outside_pitch_m.to_numpy() <= 0))
    if have_features:
        home, away = f.nearest_home_m.to_numpy(), f.nearest_away_m.to_numpy()
        observed = quality & f.ball_valid.to_numpy() & np.isfinite(home) & np.isfinite(away)
        votes[observed & (home <= params.max_player_distance_m)
              & (away - home >= params.min_team_distance_margin_m)] = "H"
        votes[observed & (away <= params.max_player_distance_m)
              & (home - away >= params.min_team_distance_margin_m)] = "A"
        contested = (observed & (np.minimum(home, away) <= params.max_player_distance_m)
                     & (np.abs(home - away) < params.min_team_distance_margin_m))
    minimum = max(1, round(params.min_team_evidence_s * fps))
    horizon = max(1, round(params.local_window_s * fps))
    pre_length = max(1, round(params.pre_evidence_s * fps))
    max_age = max(1, round(params.max_pre_evidence_age_s * fps))
    tags = [tag_team_label(row.get("tag_team"), meta) for row in ordered]
    pre_votes = votes.copy()
    pre_votes[uncertain] = "X"
    post_votes = votes.copy()
    post_votes[~live_observed | uncertain] = "X"
    # Conflicting tags veto their whole overlapping cluster, including an
    # unknown-team duplicate that would otherwise bypass the disagreement.
    conflicting_tags = set()
    for half in {row["half"] for row in ordered}:
        group, group_end = [], -1
        for index in [i for i, row in enumerate(ordered) if row["half"] == half]:
            row = ordered[index]
            if row["restart_earliest_frame"] > group_end:
                if len({tags[i] for i in group} - {"X"}) > 1:
                    conflicting_tags.update(group)
                group = []
            group.append(index)
            group_end = max(group_end, row["restart_latest_frame"])
        if len({tags[i] for i in group} - {"X"}) > 1:
            conflicting_tags.update(group)

    for index, detection in enumerate(ordered):
        lower, upper = detection["restart_earliest_frame"], detection["restart_latest_frame"]
        start = upper + 1
        end = min(n, start + horizon, detection["window_end_frame_exclusive"])
        next_uncertainty = np.flatnonzero(uncertain[start:end])
        if len(next_uncertainty):
            end = start + int(next_uncertainty[0])
        tag = tags[index]
        row = {"event_id": detection["event_id"], "half": detection["half"],
               "restart_type": detection["restart_type"],
               "restart_earliest_frame": lower, "restart_latest_frame": upper,
               "tag_team_label": tag, "pre_tracking_team": "X", "post_tracking_team": "X",
               "assigned_team": "X", "assignment_start_frame": None,
               "assignment_end_frame_exclusive": None, "source": "none",
               "pre_evidence_frames": 0, "post_evidence_frames": 0,
               "status": "abstained", "reason": ""}
        reason = ""
        if not have_features:
            reason = "Team-distance features unavailable"
        elif (upper - lower) / fps > params.max_uncertainty_s:
            reason = "Restart uncertainty exceeds local-assignment limit"
        elif start >= end:
            reason = "No post-restart frames inside event window"
        elif index in conflicting_tags:
            reason = "Overlapping restart tags disagree on team"
        if reason:
            rows.append({**row, "reason": reason})
            continue

        pre_start = max(detection["stop_observed_frame"], lower - pre_length, 0)
        pre, _, last_pre, pre_count = team_evidence(pre_votes, pre_start, lower, minimum, params.min_vote_share)
        if last_pre is not None and lower - 1 - last_pre > max_age:
            pre = "X"
        # Only use live, unambiguous observations after this restart. Do not
        # borrow an ownership vote from a later stoppage or from out-of-play ball.
        post, first_post, _, post_count = team_evidence(post_votes, start, end, minimum, params.min_vote_share)
        row.update(pre_tracking_team=pre, post_tracking_team=post,
                   pre_evidence_frames=pre_count, post_evidence_frames=post_count)
        if tag != "X":
            if (pre != "X" and pre != tag) or (post != "X" and post != tag):
                reason = "Tag and tracking disagree"
            elif tag not in {pre, post}:
                reason = "Tag has no corroborating tracking"
            else:
                team = tag
                source = "tag_and_pre_tracking" if pre == tag else "tag_and_post_tracking"
                if pre != tag:
                    start = first_post
        elif post == "X":
            reason = "No sustained post-restart team evidence"
        elif pre != "X" and pre != post:
            reason = "Pre- and post-restart tracking disagree"
        else:
            team = post
            source = "pre_and_post_tracking" if pre == post else "post_tracking"
            # Unknown-team tags never carry an owner forward through a flight
            # before we have actually observed that owner's post-restart touch.
            start = first_post
        if reason:
            rows.append({**row, "reason": reason})
            continue

        opposite = "A" if team == "H" else "H"
        barrier = (~live_observed | contested | uncertain | (votes == opposite))
        hits = np.flatnonzero(barrier[start:end])
        if len(hits):
            end = start + int(hits[0])
        if end - start < minimum:
            rows.append({**row, "reason": "Insufficient uninterrupted local evidence"})
            continue
        proposals[start:end, "HA".index(team)] = True
        rows.append({**row, "assigned_team": team, "assignment_start_frame": int(start),
                     "assignment_end_frame_exclusive": int(end), "source": source,
                     "status": "proposed", "reason": "Corroborated local team handoff"})

    conflict = proposals.all(axis=1)
    labels = np.full(n, "X", dtype="<U1")
    labels[proposals[:, 0] & ~conflict] = "H"
    labels[proposals[:, 1] & ~conflict] = "A"
    for row in rows:
        if row["status"] == "proposed":
            a, b = row["assignment_start_frame"], row["assignment_end_frame_exclusive"]
            row["selected_frames"] = int((labels[a:b] == row["assigned_team"]).sum())
            row["status"] = "assigned" if row["selected_frames"] else "abstained"
            if not row["selected_frames"]:
                row["reason"] = "Conflicting local team proposals"
        else:
            row["selected_frames"] = 0
    return {"labels": labels, "uncertain": uncertain, "conflict": conflict,
            "assignments": rows, "params": asdict(params)}


def apply_restart_overlay(base, dead_mask, restart_result, enabled=True):
    """Apply D first, then local team proposals; D and uncertainty block H/A.

    Do not resume an interrupted proposal after a newer selected dead interval.
    All labels outside these two explicitly selected masks are preserved.
    """
    base = np.asarray(base)
    dead_mask = np.asarray(dead_mask, dtype=bool)
    teams = restart_result["labels"]
    uncertain = restart_result["uncertain"]
    if (base.ndim != 1 or dead_mask.shape != base.shape or teams.shape != base.shape
            or uncertain.shape != base.shape or not np.isin(base, list("HADX")).all()):
        raise ValueError("Overlay inputs must be aligned H/A/D/X frame arrays")
    labels = np.where(dead_mask, "D", base)
    applied = np.zeros(len(base), dtype=bool)
    if enabled:
        for row in restart_result["assignments"]:
            if row["status"] != "assigned":
                continue
            a, b = row["assignment_start_frame"], row["assignment_end_frame_exclusive"]
            blocked = dead_mask[a:b] | uncertain[a:b] | (teams[a:b] != row["assigned_team"])
            hits = np.flatnonzero(blocked)
            if len(hits):
                b = a + int(hits[0])
            labels[a:b] = row["assigned_team"]
            applied[a:b] = True
    return labels, applied
