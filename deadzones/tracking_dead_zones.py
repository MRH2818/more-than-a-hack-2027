"""Fuse tactical tags and ATD evidence into dead-ball candidates, never certainty.

Run from the repository root: python -m deadzones.tracking_dead_zones --help
Thresholds are explicit physical heuristics, not fitted to public ground truth.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from deadzones.dead_zones import Config, infer, load_events, dead_mask, read_labels


@dataclass(frozen=True)
class Params:
    lookback_s: float = 30.0
    min_players: int = 18
    min_team_players: int = 7
    max_player_median_mps: float = 1.2
    max_player_p90_mps: float = 2.5
    max_ball_still_mps: float = 0.7
    min_ball_launch_mps: float = 2.0
    max_homography_error: float = 2.0
    min_setup_s: float = 1.0
    min_out_s: float = 0.6
    boundary_guard_s: float = 0.2
    max_launch_delay_s: float = 2.0
    outside_margin_m: float = 1.0
    kickoff_idle_median_mps: float = 0.5
    kickoff_idle_p90_mps: float = 1.5


def metadata(path):
    root = ET.parse(path).getroot()
    teams = [e.get("id") for e in root.findall(".//Teams/Team")]
    players = root.findall(".//Players/Player")
    spec = root.find(".//DataFormatSpecification")
    channels = {e.get("id"): e.findtext("Unit") for e in root.findall(".//Channel")}
    if channels != {"x": "normalized", "y": "normalized", "s": "kph"}:
        raise ValueError("Unexpected coordinate/speed units: " + str(channels))
    if len(teams) != 2 or any(e.get("teamId") not in teams for e in players):
        raise ValueError("Invalid track/team mapping")
    refs = [e.get("playerChannelId") for e in spec.findall(".//PlayerChannelRef")]
    expected = ["player{}_{}".format(e.get("id"), channel) for e in players for channel in ("x", "y", "speed")]
    if refs != expected or int(spec.get("startFrame")) != 1:
        raise ValueError("Unsupported ATD channel order or first frame")
    return {"fps": float(root.findtext(".//FrameRate")),
            "pitch": [float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height"))],
            "n_frames": int(spec.get("endFrame")),
            "teams": [teams.index(e.get("teamId")) for e in players],
            "team_names": [e.findtext("Name") for e in root.findall(".//Teams/Team")]}


def load_tracking(raw_path, meta):
    n, slots = meta["n_frames"], len(meta["teams"])
    data = np.empty((n, slots + 1, 3), dtype=np.float32)
    count = 0
    with open(raw_path) as stream:
        for i, line in enumerate(stream):
            parts = line.strip().split(":")
            if i >= n or len(parts) != 3 or int(parts[0]) != i + 1:
                raise ValueError("ATD frame numbering must be contiguous and one-based")
            values = np.fromstring(parts[1].replace(";", ",") + "," + parts[2], sep=",", dtype=np.float32)
            if len(values) != (slots + 1) * 3:
                raise ValueError("Wrong track count at frame " + str(i))
            data[i] = values.reshape(slots + 1, 3)
            count += 1
    if count != n:
        raise ValueError("ATD is truncated")
    data[:, :, :2] *= np.asarray(meta["pitch"], dtype=np.float32)
    data[:, :, 2] /= 3.6
    return data[:, :-1], data[:, -1]


def load_geometry(path, n):
    table = pd.read_csv(path)
    if len(table) != n or not np.array_equal(table.frame.to_numpy(), np.arange(n)):
        raise ValueError("Homography frames do not align with video")
    matrix = table[["h{}{}".format(i, j) for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    finite = np.isfinite(matrix).all(axis=(1, 2))
    determinant = np.zeros(n)
    determinant[finite] = np.linalg.det(matrix[finite])
    return table.valid.to_numpy() == 1, table.reprojection_error.to_numpy(), finite & (np.abs(determinant) > 1e-8)


def motion(xy, fps, half_window=5):
    """Centered 0.4s velocity at 25fps. Missing endpoints stay missing."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    k = half_window
    if len(xy) > 2 * k:
        out[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) * fps / (2 * k)
    return out


def features(players, ball, meta, geometry):
    fps = meta["fps"]
    teams = np.asarray(meta["teams"])
    k = max(1, round(fps * .2))
    measured = motion(players[:, :, :2], fps, k)
    # Track switches/teleports are invalid evidence, not a burst of player motion.
    supplied = players[:, :, 2]
    speed = np.fmax(measured, supplied)
    valid = np.isfinite(players[:, :, :2]).all(axis=2) & np.isfinite(measured) & (speed <= 12) & (speed >= 0)
    speed[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(speed, axis=1)
        p90 = np.nanquantile(speed, .9, axis=1)
        near = np.nanmin(np.linalg.norm(players[:, :, :2] - ball[:, None, :2], axis=2), axis=1)
    bspeed = np.fmax(motion(ball[:, :2], fps, k), ball[:, 2])
    bvalid = np.isfinite(ball[:, :2]).all(axis=1) & np.isfinite(bspeed) & (bspeed >= 0) & (bspeed <= 45)
    # Also reject each adjacent jump, including reappearance across missing data.
    adjacent = np.full(len(ball), np.nan)
    adjacent[1:] = np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) * fps
    bvalid &= np.isfinite(adjacent) & (adjacent <= 45)
    length, width = meta["pitch"]
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width])
    # Approximate kickoff formation: teams occupy opposite sides of halfway.
    # No fixed attack direction is assumed; the two halves can be reversed.
    finite_pos = np.isfinite(players[:, :, :2]).all(axis=2)
    side = []
    for t in (0, 1):
        pos = finite_pos[:, teams == t]
        x = players[:, teams == t, 0]
        denom = np.maximum(1, pos.sum(axis=1))
        side.append(((pos & (x <= length / 2 + 2)).sum(axis=1) / denom,
                     (pos & (x >= length / 2 - 2)).sum(axis=1) / denom))
    shape = np.maximum(np.minimum(side[0][0], side[1][1]), np.minimum(side[0][1], side[1][0]))
    return pd.DataFrame({"frame": np.arange(len(ball)), "players_seen": valid.sum(axis=1),
                         "home_seen": valid[:, teams == 0].sum(axis=1),
                         "away_seen": valid[:, teams == 1].sum(axis=1),
                         "player_median_mps": med, "player_p90_mps": p90,
                         "ball_x_m": ball[:, 0], "ball_y_m": ball[:, 1],
                         "ball_speed_mps": bspeed, "ball_valid": bvalid,
                         "nearest_player_m": near, "outside_pitch_m": outside,
                         "kickoff_formation_fraction": shape,
                         "homography_valid": geometry[0] & geometry[2], "homography_error": geometry[1]})


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def sustained(mask, minimum):
    out = np.zeros(len(mask), dtype=bool)
    for lo, hi in runs(mask):
        if hi - lo >= minimum:
            out[lo:hi] = True
    return out


def detect(f, events, meta, p=Params()):
    """Offline inference; no labels enter this function.

    'tracking_supported' is evidence-based, not calibrated certainty. Movement
    veto means stop overriding possession, not proof of live play or ownership.
    """
    n, fps = len(f), meta["fps"]
    length, width = meta["pitch"]
    quality = ((f.players_seen >= p.min_players) & (f.players_seen <= 24)
               & (f.home_seen >= p.min_team_players) & (f.away_seen >= p.min_team_players)
               & f.homography_valid & f.homography_error.between(0, p.max_homography_error)).to_numpy()
    calm = ((f.player_median_mps <= p.max_player_median_mps)
            & (f.player_p90_mps <= p.max_player_p90_mps)).to_numpy()
    ball_ok = f.ball_valid.to_numpy()
    still = (f.ball_speed_mps <= p.max_ball_still_mps).to_numpy() & ball_ok
    moving = ((f.ball_speed_mps >= p.min_ball_launch_mps) & (f.outside_pitch_m <= .3)).to_numpy() & ball_ok
    # Require multiple consecutive frames so one bad observation cannot revoke a core.
    resume = sustained(quality & moving & ~calm, max(1, round(.24 * fps)))
    scope = np.zeros(n, dtype=bool)
    supported = np.zeros(n, dtype=bool)
    medium = np.zeros(n, dtype=bool)
    idle = np.zeros(n, dtype=bool)
    intervals, summaries = [], []
    guard = round(p.boundary_guard_s * fps)
    restarts = [e for e in events if e.code == "SET PIECES"]

    def add(event, lo, hi, rule, tier, launch=None):
        lo, hi = lo + guard, hi - guard
        if hi <= lo:
            return
        mask = supported if tier == "tracking_supported" else medium
        mask[lo:hi] = True
        intervals.append({"event_id": event.id, "restart_type": event.kind, "half": event.half,
                          "rule": rule, "tier": tier, "frame_verified": False,
                          "start_frame": lo, "end_frame_exclusive": hi,
                          "duration_s": (hi - lo) / fps, "restart_motion_frame": launch})

    for e in restarts:
        lo, hi = max(0, round((e.start - p.lookback_s) * fps)), min(n, round(e.end * fps))
        # Windows may extend before half kickoff for the kickoff itself; later
        # restarts never borrow evidence from the previous half's tags.
        starts = [k for k in restarts if k.half == e.half and k.kind == "kick off (start)"]
        if starts:
            lo = max(lo, round((min(k.start for k in starts) - p.lookback_s) * fps))
        if hi <= lo:
            continue
        scope[lo:hi] = True
        quiet = quality[lo:hi] & calm[lo:hi]
        idle[lo:hi] |= sustained(quiet, round(p.min_setup_s * fps))
        x, y = f.ball_x_m.to_numpy(), f.ball_y_m.to_numpy()
        if e.kind.startswith("kick off"):
            location = (np.hypot(x - length / 2, y - width / 2) <= 2.0)
            location &= (f.kickoff_formation_fraction >= .8).to_numpy()
        elif e.kind == "corner kick":
            location = (np.minimum(np.abs(x), np.abs(x - length)) <= 3) & (np.minimum(np.abs(y), np.abs(y - width)) <= 3)
        elif e.kind == "goal kick":
            location = (np.minimum(np.abs(x), np.abs(x - length)) <= 8) & (np.abs(y - width / 2) <= 12)
        elif e.kind == "throw-in":
            location = np.minimum(np.abs(y), np.abs(y - width)) <= 2
        elif e.kind == "free kick":
            location = (f.outside_pitch_m <= .3).to_numpy()
        else:
            location = np.zeros(n, dtype=bool)
        if e.kind.startswith("kick off"):
            # Player formation supplies evidence while the ball is occluded,
            # but only a subsequent centre-area ball departure corroborates it.
            # A detected moving/off-centre ball interrupts this special rule.
            radius = np.hypot(x - length / 2, y - width / 2)
            waiting = (quality & (f.player_median_mps <= p.kickoff_idle_median_mps).to_numpy()
                       & (f.player_p90_mps <= p.kickoff_idle_p90_mps).to_numpy()
                       & (f.kickoff_formation_fraction >= .9).to_numpy()
                       & (~ball_ok | ((radius <= 2) & still)))
            departure = (quality & ball_ok & moving & (radius <= 15)
                         & ((f.player_median_mps > p.kickoff_idle_median_mps)
                            | (f.player_p90_mps > p.kickoff_idle_p90_mps)).to_numpy())
            for a, b in runs(waiting[lo:hi]):
                a, b = lo + a, lo + b
                if b - a < round(p.min_setup_s * fps):
                    continue
                search_end = min(hi, b + round(p.max_launch_delay_s * fps))
                for c, d in runs(departure[b:search_end]):
                    if d - c >= round(.24 * fps):
                        add(e, a, b, "kickoff_formation_then_departure", "tracking_supported", b + c)
                        break
        setup = quiet & still[lo:hi] & location[lo:hi] & (f.nearest_player_m.to_numpy()[lo:hi] <= 4)
        for a, b in runs(setup):
            a, b = lo + a, lo + b
            if b - a < round(p.min_setup_s * fps):
                continue
            # Slow carrying is not a stationary restart: require a tight cluster.
            if np.hypot(np.ptp(x[a:b]), np.ptp(y[a:b])) > 1.0:
                continue
            launch = None
            end_search = min(hi, b + round(p.max_launch_delay_s * fps))
            displaced = np.hypot(x[b:end_search] - np.median(x[a:b]), y[b:end_search] - np.median(y[a:b])) >= 1.5
            for c, d in runs(resume[b:end_search] & displaced):
                if d - c >= round(.24 * fps):
                    launch = b + c
                    break
            add(e, a, b, "stationary_setup_then_launch" if launch is not None else "stationary_setup_unresolved",
                "tracking_supported" if launch is not None else "needs_confirmation", launch)
            # Only the first observed launch anchors this event. Subsequent
            # stationary spells within its padded clip could be ordinary live play.
            if launch is not None:
                break
        # Ball outside with a generous geometric margin and persistence. This
        # remains inferred: pitch coordinates do not encode ball height or whistle.
        outside = quality[lo:hi] & ball_ok[lo:hi] & (f.outside_pitch_m.to_numpy()[lo:hi] >= p.outside_margin_m)
        for a, b in runs(outside):
            if b - a >= round(p.min_out_s * fps):
                add(e, lo + a, lo + b, "ball_outside_near_restart", "tracking_supported")
        summaries.append({"event_id": e.id, "type": e.kind, "start_frame": lo, "end_frame_exclusive": hi,
                          "quality_frames": int(quality[lo:hi].sum()),
                          "ball_observed_frames": int(ball_ok[lo:hi].sum()),
                          "idle_players_frames": int(quiet.sum())})

    resume &= scope
    supported &= ~resume
    medium &= ~supported & ~resume
    idle &= scope & ~supported & ~medium & ~resume
    # Original goal clips are retained as a separate, weaker source, and can be
    # revoked by positive movement evidence anywhere in a goal core.
    original = np.asarray(dead_mask(infer(events, Config(fps=fps, n_frames=n))), dtype=bool)
    if len(original) != n:
        raise ValueError("Smart-tagging and tracking frame counts disagree")
    goal_resume = sustained(quality & moving & ~calm, max(1, round(.24 * fps))) & original
    resume |= goal_resume
    combined = (original & ~resume) | supported
    for row in summaries:
        evidence = [r for r in intervals if r["event_id"] == row["event_id"]]
        selected = [r for r in evidence if r["tier"] == "tracking_supported"]
        event_mask = np.zeros(n, dtype=bool)
        for r in selected:
            event_mask[r["start_frame"]:r["end_frame_exclusive"]] = True
        row["supported_frames"] = int((event_mask & supported).sum())
        row["rules"] = ";".join(sorted({r["rule"] for r in selected}))
        row["restart_motion_frames"] = ";".join(str(r["restart_motion_frame"]) for r in selected if r["restart_motion_frame"] is not None)
        row["decision"] = "tracking_supported_candidates" if selected else "needs_more_evidence"
    return {"supported": supported, "setup_review": medium, "idle_review": idle,
            "resume": resume, "scope": scope, "quality": quality, "original": original,
            "combined": combined, "intervals": intervals, "restarts": summaries,
            "params": asdict(p)}


def public_metrics(mask, truth):
    known = truth != "X"
    tp = int((mask & (truth == "D")).sum())
    fp = int((mask & known & (truth != "D")).sum())
    total_dead = int((truth == "D").sum())
    return {"true_dead": tp, "false_dead": fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / total_dead if total_dead else None,
            "graded_predictions": tp + fp}


def export(result, f, meta, args, truth=None, base=None):
    out, fps = args.out, meta["fps"]
    original, supported, combined = (result[k] for k in ("original", "supported", "combined"))
    additions = supported & ~original
    scored = np.zeros(len(f), dtype=bool)
    for _, lo, hi in half_video_blocks(GAME):
        scored[lo:hi + 1] = True
    summary = {"frames": len(f), "fps": fps, "params": result["params"],
               "tracking_supported_frames": int(supported.sum()),
               "new_frames_beyond_tag_only": int(additions.sum()),
               "new_seconds_beyond_tag_only": float(additions.sum() / fps),
               "new_frames_in_scored_periods": int((additions & scored).sum()),
               "new_frames_outside_scored_periods": int((additions & ~scored).sum()),
               "original_goal_frames": int(original.sum()),
               "original_goal_frames_revoked": int((original & result["resume"]).sum()),
               "combined_dead_frames": int(combined.sum()),
               "setup_needs_confirmation_frames": int(result["setup_review"].sum()),
               "idle_only_review_frames": int(result["idle_review"].sum()),
               "tagged_restarts_with_supported_evidence": len({r["event_id"] for r in result["intervals"] if r["tier"] == "tracking_supported"}),
               "certainty": "All tracking-derived intervals are heuristic, not frame-verified or calibrated probabilities.",
               "offline": "Uses centered velocities and future launch evidence; not a live streaming detector.",
               "inputs": {k: str(getattr(args, k).resolve()) for k in ("raw", "metadata", "homography", "tags")}}
    if truth is not None:
        summary["public_evaluation"] = {name: public_metrics(mask, truth) for name, mask in
                                        (("tag_only", original), ("tracking_supported", supported),
                                         ("new_frames", additions), ("combined", combined))}
        summary["public_evaluation"]["by_rule"] = {}
        for rule in sorted({r["rule"] for r in result["intervals"]}):
            mask = np.zeros(len(f), dtype=bool)
            for r in result["intervals"]:
                if r["rule"] == rule:
                    mask[r["start_frame"]:r["end_frame_exclusive"]] = True
            summary["public_evaluation"]["by_rule"][rule] = public_metrics(mask, truth)
    pd.DataFrame(result["intervals"], columns=["event_id", "restart_type", "half", "rule", "tier", "frame_verified", "start_frame", "end_frame_exclusive", "duration_s", "restart_motion_frame"]).to_csv(out / "intervals.csv", index=False)
    pd.DataFrame(result["restarts"]).to_csv(out / "restart_review.csv", index=False)
    for name, mask in (("tracking_supported", supported), ("combined", combined), ("new_frames", additions)):
        pd.DataFrame({"frame": f.frame, "label": np.where(mask, "D", "X")}).to_csv(out / (name + ".csv"), index=False)
    if base is not None:
        labels = np.where(combined, "D", base)
        pd.DataFrame({"frame": f.frame, "label": labels}).to_csv(out / "overlaid.csv", index=False)
        if truth is not None:
            summary["public_overlay_dead_evaluation"] = {
                "before": public_metrics(base == "D", truth), "after": public_metrics(labels == "D", truth),
                "note": "Frames outside the combined override retain the original prediction, including its D labels."}
            from evaluation import possession_metrics as pm
            summary["public_possession_scores"] = {}
            for name, prediction in (("before", base), ("after", labels)):
                scores = pm.evaluate(pm.label_codes(truth), pm.label_codes(prediction), bootstrap_n=0)
                summary["public_possession_scores"][name] = {key: scores[key] for key in ("macro_f1", "spell_f1_mean")}
    review = f.copy()
    review["status"] = np.select([supported, result["resume"], original, result["setup_review"], result["idle_review"], result["scope"]],
                                  ["tracking_supported_dead", "release_to_possession", "tag_only_goal_candidate",
                                   "stationary_setup_needs_confirmation", "idle_players_only", "unresolved_restart_window"], default="outside_review")
    review["selected_dead"] = combined
    review.to_csv(out / "frame_evidence.csv.gz", index=False, compression="gzip")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Tracking-supported dead zones", "",
             "**{} additional candidate dead frames ({:.2f}s)** beyond the tag-only goal cores. "
             "Tracking evidence supports {} frames in total, across {} tagged restarts. "
             "These are heuristic detections, not certain labels.".format(summary["new_frames_beyond_tag_only"], summary["new_seconds_beyond_tag_only"],
                summary["tracking_supported_frames"], summary["tagged_restarts_with_supported_evidence"]), "",
             "{} added frames are inside the published scored periods; {} are outside them "
             "(for example, before the first kickoff). Overlapping tags can support the same interval; "
             "frame counts are unions, not sums of clips.".format(summary["new_frames_in_scored_periods"], summary["new_frames_outside_scored_periods"]), "",
             "The combined mask contains {} D frames, retaining the original goal candidates except "
             "for {} frames with sustained movement evidence. Unselected frames are unknown, not live.".format(summary["combined_dead_frames"], summary["original_goal_frames_revoked"]), "",
             "## Evidence rules", "",
             "- **Stationary setup → launch:** enough players from both teams, low median and upper-percentile "
             "player speed, a stationary ball near a player in the appropriate restart area, followed by "
             "sustained ball displacement and player movement. Kickoffs additionally require opposite-half formation.",
             "- **Ball outside near restart:** tracked ball at least 1m beyond the pitch, persisting at least "
             "0.6s, with sufficient players and valid low-error geometry. A 0.2s guard trims each end.",
             "- **Kickoff formation → departure:** players remain nearly still in opposite-half formation, "
             "then a sustained moving ball near the centre and increased player movement corroborate a restart. "
             "This specific rule permits an occluded ball during the waiting period; conflicting visible ball evidence breaks it.",
             "- **Abstention:** missing ball alone, player stillness alone, unresolved stationary setup, bad geometry, "
             "too few players, and implausible jumps do not become D. Other rules never bridge missing-ball gaps.",
             "- **Resume:** sustained moving ball inside the pitch plus player activity releases the override. "
             "Players moving by themselves never establish a restart or possession.", "",
             "{} stationary-setup frames and {} idle-only frames remain review candidates. All movement "
             "thresholds are fixed heuristics, not trained on public labels. Centered velocities and future "
             "launch checks make this an offline analysis.".format(summary["setup_needs_confirmation_frames"], summary["idle_only_review_frames"]), "",
             "## Public five-minute diagnostic", ""]
    if truth is not None:
        lines += ["| Mask | Correct D | False D | Precision | Dead recall |", "|---|---:|---:|---:|---:|"]
        for key in ("tag_only", "tracking_supported", "new_frames", "combined"):
            s = summary["public_evaluation"][key]
            lines.append("| {} | {} | {} | {} | {} |".format(key, s["true_dead"], s["false_dead"],
                "n/a" if s["precision"] is None else "{:.1%}".format(s["precision"]),
                "n/a" if s["recall"] is None else "{:.1%}".format(s["recall"])))
        lines += ["", "This tiny public slice is a diagnostic, not full-match validation. Per-rule results "
                  "are in summary.json. Public labels never enter inference."]
        if "public_possession_scores" in summary:
            lines += ["", "| Existing possession overlay | macroF1 | spellF1 |", "|---|---:|---:|"]
            for name, scores in summary["public_possession_scores"].items():
                lines.append("| {} | {:.4f} | {:.4f} |".format(name, scores["macro_f1"], scores["spell_f1_mean"]))
    else:
        lines.append("Not evaluated; pass --public-gt for a separate diagnostic.")
    lines += ["", "## Files", "", "- `intervals.csv`: event-linked evidence ranges and inferred launch frames; exclusive ends.",
              "- `new_frames.csv`: only additions beyond the old goal cores, D/X on every video frame.",
              "- `tracking_supported.csv`: tracking-only D/X mask.", "- `combined.csv`: original goal candidates plus tracking additions, minus movement conflicts.",
              "- `frame_evidence.csv.gz`: per-frame features, quality fields, status and selected D mask.",
              "- `restart_review.csv`: coverage and visibility for every tagged restart.",
              "- `overlaid.csv`, if requested: existing possession predictions with the combined D override.", "",
              "Homography is a quality gate. ATD coordinates are already pitch coordinates; applying the "
              "matrix again would be incorrect. Frame alignment: ATD row ID minus one; tags round(seconds × 25); "
              "homography frame already zero-based. Pitch size and km/h speed units come from the supplied XML.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths = asset_paths()
    parser.add_argument("--raw", type=Path, default=Path(paths["atd_raw"]))
    parser.add_argument("--metadata", type=Path, default=Path(paths["atd_meta"]))
    parser.add_argument("--homography", type=Path, default=Path(paths["homography"]))
    parser.add_argument("--tags", type=Path, default=Path(paths["smart_tagging"]))
    parser.add_argument("--out", type=Path, default=Path("data/tracking_dead_zones"))
    parser.add_argument("--public-gt", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--reuse-features", action="store_true", help="Reuse only a cache matching the input file fingerprints")
    args = parser.parse_args()
    meta = metadata(args.metadata)
    if meta["n_frames"] != GAMES[GAME]["n_video_frames"] or meta["fps"] != GAMES[GAME]["fps"]:
        raise ValueError("ATD metadata does not match the tactical match timeline")
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = {k: [str(getattr(args, k).resolve()), getattr(args, k).stat().st_size, getattr(args, k).stat().st_mtime_ns]
                   for k in ("raw", "metadata", "homography")}
    fingerprint["feature_version"] = 1
    cache, manifest = args.out / "features.npz", args.out / "features_manifest.json"
    if args.reuse_features and cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == fingerprint:
        with np.load(cache, allow_pickle=False) as arrays:
            f = pd.DataFrame({key: arrays[key] for key in arrays.files})
    else:
        players, ball = load_tracking(args.raw, meta)
        f = features(players, ball, meta, load_geometry(args.homography, meta["n_frames"]))
        np.savez_compressed(cache, **{key: f[key].to_numpy() for key in f})
        manifest.write_text(json.dumps(fingerprint, indent=2) + "\n")
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    events = load_events(args.tags)
    result = detect(f, events, meta)
    truth = np.asarray(read_labels(args.public_gt, len(f))) if args.public_gt else None
    base = np.asarray(read_labels(args.overlay, len(f))) if args.overlay else None
    print(json.dumps(export(result, f, meta, args, truth, base), indent=2))


if __name__ == "__main__":
    main()
