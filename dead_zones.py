"""Evidence-aware dead-ball candidates from SportsCode Smart Tagging (stdlib only).

Times use the tactical video clock; frames are zero-based, intervals [start, end).
Tags are windows, NOT whistle/ball-contact timestamps. No inferred interval is
frame-verified. See DEAD_ZONES.md for the model's assumptions and integration.
"""

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from games import GAME, GAMES, asset_paths


PLAY_CODES = {
    "PLAYERS IN THE BOX", "BALL IN FINAL THIRD", "BALL IN THE BOX",
    "PROGRESSION", "ATTACKING TRANSITION", "BUILD UP", "LONG BALL",
    "CREATING CHANCES", "DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION",
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN DEFENSIVE THIRD",
}


@dataclass(frozen=True)
class Event:
    id: str
    start: float
    end: float
    code: str
    half: str
    team: str
    kind: str


@dataclass(frozen=True)
class Config:
    fps: float = GAMES[GAME]["fps"]
    n_frames: int = GAMES[GAME]["n_video_frames"]
    guard_s: float = 1.0
    max_goal_span_s: float = 360.0
    max_restart_gap_s: float = 60.0

    def __post_init__(self):
        for name in ("fps", "max_goal_span_s", "max_restart_gap_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.guard_s) or self.guard_s < 0:
            raise ValueError("guard_s must be finite and nonnegative")
        if not isinstance(self.n_frames, int) or self.n_frames <= 0:
            raise ValueError("n_frames must be a positive integer")


def load_events(path):
    """Validate temporal fields, preserve IDs, normalize categorical matching."""
    events, seen = [], set()
    for instance in ET.parse(path).iter("instance"):
        event_id = (instance.findtext("ID") or "").strip()
        if not event_id or event_id in seen:
            raise ValueError("Missing or duplicate instance ID: " + event_id)
        seen.add(event_id)
        start, end = float(instance.findtext("start")), float(instance.findtext("end"))
        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
            raise ValueError("Invalid time window on instance " + event_id)
        tags = {}
        for label in instance.findall("label"):
            key = (label.findtext("group") or "").strip().lower()
            value = (label.findtext("text") or "").strip()
            if key in tags and tags[key] != value:
                raise ValueError("Conflicting label on instance " + event_id + ": " + key)
            tags[key] = value
        events.append(Event(event_id, start, end,
                            (instance.findtext("code") or "").strip().upper(),
                            tags.get("half", "").lower(), tags.get("team", ""),
                            tags.get("type", "").lower()))
    if not events:
        raise ValueError("No SportsCode instances found")
    return sorted(events, key=lambda e: (e.start, e.end, e.id))


def infer(events, config=None):
    """Return auditable pairs; only unambiguous goal cores qualify for 'goals'.

    Pair goals to the first subsequent restart (same half), which must be a
    post-goal kickoff. Internal set-piece clips are allowed inside a goal clip;
    a restart between its end and the proposed kickoff invalidates the pair.
    Multiple goals competing for a kickoff are rejected, never guessed apart.
    """
    c = config or Config()
    goals = [e for e in events if e.code == "GOALS"]
    restarts = sorted((e for e in events if e.code == "SET PIECES"),
                      key=lambda e: (e.start, e.id))
    kicks = [e for e in restarts if e.kind == "kick off (after goal)"]
    phases = [e for e in events if e.code in PLAY_CODES]
    pairs, issues, proposals = [], [], {}

    for goal in goals:
        matches = [k for k in kicks if goal.half and k.half == goal.half
                   and goal.start < k.start and goal.end <= k.end
                   and k.start - goal.start <= c.max_goal_span_s]
        if not matches:
            issues.append({"event_id": goal.id, "reason": "No compatible same-half kickoff"})
            continue
        kick = min(matches, key=lambda e: e.start)
        intervening = [r for r in restarts if r.half == goal.half
                       and goal.end <= r.start < kick.start]
        if intervening:
            issues.append({"event_id": goal.id, "reason": "Intervening restart before kickoff"})
            continue
        proposals.setdefault(kick.id, []).append(goal)

    def candidate(rule, left, right, start, end, confidence, evidence, reason):
        # Round to the repository's video-frame convention, then clip to the file.
        sf = max(0, min(c.n_frames, round(start * c.fps)))
        ef = max(0, min(c.n_frames, round(end * c.fps)))
        if ef <= sf:
            return None
        row = {
            "id": "pair-{:03d}".format(len(pairs) + 1), "rule": rule,
            "half": right.half, "stop_event_id": left.id,
            "restart_event_id": right.id, "restart_type": right.kind,
            "team": left.team, "start_frame": sf, "end_frame_exclusive": ef,
            "start_s": sf / c.fps, "end_s": ef / c.fps,
            "duration_s": (ef - sf) / c.fps,
            "pair_confidence": "high" if rule == "goal_to_kickoff" else "low",
            "boundary_confidence": confidence, "frame_verified": False,
            "evidence_ids": sorted(set(evidence)), "reason": reason,
            "stop_window_s": [left.start, left.end],
            "restart_window_s": [right.start, right.end],
        }
        pairs.append(row)
        return row

    for kick in kicks:
        matched = proposals.get(kick.id, [])
        if len(matched) != 1:
            issues.append({"event_id": kick.id, "reason":
                           "Ambiguous goals for kickoff" if matched else "Unmatched post-goal kickoff"})
            continue
        goal = matched[0]
        context = [e for e in phases + restarts if e.half == goal.half
                   and e.id != kick.id and e.end > goal.start and e.start < kick.start]
        play = [e for e in context if e.code in PLAY_CODES]
        if not play:
            issues.append({"event_id": goal.id, "reason": "No play evidence to trim goal buildup"})
            continue
        # Trim the entire goal buildup AND all internal restart/phase windows.
        # This is a conservative heuristic, not proof that untagged play is absent.
        last = max(context, key=lambda e: e.end)
        start = max(goal.start, last.end) + c.guard_s
        end = kick.start - c.guard_s
        if candidate("goal_to_kickoff", goal, kick, start, end, "medium",
                     [goal.id, kick.id] + [e.id for e in context],
                     "Goal and post-goal kickoff agree; use only the gap after all "
                     "overlapping play/set-piece clips, trimmed by the guard. "
                     "The actual goal frame and restart frame remain unknown.") is None:
            issues.append({"event_id": goal.id, "reason": "No positive core after context/guard trimming"})

    # A phase ending does not establish ball-out or a whistle. Such pairs are
    # useful review locations, but are excluded from the default D mask.
    for restart in restarts:
        if restart.kind.startswith("kick off"):
            continue
        previous = [p for p in phases if restart.half and p.half == restart.half
                    and p.end <= restart.start]
        if not previous:
            continue
        left = max(previous, key=lambda e: (e.end, e.start))
        if restart.start - left.end > c.max_restart_gap_s:
            continue
        blockers = [p for p in phases if p.half == restart.half
                    and p.start < restart.start and p.end > left.end]
        intervening = [r for r in restarts if r.half == restart.half
                       and left.end <= r.start < restart.start]
        if blockers or intervening:
            continue
        candidate("phase_to_restart", left, restart, left.end + c.guard_s,
                  restart.start - c.guard_s, "low", [left.id, restart.id],
                  "Gap before a restart, but the phase end is not a stop event. "
                  "Untagged live play may occupy this whole gap. Review only.")

    linked = {p["restart_event_id"] for p in pairs}
    review = [{"event_id": r.id, "half": r.half, "type": r.kind,
               "start_s": r.start, "end_s": r.end,
               "paired": r.id in linked,
               "reason": "Start-of-half window; no preceding stop supplied"
               if r.kind == "kick off (start)" else
               "See linked pair; restart instant remains uncertain" if r.id in linked else
               "Restart window only; no defensible preceding gap found"}
              for r in restarts]
    return {"config": asdict(c), "event_count": len(events),
            "code_counts": dict(Counter(e.code for e in events)),
            "restart_counts": dict(Counter(e.kind for e in restarts)),
            "pairs": pairs, "restart_review": review, "issues": issues}


def dead_mask(result, policy="goals"):
    """Boolean mask on VIDEO frames. False means unknown, never 'ball alive'.

    goals: medium-confidence goal cores; exploratory: includes weak phase gaps;
    strict: only frame-verified intervals (none can be supplied by XML alone).
    """
    if policy not in {"strict", "goals", "exploratory"}:
        raise ValueError("Unknown policy: " + policy)
    mask = [False] * result["config"]["n_frames"]
    for pair in result["pairs"]:
        selected = (pair["frame_verified"] if policy == "strict" else
                    pair["rule"] == "goal_to_kickoff" if policy == "goals" else True)
        if selected:
            lo, hi = pair["start_frame"], pair["end_frame_exclusive"]
            mask[lo:hi] = [True] * (hi - lo)
    return mask


def overlay_labels(labels, result, policy="goals"):
    """Copy H/A/D/X predictions and overwrite selected candidates with D."""
    mask = dead_mask(result, policy)
    if len(labels) != len(mask) or any(x not in {"H", "A", "D", "X"} for x in labels):
        raise ValueError("Expected one H/A/D/X label per video frame")
    return ["D" if selected else label for selected, label in zip(mask, labels)]


def read_labels(path, n_frames):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != n_frames or any(int(r["frame"]) != i for i, r in enumerate(rows)):
        raise ValueError("Labels must cover consecutive video frames 0..n_frames-1")
    labels = [r["label"] for r in rows]
    if any(label not in {"H", "A", "D", "X"} for label in labels):
        raise ValueError("Invalid label; expected H/A/D/X")
    return labels


def evaluate_public(result, labels):
    """Diagnostic only. Never called by inference; X truth is excluded."""
    if len(labels) != result["config"]["n_frames"]:
        raise ValueError("Ground truth length differs from video length")
    report = {}
    for policy in ("goals", "exploratory"):
        mask = dead_mask(result, policy)
        tp = sum(pred and gt == "D" for pred, gt in zip(mask, labels))
        fp = sum(pred and gt in {"H", "A"} for pred, gt in zip(mask, labels))
        n_dead = labels.count("D")
        report[policy] = {"graded_predicted_dead": tp + fp, "true_dead_frames": tp,
                          "false_dead_frames": fp, "public_dead_frames": n_dead,
                          "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / n_dead if n_dead else None,
                          "note": "No candidate coverage; accuracy is unvalidated"
                          if not tp + fp else "Public-window diagnostic only; not full-match accuracy"}
    return report


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def write_report(result, out):
    c = result["config"]
    goals = [p for p in result["pairs"] if p["rule"] == "goal_to_kickoff"]
    weak = [p for p in result["pairs"] if p["rule"] == "phase_to_restart"]
    seconds = sum(dead_mask(result)) / c["fps"]
    lines = ["# Smart-tagging dead-zone analysis", "",
             "Source: `{}`".format(result["source"]), "",
             "{} events; {} restart windows; {} goal→kickoff cores; {} weak phase→restart gaps.".format(
                 result["event_count"], len(result["restart_review"]), len(goals), len(weak)), "",
             "Default goal-core mask: **{:.2f} seconds / {} frames**. All other frames are unknown, not live.".format(
                 seconds, sum(dead_mask(result))), "",
             "**These are heuristic candidates, not verified dead frames.** Semantic goal/kickoff pairing has "
             "high confidence; timing has medium confidence. No numerical probability is claimed. "
             "The strict policy marks zero frames because XML alone cannot verify exact boundaries.", "",
             "## Goal pairs", "",
             "All times below are seconds from the tactical video's start, not match-clock times. "
             "Frame intervals include the start and exclude the end.", "",
             "| Goal → kickoff IDs | Half | Candidate seconds | Video frames [start, end) | Duration |",
             "|---|---|---|---|---|"]
    for p in goals:
        lines.append("| {} → {} | {} | {:.2f}–{:.2f} | {}–{} | {:.2f}s |".format(
            p["stop_event_id"], p["restart_event_id"], p["half"], p["start_s"], p["end_s"],
            p["start_frame"], p["end_frame_exclusive"], p["duration_s"]))
    lines += ["", "## Why these boundaries", "",
              "A GOALS clip can include the buildup and celebration; its start is not the scoring instant. "
              "For each unique same-half goal/kickoff pair, start after the latest overlapping tactical "
              "phase or internal set-piece window ends, plus a {:.2f}s guard. End {:.2f}s before the kickoff "
              "window starts. This deliberately discards uncertain edges, but cannot rule out missing or "
              "incorrect tags. Guard sizes and maximum pairing gaps are explicit heuristics, not calibrated values.".format(
                  c["guard_s"], c["guard_s"]), "",
              "Ordinary restarts have no explicit preceding foul, offside, or ball-out event in this file. "
              "Phase→restart gaps are low-confidence review candidates only; the default mask excludes them. "
              "Never mark a whole 35-second set-piece window dead, or all time between consecutive restarts dead. "
              "Attack/defence tags can describe the same play and do not provide independent corroboration.", "",
              "## Public-window diagnostic", ""]
    if "public_evaluation" in result:
        for policy, stats in result["public_evaluation"].items():
            lines.append("- **{}**: {} correctly dead, {} falsely dead, {} known dead frames; "
                         "precision {}, recall {}. {}.".format(
                             policy, stats["true_dead_frames"], stats["false_dead_frames"],
                             stats["public_dead_frames"], stats["precision"], stats["recall"], stats["note"]))
    else:
        lines.append("Not evaluated; pass --public-gt to check the supplied public window.")
    lines += ["", "## Files", "",
              "- `analysis.json`: all evidence, parameters, restart windows, and unmatched/ambiguous events.",
              "- `intervals.csv`: candidate pairs with confidence and source IDs.",
              "- `restart_review.csv`: every restart, including those with insufficient evidence.",
              "- `dead_frames.csv`: selected policy's D/X prior; X means no conclusion.",
              "- `overlaid.csv` (when --overlay is supplied): existing labels with selected candidates set to D.",
              "", "Selected export policy: **{}**.".format(result["policy"]), ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", nargs="?", default=asset_paths()["smart_tagging"])
    parser.add_argument("--out", type=Path, default=Path("data/dead_zones"))
    parser.add_argument("--policy", choices=("strict", "goals", "exploratory"), default="goals")
    parser.add_argument("--guard-seconds", type=float, default=1.0)
    parser.add_argument("--max-goal-span", type=float, default=360.0)
    parser.add_argument("--max-restart-gap", type=float, default=60.0)
    parser.add_argument("--overlay", type=Path, help="Existing video-frame H/A/D/X CSV")
    parser.add_argument("--public-gt", type=Path, help="Evaluation only; never used in inference")
    args = parser.parse_args()
    config = Config(guard_s=args.guard_seconds, max_goal_span_s=args.max_goal_span,
                    max_restart_gap_s=args.max_restart_gap)
    result = infer(load_events(args.xml), config)
    result.update(source=str(Path(args.xml).resolve()), policy=args.policy)
    # Validate optional inputs before creating any output.
    overlaid = overlay_labels(read_labels(args.overlay, config.n_frames), result, args.policy) if args.overlay else None
    if args.public_gt:
        result["public_evaluation"] = evaluate_public(result, read_labels(args.public_gt, config.n_frames))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = list(result["pairs"][0]) if result["pairs"] else [
        "id", "rule", "start_frame", "end_frame_exclusive", "boundary_confidence"]
    write_csv(args.out / "intervals.csv", result["pairs"], fields)
    write_csv(args.out / "restart_review.csv", result["restart_review"],
              ["event_id", "half", "type", "start_s", "end_s", "paired", "reason"])
    mask = dead_mask(result, args.policy)
    write_csv(args.out / "dead_frames.csv",
              ({"frame": i, "label": "D" if yes else "X"} for i, yes in enumerate(mask)),
              ["frame", "label"])
    if overlaid is not None:
        write_csv(args.out / "overlaid.csv",
                  ({"frame": i, "label": label} for i, label in enumerate(overlaid)), ["frame", "label"])
    write_report(result, args.out)
    print("{} pairs; {} selected candidate frames ({:.2f}s). Report: {}".format(
        len(result["pairs"]), sum(mask), sum(mask) / config.fps, args.out / "report.md"))


if __name__ == "__main__":
    main()
