# Smart-tagging dead-zone analysis

Source: `/Users/danieleitan/Desktop/more-than-a-hack-2027/data/tactical/BAR-ATH_tactical_smart_tagging.xml`

638 events; 59 restart windows; 4 goal→kickoff cores; 15 weak phase→restart gaps.

Default goal-core mask: **463.32 seconds / 11583 frames**. All other frames are unknown, not live.

**These are heuristic candidates, not verified dead frames.** Semantic goal/kickoff pairing has high confidence; timing has medium confidence. No numerical probability is claimed. The strict policy marks zero frames because XML alone cannot verify exact boundaries.

## Goal pairs

All times below are seconds from the tactical video's start, not match-clock times. Frame intervals include the start and exclude the end.

| Goal → kickoff IDs | Half | Candidate seconds | Video frames [start, end) | Duration |
|---|---|---|---|---|
| 269 → 2 | 1st half | 1889.80–1984.24 | 47245–49606 | 94.44s |
| 270 → 3 | 1st half | 2330.72–2479.48 | 58268–61987 | 148.76s |
| 271 → 4 | 1st half | 2980.56–3017.28 | 74514–75432 | 36.72s |
| 272 → 6 | 2nd half | 6335.60–6519.00 | 158390–162975 | 183.40s |

## Why these boundaries

A GOALS clip can include the buildup and celebration; its start is not the scoring instant. For each unique same-half goal/kickoff pair, start after the latest overlapping tactical phase or internal set-piece window ends, plus a 1.00s guard. End 1.00s before the kickoff window starts. This deliberately discards uncertain edges, but cannot rule out missing or incorrect tags. Guard sizes and maximum pairing gaps are explicit heuristics, not calibrated values.

Ordinary restarts have no explicit preceding foul, offside, or ball-out event in this file. Phase→restart gaps are low-confidence review candidates only; the default mask excludes them. Never mark a whole 35-second set-piece window dead, or all time between consecutive restarts dead. Attack/defence tags can describe the same play and do not provide independent corroboration.

## Public-window diagnostic

- **goals**: 0 correctly dead, 0 falsely dead, 1469 known dead frames; precision None, recall 0.0. No candidate coverage; accuracy is unvalidated.
- **exploratory**: 0 correctly dead, 0 falsely dead, 1469 known dead frames; precision None, recall 0.0. No candidate coverage; accuracy is unvalidated.

## Files

- `analysis.json`: all evidence, parameters, restart windows, and unmatched/ambiguous events.
- `intervals.csv`: candidate pairs with confidence and source IDs.
- `restart_review.csv`: every restart, including those with insufficient evidence.
- `dead_frames.csv`: selected policy's D/X prior; X means no conclusion.
- `overlaid.csv` (when --overlay is supplied): existing labels with selected candidates set to D.

Selected export policy: **goals**.
