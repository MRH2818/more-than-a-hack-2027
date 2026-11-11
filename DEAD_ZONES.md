# Dead zones from Smart Tagging

`dead_zones.py` turns event pairs into auditable dead-ball **candidates** on the
tactical video timeline. It runs with Python's standard library and uses the
match's 25 fps / 199,319 frame constants from `games.py`.

## Run it

The supplied XML has been copied into `data/tactical/`. To reproduce the analysis:

```sh
python3 dead_zones.py --public-gt evaluation/gt/BAR-ATH_possession_gt_public.csv
```

Or specify an XML path and output folder:

```sh
python3 dead_zones.py /path/to/smart_tagging.xml --out data/dead_zones
```

Read `data/dead_zones/report.md` first. `intervals.csv` gives the pairs, source IDs,
timing, confidence, and reasons. `analysis.json` includes all restart windows and
pairing failures; `restart_review.csv` lists all 59 restarts. `dead_frames.csv`
contains one row per video frame, labeled `D` for selected candidates and `X`
for **unknown**. It is a partial prior, not a complete possession prediction.

## What the data supports

The delivered file contains 638 tags, including four goals, four post-goal
kickoffs, two half-start kickoffs, and 53 other restarts. All four goal clips end
one second after their matching kickoff window starts. Goals span 70.88–250.72
seconds, ordinary set-piece clips are 35 seconds, and kickoff clips are five
seconds. None of these windows is an exact ball-contact timestamp.

The detector pairs a goal with a same-half post-goal kickoff. It rejects reused
kickoffs, missing half labels, excessive pairing spans, and intervening restarts
after the goal clip. The candidate starts after all overlapping play and internal
set-piece clips have ended, plus a one-second guard. It ends one second before
the kickoff window starts. This avoids using the goal's live buildup or the
restart window as dead time. The guard is configurable; it is not a measured
error bound.

On the supplied file, the four trimmed goal cores total **463.32 seconds
(11,583 frames)**. There are also 15 weaker phase-end → restart-window gaps.
The XML has no explicit foul/whistle/ball-out/half-end events, so it cannot
establish the stop boundary for ordinary restarts. The weak gaps may contain
untagged live play and are excluded from the default mask. Tactical attack and
defence tags often duplicate the same phase and do not count as independent votes.

Pair confidence and boundary confidence are separate qualitative assessments,
not calibrated probabilities. Goal pairing is high-confidence, inferred goal
boundaries medium-confidence, and phase gaps low-confidence. **No candidate is
frame-verified.** All unmarked frames remain unknown; missing tags do not imply
dead play. Half-start windows are retained for review, without guessing the
previous half's final whistle or labeling halftime as play.

## Apply it to possession predictions

```sh
python3 dead_zones.py \
  --overlay evaluation/results/baseline_smooth125.csv \
  --public-gt evaluation/gt/BAR-ATH_possession_gt_public.csv
python3 evaluation/validate_submission.py data/dead_zones/overlaid.csv
```

This writes a new `overlaid.csv`, preserving the input. Only the selected
candidate frames become `D`; all other H/A/D/X labels are retained.

Policies:

- `--policy goals` (default): inferred goal cores only. Review before treating
  these as reliable labels in a final submission.
- `--policy strict`: frame-verified intervals only; XML alone supplies none,
  so this policy abstains everywhere and leaves overlays unchanged.
- `--policy exploratory`: includes the weak phase gaps; intended for inspection
  and experiments, not a claim of greater accuracy.

`--guard-seconds`, `--max-goal-span` (360 seconds), and `--max-restart-gap`
(60 seconds) expose the heuristics. Do not tune them to the public five minutes.

For direct use in the existing tracking pipeline:

```python
from dead_zones import load_events, infer, dead_mask, overlay_labels

analysis = infer(load_events("data/tactical/BAR-ATH_tactical_smart_tagging.xml"))
mask = dead_mask(analysis)  # list[bool], indexed by zero-based tactical video frame
predictions = overlay_labels(predictions, analysis)  # full-length H/A/D/X sequence
```

Do not index the mask with a trimmed array's row number or a tracking identifier.
ATD rows use `video_frame = frame_id - 1` per `DATA.md`; other tracking ID formats
must first be mapped to tactical video frames. All exported intervals are
`[start_frame, end_frame_exclusive)` and XML seconds convert with `round(s * 25)`.

## Validation and next source of accuracy

```sh
python3 -m unittest discover -s tests -v
```

Tests cover uncertain boundaries, ambiguous matches, missing metadata,
cross-half pairing, intervening restarts, overlap, frame indexing, union of
masks, input validation, and preservation of existing predictions.

The optional public-label diagnostic is separate from inference and ignores
ground-truth `X` frames. None of the current candidates overlaps the public
five-minute labeled window. Therefore precision is undefined and dead-frame
recall there is zero; this does **not** validate full-match accuracy. The public
window contains 1,469 known dead frames that these candidates do not recover.

To recover ordinary stoppages and refine goal edges, inspect synchronized
tactical video or add tracking evidence for ball-out/whistle and actual restart
contact. A missing ball track by itself is not a stoppage. The exported event IDs,
review windows, and candidate frames identify where to investigate.

Raw data and generated results stay in the repository's existing ignored `data/`
directory. Existing possession code is unchanged.
