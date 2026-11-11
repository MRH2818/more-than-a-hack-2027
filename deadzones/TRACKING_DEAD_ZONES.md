# Extending dead zones with tracking

Run from the repository root with NumPy and pandas installed:

```sh
python -m deadzones.tracking_dead_zones \
  --public-gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --overlay evaluation/results/baseline_smooth125.csv
```

The supplied files have been copied into the ignored `data/tactical/` directory.
Use `--raw`, `--metadata`, `--homography`, and `--tags` to choose different input
paths. Results are in `data/tracking_dead_zones/`; begin with `report.md`.
Use `--reuse-features` on subsequent runs to reuse the features if file paths,
sizes, modification times, and feature version still match the cache manifest.

The temporary environment used for this analysis is
`/private/tmp/mtah-tracking-env/bin/python`. For a persistent environment, install
the repository's `requirements.txt` (NumPy and pandas).

## What makes a new dead-frame candidate

This adds evidence rules to the existing tag-only detector. It is offline and
rule-based, without a fitted model or calibrated probability. Public reference
labels are read only after inference for diagnostics.

1. **Stationary setup followed by a launch.** Within a restart clip or the
   preceding 30 seconds, the ball stays within a one-metre cluster for at least
   one second, at no more than 0.7 m/s. At least 18 reliable player tracks are
   visible, with at least seven from each team. Median player speed is at most
   1.2 m/s, and the 90th percentile at most 2.5 m/s. The ball must be close to a
   player and in a location appropriate to the tag: centre for kickoff, corner
   for corner kick, sideline for throw-in, goal-area region for goal kick, or
   inside the pitch for a free kick. Within two seconds after this setup, require
   a sustained moving ball displaced at least 1.5m and increased player motion.
   The setup interior becomes a candidate; the uncertain time between setup
   and detected launch is left unmarked.
2. **Ball outside near a restart.** The observed ball is at least one metre
   beyond a pitch boundary for at least 0.6 seconds, with adequate player
   visibility and valid low-error geometry. Trim 0.2 seconds off both ends.
   This is tracking evidence for a stoppage, not a referee observation. A false
   ball track or projection of an airborne ball can still be misleading.
3. **Kickoff formation followed by departure.** The ball is often invisible
   during kickoff preparation. A special rule permits this when each team
   occupies its own half (at least 90% of visible tracks), median speed is at
   most 0.5 m/s, and the 90th percentile is at most 1.5 m/s. A subsequent moving
   ball within 15m of the centre, together with increased player movement,
   must corroborate the restart within two seconds. A conflicting visible
   ball breaks the waiting run. This rule does not extend to ordinary restarts.

All rules trim uncertain edges and require persistent observations. Thresholds
are explicit in `Params`, selected as physical heuristics rather than optimized
against the public window. Sensitivity to those values is not calibrated.

Players moving **alone** cannot establish that play restarted: they can move
into position during stoppages. Player stillness alone also cannot establish a
dead ball. It stays in the review output. A moving ball inside the pitch plus
player activity releases the D override; possession then comes from the existing
tracker. This movement signal does not establish H/A ownership, and an input
prediction that already says D is not automatically converted to H or A.

## Data quality and alignment

- ATD frame IDs are checked one by one and mapped with `video_frame = id - 1`.
  Homography indices must exactly cover the same zero-based frames. Tag seconds
  use `round(seconds * fps)`; no halftime offset is added.
- The metadata declares normalized coordinates on a 105×68m pitch and speeds
  in km/h. Coordinates are scaled to metres and speeds divided by 3.6.
- Player motion uses the larger of reported speed and speed measured from
  positions over a centered 0.4-second window. This prevents a falsely low
  reported speed from making a moving track look idle. Track switches with
  implausible speeds are excluded from the reliable player count.
- More than 24 or fewer than 18 reliable players, an underrepresented team,
  an invalid/singular homography, or reprojection error above 2 reject evidence.
- Ball jumps above 45 m/s are rejected. Ordinary rules never bridge missing
  ball observations. The kickoff-formation rule is the explicit exception
  described above, requiring corroboration after the occlusion.
- ATD is already in pitch coordinates. Homography supplies a quality check,
  not another transform of those positions. It is not independent ground truth;
  the supplied data can share upstream errors.
- Centered velocities and later launch observations require future frames.
  This implementation is for offline analysis, not real-time operation.

## Outputs and confidence

`new_frames.csv` contains only the new D/X candidates beyond the old goal cores.
`tracking_supported.csv` contains all candidates with tracking support.
`combined.csv` merges them with the old goal candidates and removes positive
movement conflicts. Existing goal candidates retain their original uncertainty;
they are not upgraded merely because tracking was loaded.

`frame_evidence.csv.gz` contains every frame's measurements and decision. Statuses
distinguish tracking-supported dead candidates, inherited goal candidates,
unresolved stationary setups, idle-player-only evidence, and movement releasing
the override. `intervals.csv` links evidence to tag IDs and observed departure
frames. Intervals have exclusive ends. Multiple overlapping tags can refer to
the same stoppage; masks count each frame once and do not count overlapping tags
as independent corroboration.

No new interval is marked `frame_verified=True`. An automatic ball track outside
the pitch and players standing still are useful evidence, not absolute certainty.
The small public evaluation should not be interpreted as full-match precision.

## Checks

```sh
python -m unittest deadzones.test_tracking_dead_zones -v
python evaluation/validate_submission.py data/tracking_dead_zones/combined.csv
python evaluation/validate_submission.py data/tracking_dead_zones/overlaid.csv
```

The tests cover missing balls, false stillness, kickoff formation and departure,
motion alone, bad geometry, sparse tracking, outlier jumps, boundary guards,
repeated tag coverage, unit conversion, frame alignment, and revocation of an
old goal candidate when positive movement evidence conflicts with it.
