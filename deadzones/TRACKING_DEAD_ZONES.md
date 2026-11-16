# Stoppage detection from tracking

Run from the repository root with NumPy and pandas installed:

```sh
python -m deadzones.tracking_dead_zones \
  --out data/restart_detection_v3 \
  --overlay evaluation/results/baseline_smooth125.csv
```

The default D policy is **`fragments`**. The stricter `--overlay-policy episodes`
is **experimental and opt-in** until independent evaluation supports it. Both
masks are exported on every run. Restart detection and local team assignment
operate with either D policy; selecting a restart does not opt into episode masks.

The default `--restart-team-policy local` adds corroborated H/A assignments only
after a restart's uncertainty range, for at most one second. Use
`--restart-team-policy off` to disable these assignments. `overlaid_dead_only.csv`
also provides the D-only result for direct comparison. All other input labels
are preserved; a selected D override takes precedence over H/A proposals.
Start from the original possession CSV when switching policies, since overlaying
an already modified CSV cannot undo its existing D labels.

Use `--public-gt evaluation/gt/BAR-ATH_possession_gt_public.csv` for a separate
pipeline diagnostic. Labels are read only after inference. Do not use them to
choose rules or tune thresholds. Use `--reuse-features` for a cache matching the
input fingerprints and feature version; v3 recomputes older caches to include
separate reliable-player distances for each team.
Raw input overrides remain `--raw`, `--metadata`, `--homography`, and `--tags`.

## Why bounded intervals matter

The previous detector exported visible outside-ball and setup fragments. A short
D patch can divide a predicted H/A run into two possessions even when every
patched frame is correct. Dead-frame precision alone does not show this cost.
`stoppage_episodes.py` now reconstructs continuous intervals with evidence for
both waiting and restart. These are still heuristic intervals, not proof of the
actual whistle or the entire stoppage. Incomplete evidence stays available for
inspection in experimental episode mode. The default fragment policy retains
the original selection behavior while this tradeoff is evaluated independently.

## Restart uncertainty and local team assignment

`restart_detection.py` uses the observed waiting/departure transitions from
`stoppage_episodes.py`, independently of which D mask is selected.

- **Bounds:** `restart_earliest_frame` and `restart_latest_frame` are inclusive.
  The range runs from waiting end minus 0.2s to observed motion onset plus 0.2s,
  clipped to the event search scope. The margin accounts for centered motion
  features; these are heuristic observation bounds, not guaranteed contact times
  or calibrated confidence intervals. The raw waiting end and motion onset are
  also exported.
- **Confirmation:** `confirmation_frame` is the last observation needed for
  sustained departure and displacement. `confirmation_available_frame` includes
  the centered-feature lookahead. Local assignments are offline and may use later
  observations to support an earlier frame; this is not streaming inference.
- **Timing limits:** no H/A proposal is made within any detected restart's
  uncertainty range. Brackets wider than one second abstain. The assignment window
  starts after the latest possible restart and ends within one second, the tagged
  event's end, or the next restart range, whichever comes first.
- **Team evidence:** per-team distances use reliable player tracks. A vote needs
  a player within 2.5m of the ball and at least a 1m advantage over the other team.
  Evidence must include a continuous 0.2s run and an 80% share of eligible votes.
  The one-second pre-restart window excludes uncertain frames and rejects stale
  evidence; the post-restart window includes only valid in-pitch observations.
- **Tags:** club names map to H/A by exact normalized metadata names, using the
  repository's first-team/home convention. A known tag requires agreeing tracking.
  An unknown tag requires post-restart tracking; it cannot carry a pre-restart
  guess through an unseen flight. Conflicting tag/track evidence, contested ball
  proximity, or conflicting overlapping tags cause abstention.
- **Handoff:** assignments stop at the first missing ball, quality failure,
  ball-out observation, contested/opposing proximity, new restart range or selected
  D override. They do not resume after that interruption. Overlapping proposals
  that disagree on H/A abstain rather than depend on iteration order.

No baseline possession labels or reference labels enter restart/team inference.
The outputs preserve unmodified bulk live play. These rules remain physical
heuristics and have not been independently validated as a spellF1 improvement.

## Evidence and boundaries

1. A stop anchor is a persistent outside-ball observation on the boundary
   appropriate to the restart type, a stationary setup with calm players, or
   kickoff formation. Sideline exits cannot seed goal kicks. Ordinary player
   stillness and missing ball observations alone cannot seed a stop.
2. Waiting is supported by compatible outside-ball observations or a stationary
   ball near a reliable player in the appropriate restart area. Up to two seconds
   of missing ball may join compatible observations, with good geometry and
   player coverage throughout. Visible in-pitch movement and quality failures
   break the interval. No unbracketed missing-ball tail is filled. Kickoff
   formation retains its explicit occlusion exception.
3. Restart confirmation requires at least 0.4 seconds of observed in-pitch ball
   motion and at least 0.5m of actual coordinate displacement. Reported speed
   alone is insufficient. The restart must occur within the tagged clip, no more
   than two seconds after waiting evidence ends.
4. Throw-ins require an inward trajectory from the same touchline, within 15m
   of its observed origin. Goal kicks, corners and free kicks require a recent
   stationary setup at an appropriate location and subsequent displacement of
   at least 1.5m. Kickoffs require departure near the centre. A restart need not
   make the whole team accelerate. These gates reduce, but cannot eliminate,
   confusion between ball retrieval and a real restart.
5. An outside-ball stop can be refined backwards to a contiguous observed line
   crossing, within two seconds, without extrapolating through missing tracks.
   The 0.2-second start guard remains. In experimental episode mode the D interval ends at the earlier of the
   last waiting boundary and guarded motion onset. The uncertain gap between
   waiting and motion is reported, not automatically labeled dead.
6. Original tag-derived goal cores remain a separate, weaker prior. Sustained
   motion can release a core. Once released, a later occlusion does not turn the
   old core back on. A separately detected new stoppage can still be selected.

Thresholds are physical heuristics, not calibrated probabilities. Automatic
projection of an airborne ball may look out of bounds; a moving retrieved ball
may resemble a restart. Withholding unresolved fragments trades coverage for
stronger endpoint evidence. It is not a demonstrated full-match score gain.

## Tracking quality

ATD row IDs must be contiguous and map to `video_frame = id - 1`. Homography
frames cover the same zero-based timeline. XML times use `round(seconds * fps)`.
ATD metadata supplies normalized-coordinate scaling and km/h-to-m/s conversion.
Homography is a quality gate; ATD positions are already in pitch coordinates.

Player motion uses the larger of supplied speed and speed measured from positions
over a centered 0.4-second window. Implausible tracks are excluded from player
counts **and nearest-player distances**, so a teleporting ghost cannot anchor a
restart. Gates require 18–24 reliable players, at least seven per team, valid
nonsingular geometry, and reprojection error at most two. Ball jumps above 45m/s
are rejected, and reconstructed departures also check reappearance displacement.
Centered motion and future confirmation make this an offline detector.

## Outputs

- `combined.csv`: selected D/X policy, fragments by default.
- `episode_combined.csv`: experimental goal-plus-episode D/X mask.
- `episode_mask.csv`: reconstructed tracking intervals without the goal prior.
- `episodes.csv`: event IDs, stop evidence, waiting boundary, motion onset,
  restart uncertainty, and half-open selected intervals.
- `episode_review.csv`: stop anchors without a corroborated restart endpoint.
- `fragment_combined.csv`: previous fragment policy with current tracking features.
- `tracking_supported.csv`: visible fragment evidence, retained as diagnostics.
- `new_frames.csv`: selected policy's D frames beyond the original goal cores.
- `restart_detections.csv`: inclusive timing bounds, confirmation timing, tag team,
  source evidence and event scope for every corroborated restart hypothesis.
- `restart_assignments.csv`: team evidence, selected local span, source, abstention
  reason and selected/applied/changed frame counts. Per-row counts can overlap;
  summary counts are unions. Applied counts are blank without an input overlay.
- `restart_team.csv`: H/A/X proposals; X means leave the base unchanged. This is
  an inference diagnostic, not a standalone possession submission.
- `restart_review.csv`: fragment and episode coverage for each restart tag.
- `frame_evidence.csv.gz`: features, statuses, episode and goal masks on every frame.
- `overlaid_dead_only.csv`: input possession labels with the selected D override.
- `overlaid.csv`: D-only result plus enabled, unblocked local H/A assignments.
- `summary.json`, `report.md`: counts, parameters, provenance and optional metrics.

Overlapping tags may support the same interval. Counts use the union of frames;
tag count is not a count of independent stoppages. Existing `intervals.csv` and
`public_evaluation.by_rule` describe fragment evidence, not the selected episode
mask. No inferred interval is marked frame-verified.

## Checks

```sh
python -m unittest deadzones.test_dead_zones deadzones.test_tracking_dead_zones deadzones.test_stoppage_episodes deadzones.test_restart_detection -v
python evaluation/validate_submission.py data/restart_detection_v3/combined.csv
python evaluation/validate_submission.py data/restart_detection_v3/overlaid.csv
```

Synthetic tests cover short and long occlusions, quality gaps, visible live-motion
conflicts, false speed channels, wrong restart locations, teleporting reappearances,
observed line crossings, half boundaries, duplicate tags, stationary teammates,
and goal overrides after a restart. The original detector tests remain in place.
Full discovery still encounters the pre-existing missing `audit_dead_zones.py`
implementation; these commands explicitly run the maintained detector suites.
