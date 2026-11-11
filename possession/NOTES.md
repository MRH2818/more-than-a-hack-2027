# Decision log (aggregates only, no ground truth used)

## 2026-11-07 — cleandead adopted; restarts, ballfree, zone, gapdead rejected
Masked-ball test (9% of graded frames hidden, seed 0), base vs candidates:

| variant | dead_segments | D share | possessions | phase agree | set piece near D | hidden team agree |
|---|---|---|---|---|---|---|
| base | 204 | 47.0 | 276 | 86.1 | 96.6 | 68.9 |
| cleandead | 168 | 42.3 | 249 | 86.4 | 94.8 | 63.2 |
| clean+gap | 170 | 44.3 | 249 | 85.9 | 94.8 | 38.1 |

Pre-set criteria: fewer stoppages, D share toward 35-45%, phase agreement >= 86, set-piece
alignment ~96, hidden team agreement not worse. cleandead met all but the last. That one is
confounded: cleandead changes no cue on ball-missing frames (both changes act only where the
ball is seen), so on hidden frames the masked run behaves as before; what changes is the
real-ball reference, which now has fewer spurious D frames to agree with.
Earlier round: zone (2 m) clearly worse (366 possessions, phase agree 79.6); restarts and
ballfree no measurable gain; gapdead redundant (D already high) and penalised by the test.

## 2026-11-09 — kickoff derived from tags; lineaware and tagrestarts rejected
- FORCED_DEAD frame constant removed. Pre-kickoff dead stretch now comes from the Smart Tagging
  "Kick Off (Start)" tag (automatic), moved forward to the first ball movement (Law 8).
  Tag start alone gave 2852 pre-kickoff frames in the second half vs 2934 published in FAQ.md.
- Masked test vs base (cleandead): lineaware D share +0.3, dead segments 168 -> 168, set piece
  near D unchanged (criterion was: up) -> rejected. tagrestarts phase agree 86.4 -> 86.5,
  possessions +2 -> effect too small to call, kept off.

## 2026-11-10 — touches, softcontrol and all combinations rejected
Masked test vs base (cleandead + derived kickoff). Main check, transition tags within 2 s
(24 graded tags): base 41.7, touches 29.2, soft 37.5, every combination 25.0-29.2.
Median offset: base 2.68 s, all others ~3.3 s. Phase agreement drops (touches 85.3,
soft 83.7, touch+soft 82.2); possessions rise to 271-311. Detected touches 658 home / 364 away.
Caveat: 24 tags means one tag = 4.2 points, and tag timing may itself lag real turnovers, but
phase agreement fell too, so the result points the same way. Nothing adopted.
Kickoff: the second-half pre-kickoff is still 2852 (no ball movement >= 3 m/s found within
10 s of the tag); left as is, an 82-frame difference at one boundary.

## 2026-11-10 — referee whistle from audio rejected
The tactical mp4 has a stereo AAC track (7972.69 s, offset 0.000 s vs video). Detector: tonal
peak in 2-5 kHz, robust z >= 6, 0.16-5 s, steady pitch. Found 4 events in graded play and none
within 5 s of the 6 kickoff tags. Diagnostic: max tonality z within 5 s of each kickoff beats
2/59/71/73/91/92% of random 10 s windows (bar set beforehand: ~95% each). Whistles are not
distinguishable from background in this recording, so a lower threshold would only catch crowd
noise. Code kept (USE_WHISTLE = False); decoder gained tested per-frame switch bonuses.

## 2026-11-10 — decoder tuned on other games with real answers (IDSSE Bundesliga)
Data: 7 open matches (Bassek et al. 2025, Scientific Data, CC-BY 4.0; Hugging Face mirror
pysport/idsse-data). Answers from the provider's per-frame BallStatus / BallPossession.
Tracking degraded to look like BAR-ATH: correlated ball noise 0.25 m (matches BAR-ATH's 1.14 m
"at the feet" mode), 53.6% of stoppages fully without ball and 73.2% mostly, live gaps
0.56/min with median 5.4 s (all from `probe gaps` aggregates on BAR-ATH, no answers).
- Old defaults: spellF1 ~0.38 on all 7; predicted only ~47% as many possessions as the truth.
- Settings search (4 tune / 3 held-out): SWITCH_TURNOVER 10 -> 0.5, SWITCH_FROM_DEAD 4 -> 8,
  MIN_POSSESSION and MIN_DEAD 25 -> 50, plus USE_LINE_AWARE. Held-out spellF1 0.371 -> 0.518,
  macroF1 0.787 -> 0.831. All 7 with another seed: 0.387 -> 0.528; with 3x ball noise
  0.328 -> 0.437.
- Real answers overturned proxy verdicts: alone, touches (+0.055) and the 2 m zone (+0.058)
  helped and cleandead hurt (-0.02); jointly with the retuned decoder, touches and zone add
  nothing (they mostly made switching easier) and cleandead stays.
- Caveat: DFL's possession definition is finer than the competition's (~4.6 vs ~3.4 per
  minute on the public window), so the tuned decoder may over-split on BAR-ATH. Guardrail:
  BAR-ATH possession count should stay roughly 300-400.

## 2026-11-10 (later) — answers converted to the challenge definition; decoder re-tuned (v5)
v4 on the public window (pass/fail look, nothing fitted to it): spellF1 0.391 vs v2 0.413
(within noise), 10/17 possessions matched (v2 8) but 26 predicted vs 17 -> over-splitting, as
expected from DFL's finer possession definition. Fix from CHALLENGE.md's wording ("controlled"
possession: keeps the ball a few seconds; touches, deflections, uncontrolled duels don't count;
loose balls stay with the last team): in IDSSE answers, an H/A spell under EXT_CONTROL_MIN_S
following the other team's spell is given back to it (restart spells kept). 3 s chosen up front
("a few seconds"), 2 s as robustness. DFL 5.3 possessions/min -> 3.1 (2 s) / 2.5 (3 s).
Searches: 3 s held-out 0.492 -> 0.595 (wider grid 0.605); 2 s held-out 0.538 -> 0.583.
Finalists on all 7 matches, fresh seed, averaged over 2 s and 3 s definitions:
v4 0.539; A 0.612 (3x noise 0.565); B 0.598; C 0.608 (3x noise 0.587). Kept A (best average,
simpler): SWITCH_TURNOVER 2, TO_DEAD 4, FROM_DEAD 8, MIN_POSSESSION 100, MIN_DEAD 25,
softcontrol on, cleandead off (lineaware then has no effect, so off).
Transparency: the public window prompted re-reading the definition; all settings were chosen on
IDSSE only.

## 2026-11-11 — HANDOFF (state at end of session; no code or settings changed after v5)

### Current best: v5 (config defaults as of 2026-11-10)
SWITCH_TURNOVER 2, SWITCH_TO_DEAD 4, SWITCH_FROM_DEAD 8, MIN_POSSESSION 100, MIN_DEAD 25,
USE_SOFT_CONTROL True, USE_CLEAN_DEAD False, USE_LINE_AWARE False (no effect without
clean-dead); every other USE_* False. Settings are LOCKED: do not change them because of the
public window.
- IDSSE, 7 matches, 3 s definition, seed 1: spellF1 0.615, macroF1 0.874 (v4 0.539, v2 ~0.38
  on raw DFL answers, organisers' baseline_smooth50 ~0.32 on raw DFL answers).
- Public 5 min (sanity look only): spellF1 0.454 (rank 1), macroF1 0.771, 20 possessions vs 17.
- BAR-ATH full match: 278 possessions, 140 stoppages, D 46.0%, valid file.
- Reproducibility: two runs of `python -m possession.run` byte-identical (with warm cache/).
- Known weak spots: transition-tag yardstick worse than v4 (33% within 2 s vs 46%); some short
  stoppages backed only by slow players are back (clean-dead off); second-half pre-kickoff
  derived as 2852 frames vs 2934 published in FAQ.md (no ball movement >= 3 m/s found within
  10 s of the kickoff tag; left as is).

### Work in progress
- Video ball detection: `cache/video_ball_yolo11s_1920.npz` exists (7.4 MB). Completeness NOT
  verified. Rerun `python -m possession.video_ball detect`: it resumes, or prints "nothing to
  do" if all graded frames are done. Detector: yolo11s @1920 on MPS, conf 0.10, top-5
  candidates per frame; MPS runs verified byte-identical on 500 real frames (det_a/det_b).
  Benchmark: median 0.97 m from the ATD ball where both exist; ball found on ~49% of frames
  where the ATD has none; many ball-like decoys (152/200 frames had >1 candidate).
- Next video steps (not run yet):
  1. `python -m possession.videotrack check` -- pass bar: >= 90% of filled frames within 5 m
     of the hidden ATD ball.
  2. `python -m possession.masktest --variants base video videofull video+touch` -- note that
     masktest's "base" is now v5. Keep video only if hidden_team_agree_pct improves clearly
     (>= +5 points) without breaking the guardrails (possessions ~250-380, stoppages
     ~100-200, D share 35-47%, phase agreement not collapsing).
  IDSSE cannot test video (no video in that dataset).
- Rejected and kept in the code switched off: restarts, ballfree, zone, gapdead, tagrestarts,
  touches (alone), whistle (audio). See the entries above for the numbers.

### Planned next steps (deadline 17 Nov 2026; method freeze ~13 Nov)
1. Finish the video steps above; adopt only if they pass.
2. Freeze. Fresh-clone reproducibility: run twice with an empty cache
   (`python -m possession.run --rebuild-cache --out ...`) and `cmp` the CSVs.
3. Public repo: code only, on the user's OWN GitHub (the cloned repo is shared with teammates;
   never commit or push to it). possession/, tests/, scripts/ are currently hidden from git via
   .git/info/exclude, not .gitignore. Must exclude data/, cache/, models/, evaluation/results/
   CSVs except the organisers' baselines. Before pushing: `git ls-files` shows no data-derived
   files.
4. One command must also fetch the YOLO weights if video is adopted (Ultralytics release
   v8.4.0 yolo11s.pt); disable Ultralytics analytics (settings sync=False, already in code).
   Pinned versions so far: possession/requirements-video.txt (numpy 2.5.4, pandas 3.1.0,
   kloppy 3.18.0, opencv(-headless) 5.0.0.93, torch 2.15.0, torchvision 0.29.2,
   ultralytics 8.4.210, av 19.0.2), Python 3.14.7. A full requirements-solution.txt is still
   to write.
5. Team-named CSV (team name not yet given) + technical document (method; IDSSE tuning with
   the degradation and definition conversion; everything tried and rejected, from this file;
   the public window prompted the definition re-read but fitted nothing; limitations).
6. Admin (user): 3-5 registered team members for eligibility.

### Where things live (all local, never committed)
- Match data: data/tactical/ (ATD, homography, Smart Tagging, BAR-ATH_tactical.mp4),
  data/broadcast/ (ATD, homography), data/DATASET_README.md. Claude must not read data/.
- BAR-ATH caches: cache/atd_tactical.npz (parsed ATD), cache/audio_features.npz (whistle
  features), cache/video_ball_yolo11s_1920.npz (detections), cache/det_a.npz, det_b.npz
  (determinism test), cache/repro_a.csv, repro_b.csv.
- Submissions: evaluation/results/possession_v1..v5.csv (v3 = v2 + derived kickoff).
- Detector weights: models/yolo11s.pt, models/yolo11m.pt.
- IDSSE raw XML (2.5 GB, 7 matches x matchinformation/events/positions):
  cache/external/idsse/. Converted arrays: cache/external/idsse_<MATCH>.npz.
  Search logs: cache/external/tune_ctrl3.txt, tune_ctrl2.txt, tune_ctrl3_wide.txt.
  Source: https://huggingface.co/datasets/pysport/idsse-data (CC-BY 4.0; kloppy's figshare
  link answers 202 to scripts). Cite Bassek, Rein, Weber & Memmert (2025), Scientific Data,
  doi:10.1038/s41597-025-04505-y.

### Reproducing the IDSSE results (IDSSE is public data: anyone may run these)
Download (about 2.5 GB):
    cd cache/external/idsse && B=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
    for m in 000001_DFL-MAT-J03WMX 000001_DFL-MAT-J03WN1 000002_DFL-MAT-J03WOH \
             000002_DFL-MAT-J03WOY 000002_DFL-MAT-J03WPY 000002_DFL-MAT-J03WQQ \
             000002_DFL-MAT-J03WR9; do
      for k in 02_01_matchinformation 03_02_events_raw 04_03_positions_raw_observed; do
        curl -sSL --retry 3 -o DFL_${k}_DFL-COM-${m}.xml $B/DFL_${k}_DFL-COM-${m}.xml; done; done
Then, from the repo root:
    python -m possession.external convert      # parse + sanity (owner agreement 75-81%)
    python -m possession.external noise        # -> sigma 0.25 m (EXT_BALL_NOISE_M)
    python -m possession.external score        # current defaults, degraded, RAW DFL answers
    python -m possession.external score --raw  # clean tracking
CAUTION when re-running the searches: "default" in their output means the CURRENT config
defaults (now v5), and `prepared()` now converts answers to the challenge definition unless
control_min_s=0 is passed. So:
- First search (v4; raw DFL answers, original start 10/4/4/25/25):
    python -c "from possession import external; external.tune(0.25, control_min_s=0, start={'SWITCH_TURNOVER':10.0,'SWITCH_TO_DEAD':4.0,'SWITCH_FROM_DEAD':4.0,'MIN_POSSESSION':25,'MIN_DEAD':25})"
  (originally run as `tune`, before the conversion existed; result 1.0/4/8/50/50, held-out
  0.371 -> 0.497.) Then `python -m possession.external tune --joint --control-min 0`
  reproduces v4's joint search (0.5/4/8/50/50 + lineaware; held-out 0.518). Option starting
  values come from the current config, so set USE_CLEAN_DEAD True / USE_SOFT_CONTROL False
  temporarily if exact replication of the logged path matters.
- Definition-converted searches (v5):
    python -m possession.external tune --joint --control-min 3     # log: tune_ctrl3.txt
    python -m possession.external tune --joint --control-min 2     # log: tune_ctrl2.txt
    python -c "from possession import external; external.tune(0.25, search={'SWITCH_TURNOVER':(1.0,2.0,4.0,8.0),'SWITCH_TO_DEAD':(2.0,4.0,8.0),'SWITCH_FROM_DEAD':(4.0,8.0,16.0),'MIN_POSSESSION':(75,100,125,150,200),'MIN_DEAD':(12,25,50)}, start={'SWITCH_TURNOVER':2.0,'SWITCH_TO_DEAD':4.0,'SWITCH_FROM_DEAD':8.0,'MIN_POSSESSION':100,'MIN_DEAD':25}, with_options=True, control_min_s=3.0)"   # log: tune_ctrl3_wide.txt
- Finalist comparison that chose v5 (all 7, seed 1, 2 s and 3 s, plus 3x noise): evaluate
  each candidate with external.prepared(external.match_ids(), 0.25, seed=1, control_min_s=s)
  for s in (2.0, 3.0) and noise 0.75 at 3 s, via external.evaluate(settings, data, options):
    A = ({'SWITCH_TURNOVER':2.0,'SWITCH_TO_DEAD':4.0,'SWITCH_FROM_DEAD':8.0,'MIN_POSSESSION':100,'MIN_DEAD':25}, {'cleandead':False,'softcontrol':True,'lineaware':True})
    B = ({'SWITCH_TURNOVER':1.0,'SWITCH_TO_DEAD':4.0,'SWITCH_FROM_DEAD':8.0,'MIN_POSSESSION':75,'MIN_DEAD':25}, {'cleandead':False,'softcontrol':True,'lineaware':True})
    C = ({'SWITCH_TURNOVER':4.0,'SWITCH_TO_DEAD':8.0,'SWITCH_FROM_DEAD':16.0,'MIN_POSSESSION':125,'MIN_DEAD':50}, {'cleandead':False,'touches':True,'lineaware':True})
  Results: v4 0.539, A 0.612, B 0.598, C 0.608 (average of the 2 s and 3 s definitions).
  With the current defaults, `external.evaluate({}, prepared(..., seed=1, control_min_s=3.0))`
  must give spellF1 0.615 / macroF1 0.874.
- Baseline on IDSSE (raw DFL answers, seed 0): smooth(nearest_player_codes(arrays), 50) from
  possession.baseline_check, scored with external.score -> spellF1 0.315, macroF1 0.711.
All of it is deterministic (fixed seeds, closed-form fits); runtimes: convert ~1 min, each
search 6-15 min on an M3 Pro.

### Commands the user runs on BAR-ATH (Claude never runs these against data/)
    python -m possession.run --out evaluation/results/<name>.csv
    python evaluation/validate_submission.py evaluation/results/<name>.csv
    python evaluation/run_evaluation.py [--bootstrap 1000]     # public window, sanity only
    python -m possession.probe [gaps]
    python -m possession.masktest --variants ...
    python -m possession.video_ball detect | info | benchmark | compare a b
    python -m possession.videotrack check
    python -m possession.audio features | check
Tests (synthetic only, safe for Claude): python -m unittest discover -s tests   (53 tests)

## 2026-11-11 — video ball adopted (videoball, team evidence only) -> v6
Detections complete: 158857 graded frames (170 min on MPS); the stretch around video frames
76000-82000 was redone after the Mac slept mid-run (NMS time-limit warnings). detect now reruns
any batch slower than 2 s, so a stall can no longer truncate NMS.
- videotrack check, seed 0 (14284 hidden ATD frames): filled 80.1%, error p50 0.90 m, p90 8.3 m,
  within 2 m 71.1%, within 5 m 85.2% -- below the 90% bar set beforehand. Sweep over chain-start
  cost (2/4/8/anchor-only) x confidence reference (0.25/0.35/0.5): within 5 m stays 82.5-87.7%
  while fill drops to 49-67%, so strictness buys no precision; the tail is likely aerial balls
  (ground-plane homography) and ATD error itself. Defaults kept (2, 0.25). The 90% bar was
  relaxed explicitly; the decision moved to the end-to-end test with its criteria unchanged.
- masktest vs base v5 (9% hidden): hidden_team_agree 65.6 -> videoball 77.2 (+11.6, bar +5),
  videofull 83.9, video+touch 77.5. Transition tags within 2 s 33.3 -> 37.5 (videoball), median
  3.32 s -> 2.70 s. Phase agreement 82.9 -> 82.7 (-0.2, within noise). Possessions 278 -> 279,
  stoppages 140 -> 140.
- Adopted videoball. videofull not adopted: the masked test only hides live play, so it cannot
  see the risk videofull carries (video ball seen in stoppages weakening dead-ball cues); it
  added 5 stoppages. video+touch rejected (phase agreement 81.0, timing not better).
- Reproducibility: run.py now downloads the pinned weights (sha256 85a76fe8...) if missing and
  runs/resumes detection before labelling; the pipeline refuses incomplete detections.
- Incident: with videoball defaulting on, the synthetic test suite briefly loaded the real
  detections and homography (pass/fail output only, nothing printed or sent). tests/isolation.py
  now disables those defaults and blocks the real paths for every test.
