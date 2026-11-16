# What's on the `alex-v10` branch

The current pipeline, **v10**, is the one we submit as **Golden Bear Analytics**. The write-up
for the organisers is `possession/TECHNICAL.md`. The full decision log, with every number and
every rejected idea, is `possession/NOTES.md`.

No data, caches, detections, model weights or CSVs belong in this branch. Everything is
regenerated from your own local copy of the competition data in `data/`.

## What v10 is

- **Evidence from two small learned models** (gradient-boosted trees), trained only on other
  matches: 7 Bundesliga matches (IDSSE) and Metrica's 3 public sample matches. One model decides
  dead ball versus live, the other decides which team has it.
- **The same decoder as v6**, which turns per-frame evidence into whole possessions.
- **The video ball and the Smart Tagging cues from v6** are kept on top, plus dead ball from a
  goal until the following kickoff (v9).
- **New in v10: the models are trained on a realistic stoppage ball.** In the Bundesliga files
  the ball is frozen in place during every stoppage, so v7-v9 learned that a ball which is seen
  and moving is always live. The training copies now move the ball to the restart spot or
  leave it with a player (`config.STOPPAGE_BALL`).
- **New in v10: turnover snapping is off** (`config.SNAP_WINDOW = 0`). It was written for the
  hand-made evidence and cost 0.02-0.04 on the other matches with the learned models.

## Current scores (v10, 2026-11-15)

spellF1 runs from 0 to 1, and 1 is perfect. It's the metric the leaderboard ranks on. The
Bundesliga rows now use the realistic stoppage ball, so they are lower than the older figures
for the same models (v9 was 0.634 on the old bench).

| Test | v6 | v7 | v9 | v10 | What it tells us |
|---|---|---|---|---|---|
| Bundesliga held-out matches, 1 s answers | 0.483 | 0.581 | 0.589 | **0.635** | Main yardstick: 3 full matches the models never saw. The 1 s rule is the closest to the real answer key (NOTES.md). |
| Bundesliga held-out, raw answers | 0.433 | 0.525 | 0.533 | **0.572** | Same, with the provider's labels as they are. |
| Bundesliga held-out, 3 s answers | 0.544 | 0.649 | 0.656 | **0.708** | Same, with a strict reading of "control". |
| Bundesliga held-out, 3x noisier ball | 0.441 | 0.528 | 0.526 | **0.594** | 1 s answers, worse ball tracking. |
| Public 5-minute window | 0.489 | 0.514 | 0.478 | 0.495 | Sanity check only: 17 possessions, so these are the same within noise. Never tune or choose on it. |
| Full match (the real leaderboard) | unknown | unknown | unknown | unknown | The official floor is smooth50 at 0.2433. |

**Other facts about v10**
- On the full match it finds 292 possessions and 129 stoppages, and labels 45.9% of graded
  frames as dead ball (v9: 315, 145, 42.8%).
- Dead-ball recall in the public window is 58% (v9: 37%).
- The video ball fills 38% of the frames where the tracking has no ball.

**Known weak spots**
- Dead ball is still under-called where the tracking shows a ball: between a goal and the
  kickoff v10 labels 63% of such frames dead (v9: 42%).
- The simulated stoppage ball is an assumption; the real tracked ball moves much more in dead
  time than the simulated one.
- Turnovers are called a little late: withheld transition tags are matched after a median 3.1 s.
- Each of the two v10 changes is below the scorer's 0.04 noise band on its own.

## 1. Regenerate v10 (one command, about 2.5 h on Apple Silicon the first time)

```bash
pip install -r requirements.txt -r possession/requirements-video.txt
python -m possession.run --out evaluation/results/possession_v10.csv
python evaluation/validate_submission.py evaluation/results/possession_v10.csv
```

- **Most of the time goes on video ball detection** (YOLO11s on every graded frame, about
  20 frames/s on MPS).
- **Keep the laptop plugged in with the lid open.** If it sleeps, the run can stall.
- **Detection checkpoints every 2000 frames** and resumes if you stop it.
- **The first run also downloads the public training matches and the detector weights**
  (pinned versions, hashes checked) and trains the two models, which takes a few minutes.
- **The output is deterministic** on one machine. If you can, run it on yours and tell us
  whether your CSV is byte-identical to ours: a second machine has not been tested.

## 2. Video ball detection (fills the gaps in the provided ball track)

The provided tracking has no ball on about 41% of graded frames. We run a local detector on
the tactical video and map each pixel to pitch metres through the per-frame homography. A
chain filter then keeps only physically plausible ball paths: max speed, position tolerance
and skip limits. The result fills about 38% of the gaps, and is used only where the tracking
has no ball.

```bash
python -m possession.video_ball detect            # writes the detections cache (local only)
python -m possession.videotrack check             # accuracy against the tracking ball, on hidden stretches
```

In your own code:

```python
from possession import pipeline
video_xy = pipeline.video_ball(arrays, graded_mask)   # (T, 2) metres, NaN where nothing is trusted
```

## 3. IDSSE test bench (score a model on 7 other matches with real answers)

This uses 7 Bundesliga matches from the IDSSE dataset (Bassek et al. 2025, CC-BY 4.0,
`pysport/idsse-data` on Hugging Face). We prepare them in two ways:

- **The tracking is degraded to look like ours:** ball noise, the ball removed during
  most stoppages and moved realistically in the rest, and missing-ball gaps at our match's
  rates.
- **The answers are converted to the challenge's possession definition** (quote the 1 s rule
  first).

We tune on 4 matches (`external.TUNE_MATCHES`) and report on the other 3
(`external.TEST_MATCHES`). Download instructions are in `NOTES.md`.

```python
from possession import external
data = external.prepared(external.TEST_MATCHES, noise=0.25, seed=1, control_min_s=1.0)
# for each match: produce your labels, then
external.score(labels, answers)        # spellF1 / macroF1 against real answers
```

This lets you compare ideas without touching the public 5 minutes.

The tables above come from `python -m possession.stoppagelab versions` and `... bench`.
Label-free checks on the provided game: `python -m possession.gate --variants v9 base` and
`python -m possession.probe deadchecks v9 base`.

## Tests

```bash
python -m unittest discover -s tests
# 83 synthetic tests, no match data needed, about a minute
```

## Rules we followed (please keep to them)

- **Never commit data, caches, detections, weights or submission CSVs.**
- **No hand labels.** Every label comes from code.
- **The public 5-minute ground truth is a sanity check, never a tuning target.**
- **Match data and video must not be uploaded to any cloud AI tool.**
