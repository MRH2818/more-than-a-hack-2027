# What's on the `alex-v9` branch

The current pipeline, **v9** (v8 plus one dead-ball rule), is the one we plan to submit. The write-up for the organisers is
`possession/TECHNICAL.md`. The full decision log, with every number and every rejected idea, is
`possession/NOTES.md`.

No data, caches, detections, model weights or CSVs are in this branch. Everything is regenerated
from your own local copy of the competition data in `data/`.

## What v9 is

- **Evidence from two small learned models** (gradient-boosted trees), trained only on other
  matches: 7 Bundesliga matches (IDSSE) and Metrica's 3 public sample matches. One model decides
  dead ball versus live, the other decides which team has it.
- **The same decoder as v6**, which turns per-frame evidence into whole possessions.
- **The video ball and the Smart Tagging cues from v6** are kept on top.
- **New in v9: dead ball from a goal until the following kickoff**, read from the tags. The
  idea and pairing rule come from the `deadzones/` work on `main`. It removes two false
  possessions inside goal celebrations and changes nothing else.

## Current scores (v9, 2026-11-15)

spellF1 runs from 0 to 1, and 1 is perfect. It's the metric the leaderboard ranks on.

| Test | v6 | v7 | v9 | What it tells us |
|---|---|---|---|---|
| Bundesliga held-out matches, strict answers | 0.576 | 0.702 | **0.715** | Main yardstick: 3 full matches the models never saw. |
| Bundesliga held-out, loose answers | 0.506 | 0.615 | **0.634** | Same, with a looser reading of "control". |
| Bundesliga held-out, 3× noisier ball | 0.516 | 0.606 | **0.621** | Same, with worse ball tracking. |
| Public 5-minute window | 0.489 | 0.514 | 0.478 | Sanity check only: 17 possessions, so these three are the same within noise. Never tune or choose on it. |
| Full match (the real leaderboard) | unknown | unknown | unknown | The official floor is smooth50 at 0.2433. |

**Other facts about v9**
- On the full match it finds 315 possessions and 145 stoppages, and labels 42.8% of graded
  frames as dead ball.
- The video ball fills 38% of the frames where the tracking has no ball.

**Known weak spots**
- Dead-ball recall in the public window is only 37%, although it is 96-98% on the other
  matches. We could not explain the difference.
- Turnovers are called a little late: withheld transition tags are matched after a median 2.7 s.
- The step from v7 to v8 is small (+0.016) and below the scorer's 0.04 noise band.

## 1. Regenerate v9 (one command, about 2.5 h on Apple Silicon the first time)

```bash
pip install -r requirements.txt -r possession/requirements-video.txt
python -m possession.run --out evaluation/results/possession_v9.csv
python evaluation/validate_submission.py evaluation/results/possession_v9.csv
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
  stoppages, and missing-ball gaps at our match's rates.
- **The answers are converted to the challenge's possession definition.**

We tune on 4 matches (`external.TUNE_MATCHES`) and report on the other 3
(`external.TEST_MATCHES`). Download instructions are in `NOTES.md`.

```python
from possession import external
data = external.prepared(external.TEST_MATCHES, noise=0.25, seed=1, control_min_s=3.0)
# for each match: produce your labels, then
external.score(labels, answers)        # spellF1 / macroF1 against real answers
```

This lets you compare ideas without touching the public 5 minutes.

## Tests

```bash
python -m unittest tests.test_pipeline tests.test_improvements tests.test_videotrack tests.test_io_atd tests.test_audio
# add tests.test_learned tests.test_gate for the learned models: 75 synthetic tests in all,
# no match data needed, about 45 s
```

## Rules we followed (please keep to them)

- **Never commit data, caches, detections, weights or submission CSVs.**
- **No hand labels.** Every label comes from code.
- **The public 5-minute ground truth is a sanity check, never a tuning target.**
- **Match data and video must not be uploaded to any cloud AI tool.**
