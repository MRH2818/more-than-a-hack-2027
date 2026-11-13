# What's on the `alex-v6` branch

The current best pipeline, **v6**, is the model to beat. It comes with two tools you can reuse
in your own models. The full decision log, with every number and every rejected idea, is in
`possession/NOTES.md`.

No data, caches, detections, model weights or CSVs are in this branch. Everything is regenerated
from your own local copy of the competition data in `data/`.

## 1. Regenerate v6 (one command, about 2.5 h on Apple Silicon)

```bash
pip install -r requirements.txt -r possession/requirements-video.txt
python -m possession.run --out evaluation/results/possession_v6.csv
python evaluation/validate_submission.py evaluation/results/possession_v6.csv
```

- **Most of the time goes on video ball detection** (YOLO11s on every graded frame, about
  20 frames/s on MPS).
- **Keep the laptop plugged in with the lid open.** If it sleeps, the run can stall.
- **Detection checkpoints every 2000 frames** and resumes if you stop it.
- **The weights are downloaded once**, and their sha256 is checked.
- **The output is deterministic.** Reruns were byte-identical, for both the detections and the CSV.

**v6 results**
- Public 5-minute window: spellF1 0.489, macroF1 0.783. That window is for checking only, so
  don't tune on it.
- IDSSE held-out matches: spellF1 about 0.61.
- `baseline_smooth50` (the official floor) scores 0.2433 on the full match.

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
# 53 synthetic tests, no match data needed, about 30 s
```

## Rules we followed (please keep to them)

- **Never commit data, caches, detections, weights or submission CSVs.**
- **No hand labels.** Every label comes from code.
- **The public 5-minute ground truth is a sanity check, never a tuning target.**
- **Match data and video must not be uploaded to any cloud AI tool.**
