# Ball possession for BAR-ATH: technical document (DRAFT, v7)

Team: `<team_name>` · Submission file: `<team_name>.csv` · Code: `possession/` in this repository.

Every number below was produced by code in this repository. The full decision log, including
every rejected idea and its numbers, is `possession/NOTES.md`.

## 1. Approach

The scorer matches whole possessions, so one wrong frame in the middle of a possession costs far
more than one wrong frame at its edge. We therefore split the problem in two:

1. **Per-frame evidence.** For every graded frame, a score for home (H), away (A) and dead (D).
2. **A segment decoder.** A segmental Viterbi decoder turns the scores into runs of H, A and D
   with minimum durations and switch costs, so it changes label only when the evidence is
   sustained. Turnovers are then snapped to the nearest clear change of control.

We never write `X` on a graded frame.

**Evidence (v7).** A gradient-boosted tree classifier (scikit-learn
`HistGradientBoostingClassifier`, 200 trees, fixed seed) predicts H / A / D from 58 features of
the tracking data only: distance from the ball to the nearest player of each team, who is
clearly closest, ball speed and position, distance to the lines, how long the ball has been
missing and where it was last and next seen, player speeds, and short windows of these. The
model is team-symmetric: every training row is also shown with the teams swapped, and
predictions average both views, so it cannot learn a preference for either side.

The classifier is trained on other matches only (section 3). Its log-odds are added to three
smaller cues computed on the provided game:

- a weak team prior from the Smart Tagging phase tags;
- a dead-ball cue at set-piece tags, and dead ball before the kickoff of each half;
- where the provided tracking has no ball, team evidence from a ball we detect in the tactical
  video (section 2).

**Why this rather than something else.** Our first six versions used hand-written evidence
rules (possession zone, duel zone, ball out of bounds). On seven Bundesliga matches with real
answers, the learned evidence scored spellF1 0.702 on held-out matches against 0.576 for the
hand-written rules, and it also won on a third dataset (Metrica's public sample matches, 0.600
against 0.534). An end-to-end sequence model was not attempted: we have no permitted labels for
the provided game, and seven matches are few for a network.

## 2. Assets

| Asset | Used? | How |
|---|---|---|
| ATD tracking (players and ball) | Yes | The main input: all 58 features. |
| Tactical video + per-frame homography | Yes | Ball detection where the tracking has no ball (41% of graded frames). A YOLO11s detector (COCO weights, sha256-checked) runs locally on every graded frame; pixels are mapped to pitch metres through the homography; a chain filter keeps only physically plausible paths. It fills 38% of the gaps and is used only as team evidence. |
| Smart Tagging | Yes | Phase tags as a weak team prior; set-piece tags as a dead-ball cue; the kickoff tag to place the start of each half. |
| Audio (referee whistle) | Tried, dropped | The tactical video's audio track was searched for whistles; they could not be told apart from crowd noise (nothing found near the kickoffs). |
| Broadcast video | Not used | Not synchronised with the tactical video; our ceiling test put the gain from a perfect ball at about +0.04. |
| Public 5-minute ground truth | Sanity check only | See section 6. |

## 3. Training data

**Nothing from the provided game was used for training or tuning.**

| Dataset | Source | Labels | Amount | Use |
|---|---|---|---|---|
| IDSSE / DFL, 7 Bundesliga matches | Bassek et al. 2025, CC-BY 4.0, `pysport/idsse-data` on Hugging Face (pinned revision, files hash-checked) | The dataset's own per-frame ball-possession and ball-status fields, produced by the league's data provider | 7 full matches | Training the classifier (all 7 for the submission); 4 for tuning and 3 held out when comparing versions |
| Metrica public sample data, 3 matches | `metrica-sports/sample-data` (pinned commit, files hash-checked) | Metrica's event data; we derive possession as the team of the latest on-ball event, dead where the tracking has no ball | 3 matches | Evaluation only. Not in the submitted model's training set. |

**Making the other matches look like ours.** The Bundesliga tracking is cleaner than the
provided game's, so before training and scoring we degrade it: correlated ball noise (0.25 m),
the ball hidden during most stoppages, random ball gaps in live play, the ball hidden near the
lines, and player dropouts. The rates were set from aggregate statistics of the provided game's
tracking (share of frames with no ball, gap lengths, distance of the ball to the lines) — no
labels from the provided game were involved.

**Converting the answers.** The Bundesliga labels switch team on first touch. The challenge
keeps possession with the previous team until the other side has control, so we convert the
answers with a control rule (a team must hold the ball for a minimum time before it owns it):
1 s for training, 3 s for the main comparison, and we report both. This rule is our reading of
the challenge definition, not something we could verify.

## 4. Possession rules implemented

- **Loose ball and challenges.** Not a label. The previous team keeps possession until the
  other team shows sustained control; a single touch or deflection is not enough. A possession
  lasts at least 3 s (75 frames) in the decoder.
- **Dead ball.** Ball out of play, stoppages and the wait before a restart are `D`, with a
  minimum of 1 s. Entering dead ball is cheaper for the decoder than leaving it, so a stoppage
  ends only when play has clearly resumed.
- **Restarts.** The restarting team's possession starts when the ball is back in play, not when
  the ball is placed.
- **Before kickoff, half-time, after the whistle.** `D` before each half's kickoff (placed from
  the kickoff tag), `X` outside the graded blocks.

## 5. Results

Other matches, real answers, models that never saw the test matches (spellF1):

| Test | Hand-written rules (v6) | Submitted (v7) |
|---|---|---|
| Bundesliga held-out, 3 s control rule | 0.576 | 0.702 |
| Bundesliga held-out, 1 s control rule | 0.506 | 0.615 |
| Bundesliga held-out, 3× ball noise | 0.516 | 0.606 |
| Bundesliga held-out, macroF1 | 0.838 | 0.908 |
| Metrica sample matches, raw event answers | 0.534 | 0.600 |

Provided game, checks that need no ground truth (each one hides something from the model and
measures how well the labels recover it):

| Check | v6 | v7 |
|---|---|---|
| Ball hidden for live-play-length spans: team unchanged | 78.4% | 82.1% |
| Phase tags withheld: labels agree with them | 73.6% | 74.2% |
| Set-piece tags withheld: dead ball at the tag | 91.4% | 89.7% |
| Transition tags withheld: a turnover within 5 s | 83.3% | 79.2% |
| Video ball withheld: agrees with clear video-ball control | 86.0% | 88.2% |

On the full match v7 finds 329 possessions (3.1 per minute) and 145 stoppages, and labels 42.6%
of graded frames dead.

## 6. How we used the public 5 minutes

Only as a pass/fail sanity check after a version had already been chosen on other data; never
to choose between versions or set a parameter. For transparency: v7 scores spellF1 0.514 and
macroF1 0.712 there (18 possessions found, 17 true). With 17 possessions the number is very
noisy.

## 7. Limitations

- **Dead ball may be under-called.** In the public window v7 marks only 37% of the truly dead
  frames as dead (v6: 55%). On the other matches dead-versus-live frame accuracy is 98%
  (Bundesliga held-out) and 96% (Metrica), but those tests are optimistic: our degraded copies
  hide the ball in most stoppages by construction, so a missing ball is a cleaner dead-ball cue
  there than in the provided game. The dead-ball convention of the training data may also
  differ from the challenge's.
- **Boundaries are a little late.** Withheld transition tags are matched after a median 3.6 s.
  Being 1 s late already costs about a third of a possession's score.
- **Short possessions.** On the Bundesliga matches about 10% of true possessions are given to
  the other team, mostly ones a few seconds long.
- **The video ball covers only 38% of the gaps**, and a ball in the air breaks the ground-plane
  projection.
- **Our control rule is a guess** at the challenge's definition.
- **Kickoff.** We place the second-half kickoff from the tags, 82 frames from the frame given in
  the FAQ.
- **Run time.** About 2.5 h on Apple Silicon, almost all of it video ball detection. Byte-identical
  reruns were verified on one machine; a different machine or GPU backend was not tested.

**Tried and not adopted** (numbers in `NOTES.md`): restart-geometry templates, a ball-free
model, whistle detection, touch detection, a learned boundary refiner, stoppage merging, decoder
re-tuning, larger and ensembled classifiers, and a richer feature set with more robust training
(+0.02 on held-out matches, below the scorer's stated 0.04 noise band, and mixed on the provided
game, so not adopted).

**With more time:** a dead-ball model trained on data that follows the challenge's convention,
and a ball detector fine-tuned for small balls in tactical footage.

## 8. Reproduction

```bash
pip install -r requirements.txt -r possession/requirements-video.txt
python -m possession.run --out evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

The one command downloads the pinned public training matches and detector weights (hash-checked),
trains the classifier with fixed seeds, runs ball detection on the tactical video, and writes
the CSV. The first run takes about 2.5 h; keep the machine awake.
