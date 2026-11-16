# Ball possession for BAR-ATH: technical document

**Team:** Golden Bear Analytics · **Submission file:** `Golden_Bear_Analytics.csv` · **Code:** `possession/` in this repository

Every number below was produced by code in this repository. The full decision log, including every rejected idea and its numbers, is `possession/NOTES.md`.

## Summary

- **What it is.** Two learned classifiers score every graded frame (dead vs live, then home vs away). A segmental Viterbi decoder turns those scores into whole stoppages and possessions. A few Smart Tagging cues and a video-based ball detector add evidence on top.
- **What it was trained on.** Seven Bundesliga matches (IDSSE / DFL, CC-BY 4.0), plus three Metrica sample matches for one of the two classifiers. No labels from the provided game were used to train or tune anything, and nobody labelled anything by hand.
- **How well it works.** On held-out Bundesliga matches it scores spellF1 0.635 against 0.483 for our earlier hand-written rules. The provided game's own ground truth is public for only 5 minutes, so on the full match we can only run label-free checks (section 5).
- **Where it is weak.** Dead ball is under-called whenever the tracking still shows a ball during a stoppage, and turnover timing is imprecise: withheld transition tags lie a median 3.1 s from our nearest turnover (section 7).

## 1. Approach

The scorer matches whole possessions, so one wrong frame in the middle of a possession costs far more than one wrong frame at its edge. We therefore split the problem in two:

```
tracking + tags + video ball
        │
        ▼
  per-frame evidence  ──►  segmental Viterbi decoder  ──►  H / A / D per frame
  (dead, home, away        (minimum durations,
   log-odds)                switch costs)
```

We never write `X` on a graded frame.

**Per-frame evidence.** Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, 200 trees, fixed seed) use features of the tracking data only: distance from the ball to the nearest player of each team, who is clearly closest, ball speed and position, distance to the lines, how long the ball has been missing and where it was last and next seen, player speeds, and short past and future windows of these. Both models are team-symmetric: every training row is also shown with the teams swapped, and predictions average both views, so they cannot learn a preference for either side.

Two classifiers share the work:

1. **Dead vs live** (58 features).
2. **Home vs away, given live play.** It adds features about the last and next clear controller of the ball and ball direction changes, and is trained on more varied noise.

**Cues computed on the provided game.** The classifiers' log-odds are added to smaller cues:

- a weak team prior from the Smart Tagging phase tags;
- a dead-ball cue at set-piece tags, and dead ball before each half's kickoff;
- dead ball between a goal tag and the kickoff tag that follows it (4 stretches);
- where the provided tracking has no ball, team evidence from a ball we detect in the tactical video (section 2).

**Decoder.** A possession lasts at least 3 s (75 frames) and a stoppage at least 1 s. Entering dead ball costs 4 and leaving it costs 8, so a stoppage ends only once play has clearly resumed. The classifiers' output enters as log-odds against their training prior.

**Why this rather than something else.** Our first six versions used hand-written evidence rules (possession zone, duel zone, ball out of bounds). On held-out Bundesliga matches with real answers, the learned evidence scores spellF1 0.635 against 0.483 for those rules. It also wins on a third dataset (Metrica's public sample matches, 0.618 against 0.534). An end-to-end sequence model was not attempted: we have no permitted labels for the provided game, and seven matches are few for a network.

## 2. Assets

| Asset | Used? | How |
| :---- | :---- | :---- |
| ATD tracking (players and ball) | Yes | The main input: all classifier features. |
| Tactical video + per-frame homography | Yes | Ball detection where the tracking has no ball (41% of graded frames). A YOLO11s detector (COCO weights, sha256-checked) runs locally on every graded frame. Pixels are mapped to pitch metres through the homography, and a chain filter keeps only physically plausible paths. It fills 38% of the gaps and is used only as team evidence, not for dead vs live. |
| Smart Tagging | Yes | Phase tags as a weak team prior; set-piece tags as a dead-ball cue; the kickoff tag to place the start of each half; goal tags paired with the following kickoff tag to mark the celebration as dead. Also used as a label-free check, with the tags in question withheld from the model (section 5). |
| Audio (referee whistle) | Tried, dropped | The tactical video's audio track was searched for whistles. They could not be told apart from crowd noise (nothing found near the kickoffs). |
| Broadcast video | Not used | Not synchronised with the tactical video. Our ceiling test put the gain from a perfect ball at about +0.04. |
| Public 5-minute ground truth | Sanity check only | See section 6. |

## 3. Training data

**No labels from the provided game were used to train models or set parameters.** From the provided game, training uses only unlabelled aggregate statistics of its tracking (to set the noise rates below). Its tracking, video and provided Smart Tagging file are inputs to the system. Label-free checks on the provided game (section 5) were also used when deciding whether to adopt a version; they involve no ground truth. One decoder setting, the 3 s minimum possession, was picked among settings that scored alike on the Bundesliga matches because it passed those checks.

| Dataset | Source | Who labelled it | Amount | Use |
| :---- | :---- | :---- | :---- | :---- |
| IDSSE / DFL, 7 Bundesliga matches | Bassek et al. 2025, CC-BY 4.0, `pysport/idsse-data` on Hugging Face (pinned revision, files hash-checked) | The league's data provider, via the dataset's own per-frame ball-possession and ball-status fields | 7 full matches | Both classifiers. All 7 for the submission; 4 for tuning and 3 held out when comparing versions. |
| Metrica public sample data | `metrica-sports/sample-data` (pinned commit, files hash-checked) | Metrica's event data. We derive possession as the team of the latest on-ball event, and dead where the tracking has no ball. | 3 matches | Evaluation. In the submission, added to the training set of the home-vs-away classifier only. |

No person labelled anything for this work; both label sets come with the public datasets.

**Making the other matches look like ours.** The Bundesliga tracking is cleaner than the provided game's, so before training and scoring we degrade it: correlated ball noise (0.25 m), the ball hidden during most stoppages, random ball gaps in live play, the ball hidden near the lines, and player dropouts. The rates were set from aggregate statistics of the provided game's tracking (share of frames with no ball, gap lengths, distance of the ball to the lines). No labels from the provided game were involved.

**The stoppage ball (new in v10).** In the Bundesliga files the ball stands at one position for the whole of every stoppage and then jumps to the restart spot (100% of dead frames in all seven matches). A camera-tracked ball never does that, and a model trained on it learns that a ball which is seen and moving is always live. So in the training copies we replace the frozen ball: it rolls on briefly, and is then either moved to the restart spot and left there, or stays with the player nearest the restart spot when play resumes (one of the two per stoppage, at random). The same noise and hiding are applied afterwards. This simulated ball is our assumption, not a measurement (section 7).

**Converting the answers.** The Bundesliga labels switch team on first touch. The challenge keeps possession with the previous team until the other side has control. We convert the answers with a control rule (a team must hold the ball for a minimum time before it owns it): 1 s for training. We report results under 1 s, under the raw labels and under 3 s.

To choose which to quote first, we used the table in `EVALUATION.md` giving the score of a perfect submission 3, 8 and 25 frames late on the full match (0.9818 / 0.8933 / 0.6539). That table depends only on how long the true possessions are. Computed on the Bundesliga answers it gives:

| Bundesliga answers | 3 frames late | 8 frames late | 25 frames late |
| :---- | :---- | :---- | :---- |
| Raw labels | 0.960 | 0.862 | 0.557 |
| 1 s control rule | 0.998 | 0.958 | 0.663 |
| 3 s control rule | 0.999 | 0.991 | 0.874 |
| *Provided game (`EVALUATION.md`)* | *0.9818* | *0.8933* | *0.6539* |

The provided game's possessions look most like the raw-to-1 s end. This changed which column we lead with and no setting: training on raw, 0.5 s or 1 s answers gave results within 0.02 of each other.

## 4. Possession rules implemented

| Situation | Rule |
| :---- | :---- |
| **Loose ball, deflection, single touch** | Not a label. The previous team keeps possession until the other team shows sustained control. A possession lasts at least 3 s (75 frames) in the decoder. |
| **Challenges and aerial duels** | Same: possession changes only on sustained control, not on a duel won without control. |
| **Dead ball** | Ball out of play, stoppages and the wait before a restart are `D`, with a minimum of 1 s. Entering dead ball is cheaper than leaving it, so a stoppage ends only when play has clearly resumed. |
| **Restarts** | The restarting team's possession starts when the ball is back in play, not when the ball is placed. |
| **After a goal** | `D` from one second after the last tag overlapping the goal clip ends until one second before the following kickoff clip starts. |
| **Before kickoff, half-time, after the whistle** | `D` before each half's kickoff (placed from the kickoff tag). `X` outside the graded blocks. |
| **Turnover timing** | The change of team is placed where the decoder puts it. Earlier versions moved it to the nearest frame of clear control by the new team. v10 drops that step (section 5). |
| **Dead-ball ownership** | Not asked for. When the ball is dead the answer is `D`. |

## 5. Results

**Other matches, real answers, models that never saw the test matches (spellF1).** The first five rows are the three held-out Bundesliga matches with the simulated stoppage ball of section 3 (`python -m possession.stoppagelab versions`).

| Test | Hand-written rules (v6) | One classifier (v7) | Two classifiers (v9) | Submitted (v10) |
| :---- | :---- | :---- | :---- | :---- |
| Bundesliga held-out, 1 s control rule | 0.483 | 0.581 | 0.589 | **0.635** |
| Bundesliga held-out, raw labels | 0.433 | 0.525 | 0.533 | **0.572** |
| Bundesliga held-out, 3 s control rule | 0.544 | 0.649 | 0.656 | **0.708** |
| Bundesliga held-out, 1 s rule, 3× ball noise | 0.441 | 0.528 | 0.526 | **0.594** |
| Bundesliga held-out, macroF1 (1 s rule) | 0.801 | 0.868 | 0.870 | **0.901** |
| Same matches, ball frozen in stoppages as recorded, 1 s rule | 0.506 | 0.615 | 0.634 | 0.645 |
| Metrica sample matches as recorded, raw event answers\* | 0.534 | 0.611 | 0.617 | 0.618 |

\*Models trained on all seven Bundesliga matches and no Metrica match.

**What changed from v9 to v10.** v9 is the same two classifiers trained on the ball as recorded (frozen in stoppages), with turnover snapping. The two changes, measured separately on the first row (`python -m possession.stoppagelab bench`):

- **Training on the simulated stoppage ball: 0.589 → 0.613.** On those matches v9 finds 63% of the truly dead frames where the ball is visible and places 68 possessions inside dead time. The retrained models find 92% and place 23. Where the old assumption holds (ball frozen), the retrained models give up 0.009.
- **No turnover snapping: 0.613 → 0.635.** The step was written for the hand-written rules. Without it, every comparison we ran improved by 0.02 to 0.04.

Each step on its own is below the scorer's stated noise band of 0.04 (together they add 0.046). We adopted the first because it removes a known artefact of the training data and passed the checks below, and the second as a judgement call.

**Provided game, checks that need no ground truth.** Each one hides something from the model and measures how well the labels recover it (`python -m possession.gate`).

| Check | v9 | v10 |
| :---- | :---- | :---- |
| Ball hidden for live-play-length spans: team unchanged | 82.2% | 84.8% |
| Phase tags withheld: labels agree with them | 75.3% | 77.3% |
| Set-piece tags withheld: dead ball at the tag | 89.7% | 94.8% |
| Transition tags withheld: a turnover within 5 s | 79.2% | 91.7% |
| Transition tags withheld: a turnover within 2 s | 41.7% | 37.5% |
| Transition tags withheld: median distance to our nearest turnover | 2.7 s | 3.1 s |
| Video ball withheld: agrees with clear video-ball control | 88.1% | 88.0% |

The last three rows did not improve, and the 2 s turnover rate got slightly worse. Earlier versions (v6, v7) are in `possession/NOTES.md`.

**Provided game, dead ball where the tracking still shows a ball.** Set-piece tags and the goal rule withheld (`python -m possession.probe deadchecks`).

| Share of frames labelled dead | v9 | v10 |
| :---- | :---- | :---- |
| Goal to kickoff, ball seen (301 frames; dead by the Laws) | 42% | 63% |
| Goal to kickoff, ball not seen (11,282 frames) | 100% | 100% |
| 8 s before goal-kick, corner and free-kick tags, ball seen (1,206 frames) | 8% | 23% |
| Same, ball not seen (4,127 frames) | 99% | 97% |
| Ball seen well inside the pitch, away from tagged set pieces and long ball gaps (22,241 frames; mostly live play, though untagged free kicks fall in it) | 8% | 10% |

On the full match the submission finds 292 possessions (2.8 per minute) and 129 stoppages, and labels 45.9% of graded frames dead (v9: 315 possessions, 145 stoppages, 42.8% dead).

## 6. How we used the public 5 minutes

Only as a pass/fail sanity check after a version had already been chosen on other data; never to choose between versions or set a parameter. For transparency, the submitted v10 scores spellF1 0.495 and macroF1 0.792 there (21 possessions found, 17 true). v9 scored 0.478 and 0.713, v7 0.514, v6 0.489. With 17 possessions these differ by one matched possession, and the scorer itself reports that gaps below about 0.36 are within noise in this window.

Two further uses, both disclosed here:

1. The low dead-ball recall of v7 to v9 in this window (37%) was the reason we went looking for a fault in the dead-ball training data. The fault (the frozen stoppage ball) was found and fixed using the other matches (section 3), and recall in the window is 58% for v10. The window therefore prompted the investigation, but no parameter was set from it.
2. We shifted the public ground truth against itself to confirm that the `EVALUATION.md` table quoted in section 3 describes the full match and not this window.

## 7. Limitations

- **Dead ball while the ball is visible is still under-called.** v10 labels only 63% of the seen-ball frames between a goal and the kickoff as dead, and 23% of those just before slow set pieces. The ball the tracking shows in dead time on the provided game moves far more (60% of frames at 3 m/s or more) than our simulated stoppage ball (7%), so the simulation is a partial fix. The second of those two windows probably also contains some live play.
- **Dead-ball figures on the other matches are optimistic.** On the held-out matches v10's dead-ball frame recall is 97% and precision 97%, but that is against our own simulated stoppage ball. Versions up to v9 reported 98%, which was inflated by the frozen ball.
- **Turnover timing is imprecise.** Withheld transition tags lie a median 3.1 s from our nearest matching turnover. That is a distance, not a direction (it does not say early or late), and the tags are automatic and coarse themselves. A boundary 1 s off already costs about a third of a possession's score.
- **Short possessions.** The provided game appears to have many possessions under 3 s (section 3). The decoder's minimum is 3 s, so those are missed or merged. Lowering the minimum did not help on the other matches.
- **Possible over-splitting.** In the public window v10 finds 21 possessions against 17.
- **The video ball covers only 38% of the gaps**, and a ball in the air breaks the ground-plane projection.
- **Our control rule is a guess** at the challenge's definition.
- **Kickoff.** We place the second-half kickoff from the tags, 82 frames from the frame given in the FAQ. We did not use the FAQ frame because it reads as a human observation about the provided game.
- **Run time and portability.** About 2.5 h on Apple Silicon, almost all of it video ball detection. Byte-identical reruns were verified on one machine (models retrained from scratch, tracking cache rebuilt, detections reused). A different machine or GPU backend was not tested.

**Tried and not adopted** (numbers in `NOTES.md`): restart-geometry templates, a ball-free model, whistle detection, touch detection, a learned boundary refiner, stoppage merging, decoder re-tuning, larger and ensembled classifiers, training on a mix of stoppage-visibility rates, training on raw or 0.5 s answers, a shorter minimum possession, and using the richer classifier for dead ball as well.

**With more time:**

1. A stoppage ball modelled on how the tracked ball really behaves in dead time.
2. A dead-ball model that leans less on ball motion.
3. A ball detector fine-tuned for small balls in tactical footage.

## 8. Reproduction

Python 3.14. Dependencies are pinned in `possession/requirements-video.txt`.

```sh
pip install -r requirements.txt -r possession/requirements-video.txt
python -m possession.run --out Golden_Bear_Analytics.csv
python evaluation/validate_submission.py Golden_Bear_Analytics.csv
```

The one command downloads the pinned public training matches and detector weights (hash-checked), trains the classifiers with fixed seeds, runs ball detection on the tactical video, and writes the CSV. The first run takes about 2.5 h, so keep the machine awake. The provided data goes in `data/` as described in `DATA.md`. No data, detections, model weights or label files are part of the repository.

Tests (synthetic, no match data): `python -m unittest discover -s tests`.
