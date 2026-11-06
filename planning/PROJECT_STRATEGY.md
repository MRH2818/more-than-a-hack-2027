# Project strategy

Planning date: **November 5, 2026**. This document proposes future work; no implementation is included. Read the [competition analysis](/Users/danieleitan/Desktop/more-than-a-hack-2027/planning/COMPETITION_ANALYSIS.md) for the complete requirements and the [technical architecture](/Users/danieleitan/Desktop/more-than-a-hack-2027/planning/TECHNICAL_ARCHITECTURE.md) for implementation contracts.

## Decision

Build a **tactical-only, offline possession segmenter** that distinguishes missing observations from dead ball, combines team control evidence with cautiously weighted automatic tactical tags, and reconstructs coherent `H` / `A` / `D` intervals. Establish a deterministic state-machine MVP first, then add a three-state Viterbi decoder and evidence-based boundary refinement if their validation and schedule gates pass.

This combines **Concept A's reliable execution** with **Concept C's focus on possession boundaries**. The competitive target uses no learned model, external training dataset, GPU, API, cloud service, broadcast alignment, or custom hardware. The supplied tactical tracking and Smart Tagging are required inputs for the full target; their actual availability must be verified. Tactical video and homography support local diagnosis and may be omitted from the inference path.

The strategic objective is to recover whole possessions under the actual ranking metric. It is **not possible to establish that this proposal beats the published full-match `spellF1` floor of 0.2433 on the competition match before the organisers evaluate it**. Permitted independent evaluation could support an expectation of improvement, but would not reveal this match's result. The score is a benchmark and ambition, not an acceptance criterion that can be measured on the private match during development.

## What can actually be selected

There is **one documented technical challenge**: Phase 1, frame-level ball possession. Each submission labels all 199,319 tactical-video frames, with 158,857 frames graded across the two halves. The real decision is which approach to that challenge offers the strongest feasible submission. There are no alternative technical challenge briefs in the supplied repository.

`RULES.md` mentions an **Innovation Challenge** and a **Final Pitch** in its team-eligibility paragraph, but supplies no problem statements, judging rubrics, delivery contracts, or schedules for them. They are **unrankable pending further documentation**, rather than second and third documented challenges. The 2026 open-ended format described in `OVERVIEW.md` is historical context, not an available 2027 option.

The documented Phase 1 ranking is lexicographic: `spellF1`, then `macroF1`, then transition F1 at the loose tolerance, mean absolute transition latency, and filename. The score is computed automatically. Originality, hardware, presentation, use of AI, and the technical document receive no separate judging points. Reproducibility and compliance are eligibility requirements.

Sources: [CHALLENGE.md — The task; Deliverables and scoring](/Users/danieleitan/Desktop/more-than-a-hack-2027/CHALLENGE.md), [OVERVIEW.md — Objective; Evaluation criteria](/Users/danieleitan/Desktop/more-than-a-hack-2027/OVERVIEW.md), [RULES.md — Eligibility and teams](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md), and [run_evaluation.py — rank_key](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/run_evaluation.py).

## Why this problem rewards temporal reasoning

The automatic ball track is absent on approximately **41% of graded frames**. Track slots are not stable player identities: 46 slots represent 22 players, with fragmentation, duplication, and identity switches. Smart Tagging supplies 638 automatic intervals, but they describe phases and set pieces, not exact possessions. Every throw-in has `Team=N/A`; defensive phases refer to the defending team. The broadcast recording uses a separate, unsynchronised timeline and has usable homography on only 65.8% of its frames.

These constraints favor a team-level temporal method over a player-identity system or per-frame nearest-player classifier. The governing football rule is that a loose or contested ball remains with the previous possessor until the opposing team gains control. A touch, deflection, or aerial-duel contact alone does not establish control. A dead-ball interval separates possessions even when the same team restarts.

The baseline makes the opportunity concrete. It assigns the nearest visible player's team and otherwise records no team observation. Its smoothed version counts home and away observations in a centered window, and outputs `D` only if **neither team has any observation in the window**. It is not a three-class majority vote. `baseline_smooth50` uses a half-width of 50 frames, therefore a full window of 101 frames, approximately 4.04 seconds. This reduces fragmentation but still confounds missing-ball evidence with dead ball and does not use the tags.

The proposed improvement is therefore structural: treat **observation quality, controlled possession, stoppage detection, and transition timing as distinct decisions**, then resolve them jointly. Adding a more expensive detector or language model before solving those distinctions is unlikely to be the best use of the available time.

Sources: [DATA.md — Automatic Tracking Data; Smart Tagging; The broadcast recording](/Users/danieleitan/Desktop/more-than-a-hack-2027/DATA.md), [CHALLENGE.md — Possession rules](/Users/danieleitan/Desktop/more-than-a-hack-2027/CHALLENGE.md), and [starter/baseline.py — nearest_player_labels; smooth](/Users/danieleitan/Desktop/more-than-a-hack-2027/starter/baseline.py).

## Weighted decision matrix

The following weights and scores are **strategic recommendations, not official judging weights or measured model results**. The official scorer has an ordering of metrics, not an eight-factor weighted rubric. “Judging performance” here means expected potential to improve the automated ranking. Demonstration quality receives a small planning weight because a clear diagnostic demonstration helps engineering and communication; it earns no Phase 1 points.

Ratings are ordinal, from 1 (weak) to 5 (strong). For implementation risk, **5 means lower risk**. The weighted total is `sum(weight × rating) / 100`. Precision to two decimals makes the arithmetic reproducible; it does not imply that qualitative judgments are accurate to hundredths.

| Criterion | Weight | Basis for the recommendation |
|---|---:|---|
| Technical feasibility | 20% | A complete algorithmic submission with available inputs is necessary. |
| Innovation and originality | 5% | Useful when it addresses the actual failure modes; unscored by itself. |
| Engineering sophistication | 5% | Reward substantive control of uncertainty and reproducibility, not component count. |
| Potential judging performance | 35% | Largest share because private full-match `spellF1` determines rank. |
| Demonstration quality | 2% | Useful for diagnosis and explanation; no official presentation score. |
| Low implementation risk | 15% | Missing data, unsupported labels, and unreproducible outputs can invalidate the effort. |
| Time efficiency | 13% | 12 calendar days remain until the dated deadline. |
| Extensibility and real-world impact | 5% | Generic team-level inference can transfer beyond this match, if later validated. |
| **Total** | **100%** | **All weights are strategic.** |

| Rank | Direction within the single official challenge | Feasibility | Innovation | Depth | Judging | Demo | Low risk | Efficiency | Impact | Total |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Selected A+C: state-machine foundation plus capped evidence fusion and offline Viterbi** | 4 | 4 | 4 | 5 | 4 | 4 | 4 | 4 | **4.35** |
| **2** | **A: deterministic control/stoppage state machine** | 5 | 2 | 3 | 3 | 3 | 5 | 5 | 3 | **3.91** |
| **3** | **B: learned emissions and explicit-duration semi-Markov model, conditional on suitable other-game labels** | 2 | 4 | 5 | 4 | 4 | 2 | 2 | 5 | **3.14** |
| 4 | Reproduce the published smooth50 baseline | 5 | 1 | 1 | 1 | 2 | 5 | 5 | 2 | **2.99** |
| 5 | Tactical/broadcast alignment and cross-view fusion | 2 | 4 | 5 | 3 | 4 | 1 | 1 | 4 | **2.46** |
| 6 | Rebuild ball/player detection and tracking from video | 1 | 4 | 5 | 2 | 5 | 1 | 1 | 4 | **1.93** |

The score of 5 for the selected approach's judging potential is a **comparative hypothesis**, not a forecast of winning or of a particular `spellF1`. Its advantage is that it attacks fragmentation and stoppage errors using already documented inputs. A learned model may have the higher ultimate ceiling, but its data and calibration prerequisites are not currently met.

### Sensitivity and when the recommendation changes

- If delivery risk dominates, reduce judging weight from 35% to 20% and add five percentage points each to feasibility, low risk, and time efficiency. Concept A becomes **4.21**, narrowly ahead of the hybrid's **4.20**. This supports making A the protected fallback rather than a discarded prototype.
- If the hybrid's judging-potential rating falls from 5 to 4, its original-weight total becomes **4.00**, still slightly above A's 3.91. If its low-risk rating also falls from 4 to 3, it drops to **3.85** and A should be preferred.
- If suitable independent labels, substantial model experience, and compute are verified early, reassess B. Do not promote it solely because it has more sophisticated mathematics.
- A missing tactical data bundle is a blocker for all new match-specific methods. The committed baseline remains a pipeline demonstration, but copying it is not a competitive substitute for access to the actual inputs.

The top two competitive alternatives are therefore the **fully executable deterministic A** and the **conditional learned B**. Broadcast fusion is a later research direction, not the main fallback.

## Evaluation of the candidate directions

| Direction | Intended outcome and engineering opportunities | Required resources | Main bottleneck | Distinguishing value and demonstration | Schedule assessment |
|---|---|---|---|---|---|
| Selected A+C | Recover coherent team possessions with separate stoppage reasoning; fuse noisy sources and localize boundaries offline. | Tactical ATD metadata/raw file; Smart Tagging; Python, NumPy, pandas. Video/homography optional for local inspection. | Dead ball is only indirectly observed; no independent possession labels are confirmed. | Show an occlusion that does not create a false stoppage, then a controlled handover and a real restart. Explain each boundary from evidence. | Best balance if A is complete by Nov 8 and advanced work freezes by Nov 12. |
| A, deterministic state machine | Establish a complete, understandable algorithm with persistence, hysteresis, and restart handling. | Same tactical inputs; no training or GPU. Tracking-only degraded mode if tagging is unavailable. | Generic thresholds may miss quick valid turnovers or long unobserved stoppages. | Transparent behavior and easy reproducibility. Small causal implementation helps isolate errors. | Highest delivery confidence; 70–90 person-hours estimated. |
| B, learned semi-Markov | Learn nonlinear control evidence and possession-duration behavior from other games, then decode spells globally. | Accessible and compatible labeled other games; documented provenance and license; enough engineering time. GPU only if a chosen model actually requires one. | Most public event labels are not automatically equivalent to frame-level `H/A/D`; label access and semantic conversion remain unverified. | Strong potential if calibrated across games; can compare emissions, durations, and boundaries with independent labels. | 220–320 person-hours after usable assets are available; exceeds the assumed 144-hour capacity. |
| Published baseline | Produce valid output from nearest-player observations and fixed temporal smoothing. | Tactical ATD for regeneration; repository CSVs already exist. | Missing ball maps toward `D`; no actual control or restart reasoning. | Valuable regression reference and emergency fallback; little differentiation. | Fastest reliable reference. Do not choose a different smoothing window from public scores. |
| Broadcast fusion | Recover evidence where tactical tracking fails by aligning another view. | Both streams, their tracking/homography, automatic shot/replay handling and alignment. | Alignment can confidently match the wrong moment or replay; no provided mapping. | Visually compelling cross-view demonstration, but alignment errors propagate directly to labels. | Too many coupled unknowns for the default plan. Consider only after the competitive target is complete. |
| Rebuilt computer vision | Improve ball observations and optionally player detection before possession inference. | Local video, licensed pretrained models or suitable other-game training data, verified compute, inference tooling. | Small, fast, occluded ball; camera motion; detector errors; annotation constraints; training/inference time. | Useful only if missing observations are recovered reliably enough to improve downstream spells. | Highest setup and validation burden. No evidence currently supports making this the critical path. |

The absence of a live leaderboard means expensive approaches cannot be justified by hoped-for private-score feedback. Their adoption needs a permitted independent evaluation or a narrow, demonstrable engineering improvement whose limits are stated.

## Concept A — Reliable: deterministic possession state machine

### Product and user experience

A local command reads the tactical assets, runs a deterministic state machine over each half, and writes the required full-length possession CSV. A compact local diagnostic report shows the output state, ball visibility, team control evidence, and candidate stoppages. There is no web backend or interactive annotation workflow.

The demonstration shows three behaviors: missing observations preserve an existing possession when no contradictory evidence exists; a fleeting opposition touch does not automatically cause a handover; and supported dead-ball evidence divides two possessions even when the same team resumes.

### Mechanisms and architecture

1. Parse ATD as a stream, map `frame_id - 1` to tactical frame, and map slots to teams using metadata.
2. Convert normalized geometry to meters using the repository's pitch convention. Preserve missing values and observation-quality masks.
3. Compute per-team proximity, nearest-opponent margin, ball motion on valid short observation runs, and aggregate team movement. Do not treat a slot as a player identity.
4. Convert offensive phase tags into weak same-team evidence and defensive phase tags into weak opposing-team evidence only where code semantics are verified; otherwise disable that family. Ignore unknown team values. Treat set pieces as restart neighborhoods, never exact restart instants or complete dead-ball intervals.
5. Use states `H`, `A`, and `D`, with control-evidence hysteresis. Retain the previous live owner through loose-ball or missing-ball intervals. Enter `D` only with affirmative combined evidence; missing ball alone is insufficient.
6. Confirm a transition with sustained evidence, then use a bounded offline look-back to locate its likely onset rather than writing every transition late.
7. Produce `X` outside graded halves. Within graded halves, write the estimated state; uncertainty is diagnostic metadata, not a promise that abstention avoids the ranking penalty.

Tools: Python with NumPy and pandas; standard-library XML parsing; reuse the supplied parsers and validator where appropriate. Optional local Matplotlib diagnostics are helpful but not required. Inputs, outputs, and runtime contracts remain simple arrays and files. No hardware is required beyond an available general-purpose computer, whose capacity must be checked.

### Risks, effort, and tradeoffs

Estimated **70–90 person-hours**, assuming three experienced contributors and prompt data access. Complexity is moderate. The main risks are incorrect evidence interpretation, threshold sensitivity, and dead-ball ambiguity. The state machine must not impose a hard minimum possession duration: true possessions can be short, and deleting them to reduce fragmentation can lower `spellF1`.

A is easy to audit and debug, provides a complete fallback, and has no training dependency. It cannot fully resolve long occlusions or whistles that are not visible in geometry. It can also accumulate causal delay unless confirmation and boundary placement are separated. It is less flexible than C when different sources disagree across an interval.

## Concept B — Ambitious: learned emissions with explicit-duration decoding

### Product and user experience

The same local batch interface produces the same CSV, but calibrated learned evidence feeds a semi-Markov decoder that explicitly scores possible state durations. Local diagnostics show whether a boundary is supported by geometry, a learned emission, or a duration preference. A demonstration would include an independent other-match evaluation and an ablation of learned emissions and durations.

### Mechanisms and prerequisites

The preferred form uses an accessible other-game dataset with compatible possession and dead-ball labels. Train a compact tabular model, such as logistic regression or gradient-boosted trees, on team-level geometric and tagging features. Fit duration distributions and transition behavior on that independent data, then decode intervals with dynamic programming. A small temporal neural model is an alternative only if its additional training and reproducibility costs are justified by permitted evidence.

The feature schema and state alphabet should match A so the learned component can be removed without breaking ingestion or export. Possible additional libraries include scikit-learn or a model-specific library, but their exact versions, model access, license, and reproducibility must be verified before adoption. The deterministic target does not depend on these tools.

No compatible labeled dataset is currently confirmed. The public [Metrica sample-data collection](https://github.com/metrica-sports/sample-data) **is an external feasibility lead**, not an assumed source of ready-to-use frame-level possession truth. Its specific files, event semantics, dead-ball coverage, accessibility to organisers, and permitted uses require verification by **November 6**. Event data alone must not be relabeled as frame-exact possession ground truth without a documented, defensible conversion and known limitations.

If supervised labels are unavailable, the rules permit unsupervised, self-supervised, or purely algorithmic pseudo-label learning on the supplied match. A possible experiment would initialize high-confidence control intervals from generic geometry and fit a restrained observation model without human decisions about individual intervals. However:

- Pseudo-label agreement is not independent accuracy evidence.
- Learning durations from a flickering initial system can encode its errors.
- An unconstrained model can collapse toward long, easy states or exchange state meanings.
- Observed-data likelihood and synthetic success do not prove better possession recovery.
- No public-GT statistic, score, boundary, duration, or possession share may calibrate the experiment.

This unsupervised route is an **optional research fallback within Concept B**, not the fallback for delivering the competition entry. If it lacks permitted evidence of benefit, revert to fixed generic transition regularization in A+C; do not present unsupervised parameter fitting as ground-truth calibration.

### Risks, effort, and tradeoffs

Estimated **220–320 person-hours once usable data exists**, excluding a substantial new annotation campaign. Complexity is high. The design has meaningful engineering depth, but the time budget is insufficient by default. Additional risks include label-definition mismatch, domain shift, data licensing, insufficient independent matches, and exact reproduction of trained checkpoints.

B's advantage is potential calibration and cross-game generalization. Its disadvantage is that the resources needed to justify it are unverified. No hard minimum spell duration should be introduced even here: duration preferences must allow short, well-supported possessions. Any duration model must have a documented origin independent of the competition's human labels.

## Concept C — Differentiated: boundary-focused evidence fusion

### Product and user experience

An offline segmenter concentrates on the moments that create or destroy possessions: controlled changes, stoppages, and restarts. It includes a local evidence timeline explaining why each inferred boundary exists and where uncertainty remains. The explanation is a diagnostic artifact for engineers; it is not an extra output state and is not an officially scored deliverable.

The distinctive demonstration is a short sequence containing an occlusion, an opponent contact without control, a real controlled turnover, and a restart. The timeline shows how the method preserves continuity in the first two cases while accepting the latter two changes. Use synthetic examples or properly permitted external material for public presentations; competition video, tracking exports, or reconstructable derivatives must not be published without the applicable permission.

### Mechanisms and architecture

Use a three-state Viterbi decoder over each full graded half:

`best sequence = argmax(sum of state evidence − sum of evidence-dependent transition penalties)`.

State evidence contains separate terms for team control, dead-ball support, and weak tag support. Missing observations reduce confidence rather than voting directly for `D`. Conflicting or overlapping automatic phase tags are capped so many correlated tags cannot overwhelm independent geometry. Treat numerical evidence scores as **scores**, not calibrated probabilities unless a valid calibration procedure exists.

Transition penalties discourage unsupported flicker while allowing a short spell with strong evidence. They can decrease near supported control changes and restart neighborhoods. The decoder considers every frame; candidate boundary signals do not prohibit transitions elsewhere. An optional, separately gated bounded local refinement uses available evidence around an inferred boundary to improve timing, without arbitrarily shifting it by a public-GT-derived offset.

Global offline context is legitimate because the competition requests a file for a completed match and imposes no real-time requirement. The three-state dynamic program is small; the engineering difficulty lies in defensible evidence and reliable data alignment, not large-scale computation.

Inputs and technology match A. The public code can include the state definitions, evidence mapping, fixed configuration, and deterministic decoding rules. Detailed per-frame diagnostics remain local because they are derived from restricted match data.

### Risks, effort, and tradeoffs

Standalone estimate: **150–190 person-hours**. Complexity is moderate to high. Failure modes include over-smoothing, using tag boundaries as truth, over-weighting correlated signals, and overconfident dead-ball decisions. Removing hard duration floors and retaining a working A implementation limits these risks.

C uses the structure of the official metric directly: possession intervals and boundaries matter. The novelty is a well-matched engineering synthesis for this dataset, not a claim of a new segmentation algorithm or a literature-first result. Its diagnostic evidence makes mistakes explainable, but explanation alone does not improve the score.

## Selected strategy: A first, A+C only through explicit gates

| Dimension | Reliable A | Ambitious B | Differentiated C | Selected A+C |
|---|---|---|---|---|
| Core inference | Deterministic state machine | Learned emissions plus explicit-duration decoding | Capped evidence with offline sequence decoding | A as complete fallback; C as a gated decoder upgrade |
| Human labels required | None | Suitable other-game labels for the preferred supervised form | None | None |
| External data required | None | Yes for defensible supervised calibration | None | None |
| Specialized hardware | None | Only if a justified model requires it | None | None |
| Main strength | Delivery confidence | Potential learned generalization | Interval and boundary reasoning | Competitive depth with a reversible path |
| Main weakness | Threshold and causal-delay limitations | Resource and validation dependency | More integration and evidence-design work | Cannot prove private-score improvement without independent evidence |
| Estimated effort | 70–90 person-hours | 220–320 person-hours after data access | 150–190 person-hours | **120 planned person-hours (reduced scope)** |

The twelve-day plan reduces the original hybrid scope: no separate refinement module by default, no learned/video/broadcast research, and minimal diagnostics. The full C estimate above remains larger; the selected plan delivers a narrower A+C subset.

The capacity assumption is **three people × four focused hours per day × November 5–16 = 144 person-hours**, with **120 planned and 24 reserved**. This is an assumption, not known team availability or an official hackathon-duration limit. If actual capacity is lower, preserve A and reduce the advanced scope first. Confirm team composition separately because the rules state three to five members and contain an additional qualification concerning teams below three.

Milestones are shared with the roadmap: **MVP by November 8; method freeze November 12; code freeze November 14; upload target November 15; contingency November 16**. The official deadline is November 17, 2026, but its exact time and timezone are unspecified. These internal dates do not establish or extend the official cutoff.

### Feature tiers

| Tier | Included | Deliberately conditional |
|---|---|---|
| **MVP** | Correct tactical frame contract; validated team mapping; deterministic state machine; missingness-aware continuity; basic tag-supported restart/dead-ball handling; exact CSV regeneration; official format validation; initial technical write-up. | No learned model, UI application, broadcast alignment, or replacement tracking. |
| **Competitive target** | Capped tag fusion; three-state Viterbi; evidence-dependent transition costs; optional boundary refinement only if independently justified; robust synthetic and metamorphic testing; deterministic full-run reproduction; clear limitations and provenance. | Adopt advanced decoding only if its engineering checks pass before the freeze. |
| **Stretch** | Permitted independent validation; narrow local video features or tracking recovery; carefully controlled learned-emission experiment; stronger diagnostics. | No stretch item may delay a valid, reproducible entry or require unavailable labels/compute. |

### Decision gates

1. **By November 6: data and execution gate.** Verify the licensed tactical bundle, full frame sequence, team mapping, terms, team eligibility, and a reproducible baseline path. If assets are unavailable, escalate access and focus only on generic contracts and synthetic tests; do not pretend new match inference can proceed.
2. **By November 6: external-label gate.** Either demonstrate accessible other-game data with suitable semantics and a feasible evaluation protocol, or remove learned calibration and external-data-dependent claims from the delivery path. A promising dataset website is not enough.
3. **By November 8: MVP gate.** A must generate all 199,319 rows, pass the official validator, obey half boundaries, and pass synthetic continuity/restart tests. If it cannot, suspend advanced work until it can.
4. **By November 11: decoder gate.** The C decoder must preserve short, strongly supported spells, avoid false D from missingness alone, handle conflicting tags, and reproduce outputs deterministically. Passing these tests proves engineering behavior, not private-match superiority. Any independent quality comparison must use permitted other-game labels.
5. **By November 12: method freeze.** Select the simplest method justified by permitted evidence and predeclared checks. If independent labels remain absent, use conservative predeclared settings and record that quality gains are unverified. Do not choose between versions using the public window.
6. **By November 14: reproduction gate.** Exact regenerated CSV bytes, packaging, pinned dependencies, input provenance, and technical-document instructions must agree. A failed advanced pipeline falls back to the already reproducible A implementation; the published baseline remains the last-resort reference.

## Validation that respects the rules

The supplied five-minute ground truth is for **understanding label semantics and checking pipeline correctness**, such as alignment and file handling. It must not rank concepts, choose thresholds or smoothing windows, fit durations, select checkpoints, compare ablations for adoption, or choose the final submission. The same prohibition applies to decisions based on its possession counts, mean duration, or possession share.

Freeze an approach and its configuration before a limited public sanity check. Treat a discovered indexing or parsing bug as a pipeline correction, document the cause, and avoid turning residual public-score differences into feature selection. Do not create a hidden validation set by manually labeling a few transitions from the provided match; no amount is allowed. Do not import human-produced event or possession data about this match from any source.

For quality comparisons, the preferred route is permitted, accessible labels from other matches, with matched possession semantics and game-level train/validation separation. If those do not exist, rely on **synthetic tests, transformation invariance, input-quality checks, deterministic reproduction, and explicitly limited unlabeled diagnostics**. These can establish that a method behaves as designed; they cannot establish its real-match `spellF1`.

Synthetic tests should include genuine short possessions, long possessions, same-team restarts separated by D, asymmetric and bursty missingness, gaps centered on turnovers, transient opponent touches, conflicting tags, unknown throw-in teams, slot permutation, team swapping, and half boundaries. Do not tune toward a synthetic duration distribution copied from the public match statistics. Do not use a reduced predicted possession count as proof of accuracy: a constant label has very few boundaries and can score disastrously.

Inspection of supplied video is permitted for understanding and generic feature design. It must not produce hand corrections, frame-specific constants, or rules that encode decisions about particular moments. Every submitted label must come from the reproducible algorithm.

Sources: [RULES.md — What you may not do; Training and validation data](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md), [EVALUATION.md — The public window is not a leaderboard](/Users/danieleitan/Desktop/more-than-a-hack-2027/EVALUATION.md), and [FAQ.md — Can I hand-label some frames?](/Users/danieleitan/Desktop/more-than-a-hack-2027/FAQ.md).

## Competitive differentiators and their limits

| Differentiator | Why it matters to the actual score | Evidence required before making a performance claim |
|---|---|---|
| Missing ball is treated as uncertainty, not automatic D | False stoppages split real possessions and reduce one-to-one spell matches. | Synthetic occlusion tests; independent other-game spell evaluation if available. |
| Controlled changes are distinguished from touches | Avoids flicker from nearest-player switches and contested balls. | Short controlled-turnover and non-control-contact tests; compatible independent labels. |
| Separate dead-ball and restart reasoning | Same-team possessions across a stoppage must remain separate. | Restart tests and independent dead/alive evaluation; acknowledge invisible whistles. |
| Capped, semantically correct Smart Tagging fusion | Uses a supplied source without treating its automatic phase intervals as truth. | Defensive-team mapping, N/A, overlap, and restart-window tests. |
| Offline boundary placement | Confirmation can be conservative without imposing equal delay on every transition. | Boundary-offset tests and transition metrics on permitted independent labels. |
| Deterministic, inspectable output | Makes the score reproducible and satisfies the submission contract. | Clean regeneration yielding the same CSV hash. |

The most dangerous threats are **unavailable inputs**, **no defensible independent quality validation**, and **misclassifying stoppages under missing observations**. The plan reduces them through early access checks, explicit limits on what testing proves, a protected A implementation, and early dead-ball/restart work. It does not hide them behind a dashboard or a more elaborate model.

The best public explanation is concise: this system reconstructs **possession intervals from incomplete evidence**, uses the provided assets carefully, and can reproduce every submitted byte. A polished local visualization supports that explanation only after the batch system works. No presentation feature should displace the critical path to the valid CSV, public code, and short technical document required by [SUBMISSION.md — The three deliverables](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md).
