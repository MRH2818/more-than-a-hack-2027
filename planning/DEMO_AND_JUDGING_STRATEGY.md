# Demo and judging strategy

Planning date: **November 5, 2026**. This document describes a future demonstration and evidence package; no system or results have been produced in this planning session.

## What actually determines success

**Phase 1 has no jury or presentation score.** The possession CSV determines the ranking; the public source repository and short technical document establish reproducibility. A live demonstration is optional communication, not a fourth mandatory deliverable. The references to an Innovation Challenge and Final Pitch do not provide enough information to plan a later-round pitch as an official requirement. [OVERVIEW — evaluation criteria](/Users/danieleitan/Desktop/more-than-a-hack-2027/OVERVIEW.md:207), [SUBMISSION — deliverables](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md:45).

The objective is therefore a technically convincing **working inference pipeline**, with a small explanation of why its decisions fit the metric. Spend approximately one person-hour on the final optional walkthrough, after the release is ready; diagnostic tooling has a separate small engineering allocation in the roadmap. Do not build a dashboard application or pitch deck before the CSV can be regenerated.

## Alignment with the actual evaluation

| Criterion / obligation | Feature that addresses it | Evidence to present | Limit on the claim |
|---|---|---|---|
| Primary: spellF1 | Complete H/A intervals; continuity through observation gaps; supported control changes; correct stoppage separation | Official spellF1 on permitted independent labels, if available; synthetic examples of metric behavior | Hidden competition score is unknown until organizers evaluate |
| First tie-break: macroF1 | Accurate per-frame H/A/D classification; avoid unnecessary graded X | Per-class F1 and confusion matrix on permitted labels | A high frame score does not establish good possession segmentation |
| Further code tie-breaks | Transition timing without unnecessary smoothing lag | Transition F1 at ±25 frames and mean absolute latency | Secondary to spellF1/macroF1; not a separate weighted judging rubric |
| Mandatory reproducibility | Deterministic local inference, pinned environment, canonical writer | Clean run command, matching output SHA-256, validator pass | A valid repeatable file may still have low prediction accuracy |
| Mandatory algorithmic provenance | No manual corrections, no competition-match human labels used for fitting | Input/provenance table, configuration origin, no-GT inference audit | Public label semantics checks do not authorize tuning |
| Documentation | Honest rules, assets, training data, limitations and reproduction | Short technical document with exact command | Documentation is required but unscored |

The actual tie order is descending spellF1, descending macroF1, descending transition F1 at the default loose tolerance of 25 frames, ascending mean absolute transition latency, then filename. It is not a weighted average. [run_evaluation — `rank_key`](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/run_evaluation.py:165).

For each IoU threshold 0.3, 0.4, 0.5, 0.6 and 0.7, the scorer greedily matches same-team overlapping spells one-to-one in descending IoU order. The average of the five F1 values is spellF1. False D can split a real possession; missed D can merge two real possessions. Excessive smoothing can therefore hurt even if the output looks cleaner. [possession_metrics — spell matching](/Users/danieleitan/Desktop/more-than-a-hack-2027/evaluation/possession_metrics.py:464).

## The central explanation

“The ball disappears from the supplied tracking on about 41% of scored frames. Our system separates missing observations from stoppages, combines team-control and automatic restart evidence, and reconstructs possession intervals. Every output label comes from a repeatable local algorithm.”

This is a proposed explanation, not a claim that the system already works. The novelty is the integration of missingness-aware control evidence, separate live/dead reasoning, capped automatic tags and evidence-dependent temporal decisions. Viterbi decoding itself is established; do not claim a new algorithm or literature-first invention. The proposal is differentiated from the provided nearest-player baseline by how it interprets uncertain observations and boundaries, not by adding fashionable technology.

The strongest engineering decisions to explain are:

1. Tactical frame IDs are authoritative, with strict full-sequence checks and no broadcast-clock assumptions.
2. Missing ball is uncertainty, not automatic D; nearest player is a cue, not proof of control.
3. Automatic phase tags are weak correlated observations, with explicit defensive-team semantics and N/A handling.
4. Temporal penalties suppress unsupported flicker while preserving genuinely short possessions.
5. A simpler state-machine route remains executable if advanced decoding cannot be justified.
6. The model-selection process respects the ban on public-GT tuning.

## Optional three-minute walkthrough

This duration is a **team recommendation**, not an official presentation limit. Adapt only if later-round instructions are supplied.

| Time | Narrative / action | Essential evidence |
|---|---|---|
| 0:00–0:25 | State the exact task and show the CSV contract | All 199,319 tactical frames; H/A/D/X; explain whole-possession scoring |
| 0:25–0:55 | Show a synthetic occlusion during live possession | Ball disappears visually; observation-quality signal falls; possession stays continuous without an invented D |
| 0:55–1:25 | Show a transient touch followed by a controlled turnover | Touch alone does not switch; strong short control can switch; explain boundary evidence |
| 1:25–1:55 | Show a stoppage and same-team restart | H→D→H is two possessions; tag window is broader than inferred restart |
| 1:55–2:25 | Show actual validation and runtime evidence, if measured | Validator pass, clean-run hash, machine/runtime, independent metrics with dataset names |
| 2:25–3:00 | Explain limitations and regeneration | Whistles/long occlusions remain hard; one command reproduces the submitted bytes |

Use synthetic cases by default for public demonstrations. They clearly explain behavior but must be labeled “synthetic,” not passed off as evidence of match accuracy. If an authorized private demonstration uses licensed match assets, select examples through generic input-quality or model-evidence criteria; viewing them must not become manual labeling, correction, or frame-specific rule design. A selected appealing clip is illustrative, not a representative evaluation.

## Minimal visuals that improve understanding

A single local diagnostic figure is sufficient:

- Top: schematic pitch with team positions and a ball marker only when observed; optionally a permitted local video overlay.
- Middle: ball-observation mask and bounded H/A/D evidence scores. Label scores as uncalibrated unless separately calibrated on lawful data.
- Bottom: predicted state timeline, automatic tag intervals and inferred transition markers.
- Side text: frame index, state, evidence reason and input quality. Do not imply a slot number identifies a player.

On synthetic fixtures, include the known scripted state timeline. On the competition match, do not fabricate a hand-annotated reference overlay or show hidden GT. Public five-minute reference use is limited to correctness/semantics; it cannot be used to select the “best” variant or tune the demo's method.

Homography maps normalized pitch coordinates to pixels; check validity, projection denominator and quality. If an overlay is unreliable, show a schematic pitch instead. Plotting must read immutable output and have no label-editing controls. Visualization does not enter the inference critical path.

## Quantitative evidence plan

| Measurement | How to obtain | Reporting rule |
|---|---|---|
| Output contract | Official validator on exact uploaded CSV | Show row count, index range, allowed labels and validation status |
| Reproduction | Two clean raw-input runs with the same frozen configuration | Show matching SHA-256, code commit and pinned environment; keep timestamp outside CSV |
| Runtime and peak memory | Measure full CPU pipeline on identified hardware | Target <10 minutes and <2 GB without video; report actual values, not targets as achievements |
| Synthetic behavior | Fixed fixtures for occlusion, touches, short spells, stoppages, swaps and corrupt inputs | Report pass/fail count; state this tests behavior, not real-match generalization |
| Other-match accuracy, if available | Compatible independently labeled matches; match-level split; official scorer | spellF1, macroF1, class F1, transition F1/latency, failure slices, sample size and label provenance |
| Component ablations | Only on lawful independent data or predeclared synthetic tests | Explain effect of removing tags/temporal logic; never adopt a variant by public-window score |
| Public-window check | Frozen method and limited correctness audit | Label as 7,500-frame correctness check; do not claim a private-score estimate or leaderboard position |
| Full-match accuracy | Organizer evaluation | Unknown until official results; published smooth50 reference is spellF1 0.2433 / macroF1 0.6724 |

The published baseline table is useful context, with **public and private results clearly separated**. Do not compare our future public score against the baseline's private score. Public smooth125 ranks ahead of smooth50 while the private ordering reverses, illustrating why public tuning is prohibited. [EVALUATION — public window](/Users/danieleitan/Desktop/more-than-a-hack-2027/EVALUATION.md:125).

Predicted possession counts/durations, tag agreement and confidence are diagnostics, not accuracy. A reduction in predicted possessions can mean fewer false fragments or more missed true spells. Neither interpretation is justified by counts alone. When independent labels exist, report per-match results and sample sizes; the scorer's 5th–95th percentile bootstrap is a 90% interval, not proof of significance or cross-match generalization. No public bootstrap comparison can be used to select models.

## Backup strategy

| Failure | Immediate substitute | What remains demonstrable |
|---|---|---|
| Video/overlay renderer fails | Synthetic/static schematic and frozen timeline | Evidence flow and possession semantics |
| Network unavailable | Offline environment and local frozen output | Entire target inference has no API dependency |
| Full rerun exceeds walkthrough time | Display previously generated CSV/hash and separate recorded runtime; offer the actual reproduction command | Honest reproducibility evidence, without pretending cached playback is live inference |
| Advanced decoder fails release checks | Frozen state-machine route, regenerated and revalidated | Complete algorithmic entry; explain omitted advanced component |
| Restricted-content sharing rights unclear | Synthetic examples only, source/schema diagrams and permitted aggregates | Method explanation without redistribution |
| No independent labels | Show engineering tests and disclose unknown accuracy | Correctness/reliability evidence only; no invented gain |

Do not post competition video clips, frame-level tracking exports, reconstructive feature caches or similar material to the public repository. Do not send restricted match material or extracts to hosted/cloud LLMs. Enrollment terms govern additional sharing rights and are not present in the checkout. [RULES — data](/Users/danieleitan/Desktop/more-than-a-hack-2027/RULES.md:100), [FAQ — hosted LLM restriction](/Users/danieleitan/Desktop/more-than-a-hack-2027/FAQ.md:101).

## Submission narrative and final checklist

The short technical document should be a few pages in PDF or Markdown, covering the actual final implementation, not every proposed stretch feature:

1. Problem, input constraints and chosen method.
2. Assets used and dropped, with reasons.
3. Team-control, loose-ball, dead-ball and restart rules; missing/noisy observation handling.
4. Every labeled dataset used, its source, annotators and quantity; explicitly state none if applicable.
5. Measured validation and its limits; clear distinction between synthetic, independent, public and hidden results.
6. Failure cases and exact clean reproduction command/environment.

Before the November 15 internal upload target, verify the **team-named CSV**, **public repository URL** and **short technical document**, correct registered team name, Google-account form access, exact final-file validation and hash. One editable response per team is allowed; the last response before the actual cutoff counts. The official deadline is November 17, 2026; cutoff time/timezone remain unconfirmed. There is no live leaderboard or feedback. [SUBMISSION — how to submit](/Users/danieleitan/Desktop/more-than-a-hack-2027/SUBMISSION.md:77).

The priority order is working inference, compliant reproducibility, correct hand-in, then optional demonstration polish.
